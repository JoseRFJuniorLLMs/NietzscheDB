//! Convert a NietzscheDB `AdjacencyIndex` into CSR (Compressed Sparse Row) format
//! suitable for upload to GPU via cuGraph or a custom CUDA kernel.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::AdjacencyIndex;

use crate::CuGraphError;

/// CSR representation of the graph.
///
/// Vertices are assigned stable `u32` indices in sorted UUID order.
/// cuGraph requires vertex IDs to be contiguous integers starting at 0.
pub struct Csr {
    /// Ordered list of node UUIDs — `node_ids[i]` is the UUID for vertex `i`.
    pub node_ids: Vec<Uuid>,
    /// UUID → vertex index mapping.
    pub uuid_to_idx: HashMap<Uuid, u32>,
    /// CSR row offsets, length = N + 1.
    /// Row `i` spans `col_idx[offsets[i]..offsets[i+1]]`.
    pub offsets: Vec<u32>,
    /// CSR column indices (neighbor vertex indices).
    pub col_idx: Vec<u32>,
    /// Edge weights parallel to `col_idx`.
    pub weights: Vec<f32>,
}

impl Csr {
    pub fn vertex_count(&self) -> usize {
        self.node_ids.len()
    }
    pub fn edge_count(&self) -> usize {
        self.col_idx.len()
    }
}

/// Build a CSR from a NietzscheDB adjacency snapshot.
///
/// Vertex ordering: sorted UUID (deterministic across calls for the same graph).
pub fn build(adjacency: &AdjacencyIndex) -> Result<Csr, CuGraphError> {
    // ── Collect all node IDs ──────────────────────────────────────────────────
    let mut node_ids: Vec<Uuid> = adjacency.all_nodes();
    if node_ids.is_empty() {
        return Ok(Csr {
            node_ids: vec![],
            uuid_to_idx: HashMap::new(),
            offsets: vec![0],
            col_idx: vec![],
            weights: vec![],
        });
    }
    node_ids.sort_unstable(); // deterministic ordering

    let uuid_to_idx: HashMap<Uuid, u32> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as u32))
        .collect();

    // ── Build CSR from outgoing adjacency snapshot ────────────────────────────
    let snapshot = adjacency.snapshot_outgoing();

    // We need rows in sorted-UUID order, not DashMap iteration order.
    // Build a per-node edge list, then emit in UUID-sorted order.
    let mut per_node: HashMap<Uuid, Vec<(u32, f32)>> =
        HashMap::with_capacity(node_ids.len());

    for (src_id, adj_entries) in snapshot {
        let src_idx = match uuid_to_idx.get(&src_id) {
            Some(&i) => i,
            None => continue,
        };
        let row = per_node.entry(src_id).or_default();
        for entry in adj_entries {
            if let Some(&dst_idx) = uuid_to_idx.get(&entry.neighbor_id) {
                row.push((dst_idx, entry.weight));
            }
        }
        // Ensure the source is in the per_node map even with zero out-edges
        let _ = src_idx;
    }

    let mut offsets = Vec::with_capacity(node_ids.len() + 1);
    let mut col_idx = Vec::new();
    let mut weights = Vec::new();
    offsets.push(0u32);

    for &node_id in &node_ids {
        if let Some(mut edges) = per_node.remove(&node_id) {
            edges.sort_unstable_by_key(|&(dst, _)| dst); // sorted neighbors
            for (dst, w) in edges {
                col_idx.push(dst);
                weights.push(w);
            }
        }
        offsets.push(col_idx.len() as u32);
    }

    Ok(Csr {
        node_ids,
        uuid_to_idx,
        offsets,
        col_idx,
        weights,
    })
}
