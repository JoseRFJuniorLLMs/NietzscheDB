use dashmap::DashMap;
use uuid::Uuid;

use crate::model::{Edge, EdgeType};

// ─────────────────────────────────────────────
// AdjacencyEntry
// ─────────────────────────────────────────────

/// One entry in the adjacency list: (edge_id, neighbor_id, weight).
#[derive(Debug, Clone)]
pub struct AdjEntry {
    pub edge_id:     Uuid,
    pub neighbor_id: Uuid,
    pub weight:      f32,
    pub edge_type:   EdgeType,
}

// ─────────────────────────────────────────────
// AdjacencyIndex
// ─────────────────────────────────────────────

/// In-memory, lock-free bidirectional adjacency index.
///
/// Backed by `DashMap` — supports concurrent reads and writes
/// without a global lock (fine-grained sharded locking internally).
///
/// Separate from `GraphStorage` (RocksDB). On startup this index
/// is reconstructed by scanning the edge column family.
#[derive(Debug, Default)]
pub struct AdjacencyIndex {
    /// node_id → [(edge_id, neighbor_id, weight, edge_type)]
    outgoing: DashMap<Uuid, Vec<AdjEntry>>,
    /// node_id → [(edge_id, source_id, weight, edge_type)]
    incoming: DashMap<Uuid, Vec<AdjEntry>>,
}

impl AdjacencyIndex {
    pub fn new() -> Self {
        Self::default()
    }

    // ── Mutations ──────────────────────────────────────

    /// Register an edge in both directions.
    pub fn add_edge(&self, edge: &Edge) {
        self.outgoing
            .entry(edge.from)
            .or_default()
            .push(AdjEntry {
                edge_id:     edge.id,
                neighbor_id: edge.to,
                weight:      edge.weight,
                edge_type:   edge.edge_type.clone(),
            });

        self.incoming
            .entry(edge.to)
            .or_default()
            .push(AdjEntry {
                edge_id:     edge.id,
                neighbor_id: edge.from,
                weight:      edge.weight,
                edge_type:   edge.edge_type.clone(),
            });
    }

    /// Remove all adjacency entries for an edge (called on prune/delete).
    pub fn remove_edge(&self, edge: &Edge) {
        if let Some(mut out) = self.outgoing.get_mut(&edge.from) {
            out.retain(|e| e.edge_id != edge.id);
        }
        if let Some(mut inc) = self.incoming.get_mut(&edge.to) {
            inc.retain(|e| e.edge_id != edge.id);
        }
    }

    /// Remove all edges connected to a node (called on node deletion).
    pub fn remove_node(&self, node_id: &Uuid) {
        // Remove outgoing entries and clean up their reverse pointers
        if let Some((_, out_edges)) = self.outgoing.remove(node_id) {
            for entry in &out_edges {
                if let Some(mut inc) = self.incoming.get_mut(&entry.neighbor_id) {
                    inc.retain(|e| e.neighbor_id != *node_id);
                }
            }
        }
        // Remove incoming entries and clean up their forward pointers
        if let Some((_, in_edges)) = self.incoming.remove(node_id) {
            for entry in &in_edges {
                if let Some(mut out) = self.outgoing.get_mut(&entry.neighbor_id) {
                    out.retain(|e| e.neighbor_id != *node_id);
                }
            }
        }
    }

    // ── Queries ────────────────────────────────────────

    /// IDs of all outgoing neighbors.
    pub fn neighbors_out(&self, node_id: &Uuid) -> Vec<Uuid> {
        self.outgoing
            .get(node_id)
            .map(|v| v.iter().map(|e| e.neighbor_id).collect())
            .unwrap_or_default()
    }

    /// IDs of all incoming neighbors.
    pub fn neighbors_in(&self, node_id: &Uuid) -> Vec<Uuid> {
        self.incoming
            .get(node_id)
            .map(|v| v.iter().map(|e| e.neighbor_id).collect())
            .unwrap_or_default()
    }

    /// Both outgoing and incoming neighbors (deduplicated).
    pub fn neighbors_both(&self, node_id: &Uuid) -> Vec<Uuid> {
        let mut result = self.neighbors_out(node_id);
        for n in self.neighbors_in(node_id) {
            if !result.contains(&n) {
                result.push(n);
            }
        }
        result
    }

    /// Full outgoing adjacency entries (edge_id + weight included).
    pub fn entries_out(&self, node_id: &Uuid) -> Vec<AdjEntry> {
        self.outgoing
            .get(node_id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Full incoming adjacency entries.
    pub fn entries_in(&self, node_id: &Uuid) -> Vec<AdjEntry> {
        self.incoming
            .get(node_id)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Outgoing degree of a node.
    pub fn degree_out(&self, node_id: &Uuid) -> usize {
        self.outgoing
            .get(node_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Incoming degree of a node.
    pub fn degree_in(&self, node_id: &Uuid) -> usize {
        self.incoming
            .get(node_id)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Total number of unique nodes tracked.
    pub fn node_count(&self) -> usize {
        // Union of keys in both maps
        let mut count = self.outgoing.len();
        self.incoming.iter().for_each(|kv| {
            if !self.outgoing.contains_key(kv.key()) {
                count += 1;
            }
        });
        count
    }

    /// Total number of edges (each directed edge counted once).
    pub fn edge_count(&self) -> usize {
        self.outgoing.iter().map(|kv| kv.value().len()).sum()
    }

    /// Clear all entries (used in tests).
    pub fn clear(&self) {
        self.outgoing.clear();
        self.incoming.clear();
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Edge;

    fn make_edge(from: Uuid, to: Uuid) -> Edge {
        Edge::association(from, to, 0.8)
    }

    #[test]
    fn add_and_query_outgoing() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        idx.add_edge(&make_edge(a, b));

        let out = idx.neighbors_out(&a);
        assert_eq!(out, vec![b]);
        assert!(idx.neighbors_out(&b).is_empty());
    }

    #[test]
    fn add_and_query_incoming() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        idx.add_edge(&make_edge(a, b));

        let inc = idx.neighbors_in(&b);
        assert_eq!(inc, vec![a]);
        assert!(idx.neighbors_in(&a).is_empty());
    }

    #[test]
    fn neighbors_both_deduplicates() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        // a → b and b → a
        idx.add_edge(&make_edge(a, b));
        idx.add_edge(&make_edge(b, a));

        let both = idx.neighbors_both(&a);
        assert_eq!(both.len(), 1);
        assert_eq!(both[0], b);
    }

    #[test]
    fn degree_counts_are_correct() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        idx.add_edge(&make_edge(a, b));
        idx.add_edge(&make_edge(a, c));

        assert_eq!(idx.degree_out(&a), 2);
        assert_eq!(idx.degree_in(&b),  1);
        assert_eq!(idx.degree_in(&c),  1);
    }

    #[test]
    fn remove_edge_cleans_both_directions() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let edge = make_edge(a, b);
        idx.add_edge(&edge);
        idx.remove_edge(&edge);

        assert!(idx.neighbors_out(&a).is_empty());
        assert!(idx.neighbors_in(&b).is_empty());
    }

    #[test]
    fn remove_node_cleans_all_connections() {
        let idx = AdjacencyIndex::new();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        idx.add_edge(&make_edge(a, b));
        idx.add_edge(&make_edge(c, a));
        idx.remove_node(&a);

        assert!(idx.neighbors_out(&a).is_empty());
        assert!(idx.neighbors_in(&a).is_empty());
        // b and c should no longer see a
        assert!(idx.neighbors_in(&b).is_empty());
        assert!(idx.neighbors_out(&c).is_empty());
    }

    #[test]
    fn concurrent_writes_do_not_panic() {
        use std::sync::Arc;
        use std::thread;

        let idx = Arc::new(AdjacencyIndex::new());
        let root = Uuid::new_v4();

        let handles: Vec<_> = (0..8).map(|_| {
            let idx = Arc::clone(&idx);
            let child = Uuid::new_v4();
            thread::spawn(move || {
                idx.add_edge(&make_edge(root, child));
            })
        }).collect();

        for h in handles { h.join().unwrap(); }

        assert_eq!(idx.degree_out(&root), 8);
    }

    #[test]
    fn edge_count_is_accurate() {
        let idx = AdjacencyIndex::new();
        let nodes: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        for i in 0..4 {
            idx.add_edge(&make_edge(nodes[i], nodes[i + 1]));
        }
        assert_eq!(idx.edge_count(), 4);
    }
}
