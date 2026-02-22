//! **Semantic CRDTs** — Conflict-Free Replicated Data Types for graph knowledge.
//!
//! When cluster nodes independently evolve their knowledge graphs (e.g., Node A
//! grows the concept "Fire" while Node B prunes it), traditional last-writer-wins
//! destroys structural intent.
//!
//! This module implements a **graph-aware CRDT** that merges knowledge graphs
//! preserving the semantic hierarchy:
//!
//! - **Add-wins** for nodes: if any peer adds a node, it survives merge
//! - **Energy: max-wins** — highest energy value wins (most active wins)
//! - **Phantom: add-wins** — if any peer phantomized a node, it stays phantom
//!   (destructive operations are irreversible in merge)
//! - **Edges: add-wins** — edges from any peer survive merge
//! - **Embedding: energy-biased** — the embedding from the higher-energy peer
//!   wins (that peer's topology is considered more authoritative)
//!
//! # CRDT Properties
//!
//! All merge operations are:
//! - **Commutative**: `merge(A, B) = merge(B, A)`
//! - **Associative**: `merge(merge(A, B), C) = merge(A, merge(B, C))`
//! - **Idempotent**: `merge(A, A) = A`

use std::collections::HashMap;
use uuid::Uuid;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// CRDT Node State
// ─────────────────────────────────────────────

/// CRDT-mergeable state for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrdtNodeState {
    pub node_id: Uuid,
    /// Lamport timestamp (monotonically increasing per-peer).
    pub timestamp: u64,
    /// Peer ID that last modified this node.
    pub peer_id: String,
    /// Node energy at this peer.
    pub energy: f32,
    /// Whether this node is phantomized at this peer.
    pub is_phantom: bool,
    /// Embedding coordinates (for energy-biased merge).
    pub embedding: Vec<f32>,
    /// Depth (derived from embedding norm).
    pub depth: f32,
    /// Content snapshot.
    pub content: serde_json::Value,
    /// Node type name (e.g., "Semantic", "Episodic").
    pub node_type: String,
}

/// CRDT-mergeable state for a single edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CrdtEdgeState {
    pub from: Uuid,
    pub to: Uuid,
    pub edge_type: String,
    pub weight: f32,
    /// Lamport timestamp.
    pub timestamp: u64,
    pub peer_id: String,
}

// ─────────────────────────────────────────────
// Merge Operations
// ─────────────────────────────────────────────

/// Merge two node states using semantic CRDT rules.
///
/// Returns the merged state.
///
/// Rules:
/// - `energy` → max(a, b)
/// - `is_phantom` → `a || b` (add-wins for deletion)
/// - `embedding` → from the peer with higher energy
/// - `content` → from the peer with higher energy
/// - `timestamp` → max(a, b)
pub fn merge_node(a: &CrdtNodeState, b: &CrdtNodeState) -> CrdtNodeState {
    assert_eq!(a.node_id, b.node_id, "cannot merge different nodes");

    let energy_winner = if a.energy >= b.energy { a } else { b };

    CrdtNodeState {
        node_id: a.node_id,
        timestamp: a.timestamp.max(b.timestamp),
        peer_id: energy_winner.peer_id.clone(),
        energy: a.energy.max(b.energy),
        is_phantom: a.is_phantom || b.is_phantom,
        embedding: energy_winner.embedding.clone(),
        depth: energy_winner.depth,
        content: energy_winner.content.clone(),
        node_type: energy_winner.node_type.clone(),
    }
}

/// Merge two edge sets using add-wins semantics.
///
/// Returns the union of edges. For duplicate edges (same from+to+type),
/// the one with higher weight wins.
pub fn merge_edges(
    a: &[CrdtEdgeState],
    b: &[CrdtEdgeState],
) -> Vec<CrdtEdgeState> {
    let mut merged: HashMap<(Uuid, Uuid, String), CrdtEdgeState> = HashMap::new();

    for edge in a.iter().chain(b.iter()) {
        let key = (edge.from, edge.to, edge.edge_type.clone());
        match merged.get(&key) {
            Some(existing) if existing.weight >= edge.weight => {}
            _ => { merged.insert(key, edge.clone()); }
        }
    }

    merged.into_values().collect()
}

// ─────────────────────────────────────────────
// Graph Delta (for gossip transmission)
// ─────────────────────────────────────────────

/// A delta representing changes since the last sync.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphDelta {
    /// Nodes that have been added or modified.
    pub nodes: Vec<CrdtNodeState>,
    /// Edges that have been added or modified.
    pub edges: Vec<CrdtEdgeState>,
    /// Peer ID of the sender.
    pub from_peer: String,
    /// Lamport timestamp of this delta.
    pub timestamp: u64,
}

/// Merge a remote delta into a local node map.
///
/// Returns `(nodes_merged, edges_merged)` counts.
pub fn apply_delta(
    local_nodes: &mut HashMap<Uuid, CrdtNodeState>,
    local_edges: &mut Vec<CrdtEdgeState>,
    delta: &GraphDelta,
) -> (usize, usize) {
    let mut nodes_merged = 0;
    let mut edges_merged = 0;

    for remote_node in &delta.nodes {
        match local_nodes.get(&remote_node.node_id) {
            Some(local) => {
                let merged = merge_node(local, remote_node);
                if merged.energy != local.energy
                    || merged.is_phantom != local.is_phantom
                    || merged.timestamp != local.timestamp
                {
                    local_nodes.insert(remote_node.node_id, merged);
                    nodes_merged += 1;
                }
            }
            None => {
                // Add-wins: new node from remote peer
                local_nodes.insert(remote_node.node_id, remote_node.clone());
                nodes_merged += 1;
            }
        }
    }

    // Merge edges
    let new_edges = merge_edges(local_edges, &delta.edges);
    edges_merged = new_edges.len().saturating_sub(local_edges.len());
    *local_edges = new_edges;

    (nodes_merged, edges_merged)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_state(id: Uuid, peer: &str, energy: f32, phantom: bool) -> CrdtNodeState {
        CrdtNodeState {
            node_id: id,
            timestamp: 1,
            peer_id: peer.into(),
            energy,
            is_phantom: phantom,
            embedding: vec![0.1, 0.0],
            depth: 0.1,
            content: serde_json::json!({"peer": peer}),
            node_type: "Semantic".into(),
        }
    }

    #[test]
    fn merge_node_max_energy_wins() {
        let id = Uuid::new_v4();
        let a = make_node_state(id, "peer-A", 0.8, false);
        let b = make_node_state(id, "peer-B", 0.5, false);

        let merged = merge_node(&a, &b);
        assert_eq!(merged.energy, 0.8);
        assert_eq!(merged.peer_id, "peer-A"); // higher energy peer
    }

    #[test]
    fn merge_node_commutative() {
        let id = Uuid::new_v4();
        let a = make_node_state(id, "peer-A", 0.8, false);
        let b = make_node_state(id, "peer-B", 0.6, true);

        let ab = merge_node(&a, &b);
        let ba = merge_node(&b, &a);

        assert_eq!(ab.energy, ba.energy);
        assert_eq!(ab.is_phantom, ba.is_phantom);
        assert_eq!(ab.peer_id, ba.peer_id);
    }

    #[test]
    fn merge_node_phantom_add_wins() {
        let id = Uuid::new_v4();
        let a = make_node_state(id, "peer-A", 0.9, false);
        let b = make_node_state(id, "peer-B", 0.3, true); // phantomized

        let merged = merge_node(&a, &b);
        assert!(merged.is_phantom, "phantom should be irreversible in merge");
        assert_eq!(merged.energy, 0.9); // energy still max
    }

    #[test]
    fn merge_node_idempotent() {
        let id = Uuid::new_v4();
        let a = make_node_state(id, "peer-A", 0.7, false);

        let merged = merge_node(&a, &a);
        assert_eq!(merged.energy, a.energy);
        assert_eq!(merged.is_phantom, a.is_phantom);
    }

    #[test]
    fn merge_edges_union() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let edge_a = CrdtEdgeState {
            from: id1, to: id2, edge_type: "Assoc".into(),
            weight: 1.0, timestamp: 1, peer_id: "A".into(),
        };
        let edge_b = CrdtEdgeState {
            from: id2, to: id3, edge_type: "Assoc".into(),
            weight: 0.5, timestamp: 1, peer_id: "B".into(),
        };

        let merged = merge_edges(&[edge_a.clone()], &[edge_b.clone()]);
        assert_eq!(merged.len(), 2, "union should have both edges");
    }

    #[test]
    fn merge_edges_duplicate_higher_weight_wins() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let edge_a = CrdtEdgeState {
            from: id1, to: id2, edge_type: "Assoc".into(),
            weight: 0.3, timestamp: 1, peer_id: "A".into(),
        };
        let edge_b = CrdtEdgeState {
            from: id1, to: id2, edge_type: "Assoc".into(),
            weight: 0.9, timestamp: 2, peer_id: "B".into(),
        };

        let merged = merge_edges(&[edge_a], &[edge_b]);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].weight - 0.9).abs() < 1e-6);
    }

    #[test]
    fn apply_delta_adds_new_nodes() {
        let id = Uuid::new_v4();
        let mut local_nodes = HashMap::new();
        let mut local_edges = Vec::new();

        let delta = GraphDelta {
            nodes: vec![make_node_state(id, "remote", 0.7, false)],
            edges: vec![],
            from_peer: "remote".into(),
            timestamp: 1,
        };

        let (n, e) = apply_delta(&mut local_nodes, &mut local_edges, &delta);
        assert_eq!(n, 1);
        assert_eq!(e, 0);
        assert!(local_nodes.contains_key(&id));
    }

    #[test]
    fn apply_delta_merges_existing_nodes() {
        let id = Uuid::new_v4();
        let mut local_nodes = HashMap::new();
        local_nodes.insert(id, make_node_state(id, "local", 0.5, false));
        let mut local_edges = Vec::new();

        let delta = GraphDelta {
            nodes: vec![make_node_state(id, "remote", 0.9, false)],
            edges: vec![],
            from_peer: "remote".into(),
            timestamp: 2,
        };

        let (n, _) = apply_delta(&mut local_nodes, &mut local_edges, &delta);
        assert_eq!(n, 1);
        assert_eq!(local_nodes[&id].energy, 0.9);
    }

    #[test]
    fn merge_node_timestamp_max() {
        let id = Uuid::new_v4();
        let mut a = make_node_state(id, "A", 0.5, false);
        a.timestamp = 10;
        let mut b = make_node_state(id, "B", 0.6, false);
        b.timestamp = 5;

        let merged = merge_node(&a, &b);
        assert_eq!(merged.timestamp, 10);
    }
}
