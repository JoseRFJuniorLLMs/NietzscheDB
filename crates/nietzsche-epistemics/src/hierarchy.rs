//! Hierarchy consistency metric.
//!
//! In Poincaré ball geometry, depth (||embedding||) represents abstraction level:
//! - Low depth (near center) = abstract/general concepts
//! - High depth (near boundary) = specific/concrete instances
//!
//! For directed edges A→B, we expect A.depth ≤ B.depth in most cases
//! (parent → child relationship).

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

/// Tolerance for depth comparison (accounts for floating-point noise).
const DEPTH_TOLERANCE: f32 = 0.05;

/// Compute hierarchy consistency for a set of edges.
///
/// Returns a score in [0.0, 1.0] where 1.0 means all edges
/// respect the depth ordering (parent.depth ≤ child.depth).
pub fn hierarchy_consistency(
    storage: &GraphStorage,
    edge_ids: &[Uuid],
) -> f32 {
    if edge_ids.is_empty() {
        return 1.0;
    }

    let mut correct = 0u64;
    let mut total = 0u64;

    for eid in edge_ids {
        let edge = match storage.get_edge(eid) {
            Ok(Some(e)) => e,
            _ => continue,
        };

        let src_meta = match storage.get_node_meta(&edge.from) {
            Ok(Some(m)) => m,
            _ => continue,
        };
        let dst_meta = match storage.get_node_meta(&edge.to) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        total += 1;
        if src_meta.depth <= dst_meta.depth + DEPTH_TOLERANCE {
            correct += 1;
        }
    }

    if total == 0 {
        return 1.0;
    }
    correct as f32 / total as f32
}

/// Compute hierarchy consistency for ALL edges in storage.
pub fn hierarchy_consistency_global(storage: &GraphStorage) -> f32 {
    let mut correct = 0u64;
    let mut total = 0u64;

    for edge_result in storage.iter_edges() {
        let edge = match edge_result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let src_meta = match storage.get_node_meta(&edge.from) {
            Ok(Some(m)) => m,
            _ => continue,
        };
        let dst_meta = match storage.get_node_meta(&edge.to) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        total += 1;
        if src_meta.depth <= dst_meta.depth + DEPTH_TOLERANCE {
            correct += 1;
        }
    }

    if total == 0 {
        return 1.0;
    }
    correct as f32 / total as f32
}

/// Compute per-node hierarchy score: what fraction of a node's outgoing edges
/// point to nodes with greater or equal depth.
pub fn node_hierarchy_score(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_id: &Uuid,
) -> f32 {
    let src_meta = match storage.get_node_meta(node_id) {
        Ok(Some(m)) => m,
        _ => return 1.0,
    };

    let neighbors = adjacency.neighbors_out(node_id);
    if neighbors.is_empty() {
        return 1.0;
    }

    let mut correct = 0u32;
    let mut total = 0u32;

    for nbr_id in &neighbors {
        let dst_meta = match storage.get_node_meta(nbr_id) {
            Ok(Some(m)) => m,
            _ => continue,
        };
        total += 1;
        if src_meta.depth <= dst_meta.depth + DEPTH_TOLERANCE {
            correct += 1;
        }
    }

    if total == 0 {
        return 1.0;
    }
    correct as f32 / total as f32
}
