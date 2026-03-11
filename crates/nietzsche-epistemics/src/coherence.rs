//! Semantic coherence metric.
//!
//! Measures whether connected nodes have appropriate geometric relationships
//! in the Poincaré ball. Two aspects:
//!
//! 1. **Depth coherence**: Association edges connect nodes at similar depths;
//!    hierarchical edges connect nodes at different depths.
//! 2. **Distance coherence**: Poincaré distance between connected nodes
//!    should be small relative to graph diameter.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

/// Compute depth-based coherence for a set of nodes and their edges.
///
/// Returns a score in [0.0, 1.0] where 1.0 means all edges have
/// geometrically appropriate depth relationships.
pub fn depth_coherence(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
) -> f32 {
    if node_ids.is_empty() {
        return 1.0;
    }

    let mut scores = Vec::new();

    for nid in node_ids {
        let src_meta = match storage.get_node_meta(nid) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        let neighbors = adjacency.neighbors_out(nid);
        for nbr_id in &neighbors {
            let dst_meta = match storage.get_node_meta(nbr_id) {
                Ok(Some(m)) => m,
                _ => continue,
            };

            // Check if there's an edge with type info
            let depth_diff = (src_meta.depth - dst_meta.depth).abs();

            // For now, without edge type lookup, use a general heuristic:
            // connected nodes with similar depth (diff < 0.3) get high coherence
            // connected nodes with large depth diff (diff > 0.5) get lower coherence
            // unless it's clearly hierarchical
            let score = if depth_diff < 0.15 {
                1.0  // very similar depth → strong coherence
            } else if depth_diff < 0.3 {
                0.8  // moderately similar → good coherence
            } else if depth_diff < 0.5 {
                0.5  // could be hierarchical → neutral
            } else {
                0.3  // very different depths → likely hierarchical edge
            };

            scores.push(score);
        }
    }

    if scores.is_empty() {
        return 1.0;
    }
    scores.iter().sum::<f32>() / scores.len() as f32
}

/// Compute coherence using Poincaré distances between connected nodes.
///
/// Uses actual hyperbolic distance to measure if edge weights are
/// proportional to geometric distance.
pub fn distance_coherence(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
) -> f32 {
    if node_ids.is_empty() {
        return 1.0;
    }

    let mut correlations = Vec::new();

    for nid in node_ids {
        let src_emb = match storage.get_embedding(nid) {
            Ok(Some(e)) => e,
            _ => continue,
        };

        let src_f64: Vec<f64> = src_emb.coords.iter().map(|&c| c as f64).collect();

        let neighbors = adjacency.neighbors_out(nid);
        for nbr_id in &neighbors {
            let dst_emb = match storage.get_embedding(nbr_id) {
                Ok(Some(e)) => e,
                _ => continue,
            };

            let dst_f64: Vec<f64> = dst_emb.coords.iter().map(|&c| c as f64).collect();
            let dist = nietzsche_hyp_ops::poincare_distance(&src_f64, &dst_f64);

            // Connected nodes should be relatively close in hyperbolic space.
            // Score inversely proportional to distance.
            // d < 1.0 → excellent (1.0)
            // d < 2.0 → good (0.7)
            // d < 4.0 → ok (0.4)
            // d > 4.0 → poor (0.1)
            let score = if dist < 1.0 {
                1.0
            } else if dist < 2.0 {
                1.0 - (dist - 1.0) * 0.3
            } else if dist < 4.0 {
                0.7 - (dist - 2.0) * 0.15
            } else {
                0.1_f64.max(0.4 - (dist - 4.0) * 0.05)
            };

            correlations.push(score as f32);
        }
    }

    if correlations.is_empty() {
        return 1.0;
    }
    correlations.iter().sum::<f32>() / correlations.len() as f32
}

/// Combined coherence score (average of depth and distance coherence).
pub fn combined_coherence(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
) -> f32 {
    let depth = depth_coherence(storage, adjacency, node_ids);
    let distance = distance_coherence(storage, adjacency, node_ids);
    (depth + distance) / 2.0
}
