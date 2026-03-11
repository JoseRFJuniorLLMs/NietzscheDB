//! Novelty metric.
//!
//! Measures how much new information a mutation adds to the graph.
//! A novel mutation should not duplicate existing knowledge — the new
//! node/edge should occupy a distinct region of the Poincaré ball.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

/// Compute novelty score for a set of newly added nodes.
///
/// Novelty = average minimum Poincaré distance from each new node
/// to any existing node. Higher distance = more novel.
///
/// Returns [0.0, 1.0] where:
/// - 0.0 = new nodes are duplicates of existing ones
/// - 1.0 = new nodes are in completely unexplored regions
pub fn embedding_novelty(
    storage: &GraphStorage,
    new_node_ids: &[Uuid],
    existing_node_ids: &[Uuid],
) -> f32 {
    if new_node_ids.is_empty() || existing_node_ids.is_empty() {
        return 0.5; // neutral
    }

    // Collect existing embeddings
    let existing_embeddings: Vec<(Uuid, Vec<f64>)> = existing_node_ids.iter()
        .filter_map(|nid| {
            storage.get_embedding(nid).ok()?.map(|e| {
                (*nid, e.coords.iter().map(|&c| c as f64).collect())
            })
        })
        .collect();

    if existing_embeddings.is_empty() {
        return 0.5;
    }

    let mut min_distances = Vec::new();

    for nid in new_node_ids {
        let new_emb = match storage.get_embedding(nid) {
            Ok(Some(e)) => e.coords.iter().map(|&c| c as f64).collect::<Vec<f64>>(),
            _ => continue,
        };

        // Find minimum distance to any existing node
        let mut min_dist = f64::MAX;
        for (_, existing) in &existing_embeddings {
            let dist = nietzsche_hyp_ops::poincare_distance(&new_emb, existing);
            if dist < min_dist {
                min_dist = dist;
            }
        }

        if min_dist < f64::MAX {
            min_distances.push(min_dist);
        }
    }

    if min_distances.is_empty() {
        return 0.5;
    }

    let avg_min_dist = min_distances.iter().sum::<f64>() / min_distances.len() as f64;

    // Map distance to [0, 1] score
    // d < 0.1 → near duplicate (0.0)
    // d ≈ 0.5 → moderate novelty (0.5)
    // d > 2.0 → high novelty (1.0)
    let score = (avg_min_dist / 2.0).min(1.0);
    score as f32
}

/// Compute structural novelty: does a new edge create a shortcut
/// between previously distant parts of the graph?
///
/// Returns [0.0, 1.0] where 1.0 = the edge connects previously
/// disconnected or distant components.
pub fn structural_novelty(
    adjacency: &AdjacencyIndex,
    from_id: &Uuid,
    to_id: &Uuid,
) -> f32 {
    // Check if nodes were already connected
    let from_neighbors = adjacency.neighbors_both(from_id);
    if from_neighbors.contains(to_id) {
        return 0.0; // already directly connected — no novelty
    }

    // Check 2-hop connectivity
    let mut two_hop_shared = 0u32;
    let to_neighbors: std::collections::HashSet<Uuid> =
        adjacency.neighbors_both(to_id).into_iter().collect();

    for nbr in &from_neighbors {
        if to_neighbors.contains(nbr) {
            two_hop_shared += 1;
        }
    }

    if two_hop_shared > 3 {
        return 0.2; // many shared neighbors — low structural novelty
    }
    if two_hop_shared > 0 {
        return 0.5; // some shared neighbors — moderate novelty
    }

    // No shared neighbors at all — high structural novelty
    0.9
}
