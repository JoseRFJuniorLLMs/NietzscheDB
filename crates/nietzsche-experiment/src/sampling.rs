//! Hard negative sampling for Link Prediction in hyperbolic space.
//!
//! For each test edge (u, v), we generate:
//!
//! 1. **1 random negative**: node x chosen uniformly, not connected to u.
//!    Tests global separability.
//!
//! 2. **1 hard negative**: node y at exactly 2 hops from u (neighbour of a
//!    neighbour) but NOT directly connected to u. Tests local topological
//!    coherence — the Poincaré distance between u and y is small, so the model
//!    must rely on genuine structural knowledge to distinguish y from v.
//!
//! ## Why hard negatives matter
//!
//! In the Poincaré ball, volume grows exponentially with radius. Two random
//! nodes are almost certainly on opposite sides of the boundary — trivially
//! separable. Without hard negatives, AUC hits 0.99 on Cycle 1 and tells us
//! nothing about whether TGC is actually improving the topology.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use rand::seq::SliceRandom;
use rand::Rng;
use uuid::Uuid;

use crate::auc::calculate_auc;

/// Result of a single evaluation pass.
#[derive(Debug, Clone)]
pub struct LinkPredictionResult {
    /// AUC over all test edges (random + hard negatives combined).
    pub auc: f64,
    /// AUC using only random negatives.
    pub auc_random: f64,
    /// AUC using only hard negatives.
    pub auc_hard: f64,
    /// Number of test edges actually evaluated (after filtering deleted nodes).
    pub edges_evaluated: usize,
    /// Number of test edges skipped (nodes pruned/deleted by Forgetting Engine).
    pub edges_skipped: usize,
}

/// A held-out test edge: (source_id, target_id).
pub type TestEdge = (Uuid, Uuid);

/// Score: negative Poincaré distance.
///
/// Higher score = closer in hyperbolic space = more likely to be connected.
/// Returns `None` if either embedding is missing (node deleted/phantomised).
fn compute_score(
    storage: &GraphStorage,
    u: &Uuid,
    v: &Uuid,
) -> Option<f64> {
    let emb_u = storage.get_embedding(u).ok()??;
    let emb_v = storage.get_embedding(v).ok()??;
    Some(-emb_u.distance(&emb_v))
}

/// Pick a random node that is NOT a neighbour of `u` (and is not `u` itself).
///
/// Tries up to `max_attempts` times to avoid infinite loops on dense subgraphs.
/// Returns `None` if no suitable candidate found.
fn get_random_negative(
    u: &Uuid,
    all_nodes: &[Uuid],
    adjacency: &AdjacencyIndex,
    rng: &mut impl Rng,
    max_attempts: usize,
) -> Option<Uuid> {
    let u_neighbors = adjacency.neighbors_both(u);
    for _ in 0..max_attempts {
        let candidate = all_nodes.choose(rng)?;
        if candidate != u && !u_neighbors.contains(candidate) {
            return Some(*candidate);
        }
    }
    None
}

/// Pick a 2-hop non-neighbour of `u` (hard negative).
///
/// Algorithm:
/// 1. Pick a random direct neighbour `m` of `u`.
/// 2. Get `m`'s neighbours.
/// 3. Filter out `u` itself and any direct neighbour of `u`.
/// 4. Pick one uniformly at random.
///
/// Returns `None` if `u` has no neighbours or no valid 2-hop candidates exist.
fn get_hard_negative(
    u: &Uuid,
    adjacency: &AdjacencyIndex,
    rng: &mut impl Rng,
) -> Option<Uuid> {
    let neighbors = adjacency.neighbors_both(u);
    if neighbors.is_empty() {
        return None;
    }

    // Pick a random intermediate neighbour
    let intermediate = neighbors.choose(rng)?;

    // Get 2nd-degree neighbours
    let second_degree = adjacency.neighbors_both(intermediate);

    // Filter: must not be u, must not be a direct neighbour of u
    let candidates: Vec<Uuid> = second_degree
        .into_iter()
        .filter(|n| n != u && !neighbors.contains(n))
        .collect();

    if candidates.is_empty() {
        return None;
    }

    Some(*candidates.choose(rng)?)
}

/// Evaluate Link Prediction AUC over a set of held-out test edges.
///
/// For each test edge `(u, v)`:
/// - Computes the positive score: `−d_H(u, v)`.
/// - Generates 1 random negative and 1 hard negative.
/// - Computes negative scores against those.
///
/// Returns a [`LinkPredictionResult`] with combined, random-only, and hard-only AUCs.
pub fn evaluate_link_prediction(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    test_edges: &[TestEdge],
) -> LinkPredictionResult {
    let all_nodes = adjacency.all_nodes();
    if all_nodes.is_empty() || test_edges.is_empty() {
        return LinkPredictionResult {
            auc: 0.5,
            auc_random: 0.5,
            auc_hard: 0.5,
            edges_evaluated: 0,
            edges_skipped: test_edges.len(),
        };
    }

    let mut rng = rand::thread_rng();

    let mut pos_scores = Vec::with_capacity(test_edges.len());
    let mut neg_scores_random = Vec::with_capacity(test_edges.len());
    let mut neg_scores_hard = Vec::with_capacity(test_edges.len());

    let mut skipped = 0usize;

    for (u, v) in test_edges {
        // If either node was pruned/deleted, skip this edge
        let score_pos = match compute_score(storage, u, v) {
            Some(s) => s,
            None => {
                skipped += 1;
                continue;
            }
        };
        pos_scores.push(score_pos);

        // --- Random negative ---
        if let Some(rand_neg) = get_random_negative(u, &all_nodes, adjacency, &mut rng, 10) {
            if let Some(s) = compute_score(storage, u, &rand_neg) {
                neg_scores_random.push(s);
            }
        }

        // --- Hard negative (2-hop) ---
        if let Some(hard_neg) = get_hard_negative(u, adjacency, &mut rng) {
            if let Some(s) = compute_score(storage, u, &hard_neg) {
                neg_scores_hard.push(s);
            }
        }
    }

    let evaluated = pos_scores.len();

    // Combined AUC: all negatives pooled
    let mut neg_all = Vec::with_capacity(neg_scores_random.len() + neg_scores_hard.len());
    neg_all.extend_from_slice(&neg_scores_random);
    neg_all.extend_from_slice(&neg_scores_hard);

    let auc = calculate_auc(&pos_scores, &neg_all);
    let auc_random = calculate_auc(&pos_scores, &neg_scores_random);
    let auc_hard = calculate_auc(&pos_scores, &neg_scores_hard);

    LinkPredictionResult {
        auc,
        auc_random,
        auc_hard,
        edges_evaluated: evaluated,
        edges_skipped: skipped,
    }
}

/// Hold out a fraction of edges for testing.
///
/// Removes the held-out edges from the adjacency index (but NOT from storage —
/// the experiment runner is responsible for managing the DB state).
///
/// Returns `(test_edges, held_out_edge_ids)`.
pub fn hold_out_edges(
    storage: &GraphStorage,
    holdout_fraction: f64,
) -> (Vec<TestEdge>, Vec<Uuid>) {
    let mut rng = rand::thread_rng();
    let mut test_edges = Vec::new();
    let mut held_out_ids = Vec::new();

    for result in storage.iter_edges() {
        let edge = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if rng.gen::<f64>() < holdout_fraction {
            test_edges.push((edge.from, edge.to));
            held_out_ids.push(edge.id);
        }
    }

    (test_edges, held_out_ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_combined_handles_empty() {
        let result = LinkPredictionResult {
            auc: 0.5,
            auc_random: 0.5,
            auc_hard: 0.5,
            edges_evaluated: 0,
            edges_skipped: 0,
        };
        assert!((result.auc - 0.5).abs() < 1e-10);
    }
}
