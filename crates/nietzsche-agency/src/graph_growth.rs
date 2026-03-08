//! Phase C — Autonomous Graph Growth.
//!
//! Automatically discovers potential edges between semantically related nodes
//! by scanning embeddings for proximity in the Poincaré ball.
//!
//! ## Pipeline
//!
//! ```text
//! 1. Sample active nodes (energy > min_energy, not phantom)
//! 2. For each candidate, find nearest neighbors via embedding distance
//! 3. Filter: skip existing edges, skip high-degree targets
//! 4. Propose new edges with weight = f(distance)
//! ```
//!
//! ## Integration
//!
//! Called from `AgencyEngine::tick()` every `growth_interval` ticks.
//! Produces `AgencyIntent::ProposeEdge` for each viable connection.
//!
//! ## Design Principles
//!
//! - Only proposes edges between nodes with no existing connection
//! - Weight inversely proportional to Poincaré distance
//! - Respects degree limits to prevent hub formation
//! - Rate-limited by max_new_edges per tick

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_hyp_ops::poincare_distance;
use uuid::Uuid;

/// Configuration for autonomous graph growth.
#[derive(Debug, Clone)]
pub struct GraphGrowthConfig {
    /// Maximum candidates to evaluate per tick (default: 100).
    pub max_candidates: usize,
    /// Poincaré distance threshold — only propose edges closer than this (default: 1.5).
    pub distance_threshold: f64,
    /// Maximum new edges to propose per tick (default: 50).
    pub max_new_edges: usize,
    /// Minimum energy for a node to be a growth source (default: 0.1).
    pub min_energy: f32,
    /// Maximum degree for a node to receive new edges (default: 100).
    pub max_target_degree: usize,
}

impl Default for GraphGrowthConfig {
    fn default() -> Self {
        Self {
            max_candidates: 100,
            distance_threshold: 1.5,
            max_new_edges: 50,
            min_energy: 0.1,
            max_target_degree: 100,
        }
    }
}

/// Build config from AgencyConfig.
pub fn build_growth_config(cfg: &crate::config::AgencyConfig) -> GraphGrowthConfig {
    GraphGrowthConfig {
        max_candidates: cfg.growth_max_candidates,
        distance_threshold: cfg.growth_distance_threshold,
        max_new_edges: cfg.growth_max_new_edges,
        min_energy: cfg.growth_min_energy,
        max_target_degree: cfg.growth_max_target_degree,
    }
}

/// A candidate edge proposal.
#[derive(Debug, Clone)]
pub struct GrowthCandidate {
    /// Source node ID.
    pub from_id: Uuid,
    /// Target node ID.
    pub to_id: Uuid,
    /// Poincaré distance between embeddings.
    pub distance: f64,
    /// Proposed edge weight (inversely proportional to distance).
    pub weight: f32,
}

/// Result of a graph growth scan.
#[derive(Debug, Clone)]
pub struct GraphGrowthReport {
    /// Number of nodes sampled as candidates.
    pub nodes_sampled: usize,
    /// Number of neighbor pairs evaluated.
    pub pairs_evaluated: usize,
    /// Number of new edges proposed.
    pub edges_proposed: usize,
    /// Average distance of proposed edges.
    pub avg_proposed_distance: f64,
    /// The proposed candidates.
    pub candidates: Vec<GrowthCandidate>,
}

/// Scan for autonomous graph growth opportunities.
///
/// Finds pairs of active nodes that are close in embedding space but
/// not yet connected by an edge, and proposes new edges.
pub fn run_graph_growth_scan(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &GraphGrowthConfig,
) -> GraphGrowthReport {
    // Step 1: Sample active nodes with embeddings
    let mut active_nodes: Vec<(Uuid, Vec<f64>)> = Vec::new();
    let mut scanned = 0usize;

    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        if meta.is_phantom || meta.energy < config.min_energy {
            continue;
        }

        // Load embedding
        if let Ok(Some(emb)) = storage.get_embedding(&meta.id) {
            if !emb.coords.is_empty() {
                let coords: Vec<f64> = emb.coords.iter().map(|&c| c as f64).collect();
                active_nodes.push((meta.id, coords));
            }
        }

        scanned += 1;
        if scanned >= config.max_candidates * 2 {
            break; // Over-sample to have enough candidates
        }
    }

    if active_nodes.len() < 2 {
        return GraphGrowthReport {
            nodes_sampled: active_nodes.len(),
            pairs_evaluated: 0,
            edges_proposed: 0,
            avg_proposed_distance: 0.0,
            candidates: Vec::new(),
        };
    }

    // Step 2: For each node, find closest unconnected neighbors
    let mut candidates: Vec<GrowthCandidate> = Vec::new();
    let mut pairs_evaluated = 0usize;
    let max_pairs = config.max_candidates * 10; // Cap total comparisons

    'outer: for i in 0..active_nodes.len().min(config.max_candidates) {
        let (from_id, ref from_emb) = active_nodes[i];
        let from_degree = adjacency.degree_out(&from_id);

        // Skip nodes that are already heavily connected
        if from_degree >= config.max_target_degree {
            continue;
        }

        // Pre-compute neighbor set for this source (O(1) lookups)
        let from_neighbors: std::collections::HashSet<Uuid> =
            adjacency.neighbors_out(&from_id).into_iter().collect();

        // Find closest neighbors by sampling
        let mut best: Vec<(usize, f64)> = Vec::new();
        for j in 0..active_nodes.len() {
            if i == j {
                continue;
            }
            pairs_evaluated += 1;
            if pairs_evaluated > max_pairs {
                break 'outer;
            }

            let (to_id, ref to_emb) = active_nodes[j];

            // Skip if already connected (from→to or to→from)
            if from_neighbors.contains(&to_id) {
                continue;
            }
            let to_neighbors = adjacency.neighbors_out(&to_id);
            if to_neighbors.contains(&from_id) {
                continue;
            }

            // Check target degree
            let to_degree = adjacency.degree_in(&to_id) + adjacency.degree_out(&to_id);
            if to_degree >= config.max_target_degree {
                continue;
            }

            let dist = poincare_distance(from_emb, to_emb);
            if dist < config.distance_threshold {
                best.push((j, dist));
            }
        }

        // Sort by distance and take top 3
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (j, dist) in best.into_iter().take(3) {
            let (to_id, _) = active_nodes[j];
            // Weight: closer = stronger, max 1.0
            let weight = (1.0 / (1.0 + dist)) as f32;
            candidates.push(GrowthCandidate {
                from_id,
                to_id,
                distance: dist,
                weight,
            });

            if candidates.len() >= config.max_new_edges {
                break 'outer;
            }
        }
    }

    // Sort by distance (best proposals first)
    candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(config.max_new_edges);

    let avg_dist = if !candidates.is_empty() {
        candidates.iter().map(|c| c.distance).sum::<f64>() / candidates.len() as f64
    } else {
        0.0
    };

    GraphGrowthReport {
        nodes_sampled: active_nodes.len(),
        pairs_evaluated,
        edges_proposed: candidates.len(),
        avg_proposed_distance: avg_dist,
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_growth_config_defaults() {
        let cfg = GraphGrowthConfig::default();
        assert_eq!(cfg.max_candidates, 100);
        assert_eq!(cfg.max_new_edges, 50);
        assert!(cfg.distance_threshold > 0.0);
    }

    #[test]
    fn test_weight_from_distance() {
        // Closer nodes should get higher weight
        let close_weight = (1.0 / (1.0 + 0.5_f64)) as f32;
        let far_weight = (1.0 / (1.0 + 2.0_f64)) as f32;
        assert!(close_weight > far_weight);
        assert!(close_weight <= 1.0);
        assert!(far_weight > 0.0);
    }

    #[test]
    fn test_empty_report() {
        let report = GraphGrowthReport {
            nodes_sampled: 0,
            pairs_evaluated: 0,
            edges_proposed: 0,
            avg_proposed_distance: 0.0,
            candidates: Vec::new(),
        };
        assert_eq!(report.edges_proposed, 0);
    }
}
