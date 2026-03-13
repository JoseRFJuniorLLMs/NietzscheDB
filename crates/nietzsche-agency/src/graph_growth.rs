// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
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
    /// Whether to use the neural edge_predictor model for scoring (default: false).
    /// Requires `AGENCY_GROWTH_NEURAL_ENABLED=true` and the model loaded in REGISTRY.
    pub neural_enabled: bool,
    /// Minimum predicted probability from edge_predictor to accept an edge (default: 0.5).
    pub neural_threshold: f32,
}

impl Default for GraphGrowthConfig {
    fn default() -> Self {
        Self {
            max_candidates: 100,
            distance_threshold: 1.5,
            max_new_edges: 50,
            min_energy: 0.1,
            max_target_degree: 100,
            neural_enabled: false,
            neural_threshold: 0.5,
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
        neural_enabled: cfg.growth_neural_enabled,
        neural_threshold: cfg.growth_neural_threshold,
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
    /// Neural edge_predictor probability (None if neural scoring disabled or unavailable).
    pub neural_score: Option<f32>,
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
    /// Number of candidates accepted by neural scoring (0 if disabled).
    pub neural_accepted: usize,
    /// Number of candidates rejected by neural scoring (0 if disabled).
    pub neural_rejected: usize,
}

/// Try to acquire the neural edge predictor if enabled and available.
///
/// Returns `None` if neural scoring is disabled, or if the model isn't loaded
/// in the registry (graceful fallback to heuristic-only mode).
fn try_get_predictor(config: &GraphGrowthConfig) -> Option<nietzsche_gnn::EdgePredictorNet> {
    if !config.neural_enabled {
        return None;
    }
    // Check if edge_predictor is loaded in the REGISTRY before constructing.
    // EdgePredictorNet::new() would try to re-load from disk, but we only want
    // to use the model if it was already loaded at startup.
    match nietzsche_neural::REGISTRY.get_session("edge_predictor") {
        Ok(_) => {
            tracing::debug!("Neural edge predictor available for graph growth scoring");
            // Construct a lightweight wrapper — it uses the existing REGISTRY session.
            Some(nietzsche_gnn::EdgePredictorNet::new(""))
        }
        Err(_) => {
            tracing::debug!("Neural edge predictor not loaded — falling back to heuristic scoring");
            None
        }
    }
}

/// Score a batch of candidate edges using the neural edge_predictor model.
///
/// For each candidate, concatenates source+target f32 embeddings and runs
/// batch inference.  Returns a Vec of probabilities in [0, 1], one per candidate.
/// If inference fails, returns None (caller falls back to heuristic).
fn neural_score_candidates(
    predictor: &nietzsche_gnn::EdgePredictorNet,
    candidates: &[GrowthCandidate],
    active_nodes: &[(Uuid, Vec<f64>)],
    node_index: &std::collections::HashMap<Uuid, usize>,
) -> Option<Vec<f32>> {
    // Build batch of (embedding_a, embedding_b) pairs
    let mut pairs: Vec<(&[f32], &[f32])> = Vec::with_capacity(candidates.len());
    // We need f32 copies since active_nodes stores f64
    let mut f32_cache: Vec<Vec<f32>> = Vec::with_capacity(candidates.len() * 2);

    for c in candidates {
        let from_idx = node_index.get(&c.from_id)?;
        let to_idx = node_index.get(&c.to_id)?;
        let from_f32: Vec<f32> = active_nodes[*from_idx].1.iter().map(|&v| v as f32).collect();
        let to_f32: Vec<f32> = active_nodes[*to_idx].1.iter().map(|&v| v as f32).collect();
        f32_cache.push(from_f32);
        f32_cache.push(to_f32);
    }

    // Build slice references into the cache
    for i in 0..candidates.len() {
        pairs.push((&f32_cache[i * 2], &f32_cache[i * 2 + 1]));
    }

    match predictor.predict_batch(&pairs) {
        Ok(scores) => Some(scores),
        Err(e) => {
            tracing::warn!(error = %e, "Neural edge prediction batch failed — falling back to heuristic");
            None
        }
    }
}

/// Scan for autonomous graph growth opportunities.
///
/// Finds pairs of active nodes that are close in embedding space but
/// not yet connected by an edge, and proposes new edges.
///
/// When `config.neural_enabled` is true and the `edge_predictor` model is loaded
/// in the neural REGISTRY, candidates are additionally scored by the neural model.
/// Only candidates above `config.neural_threshold` are kept.  If the model is
/// unavailable or inference fails, the function falls back to heuristic-only mode.
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
            neural_accepted: 0,
            neural_rejected: 0,
        };
    }

    // Build a lookup from Uuid → index in active_nodes (for neural scoring)
    let node_index: std::collections::HashMap<Uuid, usize> = active_nodes
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();

    // Step 2: For each node, find closest unconnected neighbors (heuristic pass)
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
                neural_score: None,
            });

            if candidates.len() >= config.max_new_edges {
                break 'outer;
            }
        }
    }

    // Sort by distance (best proposals first)
    candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(config.max_new_edges);

    // Step 3: Neural edge prediction — filter/rank with ONNX model
    let mut neural_accepted = 0usize;
    let mut neural_rejected = 0usize;

    if !candidates.is_empty() {
        if let Some(predictor) = try_get_predictor(config) {
            if let Some(scores) = neural_score_candidates(
                &predictor,
                &candidates,
                &active_nodes,
                &node_index,
            ) {
                // Annotate candidates with neural scores
                for (candidate, &score) in candidates.iter_mut().zip(scores.iter()) {
                    candidate.neural_score = Some(score);
                }

                // Filter: reject candidates below threshold
                let before_count = candidates.len();
                candidates.retain(|c| {
                    match c.neural_score {
                        Some(s) if s >= config.neural_threshold => true,
                        Some(_) => false,
                        None => true, // Keep if no score (shouldn't happen here)
                    }
                });
                neural_accepted = candidates.len();
                neural_rejected = before_count - neural_accepted;

                // Re-sort by neural score (highest probability first) when available
                candidates.sort_by(|a, b| {
                    let sa = a.neural_score.unwrap_or(0.0);
                    let sb = b.neural_score.unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });

                tracing::info!(
                    accepted = neural_accepted,
                    rejected = neural_rejected,
                    threshold = config.neural_threshold,
                    "Phase C: Neural edge predictor scored candidates"
                );
            }
        }
    }

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
        neural_accepted,
        neural_rejected,
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
        assert!(!cfg.neural_enabled);
        assert!((cfg.neural_threshold - 0.5).abs() < f32::EPSILON);
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
            neural_accepted: 0,
            neural_rejected: 0,
        };
        assert_eq!(report.edges_proposed, 0);
        assert_eq!(report.neural_accepted, 0);
        assert_eq!(report.neural_rejected, 0);
    }

    #[test]
    fn test_candidate_neural_score_default_none() {
        let c = GrowthCandidate {
            from_id: Uuid::new_v4(),
            to_id: Uuid::new_v4(),
            distance: 0.5,
            weight: 0.67,
            neural_score: None,
        };
        assert!(c.neural_score.is_none());
    }

    #[test]
    fn test_neural_disabled_fallback() {
        // When neural is disabled, try_get_predictor returns None
        let cfg = GraphGrowthConfig {
            neural_enabled: false,
            ..Default::default()
        };
        assert!(try_get_predictor(&cfg).is_none());
    }
}
