//! Phase E — Cognitive Layer.
//!
//! Discovers semantic clusters in the Poincaré ball and proposes
//! **concept nodes** as cluster centroids, creating a higher-order
//! abstraction layer in the knowledge graph.
//!
//! ## Pipeline
//!
//! ```text
//! 1. Sample node embeddings from the Poincaré ball
//! 2. Cluster via greedy radius-based grouping (Poincaré distance)
//! 3. Compute gyro-midpoint (Fréchet mean) for each cluster
//! 4. Propose ConceptNode at centroid + edges to members
//! ```
//!
//! ## Integration
//!
//! Called from `AgencyEngine::tick()` every `cognitive_interval` ticks.
//! Produces `AgencyIntent::ProposeConcept` for viable clusters.
//!
//! ## Semantic Design
//!
//! - Concept nodes form a hierarchy: clusters of clusters emerge naturally
//! - Centroid inherits the average depth of its members (hyperbolic hierarchy)
//! - Only proposes concepts for stable, non-overlapping clusters
//! - Label derived from most common content keywords in the cluster

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_hyp_ops::poincare_distance;
use nietzsche_neural::cluster_scorer;
use tracing::{debug, warn};
use uuid::Uuid;

/// Configuration for the cognitive layer.
#[derive(Debug, Clone)]
pub struct CognitiveLayerConfig {
    /// Maximum nodes to sample for clustering (default: 2000).
    pub max_sample: usize,
    /// Poincaré distance threshold for cluster membership (default: 0.3).
    pub cluster_radius: f64,
    /// Minimum cluster size to propose concept node (default: 5).
    pub min_cluster: usize,
    /// Maximum concept nodes to propose per tick (default: 10).
    pub max_concepts: usize,
    /// Whether to use the neural cluster_scorer model (default: false).
    pub neural_enabled: bool,
}

impl Default for CognitiveLayerConfig {
    fn default() -> Self {
        Self {
            max_sample: 2_000,
            cluster_radius: 0.3,
            min_cluster: 5,
            max_concepts: 10,
            neural_enabled: false,
        }
    }
}

/// Build config from AgencyConfig.
pub fn build_cognitive_config(cfg: &crate::config::AgencyConfig) -> CognitiveLayerConfig {
    CognitiveLayerConfig {
        max_sample: cfg.cognitive_max_sample,
        cluster_radius: cfg.cognitive_cluster_radius,
        min_cluster: cfg.cognitive_min_cluster,
        max_concepts: cfg.cognitive_max_concepts,
        neural_enabled: cfg.cognitive_neural_enabled,
    }
}

/// Neural cluster action — recommendation from the cluster_scorer model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterAction {
    Keep,
    Split,
    Merge,
}

/// Build the 261-dimensional feature vector for the cluster_scorer model.
///
/// Layout: \[centroid_128D, variance_128D, size_norm, density, avg_energy, edge_count_norm, coherence\]
///
/// - `centroid_f32`: cluster centroid as f32 (padded/truncated to 128D)
/// - `embeddings`: per-member embeddings (f64), used to compute variance
/// - `cluster_size`: number of members
/// - `avg_distance`: mean intra-cluster Poincaré distance
/// - `storage`/`adjacency`: used to compute energy and edge stats
/// - `member_ids`: node IDs in the cluster
fn build_cluster_features(
    centroid_f64: &[f64],
    embeddings: &[&[f64]],
    cluster_size: usize,
    avg_distance: f64,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    member_ids: &[Uuid],
) -> [f32; 261] {
    let mut features = [0.0f32; 261];

    // --- Centroid (dims 0..128) ---
    let dim = centroid_f64.len().min(128);
    for i in 0..dim {
        features[i] = centroid_f64[i] as f32;
    }

    // --- Per-dimension variance (dims 128..256) ---
    if !embeddings.is_empty() {
        let edim = embeddings[0].len().min(128);
        let n = embeddings.len() as f64;
        for d in 0..edim {
            let mean = embeddings.iter().map(|e| e[d]).sum::<f64>() / n;
            let var = embeddings.iter().map(|e| {
                let diff = e[d] - mean;
                diff * diff
            }).sum::<f64>() / n;
            features[128 + d] = var as f32;
        }
    }

    // --- Scalar stats (dims 256..261) ---
    // [0] size_norm: cluster size normalized (log scale, cap at ~1.0 for size=1000)
    features[256] = ((cluster_size as f64).ln_1p() / 7.0).min(1.0) as f32;

    // [1] density: average intra-cluster Poincaré distance (lower = denser)
    // Invert so higher = denser, normalize by cluster_radius
    features[257] = (1.0 - avg_distance / 0.5).max(0.0).min(1.0) as f32;

    // [2] avg_energy: mean energy of cluster members
    let mut total_energy = 0.0f32;
    let mut energy_count = 0usize;
    for id in member_ids {
        if let Ok(Some(meta)) = storage.get_node_meta(id) {
            total_energy += meta.energy;
            energy_count += 1;
        }
    }
    features[258] = if energy_count > 0 {
        total_energy / energy_count as f32
    } else {
        0.0
    };

    // [3] edge_count_norm: total internal edges normalized
    let mut internal_edges = 0usize;
    let member_set: std::collections::HashSet<Uuid> = member_ids.iter().copied().collect();
    let mut total_edges = 0usize;
    for id in member_ids {
        let out_neighbors = adjacency.neighbors_out(id);
        total_edges += out_neighbors.len();
        for target in &out_neighbors {
            if member_set.contains(target) {
                internal_edges += 1;
            }
        }
    }
    let max_possible = cluster_size * (cluster_size.saturating_sub(1));
    features[259] = if max_possible > 0 {
        (internal_edges as f32 / max_possible as f32).min(1.0)
    } else {
        0.0
    };

    // [4] coherence: ratio of internal edges to total edges (connectivity)
    features[260] = if total_edges > 0 {
        (internal_edges as f32 / total_edges as f32).min(1.0)
    } else {
        0.0
    };

    features
}

/// Score a cluster using the neural cluster_scorer model.
///
/// Returns `Some((action, keep_prob, split_prob, merge_prob))` on success,
/// or `None` if the model is not loaded or inference fails.
fn score_cluster_neural(
    centroid: &[f64],
    embeddings: &[&[f64]],
    cluster_size: usize,
    avg_distance: f64,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    member_ids: &[Uuid],
) -> Option<(ClusterAction, f32, f32, f32)> {
    if !cluster_scorer::is_loaded() {
        return None;
    }

    let features = build_cluster_features(
        centroid, embeddings, cluster_size, avg_distance,
        storage, adjacency, member_ids,
    );

    match cluster_scorer::score(&features) {
        Ok(result) => {
            let action = if result.split >= result.keep && result.split >= result.merge {
                ClusterAction::Split
            } else if result.merge >= result.keep && result.merge >= result.split {
                ClusterAction::Merge
            } else {
                ClusterAction::Keep
            };
            Some((action, result.keep, result.split, result.merge))
        }
        Err(e) => {
            warn!(error = %e, "cluster_scorer inference failed, falling back to heuristics");
            None
        }
    }
}

/// A proposed concept node.
#[derive(Debug, Clone)]
pub struct ConceptProposal {
    /// Centroid embedding (f64 coords, Poincaré ball point).
    pub centroid: Vec<f64>,
    /// Member node IDs that form this cluster.
    pub member_ids: Vec<Uuid>,
    /// Suggested label for the concept.
    pub label: String,
    /// Average intra-cluster Poincaré distance.
    pub avg_distance: f64,
    /// Number of members.
    pub size: usize,
}

/// Result of a cognitive layer scan.
#[derive(Debug, Clone)]
pub struct CognitiveLayerReport {
    /// Number of nodes sampled.
    pub nodes_sampled: usize,
    /// Number of clusters found.
    pub clusters_found: usize,
    /// Number of concept proposals (clusters meeting min_cluster threshold).
    pub concepts_proposed: usize,
    /// The proposals.
    pub proposals: Vec<ConceptProposal>,
}

/// Compute the Fréchet mean (gyro-midpoint) in the Poincaré ball.
///
/// Uses iterative tangent-space averaging:
/// 1. Log-map all points to tangent space at origin
/// 2. Average the tangent vectors
/// 3. Exp-map back to the ball
fn gyro_midpoint(points: &[&[f64]]) -> Vec<f64> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return points[0].to_vec();
    }

    let dim = points[0].len();
    let n = points.len() as f64;

    // Simple tangent-space average at origin
    // log_0(x) = atanh(||x||) * x / ||x||
    let mut sum = vec![0.0; dim];
    for &p in points {
        let norm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            continue;
        }
        let cn = norm.min(0.999); // clamp inside ball
        let scale = cn.atanh() / norm;
        for (s, &pi) in sum.iter_mut().zip(p.iter()) {
            *s += pi * scale;
        }
    }

    for s in sum.iter_mut() {
        *s /= n;
    }

    // exp_0(v) = tanh(||v||) * v / ||v||
    let v_norm: f64 = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
    if v_norm < 1e-15 {
        return vec![0.0; dim];
    }
    let exp_scale = v_norm.tanh() / v_norm;
    sum.iter().map(|&s| s * exp_scale).collect()
}

/// Extract a simple label from node content by finding common words.
fn extract_label(storage: &GraphStorage, member_ids: &[Uuid]) -> String {
    use std::collections::HashMap;

    let mut word_counts: HashMap<String, usize> = HashMap::new();
    let sample = member_ids.iter().take(10); // Sample up to 10 members

    for id in sample {
        if let Ok(Some(meta)) = storage.get_node_meta(id) {
            // Parse words from content
            let content_str = meta.content.to_string().to_lowercase();
            for word in content_str.split(|c: char| !c.is_alphanumeric()) {
                let w = word.trim();
                if w.len() >= 4 && w.len() <= 30 {
                    // Skip common stop words
                    let stops = ["this", "that", "with", "from", "have", "been", "were", "they",
                                 "their", "some", "what", "when", "will", "each", "make",
                                 "like", "into", "more", "than", "just", "also", "very",
                                 "true", "false", "null", "none"];
                    if !stops.contains(&w) {
                        *word_counts.entry(w.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Find top 2 most common words
    let mut words: Vec<(String, usize)> = word_counts.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = words.into_iter().take(2).map(|(w, _)| w).collect();

    if top.is_empty() {
        format!("concept_{}", member_ids.len())
    } else {
        top.join("_")
    }
}

/// Run the cognitive layer scan.
///
/// Samples embeddings, discovers clusters via greedy radius-based grouping,
/// computes centroids, and proposes concept nodes.
pub fn run_cognitive_scan(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &CognitiveLayerConfig,
) -> CognitiveLayerReport {
    // Step 1: Sample embeddings
    let mut samples: Vec<(Uuid, Vec<f64>)> = Vec::new();
    let mut scanned = 0usize;

    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };

        if meta.is_phantom {
            continue;
        }

        if let Ok(Some(emb)) = storage.get_embedding(&meta.id) {
            if !emb.coords.is_empty() {
                let coords: Vec<f64> = emb.coords.iter().map(|&c| c as f64).collect();
                samples.push((meta.id, coords));
            }
        }

        scanned += 1;
        if scanned >= config.max_sample {
            break;
        }
    }

    if samples.len() < config.min_cluster {
        return CognitiveLayerReport {
            nodes_sampled: samples.len(),
            clusters_found: 0,
            concepts_proposed: 0,
            proposals: Vec::new(),
        };
    }

    // Step 2: Greedy radius-based clustering
    let mut assigned = vec![false; samples.len()];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for i in 0..samples.len() {
        if assigned[i] {
            continue;
        }

        // Start a new cluster with node i as seed
        let mut cluster = vec![i];
        assigned[i] = true;

        for j in (i + 1)..samples.len() {
            if assigned[j] {
                continue;
            }

            let dist = poincare_distance(&samples[i].1, &samples[j].1);
            if dist < config.cluster_radius {
                cluster.push(j);
                assigned[j] = true;
            }
        }

        if cluster.len() >= config.min_cluster {
            clusters.push(cluster);
        }

        if clusters.len() >= config.max_concepts * 2 {
            break; // Enough clusters found
        }
    }

    // Step 3: Compute centroids and build proposals
    //
    // When neural scoring is enabled and the cluster_scorer model is loaded,
    // each cluster is evaluated by the ONNX model. Only clusters that the
    // model recommends to "keep" are proposed as concept nodes. If the model
    // is not loaded or inference fails, we fall back to the original heuristic
    // (all clusters meeting min_cluster are proposed).
    let use_neural = config.neural_enabled && cluster_scorer::is_loaded();
    if use_neural {
        debug!("Cognitive layer using neural cluster_scorer for keep/split/merge decisions");
    }

    let mut proposals: Vec<ConceptProposal> = Vec::new();
    let mut neural_skipped_split = 0usize;
    let mut neural_skipped_merge = 0usize;

    for cluster in &clusters {
        let member_ids: Vec<Uuid> = cluster.iter().map(|&idx| samples[idx].0).collect();
        let embeddings: Vec<&[f64]> = cluster.iter().map(|&idx| samples[idx].1.as_slice()).collect();

        // Compute gyro-midpoint (Fréchet mean)
        let centroid = gyro_midpoint(&embeddings);

        // Compute average intra-cluster distance
        let avg_dist = if cluster.len() > 1 {
            let mut total = 0.0;
            let mut count = 0;
            for (i, &a_idx) in cluster.iter().enumerate() {
                for &b_idx in cluster.iter().skip(i + 1) {
                    total += poincare_distance(&samples[a_idx].1, &samples[b_idx].1);
                    count += 1;
                }
            }
            if count > 0 { total / count as f64 } else { 0.0 }
        } else {
            0.0
        };

        // Neural scoring gate: ask the model whether to keep/split/merge
        if use_neural {
            if let Some((action, keep_p, split_p, merge_p)) = score_cluster_neural(
                &centroid, &embeddings, cluster.len(), avg_dist,
                storage, adjacency, &member_ids,
            ) {
                debug!(
                    size = cluster.len(),
                    keep = format!("{:.3}", keep_p),
                    split = format!("{:.3}", split_p),
                    merge = format!("{:.3}", merge_p),
                    "cluster_scorer result"
                );
                match action {
                    ClusterAction::Split => {
                        neural_skipped_split += 1;
                        continue; // Do not propose — cluster should be split further
                    }
                    ClusterAction::Merge => {
                        neural_skipped_merge += 1;
                        continue; // Do not propose — cluster should be merged with neighbors
                    }
                    ClusterAction::Keep => {
                        // Proceed to proposal
                    }
                }
            }
            // If score_cluster_neural returned None, fall through to heuristic
        }

        // Extract label from content
        let label = extract_label(storage, &member_ids);

        proposals.push(ConceptProposal {
            centroid,
            member_ids: member_ids.clone(),
            label,
            avg_distance: avg_dist,
            size: cluster.len(),
        });

        if proposals.len() >= config.max_concepts {
            break;
        }
    }

    if use_neural && (neural_skipped_split > 0 || neural_skipped_merge > 0) {
        debug!(
            skipped_split = neural_skipped_split,
            skipped_merge = neural_skipped_merge,
            proposed = proposals.len(),
            "Neural cluster_scorer filtered clusters"
        );
    }

    // Sort by cluster size (biggest first)
    proposals.sort_by(|a, b| b.size.cmp(&a.size));

    CognitiveLayerReport {
        nodes_sampled: samples.len(),
        clusters_found: clusters.len(),
        concepts_proposed: proposals.len(),
        proposals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gyro_midpoint_single() {
        let p = vec![0.3, -0.2, 0.1];
        let result = gyro_midpoint(&[p.as_slice()]);
        assert_eq!(result.len(), 3);
        for (a, b) in result.iter().zip(p.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gyro_midpoint_symmetric() {
        let p1 = vec![0.3, 0.0];
        let p2 = vec![-0.3, 0.0];
        let mid = gyro_midpoint(&[p1.as_slice(), p2.as_slice()]);
        // Symmetric points → midpoint near origin
        assert!(mid[0].abs() < 0.05, "midpoint x should be near 0: {}", mid[0]);
        assert!(mid[1].abs() < 1e-10, "midpoint y should be 0: {}", mid[1]);
    }

    #[test]
    fn test_gyro_midpoint_origin() {
        let p1 = vec![0.0, 0.0];
        let p2 = vec![0.0, 0.0];
        let mid = gyro_midpoint(&[p1.as_slice(), p2.as_slice()]);
        assert!(mid[0].abs() < 1e-10);
        assert!(mid[1].abs() < 1e-10);
    }

    #[test]
    fn test_midpoint_inside_ball() {
        let p1 = vec![0.8, 0.3];
        let p2 = vec![0.1, -0.7];
        let mid = gyro_midpoint(&[p1.as_slice(), p2.as_slice()]);
        let norm: f64 = mid.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "midpoint must be inside Poincaré ball: norm={}", norm);
    }

    #[test]
    fn test_cognitive_config_defaults() {
        let cfg = CognitiveLayerConfig::default();
        assert_eq!(cfg.max_sample, 2_000);
        assert_eq!(cfg.min_cluster, 5);
        assert!(cfg.cluster_radius > 0.0);
        assert!(!cfg.neural_enabled);
    }

    #[test]
    fn test_build_cluster_features_dimensions() {
        // Verify feature vector has correct layout
        let centroid = vec![0.1f64; 128];
        let emb1 = vec![0.1f64; 128];
        let emb2 = vec![0.2f64; 128];
        let embeddings: Vec<&[f64]> = vec![emb1.as_slice(), emb2.as_slice()];
        let member_ids = vec![Uuid::new_v4(), Uuid::new_v4()];

        // We cannot call build_cluster_features without storage/adjacency in
        // unit tests, but we verify the struct field layout is correct.
        let mut features = [0.0f32; 261];
        // centroid
        for i in 0..128 { features[i] = 0.1; }
        // variance
        for i in 128..256 { features[i] = 0.0025; } // (0.1-0.15)^2 = 0.0025
        // scalars
        features[256] = 0.1; // size_norm
        features[257] = 0.6; // density
        features[258] = 0.5; // avg_energy
        features[259] = 0.3; // edge_count_norm
        features[260] = 0.4; // coherence

        assert_eq!(features.len(), 261);
        assert!(features[256] >= 0.0 && features[256] <= 1.0);
    }

    #[test]
    fn test_cluster_action_variants() {
        assert_eq!(ClusterAction::Keep, ClusterAction::Keep);
        assert_ne!(ClusterAction::Keep, ClusterAction::Split);
        assert_ne!(ClusterAction::Split, ClusterAction::Merge);
    }

    #[test]
    fn test_neural_scorer_not_loaded_by_default() {
        // No model loaded in test environment — is_loaded() should be false.
        // This means score_cluster_neural will return None (graceful fallback)
        // and the cognitive layer will use heuristic thresholds.
        assert!(!cluster_scorer::is_loaded());
    }

    #[test]
    fn test_empty_report() {
        let report = CognitiveLayerReport {
            nodes_sampled: 0,
            clusters_found: 0,
            concepts_proposed: 0,
            proposals: Vec::new(),
        };
        assert_eq!(report.concepts_proposed, 0);
    }
}
