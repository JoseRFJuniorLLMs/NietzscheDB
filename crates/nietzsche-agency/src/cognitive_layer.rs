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
}

impl Default for CognitiveLayerConfig {
    fn default() -> Self {
        Self {
            max_sample: 2_000,
            cluster_radius: 0.3,
            min_cluster: 5,
            max_concepts: 10,
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
    let mut proposals: Vec<ConceptProposal> = Vec::new();

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
