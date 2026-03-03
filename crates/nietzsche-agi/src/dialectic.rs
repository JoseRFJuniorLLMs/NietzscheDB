//! # Dialectic — cross-cluster tension detection
//!
//! Detects "dialectical tension" between concepts in different clusters
//! that could benefit from synthesis.
//!
//! ## How it works
//!
//! 1. For each pair of clusters, compute the inter-cluster Poincaré distance
//! 2. Identify pairs where the distance is below a threshold (semantically related)
//!    but the clusters are distinct (different semantic domains)
//! 3. These "tension points" are candidates for dialectical synthesis
//!
//! The DialecticDetector acts as the "curiosity engine" — it finds places in
//! the graph where new knowledge can be created by synthesizing existing knowledge.

use uuid::Uuid;
use nietzsche_hyp_ops;

use crate::error::AgiResult;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the dialectic detector.
#[derive(Debug, Clone)]
pub struct DialecticConfig {
    /// Maximum Poincaré distance between cluster centroids to consider "related".
    /// Default: 2.0 (moderate distance in hyperbolic space).
    pub max_distance: f64,

    /// Minimum Poincaré distance between cluster centroids to consider "distinct".
    /// Default: 0.5 (clusters must be at least somewhat separated).
    pub min_distance: f64,

    /// Maximum number of tension pairs to return.
    /// Default: 10
    pub max_pairs: usize,

    /// Minimum depth (norm) for centroid nodes to participate.
    /// Very abstract nodes (near center) are excluded to prevent trivial tensions.
    /// Default: 0.1
    pub min_depth: f64,
}

impl Default for DialecticConfig {
    fn default() -> Self {
        Self {
            max_distance: 2.0,
            min_distance: 0.5,
            max_pairs: 10,
            min_depth: 0.1,
        }
    }
}

// ─────────────────────────────────────────────
// TensionPair — a detected dialectical opportunity
// ─────────────────────────────────────────────

/// A pair of nodes from different clusters that exhibit dialectical tension.
///
/// These are candidates for Fréchet synthesis: the two nodes are semantically
/// related (close in Poincaré distance) but belong to different conceptual
/// domains (different clusters).
#[derive(Debug, Clone)]
pub struct TensionPair {
    /// ID of the first node (thesis).
    pub thesis: Uuid,
    /// ID of the second node (antithesis).
    pub antithesis: Uuid,
    /// Cluster of the thesis node.
    pub cluster_a: u32,
    /// Cluster of the antithesis node.
    pub cluster_b: u32,
    /// Poincaré distance between the two nodes.
    pub distance: f64,
    /// "Tension score" = 1/distance (higher = more tension = closer but distinct).
    pub tension_score: f64,
}

// ─────────────────────────────────────────────
// DialecticDetector
// ─────────────────────────────────────────────

/// Detects dialectical tensions between clusters for synthesis candidates.
pub struct DialecticDetector {
    config: DialecticConfig,
}

impl DialecticDetector {
    pub fn new(config: DialecticConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DialecticConfig::default())
    }

    /// Detect tension pairs from a set of cluster centroids.
    ///
    /// # Arguments
    /// - `centroids`: Vec of (node_id, cluster_id, embedding_f64)
    ///
    /// # Returns
    /// A sorted list of [`TensionPair`]s (highest tension first).
    pub fn detect_tensions(
        &self,
        centroids: &[(Uuid, u32, Vec<f64>)],
    ) -> AgiResult<Vec<TensionPair>> {
        if centroids.len() < 2 {
            return Ok(vec![]);
        }

        let mut pairs = Vec::new();

        for i in 0..centroids.len() {
            let (id_a, cluster_a, emb_a) = &centroids[i];

            // Filter by minimum depth
            let norm_a: f64 = emb_a.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_a < self.config.min_depth {
                continue;
            }

            for j in (i + 1)..centroids.len() {
                let (id_b, cluster_b, emb_b) = &centroids[j];

                // Skip same cluster
                if cluster_a == cluster_b {
                    continue;
                }

                let norm_b: f64 = emb_b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_b < self.config.min_depth {
                    continue;
                }

                // Compute Poincaré distance
                let dist = nietzsche_hyp_ops::poincare_distance(emb_a, emb_b);

                // Check if within tension range
                if dist >= self.config.min_distance && dist <= self.config.max_distance {
                    let tension_score = 1.0 / dist.max(1e-10);
                    pairs.push(TensionPair {
                        thesis: *id_a,
                        antithesis: *id_b,
                        cluster_a: *cluster_a,
                        cluster_b: *cluster_b,
                        distance: dist,
                        tension_score,
                    });
                }
            }
        }

        // Sort by tension score descending (highest tension first)
        pairs.sort_by(|a, b| {
            b.tension_score
                .partial_cmp(&a.tension_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit results
        pairs.truncate(self.config.max_pairs);

        tracing::debug!(
            total_centroids = centroids.len(),
            tension_pairs = pairs.len(),
            "dialectic detection complete"
        );

        Ok(pairs)
    }

    /// Detect tensions from raw embeddings grouped by cluster.
    ///
    /// # Arguments
    /// - `cluster_groups`: Vec of (cluster_id, Vec<(node_id, embedding_f64)>)
    ///
    /// First computes the Fréchet mean (gyromidpoint) of each cluster,
    /// then runs tension detection on the centroids.
    pub fn detect_from_clusters(
        &self,
        cluster_groups: &[(u32, Vec<(Uuid, Vec<f64>)>)],
    ) -> AgiResult<Vec<TensionPair>> {
        let mut centroids = Vec::new();

        for (cluster_id, nodes) in cluster_groups {
            if nodes.is_empty() {
                continue;
            }

            // Find the node closest to the gyromidpoint as the representative
            if nodes.len() == 1 {
                centroids.push((nodes[0].0, *cluster_id, nodes[0].1.clone()));
                continue;
            }

            // Compute gyromidpoint of the cluster
            let refs: Vec<&[f64]> = nodes.iter().map(|(_, e)| e.as_slice()).collect();
            match nietzsche_hyp_ops::gyromidpoint(&refs) {
                Ok(midpoint) => {
                    // Find the closest actual node to the midpoint
                    let closest = nodes
                        .iter()
                        .min_by(|(_, a), (_, b)| {
                            let da = nietzsche_hyp_ops::poincare_distance(a, &midpoint);
                            let db = nietzsche_hyp_ops::poincare_distance(b, &midpoint);
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap();
                    centroids.push((closest.0, *cluster_id, closest.1.clone()));
                }
                Err(_) => {
                    // Fallback: use the first node
                    centroids.push((nodes[0].0, *cluster_id, nodes[0].1.clone()));
                }
            }
        }

        self.detect_tensions(&centroids)
    }

    pub fn config(&self) -> &DialecticConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tension_same_cluster() {
        let detector = DialecticDetector::with_defaults();
        let centroids = vec![
            (Uuid::new_v4(), 0, vec![0.3, 0.0, 0.0]),
            (Uuid::new_v4(), 0, vec![0.0, 0.3, 0.0]),
        ];
        let pairs = detector.detect_tensions(&centroids).unwrap();
        assert!(pairs.is_empty(), "Same cluster should produce no tensions");
    }

    #[test]
    fn test_tension_detected() {
        let detector = DialecticDetector::new(DialecticConfig {
            max_distance: 5.0,
            min_distance: 0.1,
            min_depth: 0.05,
            ..Default::default()
        });
        let centroids = vec![
            (Uuid::new_v4(), 0, vec![0.3, 0.0, 0.0]),
            (Uuid::new_v4(), 1, vec![0.0, 0.3, 0.0]),
        ];
        let pairs = detector.detect_tensions(&centroids).unwrap();
        assert_eq!(pairs.len(), 1, "Should detect one tension pair");
        assert!(pairs[0].tension_score > 0.0);
    }

    #[test]
    fn test_too_close_filtered() {
        let detector = DialecticDetector::new(DialecticConfig {
            min_distance: 10.0, // Very high bar
            ..Default::default()
        });
        let centroids = vec![
            (Uuid::new_v4(), 0, vec![0.3, 0.0, 0.0]),
            (Uuid::new_v4(), 1, vec![0.31, 0.0, 0.0]),
        ];
        let pairs = detector.detect_tensions(&centroids).unwrap();
        assert!(pairs.is_empty(), "Too close should be filtered");
    }
}
