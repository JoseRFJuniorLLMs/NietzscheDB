//! # Discovery — measuring productive friction in trajectories
//!
//! The [`DiscoveryField`] detects **innovative** trajectories — those that
//! cross semantic boundaries and create new connections between distant
//! knowledge domains.
//!
//! ## Discovery Score
//!
//! ```text
//! D(τ) = w_g · |∇E(τ)| + w_c · θ_cluster
//! ```
//!
//! ### Components
//!
//! - **|∇E(τ)|**: The energy gradient magnitude — how much the structural
//!   quality changes along the trajectory. A high gradient means the path
//!   traverses regions of very different quality, indicating a "frontier"
//!   between well-explored and unexplored territory.
//!
//! - **θ_cluster**: The cluster crossing angle — the angular separation
//!   between distinct semantic clusters traversed by the trajectory.
//!   Large angles indicate connections between distant knowledge domains.
//!
//! ## Productive Friction
//!
//! The most interesting trajectories have **moderate** gradients: too low
//! means the path stays in familiar territory (no discovery), too high
//! means the path is chaotic (no structure). The "sweet spot" is where
//! the system learns by connecting related-but-different concepts.
//!
//! ```text
//! Productive Friction = |∇E| ∈ [min_gradient, max_gradient]
//!                     AND cluster_transitions ≥ 1
//! ```

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the discovery field.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Weight for the energy gradient component |∇E|.
    /// Default: 0.6
    pub gradient_weight: f64,

    /// Weight for the cluster crossing angle θ_cluster.
    /// Default: 0.4
    pub cluster_weight: f64,

    /// Minimum gradient for productive friction.
    /// Below this, the trajectory is "too familiar" — no discovery.
    /// Default: 0.05
    pub min_gradient: f64,

    /// Maximum gradient before friction becomes destructive.
    /// Above this, the trajectory is chaotic.
    /// Default: 0.8
    pub max_gradient: f64,

    /// Normalization factor for cluster angles (radians).
    /// Default: π/2 (90°)
    pub max_cluster_angle: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            gradient_weight: 0.6,
            cluster_weight: 0.4,
            min_gradient: 0.05,
            max_gradient: 0.8,
            max_cluster_angle: std::f64::consts::FRAC_PI_2,
        }
    }
}

// ─────────────────────────────────────────────
// DiscoveryReport — analysis output
// ─────────────────────────────────────────────

/// Result of analyzing a trajectory for discovery potential.
#[derive(Debug, Clone)]
pub struct DiscoveryReport {
    /// Energy gradient magnitude |∇E(τ)| ∈ [0, 1].
    /// Computed as the maximum absolute difference between
    /// consecutive hop energy contributions.
    pub gradient_magnitude: f64,

    /// Normalized gradient: |∇E| clamped and scaled to [0, 1].
    pub gradient_normalized: f64,

    /// Cluster crossing angle θ_cluster ∈ [0, π/2].
    /// Mean angular separation between consecutive cluster centroids.
    pub cluster_angle: f64,

    /// Normalized cluster angle ∈ [0, 1].
    pub cluster_angle_normalized: f64,

    /// Total discovery score D(τ) ∈ [0, 1].
    /// D = w_g · gradient_norm + w_c · cluster_norm
    pub discovery_score: f64,

    /// Number of distinct clusters traversed.
    pub clusters_traversed: usize,

    /// Number of cluster boundaries crossed.
    pub cluster_transitions: usize,

    /// Whether the trajectory exhibits "productive friction":
    /// moderate gradient + at least one cluster crossing.
    pub is_friction_productive: bool,
}

impl std::fmt::Display for DiscoveryReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "D(τ)={:.4} [|∇E|={:.3}, θ_c={:.3}rad, clusters={}, {}]",
            self.discovery_score,
            self.gradient_magnitude,
            self.cluster_angle,
            self.clusters_traversed,
            if self.is_friction_productive {
                "PRODUCTIVE"
            } else {
                "non-productive"
            },
        )
    }
}

// ─────────────────────────────────────────────
// DiscoveryField
// ─────────────────────────────────────────────

/// Analyzes trajectories for innovation and discovery potential.
///
/// **Pure computation** — takes pre-computed data, no graph access.
pub struct DiscoveryField {
    config: DiscoveryConfig,
}

impl DiscoveryField {
    pub fn new(config: DiscoveryConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DiscoveryConfig::default())
    }

    /// Analyze a trajectory for discovery potential.
    ///
    /// # Arguments
    /// - `hop_energies`: per-hop energy contributions E_i (from stability sub-evaluations)
    ///   These can be derived from hop GCS scores or from segment-wise stability evaluation.
    ///   If empty, gradient_magnitude will be 0.
    /// - `cluster_ids`: cluster assignment for each node in the trajectory.
    ///   Length should match the number of nodes (= hop_energies.len() + 1 for a path).
    /// - `cluster_centroids`: optional (centroid_a, centroid_b) pairs for each cluster
    ///   crossing, used to compute θ_cluster. If None, uses transition count as proxy.
    ///
    /// # Returns
    /// A [`DiscoveryReport`] with the discovery score and all sub-components.
    pub fn analyze(
        &self,
        hop_energies: &[f64],
        cluster_ids: &[u64],
        cluster_centroids: Option<&[(&[f64], &[f64])]>,
    ) -> DiscoveryReport {
        // ── Component 1: Energy gradient |∇E| ──
        let gradient_magnitude = compute_energy_gradient(hop_energies);
        let gradient_normalized = if self.config.max_gradient > self.config.min_gradient {
            ((gradient_magnitude - self.config.min_gradient)
                / (self.config.max_gradient - self.config.min_gradient))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        // ── Component 2: Cluster angle θ_cluster ──
        let (cluster_transitions, clusters_traversed) = count_cluster_transitions(cluster_ids);
        let cluster_angle = match cluster_centroids {
            Some(centroids) => compute_cluster_angle(centroids),
            None => {
                // Proxy: use transition density as angle proxy
                // More transitions in fewer hops → larger effective angle
                if cluster_ids.len() > 1 {
                    let transition_density =
                        cluster_transitions as f64 / (cluster_ids.len() - 1) as f64;
                    transition_density * self.config.max_cluster_angle
                } else {
                    0.0
                }
            }
        };
        let cluster_angle_normalized =
            (cluster_angle / self.config.max_cluster_angle).clamp(0.0, 1.0);

        // ── Discovery Score ──
        let discovery_score = (self.config.gradient_weight * gradient_normalized
            + self.config.cluster_weight * cluster_angle_normalized)
            .clamp(0.0, 1.0);

        // ── Productive Friction ──
        let is_friction_productive = gradient_magnitude >= self.config.min_gradient
            && gradient_magnitude <= self.config.max_gradient
            && cluster_transitions >= 1;

        DiscoveryReport {
            gradient_magnitude,
            gradient_normalized,
            cluster_angle,
            cluster_angle_normalized,
            discovery_score,
            clusters_traversed,
            cluster_transitions,
            is_friction_productive,
        }
    }

    pub fn config(&self) -> &DiscoveryConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Compute the energy gradient magnitude |∇E(τ)|.
///
/// Defined as the maximum absolute difference between consecutive
/// hop energy contributions. Captures the "frontier" effect where
/// quality changes sharply.
///
/// Returns 0.0 for trajectories with fewer than 2 hops.
fn compute_energy_gradient(hop_energies: &[f64]) -> f64 {
    if hop_energies.len() < 2 {
        return 0.0;
    }

    let mut max_gradient = 0.0_f64;
    for window in hop_energies.windows(2) {
        let delta = (window[1] - window[0]).abs();
        if delta > max_gradient {
            max_gradient = delta;
        }
    }

    max_gradient
}

/// Count cluster transitions and distinct clusters.
fn count_cluster_transitions(cluster_ids: &[u64]) -> (usize, usize) {
    if cluster_ids.is_empty() {
        return (0, 0);
    }

    let mut transitions = 0;
    let mut seen = std::collections::HashSet::new();
    seen.insert(cluster_ids[0]);

    for window in cluster_ids.windows(2) {
        if window[0] != window[1] {
            transitions += 1;
            seen.insert(window[1]);
        }
    }

    (transitions, seen.len())
}

/// Compute the mean angular separation between cluster centroids.
///
/// For each pair of (centroid_a, centroid_b) representing a cluster crossing,
/// computes the angle between them in Klein/Euclidean space.
fn compute_cluster_angle(centroids: &[(&[f64], &[f64])]) -> f64 {
    if centroids.is_empty() {
        return 0.0;
    }

    let mut total_angle = 0.0;
    let mut count = 0;

    for (a, b) in centroids {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 1e-15 && norm_b > 1e-15 {
            let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
            total_angle += cos_theta.acos();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_angle / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_discovery_flat_trajectory() {
        let field = DiscoveryField::with_defaults();
        // All same energy, all same cluster → no discovery
        let report = field.analyze(&[0.8, 0.8, 0.8], &[1, 1, 1, 1], None);
        assert!(
            report.discovery_score < 0.1,
            "Flat trajectory should have low D: {}",
            report.discovery_score
        );
        assert!(!report.is_friction_productive);
        assert_eq!(report.cluster_transitions, 0);
    }

    #[test]
    fn test_high_discovery_cross_cluster() {
        let field = DiscoveryField::with_defaults();
        // High gradient + cluster crossing
        let report = field.analyze(&[0.9, 0.3, 0.8], &[1, 1, 2, 2], None);
        assert!(
            report.discovery_score > 0.3,
            "Cross-cluster trajectory should have high D: {}",
            report.discovery_score
        );
        assert_eq!(report.cluster_transitions, 1);
        assert_eq!(report.clusters_traversed, 2);
    }

    #[test]
    fn test_productive_friction_detection() {
        let field = DiscoveryField::with_defaults();
        // Moderate gradient (0.2) + cluster crossing
        let report = field.analyze(&[0.7, 0.5, 0.6], &[1, 2, 2, 3], None);
        assert!(report.is_friction_productive, "Should be productive friction");
    }

    #[test]
    fn test_chaotic_trajectory_not_productive() {
        let field = DiscoveryField::with_defaults();
        // Extreme gradient (> max_gradient)
        let report = field.analyze(&[0.99, 0.01], &[1, 2, 3], None);
        assert!(
            !report.is_friction_productive,
            "Extreme gradient should not be productive"
        );
    }

    #[test]
    fn test_gradient_computation() {
        // Simple case: max gradient = 0.6 (from 0.9 to 0.3)
        let g = compute_energy_gradient(&[0.9, 0.3, 0.8]);
        assert!(
            (g - 0.6).abs() < 1e-10,
            "Gradient should be 0.6: {}",
            g
        );
    }

    #[test]
    fn test_cluster_transitions_counting() {
        let (transitions, distinct) = count_cluster_transitions(&[1, 1, 2, 2, 3, 1]);
        assert_eq!(transitions, 3); // 1→2, 2→3, 3→1
        assert_eq!(distinct, 3); // clusters 1, 2, 3
    }

    #[test]
    fn test_cluster_angle_with_centroids() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let centroids = vec![(&a[..], &b[..])];
        let angle = compute_cluster_angle(&centroids);
        // Angle between (1,0,0) and (0,1,0) = π/2
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 0.01,
            "Angle should be π/2: {}",
            angle
        );
    }

    #[test]
    fn test_empty_trajectory() {
        let field = DiscoveryField::with_defaults();
        let report = field.analyze(&[], &[], None);
        assert_eq!(report.discovery_score, 0.0);
        assert_eq!(report.cluster_transitions, 0);
    }

    #[test]
    fn test_report_display() {
        let report = DiscoveryReport {
            gradient_magnitude: 0.35,
            gradient_normalized: 0.40,
            cluster_angle: 0.78,
            cluster_angle_normalized: 0.50,
            discovery_score: 0.44,
            clusters_traversed: 3,
            cluster_transitions: 2,
            is_friction_productive: true,
        };
        let s = format!("{report}");
        assert!(s.contains("PRODUCTIVE"));
        assert!(s.contains("D(τ)=0.44"));
    }
}
