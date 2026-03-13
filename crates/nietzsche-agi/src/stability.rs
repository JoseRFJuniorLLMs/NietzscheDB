// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Stability — Continuous Trajectory Energy Evaluation
//!
//! The [`StabilityEvaluator`] computes a continuous energy function E(τ)
//! that measures the structural quality of an inference trajectory.
//!
//! ## Energy Function
//!
//! ```text
//! E(τ) = w₁·H_GCS + w₂·(1 - θ_klein/π) + w₃·causal_fraction + w₄·(1 - H(τ))
//! ```
//!
//! ### Components
//!
//! | Symbol | Name | Range | Meaning |
//! |--------|------|-------|---------|
//! | H_GCS | Harmonic mean GCS | [0, 1] | Aggregate geodesic coherence |
//! | θ_klein | Mean Klein angular deviation | [0, π/2] | How far hops deviate from geodesics |
//! | causal_fraction | Timelike edge ratio | [0, 1] | Fraction of causally valid edges |
//! | H(τ) | Trajectory entropy | [0, 1] | Shannon entropy of GCS distribution |
//!
//! ## Interpretation
//!
//! - **E(τ) ≈ 1.0**: The trajectory is geometrically pristine — geodesic, causal, low-entropy.
//! - **E(τ) ≈ 0.5**: Mixed quality — some hops are good, some deviate.
//! - **E(τ) ≈ 0.0**: The trajectory is structurally broken.
//!
//! The energy value feeds into the [`certification`](crate::certification) module
//! to determine the epistemological level of the inference.

use nietzsche_graph::CausalType;
use nietzsche_hyp_ops::klein;

use crate::inference_engine::EdgeInfo;
use crate::trajectory::GeodesicTrajectory;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Weights and parameters for the stability energy function.
///
/// Default weights: w₁=0.35, w₂=0.25, w₃=0.20, w₄=0.20
/// Sum = 1.0 (energy ∈ [0, 1]).
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Weight for H_GCS (harmonic mean of per-hop GCS).
    /// Default: 0.35
    pub w_gcs: f64,

    /// Weight for Klein angular deviation component (1 - θ/π).
    /// Default: 0.25
    pub w_klein: f64,

    /// Weight for causal fraction (Timelike edge ratio).
    /// Default: 0.20
    pub w_causal: f64,

    /// Weight for trajectory entropy component (1 - H(τ)).
    /// Default: 0.20
    pub w_entropy: f64,

    /// Clamp angular deviation to this maximum (radians).
    /// Default: π/2 (90°)
    pub max_angular_deviation: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            w_gcs: 0.35,
            w_klein: 0.25,
            w_causal: 0.20,
            w_entropy: 0.20,
            max_angular_deviation: std::f64::consts::FRAC_PI_2,
        }
    }
}

impl StabilityConfig {
    /// Validate that weights sum to approximately 1.0.
    pub fn validate(&self) -> bool {
        let sum = self.w_gcs + self.w_klein + self.w_causal + self.w_entropy;
        (sum - 1.0).abs() < 0.01
    }
}

// ─────────────────────────────────────────────
// StabilityReport — output of evaluation
// ─────────────────────────────────────────────

/// Complete stability analysis of a trajectory.
///
/// Contains the final energy E(τ) and all sub-components for
/// transparency and debugging.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    /// Final energy value E(τ) ∈ [0, 1].
    pub energy: f64,

    /// Harmonic mean of per-hop GCS scores.
    pub h_gcs: f64,

    /// Mean angular deviation in Klein model (radians).
    /// 0 = perfect geodesic, π/2 = maximum deviation.
    pub theta_klein: f64,

    /// Normalized Klein component: 1 - θ/(π/2) ∈ [0, 1].
    pub klein_component: f64,

    /// Fraction of edges that are Timelike (causal) ∈ [0, 1].
    pub causal_fraction: f64,

    /// Shannon entropy of the GCS distribution ∈ [0, 1].
    /// Normalized by ln(n) where n = number of hops.
    pub trajectory_entropy: f64,

    /// Entropy component: 1 - H(τ) ∈ [0, 1].
    pub entropy_component: f64,

    /// Number of hops evaluated.
    pub hop_count: usize,

    /// Number of causal (Timelike) edges.
    pub causal_edge_count: usize,

    /// Number of acausal (non-Timelike) edges.
    pub acausal_edge_count: usize,
}

impl std::fmt::Display for StabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "E(τ)={:.4} [H_GCS={:.3}, θ_klein={:.3}rad, causal={:.1}%, H(τ)={:.3}]",
            self.energy,
            self.h_gcs,
            self.theta_klein,
            self.causal_fraction * 100.0,
            self.trajectory_entropy,
        )
    }
}

// ─────────────────────────────────────────────
// StabilityEvaluator
// ─────────────────────────────────────────────

/// Evaluates the structural quality of inference trajectories.
///
/// Produces a continuous energy E(τ) ∈ [0, 1] that captures:
/// - Geodesic coherence (are hops on geodesics?)
/// - Klein angular fidelity (how close to straight lines?)
/// - Causal validity (do edges respect Minkowski causality?)
/// - Trajectory consistency (is quality uniform across hops?)
///
/// **Pure computation** — does not access the graph.
pub struct StabilityEvaluator {
    config: StabilityConfig,
}

impl StabilityEvaluator {
    pub fn new(config: StabilityConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(StabilityConfig::default())
    }

    /// Evaluate a trajectory and produce a stability report.
    ///
    /// # Arguments
    /// - `trajectory`: a GCS-validated trajectory (with embeddings)
    /// - `edge_infos`: causal type for each edge along the path (may be empty)
    ///
    /// # Returns
    /// A [`StabilityReport`] with E(τ) and all sub-components.
    pub fn evaluate(
        &self,
        trajectory: &GeodesicTrajectory,
        edge_infos: &[EdgeInfo],
    ) -> StabilityReport {
        // ── Component 1: H_GCS (harmonic mean of hop scores) ──
        let hop_scores: Vec<f64> = trajectory.hop_scores.iter().map(|h| h.score).collect();
        let h_gcs = if hop_scores.is_empty() {
            1.0 // Direct connection → perfect
        } else {
            harmonic_mean(&hop_scores)
        };

        // ── Component 2: Klein angular deviation ──
        let theta_klein = compute_klein_angular_deviation(trajectory);
        let klein_component = (1.0 - theta_klein / self.config.max_angular_deviation)
            .clamp(0.0, 1.0);

        // ── Component 3: Causal fraction ──
        let causal_edge_count = edge_infos
            .iter()
            .filter(|e| e.causal_type == CausalType::Timelike)
            .count();
        let acausal_edge_count = edge_infos.len() - causal_edge_count;
        let causal_fraction = if edge_infos.is_empty() {
            0.5 // No edge info → neutral
        } else {
            causal_edge_count as f64 / edge_infos.len() as f64
        };

        // ── Component 4: Trajectory entropy ──
        let trajectory_entropy = compute_trajectory_entropy(&hop_scores);
        let entropy_component = 1.0 - trajectory_entropy;

        // ── Final energy ──
        let energy = (self.config.w_gcs * h_gcs
            + self.config.w_klein * klein_component
            + self.config.w_causal * causal_fraction
            + self.config.w_entropy * entropy_component)
            .clamp(0.0, 1.0);

        StabilityReport {
            energy,
            h_gcs,
            theta_klein,
            klein_component,
            causal_fraction,
            trajectory_entropy,
            entropy_component,
            hop_count: hop_scores.len(),
            causal_edge_count,
            acausal_edge_count,
        }
    }

    pub fn config(&self) -> &StabilityConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Harmonic mean of positive values. Returns 0.0 for empty/zero inputs.
fn harmonic_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_recip: f64 = values
        .iter()
        .map(|&v| if v <= 0.0 { f64::INFINITY } else { 1.0 / v })
        .sum();
    if sum_recip.is_infinite() || sum_recip == 0.0 {
        return 0.0;
    }
    values.len() as f64 / sum_recip
}

/// Compute the mean angular deviation of interior nodes from geodesics
/// in the Klein model.
///
/// For each triple (A, B, C):
/// - Convert to Klein coordinates (geodesics = straight lines)
/// - Compute the angle between AB and the perpendicular from B to line(AC)
/// - θ_i = arcsin(perp_dist / dist(A, B))
///
/// Returns the mean θ in radians ∈ [0, π/2].
fn compute_klein_angular_deviation(trajectory: &GeodesicTrajectory) -> f64 {
    if trajectory.embeddings.len() < 3 {
        return 0.0; // No interior nodes → no deviation
    }

    // Convert all points to Klein model
    let klein_points: Vec<Vec<f64>> = trajectory
        .embeddings
        .iter()
        .map(|p| klein::to_klein(p).unwrap_or_else(|_| p.clone()))
        .collect();

    let mut total_angle = 0.0;
    let mut count = 0;

    for i in 1..klein_points.len() - 1 {
        let a = &klein_points[i - 1];
        let b = &klein_points[i];
        let c = &klein_points[i + 1];

        let perp = perpendicular_distance(a, c, b);
        let ab_dist = euclidean_dist(a, b);

        if ab_dist > 1e-15 {
            // θ = arcsin(perp / AB), clamped to valid range
            let sin_theta = (perp / ab_dist).min(1.0);
            total_angle += sin_theta.asin();
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_angle / count as f64
    }
}

/// Shannon entropy of the GCS distribution, normalized to [0, 1].
///
/// H(τ) = -Σ p_i · ln(p_i) / ln(n)
///
/// where p_i = gcs_i / Σ gcs_j (normalized GCS scores).
///
/// - H = 1.0: all hops have equal GCS (maximum uniformity)
/// - H = 0.0: one hop dominates (maximum concentration)
fn compute_trajectory_entropy(hop_scores: &[f64]) -> f64 {
    if hop_scores.len() <= 1 {
        return 0.0; // Entropy undefined for 0-1 elements
    }

    let sum: f64 = hop_scores.iter().filter(|&&s| s > 0.0).sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let n = hop_scores.len() as f64;
    let ln_n = n.ln();
    if ln_n <= 0.0 {
        return 0.0;
    }

    let entropy: f64 = hop_scores
        .iter()
        .filter(|&&s| s > 0.0)
        .map(|&s| {
            let p = s / sum;
            -p * p.ln()
        })
        .sum();

    (entropy / ln_n).clamp(0.0, 1.0)
}

/// Perpendicular distance from point P to the line through A and C.
fn perpendicular_distance(a: &[f64], c: &[f64], p: &[f64]) -> f64 {
    let n = a.len();
    debug_assert_eq!(n, c.len());
    debug_assert_eq!(n, p.len());

    let mut pa = vec![0.0; n];
    let mut ca = vec![0.0; n];
    for i in 0..n {
        pa[i] = p[i] - a[i];
        ca[i] = c[i] - a[i];
    }

    let ca_sq: f64 = ca.iter().map(|x| x * x).sum();
    if ca_sq < 1e-30 {
        return vec_norm(&pa);
    }

    let pa_dot_ca: f64 = pa.iter().zip(ca.iter()).map(|(a, b)| a * b).sum();
    let t = pa_dot_ca / ca_sq;

    let mut perp = vec![0.0; n];
    for i in 0..n {
        perp[i] = pa[i] - t * ca[i];
    }

    vec_norm(&perp)
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{GcsConfig, GeodesicCoherenceScore, GeodesicTrajectory};
    use uuid::Uuid;

    fn make_trajectory(
        embeddings: Vec<Vec<f64>>,
        gcs_scores: Vec<f64>,
        radial: f64,
    ) -> GeodesicTrajectory {
        let path: Vec<Uuid> = (0..embeddings.len()).map(|_| Uuid::new_v4()).collect();
        let hop_scores: Vec<GeodesicCoherenceScore> = gcs_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| GeodesicCoherenceScore {
                hop: i + 1,
                score,
                is_coherent: score >= 0.5,
            })
            .collect();

        GeodesicTrajectory {
            path,
            embeddings,
            hop_scores,
            aggregate_gcs: if gcs_scores.is_empty() {
                1.0
            } else {
                gcs_scores.iter().sum::<f64>() / gcs_scores.len() as f64
            },
            is_valid: true,
            radial_gradient: radial,
        }
    }

    #[test]
    fn test_perfect_trajectory() {
        // Collinear points → θ_klein ≈ 0, GCS = 1.0
        let traj = make_trajectory(
            vec![
                vec![0.1, 0.0, 0.0],
                vec![0.2, 0.0, 0.0],
                vec![0.3, 0.0, 0.0],
            ],
            vec![0.99],
            0.2,
        );
        let eval = StabilityEvaluator::with_defaults();
        let report = eval.evaluate(&traj, &[]);
        assert!(report.energy > 0.5, "Perfect trajectory should have high energy: {}", report.energy);
        assert!(report.theta_klein < 0.01, "θ_klein should be near 0: {}", report.theta_klein);
    }

    #[test]
    fn test_deviant_trajectory() {
        // B is far off the geodesic A→C
        let traj = make_trajectory(
            vec![
                vec![0.1, 0.0, 0.0],
                vec![0.0, 0.5, 0.0], // Way off
                vec![0.3, 0.0, 0.0],
            ],
            vec![0.2],
            0.2,
        );
        let eval = StabilityEvaluator::with_defaults();
        let report = eval.evaluate(&traj, &[]);
        assert!(report.energy < 0.5, "Deviant trajectory should have low energy: {}", report.energy);
        assert!(report.theta_klein > 0.3, "θ_klein should be significant: {}", report.theta_klein);
    }

    #[test]
    fn test_causal_edges_boost() {
        let traj = make_trajectory(
            vec![vec![0.1, 0.0, 0.0], vec![0.3, 0.0, 0.0]],
            vec![],
            0.2,
        );
        let eval = StabilityEvaluator::with_defaults();

        // All causal
        let causal_edges = vec![EdgeInfo { causal_type: CausalType::Timelike }];
        let report_causal = eval.evaluate(&traj, &causal_edges);

        // All acausal
        let acausal_edges = vec![EdgeInfo { causal_type: CausalType::Spacelike }];
        let report_acausal = eval.evaluate(&traj, &acausal_edges);

        assert!(
            report_causal.energy > report_acausal.energy,
            "Causal {} should beat acausal {}",
            report_causal.energy,
            report_acausal.energy
        );
    }

    #[test]
    fn test_entropy_computation() {
        // All equal scores → maximum entropy
        let h_uniform = compute_trajectory_entropy(&[0.8, 0.8, 0.8, 0.8]);
        assert!(
            (h_uniform - 1.0).abs() < 0.01,
            "Uniform scores should have max entropy: {}",
            h_uniform
        );

        // One dominant score → low entropy
        let h_concentrated = compute_trajectory_entropy(&[0.99, 0.01, 0.01, 0.01]);
        assert!(
            h_concentrated < 0.5,
            "Concentrated scores should have low entropy: {}",
            h_concentrated
        );
    }

    #[test]
    fn test_config_validation() {
        let config = StabilityConfig::default();
        assert!(config.validate(), "Default config should sum to 1.0");

        let bad_config = StabilityConfig {
            w_gcs: 0.5,
            w_klein: 0.5,
            w_causal: 0.5,
            w_entropy: 0.5,
            ..Default::default()
        };
        assert!(!bad_config.validate(), "Bad config should fail validation");
    }

    #[test]
    fn test_harmonic_mean() {
        assert!((harmonic_mean(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-10);
        assert!((harmonic_mean(&[2.0, 2.0]) - 2.0).abs() < 1e-10);
        assert_eq!(harmonic_mean(&[]), 0.0);
        assert_eq!(harmonic_mean(&[0.0, 1.0]), 0.0);
    }

    #[test]
    fn test_direct_connection() {
        // 2-node path → no interior nodes → E depends on causal + defaults
        let traj = make_trajectory(
            vec![vec![0.1, 0.0, 0.0], vec![0.3, 0.0, 0.0]],
            vec![],
            0.2,
        );
        let eval = StabilityEvaluator::with_defaults();
        let report = eval.evaluate(&traj, &[]);
        assert!(report.energy > 0.0, "Should produce non-zero energy");
        assert_eq!(report.hop_count, 0);
    }

    #[test]
    fn test_report_display() {
        let report = StabilityReport {
            energy: 0.7532,
            h_gcs: 0.85,
            theta_klein: 0.12,
            klein_component: 0.92,
            causal_fraction: 0.75,
            trajectory_entropy: 0.95,
            entropy_component: 0.05,
            hop_count: 3,
            causal_edge_count: 3,
            acausal_edge_count: 1,
        };
        let s = format!("{report}");
        assert!(s.contains("E(τ)=0.7532"));
    }
}
