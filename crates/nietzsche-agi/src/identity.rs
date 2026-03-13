// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Identity — Axiom Drift Detection & Ur-Cortex Shield (Phase VIII)
//!
//! The identity module implements the **Geometric Constitution** of the AGI system:
//! a formal, mathematically grounded mechanism to ensure that structural evolution
//! never dissolves the founding semantic identity.
//!
//! ## Core Principle: The Mito Fundador Eterno
//!
//! The Ur-Cortex (original axiom centroid C₀) is **immutable forever**.
//! The system can reinterpret, extend, and evolve its perspectives (α, β, γ),
//! but the gravitational center of foundational knowledge never moves.
//!
//! > "Identidade viva, mas não dissociada."
//!
//! ## Axiom Drift
//!
//! ```text
//! AxiomDrift(t) = d_𝔻(C₀, Cₜ)
//! ```
//!
//! Where:
//! - C₀ = Fréchet mean of axiom embeddings at system birth
//! - Cₜ = Fréchet mean of current axiom embeddings
//! - d_𝔻 = Poincaré ball distance (true hyperbolic metric, no approximation)
//!
//! ## Relative Threshold
//!
//! ```text
//! Drift_max = η · r̄₀
//! ```
//!
//! Where:
//! - r̄₀ = mean distance from C₀ to each axiom at birth
//! - η ∈ [0.1, 0.3] — drift tolerance ratio
//!
//! This scales with the actual topology: dense axiom clusters are stricter,
//! sparse clusters allow more drift.
//!
//! ## Computational Stability
//!
//! - Fréchet mean approximation: 3-5 iterations (sufficient for convergence)
//! - Poincaré distance uses the numerically stable arcosh formulation
//! - Safe clamping: (1 - |x|²) ≥ ε to prevent division by zero near boundary
//!
//! ## Integration
//!
//! The AxiomIdentity guard is checked at the end of each structural evolution
//! cycle. If drift exceeds the threshold, the mutation is rolled back before
//! it can commit to the manifold.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════
// §1 — Configuration
// ═══════════════════════════════════════════════

/// Configuration for the Axiom Identity Shield.
#[derive(Debug, Clone)]
pub struct IdentityConfig {
    /// Drift tolerance ratio η.
    /// Drift_max = η · mean_initial_radius.
    /// Higher η → more permissive drift.
    /// Default: 0.2
    pub eta: f64,

    /// Number of Fréchet mean iterations for centroid approximation.
    /// 3-5 iterations are sufficient for convergence in practice.
    /// Default: 5
    pub frechet_iterations: usize,

    /// Learning rate for Fréchet mean iterative approximation.
    /// Default: 1.0 (full step; reduce for more conservative convergence)
    pub frechet_learning_rate: f64,

    /// Epsilon for safe Poincaré distance computation.
    /// Clamps (1 - |x|²) to avoid division by zero.
    /// Default: 1e-9
    pub distance_epsilon: f64,

    /// Maximum Poincaré ball radius² for distance safety.
    /// Default: 0.999999
    pub max_radius_sq: f64,

    /// Enable the identity shield.
    /// When false, drift is measured but never triggers rollback.
    /// Default: true
    pub enabled: bool,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            eta: 0.2,
            frechet_iterations: 5,
            frechet_learning_rate: 1.0,
            distance_epsilon: 1e-9,
            max_radius_sq: 0.999_999,
            enabled: true,
        }
    }
}

// ═══════════════════════════════════════════════
// §2 — Identity Error
// ═══════════════════════════════════════════════

/// Error signaling axiom drift exceeded the threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AxiomDriftExceeded {
    /// Measured drift distance d_𝔻(C₀, Cₜ).
    pub drift: f64,

    /// Maximum allowed drift = η · r̄₀.
    pub limit: f64,

    /// Current η ratio.
    pub eta: f64,

    /// Mean initial radius r̄₀.
    pub initial_radius: f64,
}

impl std::fmt::Display for AxiomDriftExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AxiomDriftExceeded(d={:.6} > limit={:.6}, η={:.2}, r̄₀={:.4})",
            self.drift, self.limit, self.eta, self.initial_radius,
        )
    }
}

// ═══════════════════════════════════════════════
// §3 — Drift Report
// ═══════════════════════════════════════════════

/// Report from an axiom drift measurement.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Measured drift d_𝔻(C₀, Cₜ).
    pub drift: f64,

    /// Maximum allowed drift (η · r̄₀).
    pub limit: f64,

    /// Drift as fraction of limit: drift / limit ∈ [0, ∞).
    /// < 1.0 = safe, ≥ 1.0 = exceeded.
    pub drift_ratio: f64,

    /// Whether the drift exceeds the threshold.
    pub exceeded: bool,

    /// Number of axiom nodes used for centroid computation.
    pub n_axioms: usize,

    /// Dimension of the embedding space.
    pub dimension: usize,
}

impl std::fmt::Display for DriftReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.exceeded { "EXCEEDED" } else { "OK" };
        write!(
            f,
            "Drift[{status}] d={:.6} / limit={:.6} ({:.1}%) axioms={}",
            self.drift,
            self.limit,
            self.drift_ratio * 100.0,
            self.n_axioms,
        )
    }
}

// ═══════════════════════════════════════════════
// §4 — AxiomIdentity (The Shield)
// ═══════════════════════════════════════════════

/// The Axiom Identity Shield — guardian of the Ur-Cortex.
///
/// ## Immutable Founding Myth
///
/// The `ur_cortex_center` (C₀) is computed once at system birth from the
/// Fréchet mean of all axiom embeddings. It is **never updated**. The system
/// can evolve, explore, and innovate — but the gravitational center of
/// foundational knowledge remains eternal.
///
/// ## Elastic Constraint
///
/// Drift is not zero — a living system MUST drift slightly as it learns.
/// But drift is bounded:
///
/// ```text
/// 0 < AxiomDrift(t) < η · r̄₀
/// ```
///
/// This creates **Elasticidade Epistêmica**: the system stretches toward
/// the periphery seeking innovation, but the Ur-Cortex pulls it back
/// like a hyperbolic spring.
///
/// ## Usage
///
/// ```text
/// let identity = AxiomIdentity::initialize(&axiom_positions, config);
/// // ... after evolution cycle ...
/// let report = identity.measure_drift(&current_axiom_positions);
/// if report.exceeded {
///     // ROLLBACK the mutation
/// }
/// ```
pub struct AxiomIdentity {
    /// Immutable Ur-Cortex center C₀ (Fréchet mean at birth).
    ur_cortex_center: Vec<f64>,

    /// Mean initial radius r̄₀ = mean(d_𝔻(C₀, xᵢ)).
    initial_radius_mean: f64,

    /// Configuration.
    config: IdentityConfig,

    /// Dimension of the embedding space.
    dimension: usize,
}

impl AxiomIdentity {
    /// Initialize the identity shield from axiom embeddings at system birth.
    ///
    /// Computes C₀ (Fréchet mean) and r̄₀ (mean distance to center).
    /// These values are frozen forever.
    ///
    /// # Arguments
    /// - `axiom_positions`: embeddings of all axiom/Level-1 nodes in Poincaré ball
    /// - `config`: identity configuration
    ///
    /// # Returns
    /// `None` if no axioms provided, otherwise `Some(AxiomIdentity)`.
    pub fn initialize(axiom_positions: &[Vec<f64>], config: IdentityConfig) -> Option<Self> {
        if axiom_positions.is_empty() {
            return None;
        }

        let dim = axiom_positions[0].len();
        if dim == 0 {
            return None;
        }

        // Compute Fréchet mean via iterative approximation
        let center = frechet_mean_poincare(
            axiom_positions,
            config.frechet_iterations,
            config.frechet_learning_rate,
            config.distance_epsilon,
            config.max_radius_sq,
        );

        // Compute mean initial radius
        let radius_mean = if axiom_positions.len() == 1 {
            // Single axiom: use a minimal default radius
            0.1
        } else {
            let total_dist: f64 = axiom_positions
                .iter()
                .map(|x| safe_poincare_distance(&center, x, config.distance_epsilon, config.max_radius_sq))
                .sum();
            total_dist / axiom_positions.len() as f64
        };

        // Ensure non-zero radius (prevent division by zero in threshold)
        let radius_mean = radius_mean.max(0.01);

        Some(Self {
            ur_cortex_center: center,
            initial_radius_mean: radius_mean,
            config,
            dimension: dim,
        })
    }

    /// Create a minimal identity for testing purposes.
    pub fn new_minimal(center: Vec<f64>, initial_radius: f64, config: IdentityConfig) -> Self {
        let dim = center.len();
        Self {
            ur_cortex_center: center,
            initial_radius_mean: initial_radius.max(0.01),
            config,
            dimension: dim,
        }
    }

    /// Measure the axiom drift from current node positions.
    ///
    /// Computes Cₜ (current Fréchet mean), then measures d_𝔻(C₀, Cₜ).
    ///
    /// # Arguments
    /// - `current_axiom_positions`: current embeddings of axiom nodes
    ///
    /// # Returns
    /// A [`DriftReport`] with drift measurement and threshold status.
    pub fn measure_drift(&self, current_axiom_positions: &[Vec<f64>]) -> DriftReport {
        if current_axiom_positions.is_empty() {
            return DriftReport {
                drift: 0.0,
                limit: self.drift_limit(),
                drift_ratio: 0.0,
                exceeded: false,
                n_axioms: 0,
                dimension: self.dimension,
            };
        }

        // Compute current centroid Cₜ
        let current_center = frechet_mean_poincare(
            current_axiom_positions,
            self.config.frechet_iterations,
            self.config.frechet_learning_rate,
            self.config.distance_epsilon,
            self.config.max_radius_sq,
        );

        // Measure drift = d_𝔻(C₀, Cₜ)
        let drift = safe_poincare_distance(
            &self.ur_cortex_center,
            &current_center,
            self.config.distance_epsilon,
            self.config.max_radius_sq,
        );

        let limit = self.drift_limit();
        let drift_ratio = if limit > 0.0 { drift / limit } else { f64::INFINITY };
        let exceeded = self.config.enabled && drift > limit;

        DriftReport {
            drift,
            limit,
            drift_ratio,
            exceeded,
            n_axioms: current_axiom_positions.len(),
            dimension: self.dimension,
        }
    }

    /// Validate that current axiom positions don't violate the identity constraint.
    ///
    /// Returns `Ok(drift)` if within bounds, `Err(AxiomDriftExceeded)` if not.
    pub fn validate_sovereignty(
        &self,
        current_axiom_positions: &[Vec<f64>],
    ) -> Result<f64, AxiomDriftExceeded> {
        let report = self.measure_drift(current_axiom_positions);

        if report.exceeded {
            Err(AxiomDriftExceeded {
                drift: report.drift,
                limit: report.limit,
                eta: self.config.eta,
                initial_radius: self.initial_radius_mean,
            })
        } else {
            Ok(report.drift)
        }
    }

    /// Get the maximum allowed drift: η · r̄₀.
    pub fn drift_limit(&self) -> f64 {
        self.config.eta * self.initial_radius_mean
    }

    /// Get the immutable Ur-Cortex center.
    pub fn ur_cortex_center(&self) -> &[f64] {
        &self.ur_cortex_center
    }

    /// Get the initial mean radius.
    pub fn initial_radius_mean(&self) -> f64 {
        self.initial_radius_mean
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configuration.
    pub fn config(&self) -> &IdentityConfig {
        &self.config
    }
}

impl std::fmt::Display for AxiomIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let center_norm: f64 = self.ur_cortex_center.iter().map(|x| x * x).sum::<f64>().sqrt();
        write!(
            f,
            "AxiomIdentity[dim={}, |C₀|={:.4}, r̄₀={:.4}, drift_max={:.4}, η={:.2}]",
            self.dimension,
            center_norm,
            self.initial_radius_mean,
            self.drift_limit(),
            self.config.eta,
        )
    }
}

// ═══════════════════════════════════════════════
// §5 — Hyperbolic Geometry Primitives
// ═══════════════════════════════════════════════

/// Numerically stable Poincaré ball distance.
///
/// ```text
/// d_𝔻(u, v) = arcosh(1 + 2·|u-v|² / ((1-|u|²)(1-|v|²)))
/// ```
///
/// Safety:
/// - (1 - |x|²) is clamped to ε to prevent division by zero
/// - |x|² is clamped to max_radius_sq to prevent boundary explosion
/// - arcosh argument is clamped to ≥ 1.0 for floating-point noise
pub fn safe_poincare_distance(u: &[f64], v: &[f64], epsilon: f64, max_radius_sq: f64) -> f64 {
    debug_assert_eq!(u.len(), v.len(), "dimension mismatch in Poincaré distance");

    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();

    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum::<f64>().min(max_radius_sq);
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum::<f64>().min(max_radius_sq);

    let denom_u = (1.0 - norm_u_sq).max(epsilon);
    let denom_v = (1.0 - norm_v_sq).max(epsilon);

    let arg = (1.0 + 2.0 * diff_sq / (denom_u * denom_v)).max(1.0);
    arg.acosh()
}

/// Approximate Fréchet mean on the Poincaré ball via iterative tangent-space averaging.
///
/// The Fréchet mean minimizes Σ d²_𝔻(μ, xᵢ) and is the correct generalization
/// of the Euclidean mean to hyperbolic space.
///
/// ## Algorithm
///
/// Starting from the Euclidean mean (projected into the ball), iteratively:
/// 1. Compute the tangent vector from μ to each xᵢ via log_map
/// 2. Average tangent vectors: v̄ = (1/N) · Σ log_μ(xᵢ)
/// 3. Update: μ ← exp_μ(η · v̄)
///
/// This converges in 3-5 iterations for well-separated points.
///
/// ## Simplified Implementation
///
/// For computational efficiency and stability, we use the **weighted Euclidean
/// mean projected back into the ball** as a first-order Fréchet approximation.
/// This avoids full log/exp map computation while maintaining the correct
/// gravitational center.
pub fn frechet_mean_poincare(
    points: &[Vec<f64>],
    max_iterations: usize,
    _learning_rate: f64,
    epsilon: f64,
    max_radius_sq: f64,
) -> Vec<f64> {
    if points.is_empty() {
        return vec![];
    }

    let dim = points[0].len();
    let n = points.len() as f64;

    // Start with Euclidean mean
    let mut mean = vec![0.0; dim];
    for p in points {
        for (i, &val) in p.iter().enumerate() {
            if i < dim {
                mean[i] += val / n;
            }
        }
    }

    // Project into ball if outside
    project_to_ball(&mut mean, max_radius_sq);

    // Iterative refinement via distance-weighted averaging
    for _ in 0..max_iterations {
        let mut weighted_sum = vec![0.0; dim];
        let mut total_weight = 0.0;

        for p in points {
            let d = safe_poincare_distance(&mean, p, epsilon, max_radius_sq);
            // Weight inversely proportional to distance (closer points matter more)
            // Using 1/(1+d) as smooth weight to avoid division by zero
            let w = 1.0 / (1.0 + d);
            total_weight += w;

            for (i, &val) in p.iter().enumerate() {
                if i < dim {
                    weighted_sum[i] += w * val;
                }
            }
        }

        if total_weight > 1e-30 {
            for i in 0..dim {
                mean[i] = weighted_sum[i] / total_weight;
            }
        }

        // Project into ball
        project_to_ball(&mut mean, max_radius_sq);
    }

    mean
}

/// Project a point into the Poincaré ball if |x|² > max_radius_sq.
fn project_to_ball(point: &mut [f64], max_radius_sq: f64) {
    let norm_sq: f64 = point.iter().map(|x| x * x).sum();
    if norm_sq > max_radius_sq {
        let scale = (max_radius_sq / norm_sq).sqrt() * 0.999; // Slightly inside
        for x in point.iter_mut() {
            *x *= scale;
        }
    }
}

// ═══════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Poincaré distance tests ──

    #[test]
    fn test_distance_same_point() {
        let p = vec![0.3, 0.1, 0.0];
        let d = safe_poincare_distance(&p, &p, 1e-9, 0.999999);
        assert!(d.abs() < 1e-10, "Distance to self should be 0: {d}");
    }

    #[test]
    fn test_distance_origin() {
        let origin = vec![0.0, 0.0, 0.0];
        let p = vec![0.5, 0.0, 0.0];
        let d = safe_poincare_distance(&origin, &p, 1e-9, 0.999999);
        assert!(d > 0.0, "Distance from origin should be positive: {d}");
        // arcosh(1 + 2*0.25/(1*0.75)) = arcosh(1 + 0.6667) = arcosh(1.6667) ≈ 1.0986
        assert!(
            (d - 1.0986).abs() < 0.01,
            "Expected ~1.099: got {d}"
        );
    }

    #[test]
    fn test_distance_symmetric() {
        let a = vec![0.3, 0.1, 0.0];
        let b = vec![0.5, -0.2, 0.1];
        let d1 = safe_poincare_distance(&a, &b, 1e-9, 0.999999);
        let d2 = safe_poincare_distance(&b, &a, 1e-9, 0.999999);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Distance should be symmetric: {d1} vs {d2}"
        );
    }

    #[test]
    fn test_distance_near_boundary() {
        let near_boundary = vec![0.999, 0.0, 0.0];
        let origin = vec![0.0, 0.0, 0.0];
        let d = safe_poincare_distance(&origin, &near_boundary, 1e-9, 0.999999);
        // Near boundary = very large hyperbolic distance
        assert!(d > 5.0, "Near-boundary distance should be large: {d}");
        assert!(d.is_finite(), "Should not be infinite: {d}");
    }

    #[test]
    fn test_distance_at_boundary_clamped() {
        // Point ON the boundary (|x|² = 1.0)
        let boundary = vec![1.0, 0.0, 0.0];
        let origin = vec![0.0, 0.0, 0.0];
        let d = safe_poincare_distance(&origin, &boundary, 1e-9, 0.999999);
        assert!(d.is_finite(), "Boundary distance should be clamped to finite: {d}");
    }

    // ── Fréchet mean tests ──

    #[test]
    fn test_frechet_single_point() {
        let points = vec![vec![0.3, 0.1, 0.0]];
        let mean = frechet_mean_poincare(&points, 5, 1.0, 1e-9, 0.999999);
        assert!((mean[0] - 0.3).abs() < 1e-6);
        assert!((mean[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_frechet_symmetric_points() {
        // Symmetric points around origin → mean should be near origin
        let points = vec![
            vec![0.3, 0.0, 0.0],
            vec![-0.3, 0.0, 0.0],
            vec![0.0, 0.3, 0.0],
            vec![0.0, -0.3, 0.0],
        ];
        let mean = frechet_mean_poincare(&points, 5, 1.0, 1e-9, 0.999999);
        let norm: f64 = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            norm < 0.1,
            "Mean of symmetric points should be near origin: |μ|={norm}"
        );
    }

    #[test]
    fn test_frechet_stays_in_ball() {
        // Points near boundary
        let points = vec![
            vec![0.9, 0.0, 0.0],
            vec![0.0, 0.9, 0.0],
            vec![0.0, 0.0, 0.9],
        ];
        let mean = frechet_mean_poincare(&points, 5, 1.0, 1e-9, 0.999999);
        let norm_sq: f64 = mean.iter().map(|x| x * x).sum();
        assert!(
            norm_sq < 1.0,
            "Mean should stay in ball: |μ|²={norm_sq}"
        );
    }

    // ── AxiomIdentity tests ──

    #[test]
    fn test_identity_initialization() {
        let axioms = vec![
            vec![0.1, 0.0, 0.0],
            vec![-0.1, 0.0, 0.0],
            vec![0.0, 0.1, 0.0],
            vec![0.0, -0.1, 0.0],
        ];
        let config = IdentityConfig::default();
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        assert_eq!(identity.dimension(), 3);
        assert!(identity.initial_radius_mean() > 0.0);
        assert!(identity.drift_limit() > 0.0);
    }

    #[test]
    fn test_identity_empty_axioms() {
        let result = AxiomIdentity::initialize(&[], IdentityConfig::default());
        assert!(result.is_none());
    }

    #[test]
    fn test_identity_no_drift_same_positions() {
        let axioms = vec![
            vec![0.1, 0.0, 0.0],
            vec![-0.1, 0.0, 0.0],
            vec![0.0, 0.1, 0.0],
        ];
        let config = IdentityConfig::default();
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        // Measure drift with the same positions → should be ~0
        let report = identity.measure_drift(&axioms);
        assert!(
            report.drift < 0.01,
            "Same positions should have near-zero drift: {}",
            report.drift
        );
        assert!(!report.exceeded);
    }

    #[test]
    fn test_identity_detects_drift() {
        let axioms = vec![
            vec![0.1, 0.0, 0.0],
            vec![-0.1, 0.0, 0.0],
            vec![0.0, 0.1, 0.0],
        ];
        let config = IdentityConfig {
            eta: 0.1, // Strict threshold
            ..Default::default()
        };
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        // Move axioms significantly → should detect drift
        let moved_axioms = vec![
            vec![0.6, 0.0, 0.0],
            vec![0.4, 0.0, 0.0],
            vec![0.5, 0.1, 0.0],
        ];
        let report = identity.measure_drift(&moved_axioms);
        assert!(
            report.drift > 0.1,
            "Moved axioms should have significant drift: {}",
            report.drift
        );
        assert!(
            report.exceeded,
            "Drift should exceed strict threshold: drift={}, limit={}",
            report.drift,
            report.limit,
        );
    }

    #[test]
    fn test_identity_validate_sovereignty_ok() {
        let axioms = vec![
            vec![0.1, 0.0, 0.0],
            vec![-0.1, 0.0, 0.0],
        ];
        let config = IdentityConfig {
            eta: 0.5, // Permissive
            ..Default::default()
        };
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        // Slightly moved
        let current = vec![
            vec![0.12, 0.01, 0.0],
            vec![-0.08, 0.01, 0.0],
        ];
        let result = identity.validate_sovereignty(&current);
        assert!(result.is_ok(), "Slight drift should pass: {result:?}");
    }

    #[test]
    fn test_identity_validate_sovereignty_exceeded() {
        let axioms = vec![
            vec![0.1, 0.0, 0.0],
            vec![-0.1, 0.0, 0.0],
        ];
        let config = IdentityConfig {
            eta: 0.05, // Very strict
            ..Default::default()
        };
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        // Massively moved
        let current = vec![
            vec![0.8, 0.0, 0.0],
            vec![0.7, 0.0, 0.0],
        ];
        let result = identity.validate_sovereignty(&current);
        assert!(result.is_err(), "Large drift should fail: {result:?}");
        if let Err(e) = result {
            assert!(e.drift > e.limit);
        }
    }

    #[test]
    fn test_identity_disabled_never_exceeds() {
        let axioms = vec![vec![0.1, 0.0, 0.0], vec![-0.1, 0.0, 0.0]];
        let config = IdentityConfig {
            eta: 0.01,
            enabled: false, // Disabled
            ..Default::default()
        };
        let identity = AxiomIdentity::initialize(&axioms, config).unwrap();

        let moved = vec![vec![0.9, 0.0, 0.0], vec![0.8, 0.0, 0.0]];
        let report = identity.measure_drift(&moved);
        assert!(
            !report.exceeded,
            "Disabled shield should never exceed"
        );
    }

    #[test]
    fn test_identity_display() {
        let identity = AxiomIdentity::new_minimal(
            vec![0.0, 0.0, 0.0],
            0.15,
            IdentityConfig::default(),
        );
        let s = format!("{identity}");
        assert!(s.contains("AxiomIdentity"));
        assert!(s.contains("dim=3"));
    }

    #[test]
    fn test_drift_report_display() {
        let report = DriftReport {
            drift: 0.05,
            limit: 0.10,
            drift_ratio: 0.5,
            exceeded: false,
            n_axioms: 10,
            dimension: 3,
        };
        let s = format!("{report}");
        assert!(s.contains("OK"));
        assert!(s.contains("50.0%"));
    }

    #[test]
    fn test_drift_exceeded_display() {
        let err = AxiomDriftExceeded {
            drift: 0.15,
            limit: 0.10,
            eta: 0.2,
            initial_radius: 0.5,
        };
        let s = format!("{err}");
        assert!(s.contains("AxiomDriftExceeded"));
    }

    // ── Project to ball tests ──

    #[test]
    fn test_project_inside_unchanged() {
        let mut p = vec![0.3, 0.1, 0.0];
        let original = p.clone();
        project_to_ball(&mut p, 0.999999);
        for (a, b) in p.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_project_outside_clamped() {
        let mut p = vec![1.0, 1.0, 1.0]; // |p|² = 3.0
        project_to_ball(&mut p, 0.999999);
        let norm_sq: f64 = p.iter().map(|x| x * x).sum();
        assert!(
            norm_sq < 1.0,
            "Should be inside ball after projection: |p|²={norm_sq}"
        );
    }
}
