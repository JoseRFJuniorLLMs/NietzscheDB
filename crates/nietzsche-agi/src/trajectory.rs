//! # Trajectory — Geodesic Coherence Score (GCS) validation
//!
//! Validates that a path through the Poincaré ball follows actual geodesics.
//!
//! ## How GCS works
//!
//! For each consecutive triple (A, B, C) in a path:
//! 1. Convert A, B, C to Klein coordinates (where geodesics = straight lines)
//! 2. Check if B lies on (or near) the geodesic segment A→C
//! 3. Compute `gcs(A,B,C) = 1 - deviation / max_deviation`
//!
//! A hop is "coherent" if B is a natural waypoint on the geodesic from A to C.
//! If B deviates significantly, it suggests a non-geodesic "jump" that breaks
//! the semantic thread of reasoning.
//!
//! ## Per-hop vs aggregate GCS
//!
//! - Per-hop GCS: `gcs[i]` for the triple `(path[i-1], path[i], path[i+1])`
//! - Aggregate GCS: harmonic mean of all per-hop scores
//!
//! The harmonic mean ensures that a single bad hop drags down the entire score,
//! preventing "dilution" of a logical rupture by many good hops.

use uuid::Uuid;
use nietzsche_hyp_ops::klein;

use crate::error::{AgiError, AgiResult};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for GCS validation.
#[derive(Debug, Clone)]
pub struct GcsConfig {
    /// Minimum acceptable GCS for a single hop. Below this = LogicalRupture.
    /// Default: 0.5 (50% geodesic coherence).
    pub hop_threshold: f64,

    /// Minimum acceptable aggregate GCS for the entire trajectory.
    /// Default: 0.6 (stricter than per-hop because harmonic mean is conservative).
    pub trajectory_threshold: f64,

    /// Epsilon for Klein collinearity check.
    /// Default: 0.15 (generous for high-dimensional spaces).
    pub collinearity_epsilon: f64,
}

impl Default for GcsConfig {
    fn default() -> Self {
        Self {
            hop_threshold: 0.5,
            trajectory_threshold: 0.6,
            collinearity_epsilon: 0.15,
        }
    }
}

// ─────────────────────────────────────────────
// GeodesicCoherenceScore — per-hop metric
// ─────────────────────────────────────────────

/// Per-hop geodesic coherence score.
#[derive(Debug, Clone, Copy)]
pub struct GeodesicCoherenceScore {
    /// The hop index (0-based, refers to the middle node of the triple).
    pub hop: usize,

    /// GCS value ∈ [0.0, 1.0].
    /// 1.0 = perfectly on the geodesic.
    /// 0.0 = maximally deviant.
    pub score: f64,

    /// Whether this hop is above the per-hop threshold.
    pub is_coherent: bool,
}

// ─────────────────────────────────────────────
// GeodesicTrajectory — validated path
// ─────────────────────────────────────────────

/// A path through the Poincaré ball with GCS validation results.
#[derive(Debug, Clone)]
pub struct GeodesicTrajectory {
    /// Ordered node IDs forming the path.
    pub path: Vec<Uuid>,

    /// Embeddings (promoted to f64) for each node in the path.
    /// `embeddings[i]` corresponds to `path[i]`.
    pub embeddings: Vec<Vec<f64>>,

    /// Per-hop GCS scores. Length = `path.len() - 2` (one per interior node).
    /// For paths of length 2, this is empty (direct connection, GCS = 1.0).
    pub hop_scores: Vec<GeodesicCoherenceScore>,

    /// Aggregate GCS (harmonic mean of hop scores, or 1.0 for direct connections).
    pub aggregate_gcs: f64,

    /// Whether the entire trajectory passes validation.
    pub is_valid: bool,

    /// Radial gradient: `depth(last) - depth(first)`.
    pub radial_gradient: f64,
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Validate a trajectory through the Poincaré ball.
///
/// # Arguments
/// - `path`: ordered node IDs
/// - `embeddings`: Poincaré coordinates for each node (f64, promoted from f32)
/// - `config`: GCS configuration
///
/// # Returns
/// A [`GeodesicTrajectory`] with all per-hop scores and the aggregate GCS.
///
/// # Errors
/// - [`AgiError::EmptyInput`] if path has fewer than 2 nodes
/// - [`AgiError::DimensionMismatch`] if path and embeddings have different lengths
pub fn validate_trajectory(
    path: &[Uuid],
    embeddings: &[Vec<f64>],
    config: &GcsConfig,
) -> AgiResult<GeodesicTrajectory> {
    if path.len() < 2 {
        return Err(AgiError::EmptyInput {
            context: "trajectory must have at least 2 nodes".into(),
        });
    }
    if path.len() != embeddings.len() {
        return Err(AgiError::DimensionMismatch {
            expected: path.len(),
            got: embeddings.len(),
        });
    }

    // Validate all embeddings have the same dimension
    let dim = embeddings[0].len();
    for (_i, emb) in embeddings.iter().enumerate().skip(1) {
        if emb.len() != dim {
            return Err(AgiError::DimensionMismatch {
                expected: dim,
                got: emb.len(),
            });
        }
    }

    // Compute radial gradient
    let norm_first = vec_norm(&embeddings[0]);
    let norm_last = vec_norm(embeddings.last().unwrap());
    let radial_gradient = norm_last - norm_first;

    // For direct connections (2 nodes), GCS is 1.0 by definition
    if path.len() == 2 {
        return Ok(GeodesicTrajectory {
            path: path.to_vec(),
            embeddings: embeddings.to_vec(),
            hop_scores: vec![],
            aggregate_gcs: 1.0,
            is_valid: true,
            radial_gradient,
        });
    }

    // Convert all points to Klein model for collinearity checks
    let klein_points: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|p| klein::to_klein(p).unwrap_or_else(|_| p.clone()))
        .collect();

    // Compute per-hop GCS for each interior node
    let mut hop_scores = Vec::with_capacity(path.len() - 2);

    for i in 1..path.len() - 1 {
        let a = &klein_points[i - 1];
        let b = &klein_points[i];
        let c = &klein_points[i + 1];

        let score = compute_gcs_klein(a, b, c, config.collinearity_epsilon);
        let is_coherent = score >= config.hop_threshold;

        hop_scores.push(GeodesicCoherenceScore {
            hop: i,
            score,
            is_coherent,
        });
    }

    // Aggregate GCS via harmonic mean
    let scores: Vec<f64> = hop_scores.iter().map(|h| h.score).collect();
    let aggregate_gcs = harmonic_mean_f64(&scores);
    let is_valid = aggregate_gcs >= config.trajectory_threshold
        && hop_scores.iter().all(|h| h.is_coherent);

    Ok(GeodesicTrajectory {
        path: path.to_vec(),
        embeddings: embeddings.to_vec(),
        hop_scores,
        aggregate_gcs,
        is_valid,
        radial_gradient,
    })
}

/// Compute GCS for a single hop (A → B → C) using Poincaré coordinates.
///
/// Promotes f32 embeddings to f64 internally.
pub fn compute_gcs_poincare(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    epsilon: f64,
) -> f64 {
    let a64: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let b64: Vec<f64> = b.iter().map(|&x| x as f64).collect();
    let c64: Vec<f64> = c.iter().map(|&x| x as f64).collect();

    let ka = klein::to_klein(&a64).unwrap_or(a64);
    let kb = klein::to_klein(&b64).unwrap_or(b64);
    let kc = klein::to_klein(&c64).unwrap_or(c64);

    compute_gcs_klein(&ka, &kb, &kc, epsilon)
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Compute the GCS for a triple (A, B, C) already in Klein coordinates.
///
/// GCS = 1 - (perpendicular_distance(B, line(A,C)) / max_distance)
///
/// where max_distance = klein_distance(A, C) / 2 (the maximum possible deviation
/// for a point "between" A and C in the Klein model).
fn compute_gcs_klein(a: &[f64], b: &[f64], c: &[f64], epsilon: f64) -> f64 {
    // If A and C are the same point, B must also be the same
    let ac_dist = euclidean_dist(a, c);
    if ac_dist < epsilon {
        let ab_dist = euclidean_dist(a, b);
        return if ab_dist < epsilon { 1.0 } else { 0.0 };
    }

    // In Klein model, geodesics are straight lines.
    // Perpendicular distance from B to line(A, C):
    let perp_dist = perpendicular_distance_to_line(a, c, b);

    // Max deviation = half the distance A→C (reasonable upper bound)
    let max_dev = ac_dist / 2.0;
    if max_dev < 1e-15 {
        return 1.0;
    }

    // Also check if B is "between" A and C (not beyond either endpoint)
    let between_bonus = if is_between_projection(a, c, b) {
        1.0
    } else {
        0.7 // Penalty for overshoot
    };

    let raw_gcs = 1.0 - (perp_dist / max_dev).min(1.0);
    (raw_gcs * between_bonus).clamp(0.0, 1.0)
}

/// Perpendicular distance from point P to the line through A and C.
///
/// Uses the formula: d = ‖(P-A) - ((P-A)·(C-A))/(‖C-A‖²) · (C-A)‖
fn perpendicular_distance_to_line(a: &[f64], c: &[f64], p: &[f64]) -> f64 {
    let n = a.len();
    debug_assert_eq!(n, c.len());
    debug_assert_eq!(n, p.len());

    // vec_pa = P - A
    // vec_ca = C - A
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

    // Projection point = A + t * (C - A)
    // Perpendicular vector = PA - t * CA
    let mut perp = vec![0.0; n];
    for i in 0..n {
        perp[i] = pa[i] - t * ca[i];
    }

    vec_norm(&perp)
}

/// Check if the projection of B onto line(A,C) falls between A and C.
fn is_between_projection(a: &[f64], c: &[f64], b: &[f64]) -> bool {
    let n = a.len();
    let mut ba = vec![0.0; n];
    let mut ca = vec![0.0; n];
    for i in 0..n {
        ba[i] = b[i] - a[i];
        ca[i] = c[i] - a[i];
    }
    let ca_sq: f64 = ca.iter().map(|x| x * x).sum();
    if ca_sq < 1e-30 {
        return true;
    }
    let t: f64 = ba.iter().zip(ca.iter()).map(|(a, b)| a * b).sum::<f64>() / ca_sq;
    t >= -0.1 && t <= 1.1 // Slight tolerance
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

fn harmonic_mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0; // No interior nodes → perfect score
    }
    let sum_recip: f64 = values.iter().map(|&v| {
        if v <= 0.0 { f64::INFINITY } else { 1.0 / v }
    }).sum();
    if sum_recip.is_infinite() || sum_recip == 0.0 {
        return 0.0;
    }
    values.len() as f64 / sum_recip
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_connection_is_valid() {
        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let embs = vec![vec![0.1, 0.0, 0.0], vec![0.3, 0.0, 0.0]];
        let config = GcsConfig::default();
        let traj = validate_trajectory(&ids, &embs, &config).unwrap();
        assert!(traj.is_valid);
        assert!((traj.aggregate_gcs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_collinear_path_high_gcs() {
        // Three points roughly on a line in Poincaré ball
        let ids = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let embs = vec![
            vec![0.1, 0.0, 0.0],
            vec![0.2, 0.0, 0.0], // On the geodesic
            vec![0.3, 0.0, 0.0],
        ];
        let config = GcsConfig::default();
        let traj = validate_trajectory(&ids, &embs, &config).unwrap();
        assert!(traj.aggregate_gcs > 0.9, "GCS should be high for collinear: {}", traj.aggregate_gcs);
        assert!(traj.is_valid);
    }

    #[test]
    fn test_deviant_path_low_gcs() {
        // B is far off the geodesic A→C
        let ids = vec![Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4()];
        let embs = vec![
            vec![0.1, 0.0, 0.0],
            vec![0.0, 0.5, 0.0], // Way off the A→C geodesic
            vec![0.3, 0.0, 0.0],
        ];
        let config = GcsConfig::default();
        let traj = validate_trajectory(&ids, &embs, &config).unwrap();
        assert!(traj.aggregate_gcs < 0.5, "GCS should be low for deviant: {}", traj.aggregate_gcs);
    }

    #[test]
    fn test_radial_gradient() {
        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        // First node near center (norm ≈ 0.1), second near boundary (norm ≈ 0.5)
        let embs = vec![vec![0.1, 0.0, 0.0], vec![0.4, 0.3, 0.0]];
        let config = GcsConfig::default();
        let traj = validate_trajectory(&ids, &embs, &config).unwrap();
        assert!(traj.radial_gradient > 0.0, "Should move outward");
    }

    #[test]
    fn test_too_few_nodes() {
        let ids = vec![Uuid::new_v4()];
        let embs = vec![vec![0.1]];
        let config = GcsConfig::default();
        assert!(validate_trajectory(&ids, &embs, &config).is_err());
    }
}
