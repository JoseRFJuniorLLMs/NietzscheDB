//! # Manifold normalization and health checking
//!
//! Repeated projections between manifolds (Poincaré → Klein → Poincaré,
//! Poincaré → Tangent → Sphere → Tangent → Poincaré) accumulate floating-point
//! error. This module provides:
//!
//! - **Post-projection normalization**: re-project vectors to their manifold
//!   after every inter-geometry transformation
//! - **Health checks**: validate that all stored embeddings satisfy their
//!   manifold invariants
//! - **Batch repair**: fix vectors that have drifted outside their manifold
//!
//! ## Invariants enforced
//!
//! | Manifold | Invariant |
//! |---|---|
//! | Poincaré ball | ‖x‖ < 1.0 (strict), clamped to MAX_NORM = 0.999 |
//! | Klein disk | ‖x‖ < 1.0 (strict), same boundary |
//! | Unit sphere | ‖x‖ = 1.0 (re-normalized after every operation) |
//!
//! ## Usage
//!
//! Insert normalization after every inter-manifold projection:
//! ```text
//! let klein = to_klein(&poincare)?;
//! let klein = normalize_klein(&klein);  // ← ensure invariant
//! // ... use klein for pathfinding ...
//! let poincare = to_poincare(&klein)?;
//! let poincare = normalize_poincare(&poincare);  // ← ensure invariant
//! ```

use crate::MAX_NORM;

// ─────────────────────────────────────────────
// Poincaré normalization
// ─────────────────────────────────────────────

/// Re-project a vector into the open Poincaré ball.
///
/// If ‖x‖ > MAX_NORM (0.999), rescale to MAX_NORM - ε.
/// If the vector contains NaN or Inf, returns the origin.
#[inline]
pub fn normalize_poincare(x: &[f64]) -> Vec<f64> {
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return vec![0.0; x.len()];
    }
    let norm = l2_norm(x);
    if norm < MAX_NORM {
        x.to_vec()
    } else {
        let scale = (MAX_NORM - 1e-7) / (norm + 1e-15);
        x.iter().map(|&xi| xi * scale).collect()
    }
}

/// Re-project f32 Poincaré coordinates.
#[inline]
pub fn normalize_poincare_f32(x: &[f32]) -> Vec<f32> {
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return vec![0.0; x.len()];
    }
    let norm: f64 = x.iter().map(|&v| { let f = v as f64; f * f }).sum::<f64>().sqrt();
    if norm < MAX_NORM {
        x.to_vec()
    } else {
        let scale = ((MAX_NORM - 1e-7) / (norm + 1e-15)) as f32;
        x.iter().map(|&xi| xi * scale).collect()
    }
}

// ─────────────────────────────────────────────
// Klein normalization
// ─────────────────────────────────────────────

/// Re-project a vector into the open Klein disk.
///
/// Klein and Poincaré share the same unit ball boundary,
/// so the same clamping applies.
#[inline]
pub fn normalize_klein(x: &[f64]) -> Vec<f64> {
    normalize_poincare(x) // Same invariant: ‖x‖ < 1.0
}

// ─────────────────────────────────────────────
// Sphere normalization
// ─────────────────────────────────────────────

/// Re-project a vector onto the unit sphere.
///
/// If the vector is zero or NaN, returns a default direction (first basis vector).
#[inline]
pub fn normalize_sphere(x: &[f64]) -> Vec<f64> {
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        let mut basis = vec![0.0; x.len()];
        if !basis.is_empty() { basis[0] = 1.0; }
        return basis;
    }
    let norm = l2_norm(x);
    if norm < 1e-15 {
        let mut basis = vec![0.0; x.len()];
        if !basis.is_empty() { basis[0] = 1.0; }
        return basis;
    }
    x.iter().map(|&xi| xi / norm).collect()
}

// ─────────────────────────────────────────────
// Health checking
// ─────────────────────────────────────────────

/// Result of a manifold health check on a single vector.
#[derive(Debug, Clone)]
pub enum HealthStatus {
    /// Vector satisfies all invariants.
    Healthy,
    /// Vector was outside the manifold and has been repaired.
    Repaired { original_norm: f64, repaired_norm: f64 },
    /// Vector contained NaN/Inf and was reset to origin.
    Reset,
}

/// Check and repair a Poincaré vector. Returns the (possibly repaired) vector
/// and its health status.
pub fn health_check_poincare(x: &[f32]) -> (Vec<f32>, HealthStatus) {
    if x.iter().any(|v| v.is_nan() || v.is_infinite()) {
        return (vec![0.0; x.len()], HealthStatus::Reset);
    }

    let norm: f64 = x.iter().map(|&v| { let f = v as f64; f * f }).sum::<f64>().sqrt();

    if norm < MAX_NORM {
        (x.to_vec(), HealthStatus::Healthy)
    } else {
        let repaired = normalize_poincare_f32(x);
        let repaired_norm: f64 = repaired.iter()
            .map(|&v| { let f = v as f64; f * f })
            .sum::<f64>()
            .sqrt();
        (repaired, HealthStatus::Repaired {
            original_norm: norm,
            repaired_norm,
        })
    }
}

/// Batch health check: returns number of healthy, repaired, and reset vectors.
pub fn batch_health_check(vectors: &[&[f32]]) -> (usize, usize, usize) {
    let mut healthy = 0usize;
    let mut repaired = 0usize;
    let mut reset = 0usize;

    for &v in vectors {
        match health_check_poincare(v).1 {
            HealthStatus::Healthy => healthy += 1,
            HealthStatus::Repaired { .. } => repaired += 1,
            HealthStatus::Reset => reset += 1,
        }
    }

    (healthy, repaired, reset)
}

// ─────────────────────────────────────────────
// Cascaded projection with normalization
// ─────────────────────────────────────────────

/// Safe Poincaré → Klein → Poincaré roundtrip with normalization at each step.
///
/// This is the pattern to use whenever you need a temporary Klein projection:
/// ```text
/// let (klein, poincare_back) = safe_klein_roundtrip(&poincare_coords);
/// ```
pub fn safe_klein_roundtrip(poincare: &[f64]) -> Result<(Vec<f64>, Vec<f64>), crate::error::HypError> {
    let p_normalized = normalize_poincare(poincare);
    let klein = crate::klein::to_klein(&p_normalized)?;
    let klein_normalized = normalize_klein(&klein);
    let back = crate::klein::to_poincare(&klein_normalized)?;
    let back_normalized = normalize_poincare(&back);
    Ok((klein_normalized, back_normalized))
}

/// Safe Poincaré → Sphere → Poincaré roundtrip with normalization.
pub fn safe_sphere_roundtrip(
    poincare: &[f64],
) -> Result<(Vec<f64>, f64, Vec<f64>), crate::error::HypError> {
    let p_normalized = normalize_poincare(poincare);
    let (sphere, magnitude) = crate::riemann::to_sphere_with_magnitude(&p_normalized)?;
    let sphere_normalized = normalize_sphere(&sphere);
    let back = crate::riemann::from_sphere(&sphere_normalized, magnitude);
    let back_normalized = normalize_poincare(&back);
    Ok((sphere_normalized, magnitude, back_normalized))
}

// ─────────────────────────────────────────────
// Post-query sanitization
// ─────────────────────────────────────────────

/// Results from a `manifold_sanitize` pass.
#[derive(Debug, Clone, Default)]
pub struct SanitizeReport {
    /// Number of vectors that were already healthy.
    pub healthy: usize,
    /// Number of vectors that were repaired (re-projected).
    pub repaired: usize,
    /// Number of vectors that contained NaN/Inf and were reset to origin.
    pub reset: usize,
}

/// Sanitize a batch of Poincaré f32 vectors after a multi-manifold operation.
///
/// This is the **post-query hook**: call it after any operation that involves
/// inter-manifold projection (Synthesis, KleinPath, etc.) to guarantee all
/// returned embeddings satisfy ‖x‖ < 1.0.
///
/// Returns the sanitized vectors and a report of how many were repaired.
///
/// # Example
///
/// ```text
/// let (clean_vectors, report) = manifold_sanitize(&raw_results);
/// if report.repaired > 0 {
///     tracing::warn!("repaired {} drifted vectors", report.repaired);
/// }
/// ```
pub fn manifold_sanitize(vectors: &[Vec<f32>]) -> (Vec<Vec<f32>>, SanitizeReport) {
    let mut report = SanitizeReport::default();
    let mut result = Vec::with_capacity(vectors.len());

    for v in vectors {
        let (repaired, status) = health_check_poincare(v);
        match status {
            HealthStatus::Healthy => report.healthy += 1,
            HealthStatus::Repaired { .. } => report.repaired += 1,
            HealthStatus::Reset => report.reset += 1,
        }
        result.push(repaired);
    }

    (result, report)
}

/// Sanitize a single f64 Poincaré vector (used after synthesis/Klein roundtrip).
///
/// Returns the sanitized vector. If already healthy, returns a clone.
pub fn sanitize_poincare_f64(v: &[f64]) -> Vec<f64> {
    if v.iter().any(|x| x.is_nan() || x.is_infinite()) {
        return vec![0.0; v.len()];
    }
    normalize_poincare(v)
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_poincare_inside_unchanged() {
        let v = vec![0.3, 0.4];
        let n = normalize_poincare(&v);
        assert_eq!(v, n);
    }

    #[test]
    fn normalize_poincare_outside_clamped() {
        let v = vec![0.9, 0.9]; // norm ≈ 1.27
        let n = normalize_poincare(&v);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "normalized norm = {norm}");
    }

    #[test]
    fn normalize_poincare_nan_resets() {
        let v = vec![f64::NAN, 0.5];
        let n = normalize_poincare(&v);
        assert_eq!(n, vec![0.0, 0.0]);
    }

    #[test]
    fn normalize_sphere_gives_unit() {
        let v = vec![3.0, 4.0];
        let n = normalize_sphere(&v);
        let norm: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    #[test]
    fn normalize_sphere_zero_gives_basis() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize_sphere(&v);
        assert_eq!(n, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn health_check_healthy() {
        let v = vec![0.3f32, 0.4];
        let (_, status) = health_check_poincare(&v);
        assert!(matches!(status, HealthStatus::Healthy));
    }

    #[test]
    fn health_check_repairs() {
        let v = vec![0.9f32, 0.9]; // norm ≈ 1.27
        let (repaired, status) = health_check_poincare(&v);
        assert!(matches!(status, HealthStatus::Repaired { .. }));
        let norm: f64 = repaired.iter().map(|&x| { let f = x as f64; f * f }).sum::<f64>().sqrt();
        assert!(norm < 1.0);
    }

    #[test]
    fn health_check_resets_nan() {
        let v = vec![f32::NAN, 0.5];
        let (repaired, status) = health_check_poincare(&v);
        assert!(matches!(status, HealthStatus::Reset));
        assert_eq!(repaired, vec![0.0, 0.0]);
    }

    #[test]
    fn safe_klein_roundtrip_preserves() {
        let p = vec![0.3, 0.4, 0.1];
        let (_, back) = safe_klein_roundtrip(&p).unwrap();
        for (a, b) in p.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-10, "Klein roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn safe_sphere_roundtrip_preserves() {
        let p = vec![0.3, 0.4, 0.1];
        let (_, _, back) = safe_sphere_roundtrip(&p).unwrap();
        for (a, b) in p.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-6, "Sphere roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn cascaded_10x_klein_roundtrip_error_bounded() {
        // Spec: 10 sequential P→K→P roundtrips: error < 1e-4
        let mut p = vec![0.3, 0.4, 0.2, -0.1];
        for i in 0..10 {
            let (_, back) = safe_klein_roundtrip(&p).unwrap();
            p = back;
        }
        let original = vec![0.3, 0.4, 0.2, -0.1];
        let max_error: f64 = original.iter().zip(p.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            max_error < 1e-4,
            "10x Klein roundtrip error {max_error} exceeds 1e-4"
        );
    }
}
