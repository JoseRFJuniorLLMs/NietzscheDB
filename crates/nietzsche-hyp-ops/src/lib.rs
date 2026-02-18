//! # nietzsche-hyp-ops
//!
//! Hyperbolic geometry operations for the Poincaré ball model.
//!
//! This crate is the **single source of truth** for all hyperbolic math in
//! NietzscheDB. Every crate that touches Poincaré coordinates imports from
//! here — no inline reimplementations allowed.
//!
//! ## Core operations
//!
//! | Function | Direction | Purpose |
//! |---|---|---|
//! | [`exp_map_zero`] | Euclidean → Poincaré | Project encoder output into the ball |
//! | [`log_map_zero`] | Poincaré → Euclidean | Unproject for decoder input |
//! | [`mobius_add`] | Poincaré × Poincaré → Poincaré | Translate a point by another |
//! | [`poincare_distance`] | Poincaré × Poincaré → ℝ⁺ | Hyperbolic distance |
//! | [`gyromidpoint`] | \[Poincaré\] → Poincaré | Fréchet mean for multimodal fusion |
//!
//! ## Safety invariant
//!
//! Every vector returned by this crate satisfies **‖x‖ < 1.0** (open unit ball).
//! Use [`assert_poincare`] to validate external inputs.

pub mod error;

use error::HypError;

// ─────────────────────────────────────────────
// Exponential map at the origin
// ─────────────────────────────────────────────

/// Map a Euclidean (tangent-space) vector to the Poincaré ball.
///
/// ```text
/// exp₀(v) = tanh(‖v‖) · v / ‖v‖
/// ```
///
/// This is the exponential map at the origin of the Poincaré ball model.
/// The result always satisfies ‖x‖ < 1.0 because tanh(·) < 1.0 for finite input.
///
/// Used by every encoder to project latent vectors into hyperbolic space.
pub fn exp_map_zero(v: &[f64]) -> Vec<f64> {
    let norm = l2_norm(v);
    if norm < 1e-15 {
        return vec![0.0; v.len()];
    }
    let scale = norm.tanh() / norm;
    v.iter().map(|&x| x * scale).collect()
}

// ─────────────────────────────────────────────
// Logarithmic map at the origin
// ─────────────────────────────────────────────

/// Map a Poincaré ball point back to the Euclidean tangent space at the origin.
///
/// ```text
/// log₀(x) = atanh(‖x‖) · x / ‖x‖
/// ```
///
/// Inverse of [`exp_map_zero`]. Used by decoders to recover Euclidean
/// representations before feeding into convolutional layers.
///
/// # Errors
///
/// Returns [`HypError::OutsideBall`] if ‖x‖ ≥ 1.0.
pub fn log_map_zero(x: &[f64]) -> Result<Vec<f64>, HypError> {
    let norm = l2_norm(x);
    if norm >= 1.0 {
        return Err(HypError::OutsideBall { norm });
    }
    if norm < 1e-15 {
        return Ok(vec![0.0; x.len()]);
    }
    let scale = norm.atanh() / norm;
    Ok(x.iter().map(|&xi| xi * scale).collect())
}

// ─────────────────────────────────────────────
// Möbius addition
// ─────────────────────────────────────────────

/// Möbius addition u ⊕ v in the Poincaré ball.
///
/// ```text
/// u ⊕ v = [(1 + 2⟨u,v⟩ + ‖v‖²) · u  +  (1 − ‖u‖²) · v]
///          ───────────────────────────────────────────────
///                   1 + 2⟨u,v⟩ + ‖u‖²·‖v‖²
/// ```
///
/// # Panics
///
/// Debug-panics if `u` and `v` have different lengths.
pub fn mobius_add(u: &[f64], v: &[f64]) -> Vec<f64> {
    debug_assert_eq!(u.len(), v.len(), "dimension mismatch in Möbius add");

    let dot_uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom = 1.0 + 2.0 * dot_uv + norm_u_sq * norm_v_sq;
    let coeff_u = (1.0 + 2.0 * dot_uv + norm_v_sq) / denom;
    let coeff_v = (1.0 - norm_u_sq) / denom;

    let result: Vec<f64> = u.iter().zip(v.iter())
        .map(|(&ui, &vi)| coeff_u * ui + coeff_v * vi)
        .collect();

    // Safety: clamp to ball if floating-point drift pushes us out
    project_to_ball(&result, MAX_NORM)
}

// ─────────────────────────────────────────────
// Poincaré distance
// ─────────────────────────────────────────────

/// Hyperbolic distance in the Poincaré ball model.
///
/// ```text
/// d(u, v) = acosh(1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)))
/// ```
///
/// # Panics
///
/// Debug-panics if `u` and `v` have different lengths.
pub fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    debug_assert_eq!(u.len(), v.len(), "dimension mismatch in distance");

    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
    if denom <= 0.0 {
        return f64::INFINITY;
    }

    // Clamp to ≥ 1.0 to avoid NaN from floating-point noise
    let arg = (1.0 + 2.0 * diff_sq / denom).max(1.0);
    arg.acosh()
}

// ─────────────────────────────────────────────
// Gyromidpoint (Fréchet mean)
// ─────────────────────────────────────────────

/// Fréchet mean of multiple points in the Poincaré ball.
///
/// Computed via the tangent-space approximation:
/// 1. Map all points to the tangent space at the origin (`log_map_zero`)
/// 2. Compute Euclidean mean
/// 3. Map back to the ball (`exp_map_zero`)
///
/// This is the operation used for **multimodal fusion** in Phase 11:
/// ```text
/// z_fused = gyromidpoint([z_audio, z_text, z_image])
/// ```
///
/// # Errors
///
/// Returns [`HypError::EmptyInput`] if the slice is empty.
/// Returns [`HypError::OutsideBall`] if any point has ‖x‖ ≥ 1.0.
pub fn gyromidpoint(points: &[&[f64]]) -> Result<Vec<f64>, HypError> {
    if points.is_empty() {
        return Err(HypError::EmptyInput);
    }

    let dim = points[0].len();
    let n = points.len() as f64;

    // 1. Project to tangent space
    let mut sum = vec![0.0; dim];
    for &p in points {
        let tangent = log_map_zero(p)?;
        for (s, t) in sum.iter_mut().zip(tangent.iter()) {
            *s += t;
        }
    }

    // 2. Euclidean mean in tangent space
    for s in sum.iter_mut() {
        *s /= n;
    }

    // 3. Project back to ball
    Ok(exp_map_zero(&sum))
}

/// Weighted Fréchet mean — each point has an importance weight.
///
/// Useful when fusing modalities with different confidence levels:
/// ```text
/// z_fused = gyromidpoint_weighted(
///     &[z_audio, z_text],
///     &[0.7, 0.3],  // audio carries more weight
/// )
/// ```
pub fn gyromidpoint_weighted(points: &[&[f64]], weights: &[f64]) -> Result<Vec<f64>, HypError> {
    if points.is_empty() || weights.is_empty() {
        return Err(HypError::EmptyInput);
    }
    if points.len() != weights.len() {
        return Err(HypError::DimensionMismatch {
            expected: points.len(),
            got: weights.len(),
        });
    }

    let dim = points[0].len();
    let total_weight: f64 = weights.iter().sum();
    if total_weight < 1e-15 {
        return Err(HypError::EmptyInput);
    }

    let mut sum = vec![0.0; dim];
    for (&p, &w) in points.iter().zip(weights.iter()) {
        let tangent = log_map_zero(p)?;
        for (s, t) in sum.iter_mut().zip(tangent.iter()) {
            *s += t * w;
        }
    }

    for s in sum.iter_mut() {
        *s /= total_weight;
    }

    Ok(exp_map_zero(&sum))
}

// ─────────────────────────────────────────────
// Validation & Projection
// ─────────────────────────────────────────────

/// Maximum allowed norm. Points at exactly tanh(large) ≈ 0.9999... are fine,
/// but we clamp to this to prevent numerical issues in distance computation.
pub const MAX_NORM: f64 = 0.999;

/// Validate that a vector lies strictly inside the Poincaré ball.
///
/// Returns `Ok(())` if ‖x‖ < 1.0, otherwise [`HypError::OutsideBall`].
///
/// Call this on every vector entering the system from external sources
/// (gRPC inputs, deserialized data, encoder outputs before storage).
pub fn assert_poincare(x: &[f64]) -> Result<(), HypError> {
    let norm = l2_norm(x);
    if norm >= 1.0 {
        Err(HypError::OutsideBall { norm })
    } else {
        Ok(())
    }
}

/// Project a vector to have ‖result‖ ≤ `max_norm`.
///
/// If already inside, returns a clone. Otherwise rescales.
/// Typical usage: `project_to_ball(&v, 0.999)`.
pub fn project_to_ball(x: &[f64], max_norm: f64) -> Vec<f64> {
    let norm = l2_norm(x);
    if norm < max_norm {
        x.to_vec()
    } else {
        let scale = (max_norm - 1e-7) / (norm + 1e-15);
        x.iter().map(|&xi| xi * scale).collect()
    }
}

// ─────────────────────────────────────────────
// Parallel transport (for sleep cycle perturbation)
// ─────────────────────────────────────────────

/// Parallel transport a tangent vector from the origin to point `x`.
///
/// ```text
/// PT_{0→x}(v) = v · (1 − ‖x‖²) / 2
/// ```
///
/// Used in the sleep cycle to perturb embeddings in the correct tangent space.
pub fn parallel_transport_zero_to(x: &[f64], v: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), v.len());
    let norm_x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let conformal = (1.0 - norm_x_sq) / 2.0;
    v.iter().map(|&vi| vi * conformal).collect()
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

    // ── exp_map_zero ───────────────────────────

    #[test]
    fn exp_map_zero_of_zero_is_origin() {
        let v = vec![0.0, 0.0, 0.0];
        let result = exp_map_zero(&v);
        assert!(l2_norm(&result) < 1e-14);
    }

    #[test]
    fn exp_map_zero_always_inside_ball() {
        // Even for very large inputs, tanh keeps it < 1.0
        for scale in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] {
            let v = vec![scale, scale, scale];
            let result = exp_map_zero(&v);
            let norm = l2_norm(&result);
            assert!(norm < 1.0, "exp_map_zero({scale}) gave norm={norm}");
        }
    }

    #[test]
    fn exp_map_zero_unit_vector_gives_tanh_1() {
        // exp₀([1, 0, 0]) = [tanh(1), 0, 0]
        let v = vec![1.0, 0.0, 0.0];
        let result = exp_map_zero(&v);
        let expected_tanh_1 = 1.0_f64.tanh(); // 0.7615941559557649
        assert!(
            (result[0] - expected_tanh_1).abs() < 1e-12,
            "expected {expected_tanh_1}, got {}",
            result[0]
        );
        assert!(result[1].abs() < 1e-14);
        assert!(result[2].abs() < 1e-14);
    }

    #[test]
    fn exp_map_zero_preserves_direction() {
        let v = vec![3.0, 4.0]; // norm = 5
        let result = exp_map_zero(&v);
        // Direction should be same: result[0]/result[1] = 3/4
        let ratio = result[0] / result[1];
        assert!((ratio - 0.75).abs() < 1e-12);
    }

    // ── log_map_zero ───────────────────────────

    #[test]
    fn log_map_zero_of_origin_is_zero() {
        let x = vec![0.0, 0.0];
        let result = log_map_zero(&x).unwrap();
        assert!(l2_norm(&result) < 1e-14);
    }

    #[test]
    fn log_map_rejects_outside_ball() {
        let x = vec![0.8, 0.8]; // norm ≈ 1.13
        assert!(log_map_zero(&x).is_err());
    }

    #[test]
    fn exp_log_roundtrip() {
        // log₀(exp₀(v)) ≈ v for any v
        let v = vec![0.5, -0.3, 0.7];
        let mapped = exp_map_zero(&v);
        let recovered = log_map_zero(&mapped).unwrap();
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "roundtrip failed: {a} vs {b}"
            );
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        // exp₀(log₀(x)) ≈ x for any x inside ball
        let x = vec![0.3, -0.4, 0.2];
        let tangent = log_map_zero(&x).unwrap();
        let recovered = exp_map_zero(&tangent);
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "roundtrip failed: {a} vs {b}"
            );
        }
    }

    // ── mobius_add ──────────────────────────────

    #[test]
    fn mobius_add_identity_at_origin() {
        let zero = vec![0.0, 0.0];
        let v = vec![0.3, 0.1];
        let result = mobius_add(&zero, &v);
        assert!((result[0] - 0.3).abs() < 1e-12);
        assert!((result[1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn mobius_add_stays_in_ball() {
        let u = vec![0.5, 0.3];
        let v = vec![0.1, 0.2];
        let w = mobius_add(&u, &v);
        assert!(l2_norm(&w) < 1.0);
    }

    #[test]
    fn mobius_add_near_boundary_clamped() {
        // Two points near boundary — result must still be in ball
        let u = vec![0.9, 0.0];
        let v = vec![0.9, 0.0];
        let w = mobius_add(&u, &v);
        assert!(l2_norm(&w) < 1.0, "norm = {}", l2_norm(&w));
    }

    // ── poincare_distance ──────────────────────

    #[test]
    fn poincare_distance_self_is_zero() {
        let v = vec![0.3, 0.4];
        assert!(poincare_distance(&v, &v) < 1e-10);
    }

    #[test]
    fn poincare_distance_is_symmetric() {
        let u = vec![0.1, 0.2];
        let v = vec![0.3, -0.1];
        let d1 = poincare_distance(&u, &v);
        let d2 = poincare_distance(&v, &u);
        assert!((d1 - d2).abs() < 1e-12);
    }

    #[test]
    fn poincare_distance_triangle_inequality() {
        let a = vec![0.1, 0.0];
        let b = vec![0.3, 0.2];
        let c = vec![0.5, -0.1];
        let d_ab = poincare_distance(&a, &b);
        let d_bc = poincare_distance(&b, &c);
        let d_ac = poincare_distance(&a, &c);
        assert!(d_ac <= d_ab + d_bc + 1e-10);
    }

    #[test]
    fn poincare_distance_increases_near_boundary() {
        // Same Euclidean distance, but closer to boundary = larger hyperbolic distance
        let a = vec![0.0, 0.0];
        let b = vec![0.1, 0.0]; // near center
        let c = vec![0.8, 0.0];
        let d = vec![0.9, 0.0]; // near boundary

        let d_center = poincare_distance(&a, &b);
        let d_boundary = poincare_distance(&c, &d);
        assert!(
            d_boundary > d_center,
            "boundary distance {d_boundary:.4} should exceed center {d_center:.4}"
        );
    }

    // ── gyromidpoint ───────────────────────────

    #[test]
    fn gyromidpoint_single_point_is_identity() {
        let p = vec![0.3, 0.4];
        let mid = gyromidpoint(&[&p]).unwrap();
        for (a, b) in p.iter().zip(mid.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn gyromidpoint_symmetric_points_gives_origin() {
        // Midpoint of p and -p should be near the origin
        let p = vec![0.3, 0.4];
        let neg_p: Vec<f64> = p.iter().map(|x| -x).collect();
        let mid = gyromidpoint(&[&p, &neg_p]).unwrap();
        assert!(l2_norm(&mid) < 1e-10, "midpoint norm = {}", l2_norm(&mid));
    }

    #[test]
    fn gyromidpoint_result_inside_ball() {
        let a = vec![0.5, 0.3];
        let b = vec![-0.2, 0.6];
        let c = vec![0.1, -0.4];
        let mid = gyromidpoint(&[&a, &b, &c]).unwrap();
        assert!(l2_norm(&mid) < 1.0);
    }

    #[test]
    fn gyromidpoint_empty_returns_error() {
        let empty: &[&[f64]] = &[];
        assert!(gyromidpoint(empty).is_err());
    }

    // ── gyromidpoint_weighted ──────────────────

    #[test]
    fn gyromidpoint_weighted_equal_weights_matches_unweighted() {
        let a = vec![0.3, 0.2];
        let b = vec![-0.1, 0.4];
        let unweighted = gyromidpoint(&[&a, &b]).unwrap();
        let weighted = gyromidpoint_weighted(&[&a, &b], &[1.0, 1.0]).unwrap();
        for (u, w) in unweighted.iter().zip(weighted.iter()) {
            assert!((u - w).abs() < 1e-10);
        }
    }

    #[test]
    fn gyromidpoint_weighted_full_weight_one_point() {
        let a = vec![0.3, 0.2];
        let b = vec![-0.1, 0.4];
        // All weight on a → result should be close to a
        let result = gyromidpoint_weighted(&[&a, &b], &[1000.0, 0.001]).unwrap();
        for (r, expected) in result.iter().zip(a.iter()) {
            assert!((r - expected).abs() < 0.01);
        }
    }

    // ── validation ─────────────────────────────

    #[test]
    fn assert_poincare_accepts_valid() {
        assert!(assert_poincare(&[0.3, 0.4]).is_ok());
    }

    #[test]
    fn assert_poincare_rejects_boundary() {
        assert!(assert_poincare(&[0.8, 0.8]).is_err());
    }

    #[test]
    fn project_to_ball_identity_when_inside() {
        let v = vec![0.3, 0.4];
        let projected = project_to_ball(&v, 0.999);
        assert_eq!(v, projected);
    }

    #[test]
    fn project_to_ball_clamps_outside() {
        let v = vec![0.9, 0.9]; // norm ≈ 1.27
        let projected = project_to_ball(&v, 0.999);
        assert!(l2_norm(&projected) < 1.0);
    }

    // ── parallel transport ─────────────────────

    #[test]
    fn parallel_transport_at_origin_is_half() {
        // At origin, conformal factor = (1-0)/2 = 0.5
        let x = vec![0.0, 0.0];
        let v = vec![1.0, 0.0];
        let result = parallel_transport_zero_to(&x, &v);
        assert!((result[0] - 0.5).abs() < 1e-14);
    }

    // ── high-dimensional stress tests ──────────

    #[test]
    fn exp_log_roundtrip_128d() {
        let v: Vec<f64> = (0..128).map(|i| (i as f64 * 0.01) - 0.64).collect();
        let mapped = exp_map_zero(&v);
        assert!(l2_norm(&mapped) < 1.0);
        let recovered = log_map_zero(&mapped).unwrap();
        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-8, "128D roundtrip drift");
        }
    }

    #[test]
    fn gyromidpoint_256d_inside_ball() {
        // Simulate multimodal fusion at Phase 11 dimensions
        let a: Vec<f64> = (0..256).map(|i| (i as f64 * 0.003) - 0.384).collect();
        let b: Vec<f64> = (0..256).map(|i| ((i * 7 % 256) as f64 * 0.003) - 0.384).collect();
        let a_hyp = exp_map_zero(&a);
        let b_hyp = exp_map_zero(&b);
        let mid = gyromidpoint(&[&a_hyp, &b_hyp]).unwrap();
        assert!(l2_norm(&mid) < 1.0, "256D gyromidpoint outside ball");
    }
}
