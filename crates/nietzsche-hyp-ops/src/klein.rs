//! # Klein model operations
//!
//! The Klein (Beltrami-Klein) model of hyperbolic space where **geodesics are
//! straight lines**. This makes pathfinding, colinearity checks, and shortest-path
//! verification trivial linear algebra instead of expensive trigonometric operations.
//!
//! ## When to use Klein
//!
//! | Operation | Poincaré cost | Klein cost |
//! |---|---|---|
//! | Colinearity check | arc intersection (trig) | determinant (O(1)) |
//! | Shortest path | atanh-based A* | Euclidean A* |
//! | Geodesic intersection | circle-circle intersect | line-line intersect |
//!
//! ## Conversion formulas
//!
//! ```text
//! Poincaré → Klein:  k_i = 2·p_i / (1 + ‖p‖²)
//! Klein → Poincaré:  p_i = k_i / (1 + √(1 − ‖k‖²))
//! ```
//!
//! ## Safety invariant
//!
//! All vectors returned by this module satisfy **‖x‖ < 1.0** (open unit ball).
//! Klein and Poincaré share the same disk — only the metric differs.

use crate::error::HypError;

// ─────────────────────────────────────────────
// Poincaré → Klein projection
// ─────────────────────────────────────────────

/// Project a Poincaré ball point into the Klein model.
///
/// ```text
/// k_i = 2·p_i / (1 + ‖p‖²)
/// ```
///
/// The Klein point is always inside the unit disk because for ‖p‖ < 1:
/// ‖k‖ = 2‖p‖/(1+‖p‖²) < 1.
///
/// # Errors
///
/// Returns [`HypError::OutsideBall`] if ‖p‖ ≥ 1.0.
#[inline]
pub fn to_klein(poincare: &[f64]) -> Result<Vec<f64>, HypError> {
    let norm_sq: f64 = poincare.iter().map(|x| x * x).sum();
    if norm_sq >= 1.0 {
        return Err(HypError::OutsideBall { norm: norm_sq.sqrt() });
    }
    let denom = 1.0 + norm_sq;
    Ok(poincare.iter().map(|&p| (2.0 * p) / denom).collect())
}

// ─────────────────────────────────────────────
// Klein → Poincaré projection
// ─────────────────────────────────────────────

/// Project a Klein model point back to the Poincaré ball.
///
/// ```text
/// p_i = k_i / (1 + √(1 − ‖k‖²))
/// ```
///
/// # Errors
///
/// Returns [`HypError::OutsideBall`] if ‖k‖ ≥ 1.0.
#[inline]
pub fn to_poincare(klein: &[f64]) -> Result<Vec<f64>, HypError> {
    let norm_sq: f64 = klein.iter().map(|x| x * x).sum();
    if norm_sq >= 1.0 {
        return Err(HypError::OutsideBall { norm: norm_sq.sqrt() });
    }
    let denom = 1.0 + (1.0 - norm_sq).sqrt();
    Ok(klein.iter().map(|&k| k / denom).collect())
}

// ─────────────────────────────────────────────
// Klein distance
// ─────────────────────────────────────────────

/// Hyperbolic distance computed in the Klein model.
///
/// ```text
/// d(a, b) = acosh( (1 − ⟨a,b⟩) / √((1−‖a‖²)(1−‖b‖²)) )
/// ```
///
/// Returns the same value as Poincaré distance for corresponding points.
pub fn klein_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch in klein_distance");

    let dot_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a_sq: f64 = a.iter().map(|x| x * x).sum();
    let norm_b_sq: f64 = b.iter().map(|x| x * x).sum();

    let denom = ((1.0 - norm_a_sq) * (1.0 - norm_b_sq)).sqrt();
    if denom <= 0.0 {
        return f64::INFINITY;
    }

    let arg = ((1.0 - dot_ab) / denom).max(1.0);
    arg.acosh()
}

// ─────────────────────────────────────────────
// Colinearity (geodesic check)
// ─────────────────────────────────────────────

/// Check if three points in Klein space are collinear (lie on the same geodesic).
///
/// In the Klein model, geodesics are straight lines. Collinearity is verified
/// by checking that the cross product (2D) or determinant (nD) of the vectors
/// `(b - a)` and `(c - a)` is zero within tolerance.
///
/// For n > 2 dimensions, we check that `c - a` is a linear combination of
/// `b - a` by verifying the rank of the matrix `[b-a | c-a]` is 1.
///
/// `epsilon` controls numerical tolerance. Recommended: `1e-7` for f64.
pub fn is_collinear(a: &[f64], b: &[f64], c: &[f64], epsilon: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());

    let dim = a.len();

    // Vectors from a to b and a to c
    let ab: Vec<f64> = a.iter().zip(b.iter()).map(|(ai, bi)| bi - ai).collect();
    let ac: Vec<f64> = a.iter().zip(c.iter()).map(|(ai, ci)| ci - ai).collect();

    let norm_ab: f64 = ab.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_ab < epsilon {
        // a ≈ b → trivially collinear
        return true;
    }

    let norm_ac: f64 = ac.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_ac < epsilon {
        // a ≈ c → trivially collinear
        return true;
    }

    if dim == 2 {
        // 2D: cross product = ab.x * ac.y - ab.y * ac.x
        let cross = ab[0] * ac[1] - ab[1] * ac[0];
        return cross.abs() < epsilon * norm_ab * norm_ac;
    }

    // nD: check that the cross product norm / (norm_ab * norm_ac) < epsilon
    // For arbitrary dimension, compute ‖ab × ac‖² = ‖ab‖²·‖ac‖² − (ab·ac)²
    let dot_ab_ac: f64 = ab.iter().zip(ac.iter()).map(|(x, y)| x * y).sum();
    let cross_sq = (norm_ab * norm_ab) * (norm_ac * norm_ac) - dot_ab_ac * dot_ab_ac;

    // Normalize: sin²(θ) = cross_sq / (‖ab‖²·‖ac‖²)
    let sin_sq = cross_sq / ((norm_ab * norm_ab) * (norm_ac * norm_ac));
    sin_sq.abs() < epsilon * epsilon
}

// ─────────────────────────────────────────────
// Shortest path check
// ─────────────────────────────────────────────

/// Check if point `c` lies on the shortest geodesic path between `a` and `b`
/// in the Klein model.
///
/// In Klein space, this is equivalent to checking:
/// 1. `c` is collinear with `a` and `b` (same geodesic)
/// 2. `c` lies **between** `a` and `b` (convex combination)
///
/// This replaces the expensive Poincaré-space computation that requires
/// arc-of-circle intersection with a simple linear algebra check.
///
/// All inputs must be in **Klein** coordinates (use [`to_klein`] first).
pub fn is_on_shortest_path(a: &[f64], b: &[f64], c: &[f64], epsilon: f64) -> bool {
    if !is_collinear(a, b, c, epsilon) {
        return false;
    }

    // c must be a convex combination of a and b: c = t·a + (1-t)·b, t ∈ [0, 1]
    // Solve for t using the dimension with largest |b_i - a_i| to avoid division by zero.
    let ab: Vec<f64> = a.iter().zip(b.iter()).map(|(ai, bi)| bi - ai).collect();
    let ac: Vec<f64> = a.iter().zip(c.iter()).map(|(ai, ci)| ci - ai).collect();

    // Find the dimension with max |ab|
    let (best_dim, max_ab) = ab.iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().partial_cmp(&y.abs()).unwrap())
        .unwrap();

    if max_ab.abs() < epsilon {
        // a ≈ b → c must also be ≈ a
        return ac.iter().map(|x| x * x).sum::<f64>().sqrt() < epsilon;
    }

    let t = ac[best_dim] / ab[best_dim];
    // t ∈ [0, 1] means c is between a and b
    t >= -epsilon && t <= 1.0 + epsilon
}

// ─────────────────────────────────────────────
// Batch projection (for pathfinding pipelines)
// ─────────────────────────────────────────────

/// Project a batch of Poincaré vectors to Klein in one pass.
///
/// Skips invalid vectors (‖p‖ ≥ 1.0) and returns `None` at those positions.
/// Useful for projecting an entire BFS frontier before pathfinding.
pub fn batch_to_klein(poincare_batch: &[&[f64]]) -> Vec<Option<Vec<f64>>> {
    poincare_batch.iter().map(|p| to_klein(p).ok()).collect()
}

/// Project a batch of Klein vectors back to Poincaré.
pub fn batch_to_poincare(klein_batch: &[&[f64]]) -> Vec<Option<Vec<f64>>> {
    klein_batch.iter().map(|k| to_poincare(k).ok()).collect()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    // ── to_klein / to_poincare roundtrip ─────────

    #[test]
    fn klein_poincare_roundtrip_2d() {
        let p = vec![0.3, 0.4];
        let k = to_klein(&p).unwrap();
        let recovered = to_poincare(&k).unwrap();
        for (a, b) in p.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-12, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn klein_poincare_roundtrip_128d() {
        let p: Vec<f64> = (0..128).map(|i| (i as f64 * 0.005) - 0.32).collect();
        assert!(l2_norm(&p) < 1.0, "test vector must be inside ball");
        let k = to_klein(&p).unwrap();
        let recovered = to_poincare(&k).unwrap();
        for (a, b) in p.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-10, "128D roundtrip drift");
        }
    }

    #[test]
    fn roundtrip_error_below_threshold() {
        // Spec: Poincaré ↔ Klein roundtrip error < 1e-6 on 10,000 random vectors
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..10_000 {
            let dim = 8;
            let p: Vec<f64> = (0..dim).map(|_| rng.gen_range(-0.9..0.9)).collect();
            let norm = l2_norm(&p);
            if norm >= 0.999 { continue; }
            let k = to_klein(&p).unwrap();
            let recovered = to_poincare(&k).unwrap();
            let error: f64 = p.iter().zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(error < 1e-6, "roundtrip error {error} exceeds 1e-6");
        }
    }

    #[test]
    fn to_klein_origin_stays_origin() {
        let p = vec![0.0, 0.0, 0.0];
        let k = to_klein(&p).unwrap();
        assert!(l2_norm(&k) < 1e-15);
    }

    #[test]
    fn to_klein_stays_inside_ball() {
        let p = vec![0.9, 0.0];
        let k = to_klein(&p).unwrap();
        assert!(l2_norm(&k) < 1.0, "Klein norm {} >= 1.0", l2_norm(&k));
    }

    #[test]
    fn to_klein_rejects_outside_ball() {
        let p = vec![0.8, 0.8]; // norm ≈ 1.13
        assert!(to_klein(&p).is_err());
    }

    // ── Klein distance ───────────────────────────

    #[test]
    fn klein_distance_self_is_zero() {
        let k = to_klein(&[0.3, 0.4]).unwrap();
        assert!(klein_distance(&k, &k) < 1e-10);
    }

    #[test]
    fn klein_distance_matches_poincare() {
        let p1 = vec![0.1, 0.2];
        let p2 = vec![0.4, -0.1];
        let d_poincare = crate::poincare_distance(&p1, &p2);
        let k1 = to_klein(&p1).unwrap();
        let k2 = to_klein(&p2).unwrap();
        let d_klein = klein_distance(&k1, &k2);
        assert!(
            (d_poincare - d_klein).abs() < 1e-8,
            "Poincaré dist {d_poincare} ≠ Klein dist {d_klein}"
        );
    }

    // ── Collinearity ─────────────────────────────

    #[test]
    fn collinear_points_on_line() {
        let a = vec![0.0, 0.0];
        let b = vec![0.5, 0.0];
        let c = vec![0.3, 0.0]; // on the x-axis between a and b
        assert!(is_collinear(&a, &b, &c, 1e-10));
    }

    #[test]
    fn non_collinear_points() {
        let a = vec![0.0, 0.0];
        let b = vec![0.5, 0.0];
        let c = vec![0.25, 0.3]; // off the x-axis
        assert!(!is_collinear(&a, &b, &c, 1e-7));
    }

    #[test]
    fn collinear_3d() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![0.3, 0.3, 0.3];
        let c = vec![0.15, 0.15, 0.15]; // midpoint
        assert!(is_collinear(&a, &b, &c, 1e-10));
    }

    // ── Shortest path check ──────────────────────

    #[test]
    fn on_shortest_path_between() {
        let a = vec![0.0, 0.0];
        let b = vec![0.6, 0.0];
        let c = vec![0.3, 0.0]; // between a and b
        assert!(is_on_shortest_path(&a, &b, &c, 1e-10));
    }

    #[test]
    fn not_on_shortest_path_outside() {
        let a = vec![0.0, 0.0];
        let b = vec![0.3, 0.0];
        let c = vec![0.5, 0.0]; // beyond b
        assert!(!is_on_shortest_path(&a, &b, &c, 1e-10));
    }

    #[test]
    fn not_on_shortest_path_off_line() {
        let a = vec![0.0, 0.0];
        let b = vec![0.5, 0.0];
        let c = vec![0.25, 0.1]; // off the geodesic
        assert!(!is_on_shortest_path(&a, &b, &c, 1e-7));
    }

    #[test]
    fn on_shortest_path_at_endpoints() {
        let a = vec![0.1, 0.2];
        let b = vec![0.4, 0.5];
        assert!(is_on_shortest_path(&a, &b, &a, 1e-10));
        assert!(is_on_shortest_path(&a, &b, &b, 1e-10));
    }
}
