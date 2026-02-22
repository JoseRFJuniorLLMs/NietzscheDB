//! # Riemann (Spherical) model operations
//!
//! The sphere S^n is the space of constant **positive curvature** (K > 0).
//! It is used in NietzscheDB for:
//!
//! - **Synthesis / Aggregation**: finding the "parent concept" that unifies
//!   two distant points (dialectical synthesis: Thesis + Antithesis → Synthesis)
//! - **GROUP BY centroid**: computing the Fréchet mean on the sphere converges
//!   faster and more stably than in hyperbolic space
//! - **Cyclic data**: encoding periodic structures (time of day, seasons)
//!
//! ## Pipeline: Poincaré → Tangent → Sphere → Tangent → Poincaré
//!
//! ```text
//! 1. log_map₀(p)       → tangent vector at origin (Euclidean)
//! 2. normalize(v)       → project onto unit sphere
//! 3. spherical_midpoint → compute average on sphere
//! 4. denormalize        → back to tangent space
//! 5. exp_map₀(v)        → back into Poincaré ball
//! ```
//!
//! ## Safety invariant
//!
//! All sphere points satisfy **‖x‖ = 1.0** (unit sphere).
//! All Poincaré points returned satisfy **‖x‖ < 1.0**.

use crate::error::HypError;

// ─────────────────────────────────────────────
// Poincaré → Sphere (via tangent space)
// ─────────────────────────────────────────────

/// Project a Poincaré ball point onto the unit sphere.
///
/// Pipeline: Poincaré → tangent space at origin (log_map₀) → normalize to S^n.
///
/// The magnitude (depth in Poincaré ball) is lost — only the direction is kept.
/// This is intentional: the sphere captures **angular relationships** between
/// concepts, not their hierarchical depth.
///
/// Returns the unit vector on S^{n-1}.
///
/// # Errors
///
/// Returns [`HypError::OutsideBall`] if ‖p‖ ≥ 1.0.
/// Returns [`HypError::EmptyInput`] if the vector is zero (no direction).
pub fn to_sphere(poincare: &[f64]) -> Result<Vec<f64>, HypError> {
    let tangent = crate::log_map_zero(poincare)?;
    let norm = l2_norm(&tangent);
    if norm < 1e-15 {
        return Err(HypError::EmptyInput);
    }
    Ok(tangent.iter().map(|&x| x / norm).collect())
}

/// Project a Poincaré ball point onto the unit sphere, preserving the tangent
/// magnitude as a separate return value.
///
/// Returns `(sphere_point, magnitude)` where magnitude can be used to
/// reconstruct the original distance from the origin in tangent space.
pub fn to_sphere_with_magnitude(poincare: &[f64]) -> Result<(Vec<f64>, f64), HypError> {
    let tangent = crate::log_map_zero(poincare)?;
    let norm = l2_norm(&tangent);
    if norm < 1e-15 {
        return Err(HypError::EmptyInput);
    }
    let sphere: Vec<f64> = tangent.iter().map(|&x| x / norm).collect();
    Ok((sphere, norm))
}

// ─────────────────────────────────────────────
// Sphere → Poincaré (via tangent space)
// ─────────────────────────────────────────────

/// Project a unit sphere point back into the Poincaré ball.
///
/// Requires a `magnitude` to reconstruct the tangent vector length
/// (lost during `to_sphere`). If `magnitude` is not available,
/// use a default like `0.5` (mid-depth in the ball).
///
/// Pipeline: scale sphere point by magnitude → exp_map₀ → Poincaré ball.
pub fn from_sphere(sphere: &[f64], magnitude: f64) -> Vec<f64> {
    let tangent: Vec<f64> = sphere.iter().map(|&x| x * magnitude).collect();
    crate::exp_map_zero(&tangent)
}

// ─────────────────────────────────────────────
// Spherical distance (great-circle)
// ─────────────────────────────────────────────

/// Great-circle distance between two points on the unit sphere.
///
/// ```text
/// d(a, b) = arccos(⟨a, b⟩)
/// ```
///
/// Both points must be unit-normalized (‖a‖ = ‖b‖ = 1).
#[inline]
pub fn spherical_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // Clamp to [-1, 1] to handle floating-point noise
    dot.clamp(-1.0, 1.0).acos()
}

// ─────────────────────────────────────────────
// Spherical midpoint (simple average)
// ─────────────────────────────────────────────

/// Compute the spherical midpoint (centroid) of multiple points on the unit sphere.
///
/// Algorithm: normalize the Euclidean mean of the points.
///
/// ```text
/// midpoint = normalize(Σ points)
/// ```
///
/// # Warning
///
/// For antipodal points (e.g. p and -p), the mean is near zero and the
/// midpoint is ill-defined. Returns an error in this case.
///
/// # Errors
///
/// Returns [`HypError::EmptyInput`] if `points` is empty.
/// Returns [`HypError::EmptyInput`] if the sum is zero (antipodal points).
pub fn spherical_midpoint(points: &[&[f64]]) -> Result<Vec<f64>, HypError> {
    if points.is_empty() {
        return Err(HypError::EmptyInput);
    }

    let dim = points[0].len();
    let mut sum = vec![0.0; dim];

    for &p in points {
        for (s, &pi) in sum.iter_mut().zip(p.iter()) {
            *s += pi;
        }
    }

    let norm = l2_norm(&sum);
    if norm < 1e-15 {
        return Err(HypError::EmptyInput);
    }

    Ok(sum.iter().map(|&x| x / norm).collect())
}

/// Weighted spherical midpoint.
///
/// Each point has an importance weight. The result is the normalized
/// weighted sum.
pub fn spherical_midpoint_weighted(
    points: &[&[f64]],
    weights: &[f64],
) -> Result<Vec<f64>, HypError> {
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
    let mut sum = vec![0.0; dim];

    for (&p, &w) in points.iter().zip(weights.iter()) {
        for (s, &pi) in sum.iter_mut().zip(p.iter()) {
            *s += pi * w;
        }
    }

    let norm = l2_norm(&sum);
    if norm < 1e-15 {
        return Err(HypError::EmptyInput);
    }

    Ok(sum.iter().map(|&x| x / norm).collect())
}

// ─────────────────────────────────────────────
// Fréchet mean on the sphere (iterative)
// ─────────────────────────────────────────────

/// Iterative Fréchet mean on the unit sphere.
///
/// For well-clustered points, `spherical_midpoint` suffices. For spread-out
/// distributions, this iterative algorithm converges to the true geodesic mean.
///
/// Algorithm (Riemannian gradient descent):
/// 1. Start with normalized Euclidean mean as initial estimate
/// 2. Compute tangent vectors from estimate to each point (log_map on sphere)
/// 3. Average the tangent vectors
/// 4. Step along that direction (exp_map on sphere)
/// 5. Repeat until convergence or max_iter
///
/// # Errors
///
/// Returns [`HypError::EmptyInput`] if `points` is empty.
pub fn frechet_mean_sphere(
    points: &[&[f64]],
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>, HypError> {
    if points.is_empty() {
        return Err(HypError::EmptyInput);
    }

    let dim = points[0].len();

    // Initial estimate: normalized Euclidean mean
    let mut mu = spherical_midpoint(points)?;

    for _ in 0..max_iter {
        // Compute mean tangent vector at mu
        let mut mean_tangent = vec![0.0; dim];
        let n = points.len() as f64;

        for &p in points {
            let tangent = sphere_log_map(&mu, p);
            for (mt, &t) in mean_tangent.iter_mut().zip(tangent.iter()) {
                *mt += t / n;
            }
        }

        let step_norm = l2_norm(&mean_tangent);
        if step_norm < tol {
            break;
        }

        // Step: exp_map on sphere
        mu = sphere_exp_map(&mu, &mean_tangent);
    }

    Ok(mu)
}

// ─────────────────────────────────────────────
// Synthesis: find the unifying concept
// ─────────────────────────────────────────────

/// Dialectical synthesis of two Poincaré ball points via the sphere.
///
/// Given two concepts (Thesis and Antithesis), finds the point that
/// represents their **unification** — the "parent concept" that
/// encompasses both.
///
/// Pipeline:
/// 1. Project both points to sphere (losing depth, keeping direction)
/// 2. Compute spherical midpoint
/// 3. Project back to Poincaré ball at a **shallower depth** than either input
///    (the synthesis is more abstract → closer to center)
///
/// Returns the synthesized Poincaré ball point.
///
/// # Errors
///
/// Returns error if either point is outside the ball or at the origin.
pub fn synthesis(a: &[f64], b: &[f64]) -> Result<Vec<f64>, HypError> {
    let (sphere_a, mag_a) = to_sphere_with_magnitude(a)?;
    let (sphere_b, mag_b) = to_sphere_with_magnitude(b)?;

    let midpoint = spherical_midpoint(&[&sphere_a, &sphere_b])?;

    // The synthesis is more abstract → use the MINIMUM magnitude
    // (closer to center = more general concept in the Poincaré hierarchy)
    let synthesis_magnitude = mag_a.min(mag_b) * 0.8;

    Ok(from_sphere(&midpoint, synthesis_magnitude))
}

/// Multi-point synthesis: find the unifying concept of N points.
///
/// Generalizes binary synthesis to N-ary. The result is the concept
/// that best unifies all input concepts.
pub fn synthesis_multi(points: &[&[f64]]) -> Result<Vec<f64>, HypError> {
    if points.is_empty() {
        return Err(HypError::EmptyInput);
    }
    if points.len() == 1 {
        return Ok(points[0].to_vec());
    }

    let mut spheres = Vec::with_capacity(points.len());
    let mut min_magnitude = f64::INFINITY;

    for &p in points {
        let (sphere, mag) = to_sphere_with_magnitude(p)?;
        spheres.push(sphere);
        if mag < min_magnitude {
            min_magnitude = mag;
        }
    }

    let sphere_refs: Vec<&[f64]> = spheres.iter().map(|s| s.as_slice()).collect();
    let midpoint = frechet_mean_sphere(&sphere_refs, 20, 1e-8)?;

    // More abstract than any input
    let synthesis_magnitude = min_magnitude * 0.7;

    Ok(from_sphere(&midpoint, synthesis_magnitude))
}

// ─────────────────────────────────────────────
// Sphere exponential / logarithmic maps
// ─────────────────────────────────────────────

/// Exponential map on the unit sphere: move from `base` along `tangent`.
///
/// ```text
/// exp_x(v) = cos(‖v‖)·x + sin(‖v‖)·(v/‖v‖)
/// ```
fn sphere_exp_map(base: &[f64], tangent: &[f64]) -> Vec<f64> {
    let norm = l2_norm(tangent);
    if norm < 1e-15 {
        return base.to_vec();
    }
    let cos_n = norm.cos();
    let sin_n = norm.sin();
    let result: Vec<f64> = base.iter()
        .zip(tangent.iter())
        .map(|(&b, &t)| cos_n * b + sin_n * (t / norm))
        .collect();
    // Re-normalize to stay exactly on the sphere
    let r_norm = l2_norm(&result);
    if r_norm < 1e-15 {
        return base.to_vec();
    }
    result.iter().map(|&x| x / r_norm).collect()
}

/// Logarithmic map on the unit sphere: tangent vector from `base` to `target`.
///
/// ```text
/// log_x(y) = (y − ⟨x,y⟩·x) / ‖y − ⟨x,y⟩·x‖ · arccos(⟨x,y⟩)
/// ```
fn sphere_log_map(base: &[f64], target: &[f64]) -> Vec<f64> {
    let dot: f64 = base.iter().zip(target.iter()).map(|(x, y)| x * y).sum();
    let dot_clamped = dot.clamp(-1.0, 1.0);
    let angle = dot_clamped.acos();

    if angle < 1e-15 {
        return vec![0.0; base.len()];
    }

    // Project target onto tangent plane at base
    let proj: Vec<f64> = target.iter()
        .zip(base.iter())
        .map(|(&t, &b)| t - dot_clamped * b)
        .collect();

    let proj_norm = l2_norm(&proj);
    if proj_norm < 1e-15 {
        return vec![0.0; base.len()];
    }

    proj.iter().map(|&x| x * angle / proj_norm).collect()
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
    fn to_sphere_gives_unit_norm() {
        let p = vec![0.3, 0.4, 0.1];
        let s = to_sphere(&p).unwrap();
        let norm = l2_norm(&s);
        assert!((norm - 1.0).abs() < 1e-12, "sphere norm = {norm}");
    }

    #[test]
    fn to_sphere_rejects_outside_ball() {
        let p = vec![0.8, 0.8];
        assert!(to_sphere(&p).is_err());
    }

    #[test]
    fn sphere_poincare_roundtrip() {
        let p = vec![0.3, 0.4];
        let (sphere, mag) = to_sphere_with_magnitude(&p).unwrap();
        let recovered = from_sphere(&sphere, mag);
        for (a, b) in p.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-8, "roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_error_below_threshold() {
        // Spec: Poincaré → Tangent → Sphere → Tangent → Poincaré error < 1e-5
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..10_000 {
            let dim = 8;
            let p: Vec<f64> = (0..dim).map(|_| rng.gen_range(-0.8..0.8)).collect();
            let norm = l2_norm(&p);
            if norm >= 0.99 || norm < 0.01 { continue; }
            let (sphere, mag) = to_sphere_with_magnitude(&p).unwrap();
            let recovered = from_sphere(&sphere, mag);
            let error: f64 = p.iter().zip(recovered.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(error < 1e-5, "roundtrip error {error} exceeds 1e-5");
        }
    }

    #[test]
    fn spherical_distance_self_is_zero() {
        let s = vec![1.0, 0.0, 0.0];
        assert!(spherical_distance(&s, &s) < 1e-12);
    }

    #[test]
    fn spherical_distance_orthogonal_is_pi_over_2() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = spherical_distance(&a, &b);
        assert!(
            (d - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "expected π/2, got {d}"
        );
    }

    #[test]
    fn spherical_distance_antipodal_is_pi() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = spherical_distance(&a, &b);
        assert!((d - std::f64::consts::PI).abs() < 1e-12, "expected π, got {d}");
    }

    #[test]
    fn spherical_midpoint_symmetric_is_north_pole() {
        // Midpoint of (1,0,0) and (0,1,0) should be (1/√2, 1/√2, 0)
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let mid = spherical_midpoint(&[&a, &b]).unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((mid[0] - expected).abs() < 1e-12);
        assert!((mid[1] - expected).abs() < 1e-12);
        assert!(mid[2].abs() < 1e-12);
    }

    #[test]
    fn spherical_midpoint_antipodal_fails() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(spherical_midpoint(&[&a, &b]).is_err());
    }

    #[test]
    fn synthesis_result_inside_ball() {
        let a = vec![0.3, 0.4];
        let b = vec![-0.2, 0.5];
        let s = synthesis(&a, &b).unwrap();
        let norm = l2_norm(&s);
        assert!(norm < 1.0, "synthesis result norm = {norm} >= 1.0");
    }

    #[test]
    fn synthesis_is_more_abstract() {
        // Synthesis should be closer to center (more abstract) than inputs
        let a = vec![0.5, 0.3];
        let b = vec![0.4, 0.5];
        let s = synthesis(&a, &b).unwrap();
        let depth_a = l2_norm(&a);
        let depth_b = l2_norm(&b);
        let depth_s = l2_norm(&s);
        assert!(
            depth_s < depth_a.min(depth_b),
            "synthesis depth {depth_s} should be < min({depth_a}, {depth_b})"
        );
    }

    #[test]
    fn frechet_mean_converges() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![0.0, 0.0, 1.0];
        let mean = frechet_mean_sphere(&[&a, &b, &c], 100, 1e-10).unwrap();
        let norm = l2_norm(&mean);
        assert!((norm - 1.0).abs() < 1e-10, "Fréchet mean not on sphere");
        // Mean of three orthogonal unit vectors = (1,1,1)/√3
        let expected = 1.0 / 3.0_f64.sqrt();
        for &m in &mean {
            assert!((m - expected).abs() < 1e-6, "Fréchet mean component {m} ≠ {expected}");
        }
    }

    #[test]
    fn synthesis_multi_result_inside_ball() {
        let a = vec![0.3, 0.4, 0.1];
        let b = vec![-0.2, 0.5, 0.3];
        let c = vec![0.1, -0.3, 0.4];
        let s = synthesis_multi(&[&a, &b, &c]).unwrap();
        let norm = l2_norm(&s);
        assert!(norm < 1.0, "multi-synthesis norm = {norm} >= 1.0");
    }
}
