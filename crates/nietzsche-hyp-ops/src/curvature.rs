//! Phase A2 — Curvature Adaptive Geometry.
//!
//! Variable curvature K(node) instead of fixed K = -1.
//!
//! ## Generalised Poincaré Distance
//!
//! For curvature K < 0 (with c = √|K|):
//!
//! ```text
//! d_K(u, v) = (1/c) · acosh(1 + 2c²·‖u-v‖² / ((1 - c²·‖u‖²)(1 - c²·‖v‖²)))
//! ```
//!
//! When K = -1 (c = 1), this reduces to the standard formula.
//!
//! ## Adaptive Curvature
//!
//! K(node) = f(local_density, hierarchy_depth):
//! - Dense regions → higher |K| (more curved, tighter packing)
//! - Sparse/deep hierarchy → lower |K| (more room for fine distinctions)
//!
//! ## Exponential/Log Maps with Variable Curvature
//!
//! ```text
//! exp_K(v) = tanh(c·‖v‖) · v / (c·‖v‖)
//! log_K(x) = atanh(c·‖x‖) · x / (c·‖x‖)
//! ```

/// Curvature descriptor for a region of hyperbolic space.
#[derive(Debug, Clone, Copy)]
pub struct Curvature {
    /// Negative curvature value (K < 0). Default: -1.0.
    pub k: f64,
    /// Derived: c = √|K|. Precomputed for performance.
    c: f64,
}

impl Default for Curvature {
    fn default() -> Self {
        Self { k: -1.0, c: 1.0 }
    }
}

impl Curvature {
    /// Create curvature from K value. K must be negative.
    ///
    /// # Panics
    /// Panics if K >= 0.
    pub fn new(k: f64) -> Self {
        assert!(k < 0.0, "Curvature K must be negative, got {k}");
        Self { k, c: (-k).sqrt() }
    }

    /// The c = √|K| scaling factor.
    #[inline]
    pub fn c(&self) -> f64 {
        self.c
    }

    /// Standard K = -1 curvature.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Compute adaptive curvature based on local graph properties.
    ///
    /// K(node) = -1 × density_factor × depth_factor
    ///
    /// - `local_density`: avg degree of neighbors (higher = more curved)
    /// - `hierarchy_depth`: ‖embedding‖ ∈ [0, 1) (deeper = less curved)
    /// - `base_k`: base curvature (default: -1.0)
    pub fn adaptive(local_density: f64, hierarchy_depth: f64, base_k: f64) -> Self {
        let base = base_k.abs();

        // Dense regions get tighter curvature (more curved)
        // Clamp density factor to [0.5, 3.0]
        let density_factor = (local_density / 10.0).max(0.5).min(3.0);

        // Deeper hierarchy gets less curvature (more room for fine distinctions)
        // depth ∈ [0, 1) → factor ∈ [1.0, 0.3]
        let depth_factor = 1.0 - 0.7 * hierarchy_depth.min(0.99);

        let k = -(base * density_factor * depth_factor);
        Self::new(k.min(-0.01)) // Never let |K| go to zero
    }
}

/// Poincaré distance with variable curvature K.
///
/// d_K(u, v) = (1/c) · acosh(1 + 2c²·‖u-v‖² / ((1-c²·‖u‖²)(1-c²·‖v‖²)))
///
/// When K = -1 (c = 1), this is identical to `poincare_distance()`.
pub fn poincare_distance_curved(u: &[f64], v: &[f64], curvature: &Curvature) -> f64 {
    let c = curvature.c;
    let c2 = c * c;

    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom = (1.0 - c2 * norm_u_sq).max(1e-15) * (1.0 - c2 * norm_v_sq).max(1e-15);
    let arg = (1.0 + 2.0 * c2 * diff_sq / denom).max(1.0);

    arg.acosh() / c
}

/// Exponential map at origin with variable curvature.
///
/// exp_K(v) = tanh(c·‖v‖) · v / (c·‖v‖)
pub fn exp_map_zero_curved(v: &[f64], curvature: &Curvature) -> Vec<f64> {
    let c = curvature.c;
    let norm = crate::l2_norm(v);
    if norm < 1e-15 {
        return vec![0.0; v.len()];
    }
    let cn = c * norm;
    let scale = cn.tanh() / (c * norm);
    v.iter().map(|&x| x * scale).collect()
}

/// Logarithmic map at origin with variable curvature.
///
/// log_K(x) = atanh(c·‖x‖) · x / (c·‖x‖)
pub fn log_map_zero_curved(x: &[f64], curvature: &Curvature) -> Result<Vec<f64>, crate::error::HypError> {
    let c = curvature.c;
    let norm = crate::l2_norm(x);
    let cn = c * norm;
    if cn >= 1.0 {
        return Err(crate::error::HypError::OutsideBall { norm: cn });
    }
    if norm < 1e-15 {
        return Ok(vec![0.0; x.len()]);
    }
    let scale = cn.atanh() / (c * norm);
    Ok(x.iter().map(|&xi| xi * scale).collect())
}

/// Möbius addition with variable curvature.
///
/// u ⊕_K v — generalised to curvature K.
pub fn mobius_add_curved(u: &[f64], v: &[f64], curvature: &Curvature) -> Vec<f64> {
    let c2 = curvature.c * curvature.c;

    let dot_uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom = 1.0 + 2.0 * c2 * dot_uv + c2 * c2 * norm_u_sq * norm_v_sq;
    let coeff_u = (1.0 + 2.0 * c2 * dot_uv + c2 * norm_v_sq) / denom;
    let coeff_v = (1.0 - c2 * norm_u_sq) / denom;

    let result: Vec<f64> = u.iter().zip(v.iter())
        .map(|(&ui, &vi)| coeff_u * ui + coeff_v * vi)
        .collect();

    crate::project_to_ball(&result, crate::MAX_NORM)
}

/// Parallel transport with variable curvature.
///
/// PT_{0→x}(v) = v · (1 − c²·‖x‖²) / 2
pub fn parallel_transport_curved(x: &[f64], v: &[f64], curvature: &Curvature) -> Vec<f64> {
    let c2 = curvature.c * curvature.c;
    let norm_x_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let conformal = (1.0 - c2 * norm_x_sq) / 2.0;
    v.iter().map(|&vi| vi * conformal).collect()
}

/// Gyromidpoint (Fréchet mean) with variable curvature.
pub fn gyromidpoint_curved(points: &[&[f64]], curvature: &Curvature) -> Result<Vec<f64>, crate::error::HypError> {
    if points.is_empty() {
        return Err(crate::error::HypError::EmptyInput);
    }

    let dim = points[0].len();
    let n = points.len() as f64;

    let mut sum = vec![0.0; dim];
    for &p in points {
        let tangent = log_map_zero_curved(p, curvature)?;
        for (s, t) in sum.iter_mut().zip(tangent.iter()) {
            *s += t;
        }
    }

    for s in sum.iter_mut() {
        *s /= n;
    }

    Ok(exp_map_zero_curved(&sum, curvature))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_curvature_matches_original() {
        let u = vec![0.3, -0.2, 0.1];
        let v = vec![0.1, 0.4, -0.3];
        let k = Curvature::standard(); // K = -1

        let d_curved = poincare_distance_curved(&u, &v, &k);
        let d_standard = crate::poincare_distance(&u, &v);

        assert!(
            (d_curved - d_standard).abs() < 1e-10,
            "K=-1 curved ({d_curved}) should match standard ({d_standard})"
        );
    }

    #[test]
    fn test_higher_curvature_increases_distance() {
        let u = vec![0.3, 0.0];
        let v = vec![0.6, 0.0];

        let k1 = Curvature::new(-1.0);
        let k2 = Curvature::new(-4.0); // higher |K| = more curved

        let d1 = poincare_distance_curved(&u, &v, &k1);
        let d2 = poincare_distance_curved(&u, &v, &k2);

        // Higher curvature should give different distance
        assert!(d1 != d2, "Different curvatures should give different distances");
    }

    #[test]
    fn test_adaptive_curvature_dense_region() {
        let k = Curvature::adaptive(50.0, 0.1, -1.0); // dense, shallow
        assert!(k.k < -1.0, "Dense region should have higher |K|: got {}", k.k);
    }

    #[test]
    fn test_adaptive_curvature_deep_region() {
        let k = Curvature::adaptive(5.0, 0.9, -1.0); // sparse, deep
        assert!(k.k > -1.0, "Deep region should have lower |K|: got {}", k.k);
    }

    #[test]
    fn test_exp_log_roundtrip_curved() {
        let k = Curvature::new(-2.0);
        let v = vec![0.5, -0.3, 0.1];
        let mapped = exp_map_zero_curved(&v, &k);
        let recovered = log_map_zero_curved(&mapped, &k).unwrap();

        for (a, b) in v.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "exp/log roundtrip failed: {v:?} vs {recovered:?}"
            );
        }
    }

    #[test]
    fn test_distance_self_zero() {
        let u = vec![0.3, 0.2];
        let k = Curvature::new(-3.0);
        let d = poincare_distance_curved(&u, &u, &k);
        assert!(d.abs() < 1e-10, "Distance to self should be 0, got {d}");
    }
}
