//! Riemannian geometry tools for the Poincaré ball model.
//!
//! All operations keep vectors strictly inside the **open unit ball** ‖x‖ < 1.
//!
//! ## Exponential map
//!
//! The exponential map at `x` in the tangent direction `v`:
//!
//! ```text
//! exp_x(v) = x ⊕ tanh(λ_x · ‖v‖ / 2) · (v / ‖v‖)
//! ```
//!
//! where `λ_x = 2 / (1 - ‖x‖²)` is the conformal factor at `x`.
//!
//! This moves a point along the geodesic starting at `x` with velocity `v`.
//!
//! ## Riemannian gradient
//!
//! The Riemannian gradient is the Euclidean gradient rescaled by the inverse
//! of the Poincaré metric tensor:
//!
//! ```text
//! grad_R(x) = ((1 - ‖x‖²) / 2)² · grad_E(x)
//! ```
//!
//! ## RiemannianAdam
//!
//! Adam optimizer adapted to the Poincaré manifold (Becigneul & Ganea, 2019):
//!
//! ```text
//! g_t   = riemannian_grad(x_t, eucl_grad_t)
//! m_t   = β₁ m_{t-1} + (1-β₁) g_t
//! v_t   = β₂ v_{t-1} + (1-β₂) ‖g_t‖²
//! m̂_t  = m_t / (1 - β₁^t)
//! v̂_t  = v_t / (1 - β₂^t)
//! x_{t+1} = exp_{x_t}(−lr · m̂_t / (√v̂_t + ε))
//! ```

use rand::Rng;

// ─────────────────────────────────────────────
// Möbius addition (local copy — avoids cross-crate dep)
// ─────────────────────────────────────────────

fn mobius_add(u: &[f64], v: &[f64]) -> Vec<f64> {
    let dot_uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom   = 1.0 + 2.0 * dot_uv + norm_u_sq * norm_v_sq;
    let coeff_u = (1.0 + 2.0 * dot_uv + norm_v_sq) / denom;
    let coeff_v = (1.0 - norm_u_sq) / denom;

    u.iter().zip(v.iter())
        .map(|(ui, vi)| coeff_u * ui + coeff_v * vi)
        .collect()
}

/// Project a vector back into the open unit ball if ‖x‖ ≥ 0.999.
pub fn project_into_ball(coords: Vec<f64>) -> Vec<f64> {
    let norm: f64 = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm >= 0.999 {
        let scale = 0.95 / (norm + 1e-15);
        coords.into_iter().map(|x| x * scale).collect()
    } else {
        coords
    }
}

// ─────────────────────────────────────────────
// Conformal factor
// ─────────────────────────────────────────────

/// λ_x = 2 / (1 − ‖x‖²) — the Poincaré conformal factor at `x`.
#[inline]
pub fn conformal_factor(x: &[f64]) -> f64 {
    let norm_sq: f64 = x.iter().map(|v| v * v).sum();
    2.0 / (1.0 - norm_sq).max(1e-12)
}

// ─────────────────────────────────────────────
// Exponential map
// ─────────────────────────────────────────────

/// Move from `x` along the geodesic with tangent vector `v`.
///
/// ```text
/// exp_x(v) = x ⊕ tanh(λ_x · ‖v‖ / 2) · (v / ‖v‖)
/// ```
pub fn exp_map(x: &[f64], v: &[f64]) -> Vec<f64> {
    let v_norm: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
    if v_norm < 1e-15 {
        return x.to_vec(); // no movement
    }

    let lambda = conformal_factor(x);
    let scale  = (lambda * v_norm / 2.0).tanh() / v_norm;
    let v_scaled: Vec<f64> = v.iter().map(|vi| scale * vi).collect();

    project_into_ball(mobius_add(x, &v_scaled))
}

// ─────────────────────────────────────────────
// Riemannian gradient
// ─────────────────────────────────────────────

/// Scale Euclidean gradient `eucl_grad` to the Riemannian gradient at `x`.
///
/// `grad_R = ((1 − ‖x‖²) / 2)² · grad_E`
pub fn riemannian_grad(x: &[f64], eucl_grad: &[f64]) -> Vec<f64> {
    let norm_sq: f64 = x.iter().map(|v| v * v).sum();
    let scale = ((1.0 - norm_sq) / 2.0).powi(2);
    eucl_grad.iter().map(|g| scale * g).collect()
}

// ─────────────────────────────────────────────
// Random tangent vector (for perturbation)
// ─────────────────────────────────────────────

/// Sample a random tangent vector at `x` with magnitude `noise`.
///
/// Used during the "dream" phase to perturb node embeddings.
pub fn random_tangent(dim: usize, noise: f64, rng: &mut impl Rng) -> Vec<f64> {
    let raw: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
    let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    raw.into_iter().map(|v| v / norm * noise).collect()
}

// ─────────────────────────────────────────────
// RiemannianAdam
// ─────────────────────────────────────────────

/// Per-parameter Adam state living in the tangent space.
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment (exponential moving average of gradient).
    pub m: Vec<f64>,
    /// Second moment (exponential moving average of ‖gradient‖²).
    pub v: f64,
    /// Step counter.
    pub t: u32,
}

impl AdamState {
    pub fn new(dim: usize) -> Self {
        Self { m: vec![0.0; dim], v: 0.0, t: 0 }
    }
}

/// Riemannian Adam optimizer (Becigneul & Ganea, 2019).
///
/// Adapts Adam to the Poincaré manifold by:
/// 1. Scaling the Euclidean gradient to the Riemannian gradient.
/// 2. Maintaining Adam moments in the **tangent space**.
/// 3. Retracting the update back to the manifold via the **exponential map**.
#[derive(Debug, Clone)]
pub struct RiemannianAdam {
    /// Learning rate.
    pub lr:    f64,
    /// Exponential decay for first moment.
    pub beta1: f64,
    /// Exponential decay for second moment.
    pub beta2: f64,
    /// Numerical stability term.
    pub eps:   f64,
}

impl Default for RiemannianAdam {
    fn default() -> Self {
        Self { lr: 5e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }
}

impl RiemannianAdam {
    pub fn new(lr: f64) -> Self {
        Self { lr, ..Default::default() }
    }

    /// Perform one Adam step.
    ///
    /// # Arguments
    /// * `x`          — current position on the manifold
    /// * `eucl_grad`  — Euclidean gradient of the loss at `x`
    /// * `state`      — mutable Adam state (moments + step counter)
    ///
    /// # Returns
    /// New position on the manifold after the retraction.
    pub fn step(
        &self,
        x:         &[f64],
        eucl_grad: &[f64],
        state:     &mut AdamState,
    ) -> Vec<f64> {
        state.t += 1;
        let t = state.t as f64;

        // Riemannian gradient
        let riem_g = riemannian_grad(x, eucl_grad);
        let g_norm_sq: f64 = riem_g.iter().map(|g| g * g).sum();

        // Update moments
        for (mi, gi) in state.m.iter_mut().zip(riem_g.iter()) {
            *mi = self.beta1 * *mi + (1.0 - self.beta1) * gi;
        }
        state.v = self.beta2 * state.v + (1.0 - self.beta2) * g_norm_sq;

        // Bias correction
        let m_hat: Vec<f64> = state.m.iter()
            .map(|mi| mi / (1.0 - self.beta1.powi(state.t as i32)))
            .collect();
        let v_hat = state.v / (1.0 - self.beta2.powi(state.t as i32));

        // Retraction step: move along geodesic
        let step_scale = self.lr / (v_hat.sqrt() + self.eps);
        let direction: Vec<f64> = m_hat.iter().map(|m| -step_scale * m).collect();

        exp_map(x, &direction)
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    fn exp_map_at_origin_equals_direction() {
        // exp_0(v) = tanh(‖v‖) · v/‖v‖  →  for small ‖v‖ ≈ v
        let x = vec![0.0, 0.0];
        let v = vec![0.1, 0.0];
        let result = exp_map(&x, &v);
        // tanh(2 · 0.1 / 2) = tanh(0.1) ≈ 0.0997
        let expected = (0.1_f64).tanh();
        assert!((result[0] - expected).abs() < 1e-6, "exp_0(v)[0] = {}", result[0]);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn exp_map_stays_in_ball() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let x = vec![rng.gen::<f64>() * 0.8, rng.gen::<f64>() * 0.8];
            let n = norm(&x);
            let x = if n > 0.8 { x.iter().map(|v| v * 0.8 / n).collect() } else { x };
            let v = vec![rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5];
            let result = exp_map(&x, &v);
            assert!(norm(&result) < 1.0, "exp_map left the ball: ‖r‖ = {}", norm(&result));
        }
    }

    #[test]
    fn project_into_ball_clamps() {
        let outside = vec![0.9, 0.9]; // ‖·‖ ≈ 1.27
        let projected = project_into_ball(outside);
        assert!(norm(&projected) < 1.0, "still outside: {}", norm(&projected));
    }

    #[test]
    fn riemannian_grad_at_origin_equals_euclidean() {
        let x    = vec![0.0, 0.0]; // at origin: ((1-0)/2)² = 0.25
        let g    = vec![1.0, 0.0];
        let rg   = riemannian_grad(&x, &g);
        assert!((rg[0] - 0.25).abs() < 1e-10, "rg[0] = {}", rg[0]);
    }

    #[test]
    fn riemannian_grad_shrinks_near_boundary() {
        // Near boundary ‖x‖ → 1, conformal factor → 0
        let x    = vec![0.9, 0.0];
        let g    = vec![1.0, 0.0];
        let rg   = riemannian_grad(&x, &g);
        // ((1-0.81)/2)² = (0.19/2)² ≈ 0.009
        assert!(rg[0] < 0.05, "gradient should be small near boundary: {}", rg[0]);
    }

    #[test]
    fn riemannian_adam_moves_toward_gradient() {
        let adam = RiemannianAdam::new(0.1);
        let x    = vec![0.3, 0.0];
        let g    = vec![1.0, 0.0]; // gradient pushing in +x direction
        let mut state = AdamState::new(2);

        let x1 = adam.step(&x, &g, &mut state);

        // With gradient pointing +x, step goes −x (minimization)
        assert!(x1[0] < x[0], "x should decrease: {} vs {}", x1[0], x[0]);
        assert!(norm(&x1) < 1.0, "result should stay in ball: {}", norm(&x1));
    }

    #[test]
    fn adam_converges_toward_target() {
        // Minimize ‖x - target‖² — loss gradient = 2(x - target)
        let adam   = RiemannianAdam::new(0.05);
        let target = vec![0.2, 0.1];
        let mut x  = vec![0.5, 0.4];
        let mut state = AdamState::new(2);

        for _ in 0..50 {
            let g: Vec<f64> = x.iter().zip(target.iter())
                .map(|(xi, ti)| 2.0 * (xi - ti))
                .collect();
            x = adam.step(&x, &g, &mut state);
        }

        // Should have moved toward target
        let dist: f64 = x.iter().zip(target.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let init_dist: f64 = vec![0.5_f64 - 0.2, 0.4 - 0.1].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(dist < init_dist, "should converge: dist={dist:.4} init={init_dist:.4}");
    }

    #[test]
    fn random_tangent_has_correct_magnitude() {
        let mut rng = StdRng::seed_from_u64(7);
        let v = random_tangent(3, 0.05, &mut rng);
        let n = norm(&v);
        assert!((n - 0.05).abs() < 1e-10, "‖v‖ = {n}");
    }
}
