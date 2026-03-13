// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Homeostasis — radial repulsion preventing center collapse
//!
//! ## Problem
//! Repeated synthesis operations tend to produce nodes ever closer to the
//! center of the Poincaré disk (since synthesis = more abstract = lower ‖x‖).
//! Eventually, all synthesis nodes would collapse to the origin, destroying
//! the hierarchical semantics of the ball.
//!
//! ## Solution
//! The [`HomeostasisGuard`] enforces a minimum radius for synthesis nodes.
//! If a synthesis point has ‖x‖ < `min_radius`, it is repelled outward
//! along the same direction until ‖x‖ = `min_radius`.
//!
//! This is analogous to biological homeostasis: the system has a "set point"
//! (minimum radius) and actively resists perturbations that would violate it.
//!
//! ## Active Radial Field (Phase V)
//!
//! The [`RadialField`] extends homeostasis from hard clamping to a smooth
//! gradient field. Instead of abruptly pushing a point to `min_radius`,
//! the field applies a continuous repulsive force that increases as the
//! point approaches the center:
//!
//! ```text
//! F(r) = strength · ((min_radius - r) / min_radius)²   for r < min_radius
//! F(r) = 0                                              for min_radius ≤ r ≤ max_radius
//! F(r) = -strength · ((r - max_radius) / (1 - max_radius))²  for r > max_radius
//! ```
//!
//! The force is radial (preserves direction) and smooth (no discontinuities).

use crate::error::{AgiError, AgiResult};
use nietzsche_hyp_ops::{exp_map_zero, log_map_zero, project_to_ball};

// ─────────────────────────────────────────────
// HomeostasisGuard
// ─────────────────────────────────────────────

/// Prevents synthesis nodes from collapsing to the center of the Poincaré ball.
///
/// Enforces `‖x‖ >= min_radius` for all synthesis outputs.
/// If violated, the point is repelled outward along its direction.
#[derive(Debug, Clone)]
pub struct HomeostasisGuard {
    /// Minimum allowed radius for synthesis nodes.
    /// Default: 0.05 (5% of the ball radius).
    pub min_radius: f64,

    /// Maximum allowed radius for synthesis nodes.
    /// Default: 0.95 (5% from the boundary — safety margin).
    pub max_radius: f64,
}

impl HomeostasisGuard {
    pub fn new(min_radius: f64) -> Self {
        Self {
            min_radius: min_radius.max(0.001),
            max_radius: 0.95,
        }
    }

    /// Check if a norm passes homeostasis. Returns Ok(()) or an error.
    pub fn check(&self, norm: f64) -> AgiResult<()> {
        if norm < self.min_radius {
            Err(AgiError::HomeostasisViolation {
                norm,
                min_radius: self.min_radius,
            })
        } else {
            Ok(())
        }
    }

    /// Apply homeostasis correction to a point in the Poincaré ball.
    ///
    /// If ‖point‖ < min_radius, the point is pushed outward to min_radius.
    /// If ‖point‖ > max_radius, the point is pulled inward to max_radius.
    /// Otherwise, the point is returned unchanged.
    ///
    /// The direction is preserved — only the magnitude changes.
    pub fn correct(&self, point: &mut [f64]) {
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm < 1e-15 {
            // Point is at the exact origin — can't determine direction.
            // Push it along the first axis.
            if !point.is_empty() {
                point[0] = self.min_radius;
            }
            return;
        }

        if norm < self.min_radius {
            let scale = self.min_radius / norm;
            for x in point.iter_mut() {
                *x *= scale;
            }
        } else if norm > self.max_radius {
            let scale = self.max_radius / norm;
            for x in point.iter_mut() {
                *x *= scale;
            }
        }
    }

    /// Returns the minimum allowed radius.
    pub fn min_radius(&self) -> f64 {
        self.min_radius
    }

    /// Returns the maximum allowed radius.
    pub fn max_radius(&self) -> f64 {
        self.max_radius
    }
}

impl Default for HomeostasisGuard {
    fn default() -> Self {
        Self {
            min_radius: 0.05,
            max_radius: 0.95,
        }
    }
}

// ─────────────────────────────────────────────
// RadialField — smooth gradient-based homeostasis
// ─────────────────────────────────────────────

/// A smooth radial force field that maintains points within the homeostatic zone.
///
/// Unlike the hard-clamping [`HomeostasisGuard`], the `RadialField` applies
/// a continuous gradient force:
///
/// - **Inside min_radius**: repulsive force pushing outward (quadratic ramp)
/// - **Inside [min, max]**: no force (homeostatic zone)
/// - **Outside max_radius**: attractive force pulling inward (quadratic ramp)
///
/// The force magnitude is smooth and differentiable at the boundaries.
#[derive(Debug, Clone)]
pub struct RadialField {
    /// Minimum radius (inner boundary of homeostatic zone).
    pub min_radius: f64,
    /// Maximum radius (outer boundary of homeostatic zone).
    pub max_radius: f64,
    /// Strength of the repulsive/attractive force.
    /// Default: 0.1 (gentle correction).
    pub strength: f64,
}

impl RadialField {
    pub fn new(min_radius: f64, max_radius: f64, strength: f64) -> Self {
        Self {
            min_radius: min_radius.max(0.001),
            max_radius: max_radius.min(0.999),
            strength: strength.max(0.0),
        }
    }

    /// Compute the scalar force at a given radius.
    ///
    /// - Positive: repulsive (push outward)
    /// - Negative: attractive (pull inward)
    /// - Zero: in homeostatic zone
    pub fn force_at(&self, radius: f64) -> f64 {
        if radius < self.min_radius {
            // Quadratic repulsion: stronger as radius → 0
            let violation = (self.min_radius - radius) / self.min_radius;
            self.strength * violation * violation
        } else if radius > self.max_radius {
            // Quadratic attraction: stronger as radius → 1
            let boundary = 1.0 - self.max_radius;
            if boundary < 1e-15 {
                return -self.strength;
            }
            let violation = (radius - self.max_radius) / boundary;
            -self.strength * violation * violation
        } else {
            0.0 // In homeostatic zone
        }
    }

    /// Compute the potential energy at a given radius.
    ///
    /// V(r) = ∫ F(r) dr — the potential well around the homeostatic zone.
    /// Returns 0 inside the zone, positive outside.
    pub fn potential(&self, radius: f64) -> f64 {
        if radius < self.min_radius {
            let violation = (self.min_radius - radius) / self.min_radius;
            self.strength * violation * violation * violation / 3.0
        } else if radius > self.max_radius {
            let boundary = 1.0 - self.max_radius;
            if boundary < 1e-15 {
                return self.strength;
            }
            let violation = (radius - self.max_radius) / boundary;
            self.strength * violation * violation * violation / 3.0
        } else {
            0.0
        }
    }

    /// Compute the displacement vector for a point in the Poincaré ball.
    ///
    /// The displacement pushes the point toward the homeostatic zone
    /// along the radial direction.
    ///
    /// # Returns
    /// A displacement vector to be **added** to the point.
    pub fn compute_displacement(&self, point: &[f64]) -> Vec<f64> {
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm < 1e-15 {
            // At origin — push along first axis
            let mut disp = vec![0.0; point.len()];
            if !disp.is_empty() {
                disp[0] = self.force_at(0.0);
            }
            return disp;
        }

        let f = self.force_at(norm);
        if f.abs() < 1e-15 {
            return vec![0.0; point.len()];
        }

        // Displacement = F(r) · (x / ‖x‖) — radial direction
        point.iter().map(|&x| f * x / norm).collect()
    }

    /// Apply the radial field correction to a point (in-place).
    ///
    /// Adds the displacement to the point, then clamps to ensure
    /// the result stays within the Poincaré ball (‖x‖ < 1).
    pub fn apply(&self, point: &mut [f64]) {
        let disp = self.compute_displacement(point);
        for (x, d) in point.iter_mut().zip(disp.iter()) {
            *x += d;
        }
        // Ensure we're still in the Poincaré ball
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm >= 1.0 {
            let scale = 0.99 / norm;
            for x in point.iter_mut() {
                *x *= scale;
            }
        }
    }
}

impl Default for RadialField {
    fn default() -> Self {
        Self {
            min_radius: 0.05,
            max_radius: 0.95,
            strength: 0.1,
        }
    }
}

// ─────────────────────────────────────────────
// DampedHomeostasis — damped dynamics in tangent space
// ─────────────────────────────────────────────

/// Configuration for the damped homeostasis engine.
///
/// ## Key Insight: Why Tangent Space Damping?
///
/// In the Poincaré disk, Euclidean displacements are **geometrically distorted**:
/// a small step near the boundary moves much further in hyperbolic distance than
/// the same step near the center. Applying forces directly as `new_pos = old_pos + F`
/// would silently destroy the metric structure.
///
/// Instead, we:
/// 1. Compute forces in the **tangent space** at the origin (via `log_map₀`)
/// 2. Apply damping: `F_damped = F - λ·v` (dissipates energy, prevents oscillation)
/// 3. Update velocity: `v' = v + η·F_damped`
/// 4. Reproject via **exponential map**: `x_new = exp₀(v')`
///
/// This preserves the Riemannian metric and prevents the "slingshot effect"
/// where high-velocity points overshoot the disk boundary.
#[derive(Debug, Clone)]
pub struct DampingConfig {
    /// Damping coefficient λ ∈ [0, 1].
    /// Higher = more dissipation, faster settling.
    /// λ = 0: undamped (oscillatory), λ = 1: critically damped.
    /// Default: 0.3 (underdamped — allows gentle exploration)
    pub damping_coeff: f64,

    /// Integration step size η.
    /// Controls how strongly forces translate to velocity.
    /// Default: 0.01
    pub step_size: f64,

    /// Maximum allowed velocity magnitude in tangent space.
    /// Prevents runaway acceleration. Clipped to this value.
    /// Default: 0.1
    pub max_velocity: f64,

    /// Maximum allowed radius in the Poincaré ball.
    /// Points are clamped to this radius after projection.
    /// Default: 0.95 (safe distance from boundary ‖x‖ = 1)
    pub max_radius: f64,

    /// Minimum allowed radius in the Poincaré ball.
    /// Default: 0.01
    pub min_radius: f64,
}

impl Default for DampingConfig {
    fn default() -> Self {
        Self {
            damping_coeff: 0.3,
            step_size: 0.01,
            max_velocity: 0.1,
            max_radius: 0.95,
            min_radius: 0.01,
        }
    }
}

/// Node state tracked by the damped homeostasis engine.
///
/// Each node has a position (in Poincaré ball) and a velocity
/// (in tangent space at the origin). The velocity decays over time
/// due to damping, preventing oscillation.
#[derive(Debug, Clone)]
pub struct NodeDynamics {
    /// Current velocity in tangent space T₀B^n.
    /// Updated each tick by forces minus damping.
    pub velocity: Vec<f64>,
}

impl NodeDynamics {
    pub fn new(dim: usize) -> Self {
        Self {
            velocity: vec![0.0; dim],
        }
    }

    /// Kinetic energy: ½‖v‖²
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.velocity.iter().map(|v| v * v).sum::<f64>()
    }
}

/// Damped homeostasis engine for the Poincaré ball.
///
/// Applies forces with viscous damping in tangent space, then reprojects
/// via exponential map. This is the correct way to do dynamics on a
/// Riemannian manifold.
///
/// ## Equations
///
/// ```text
/// F_damped = F_radial - λ · v        (force minus damping)
/// v' = v + η · F_damped               (velocity update in tangent space)
/// v' = clip(v', max_velocity)          (prevent runaway)
/// x_new = exp₀(v')                    (reproject to Poincaré ball)
/// x_new = clamp(x_new, max_radius)    (safety boundary)
/// ```
///
/// **Pure computation** — takes positions and returns new positions.
pub struct DampedHomeostasis {
    pub config: DampingConfig,
    radial_field: RadialField,
}

impl DampedHomeostasis {
    pub fn new(config: DampingConfig) -> Self {
        let radial_field = RadialField::new(
            config.min_radius,
            config.max_radius,
            0.1, // strength for the radial force
        );
        Self {
            config,
            radial_field,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(DampingConfig::default())
    }

    /// Compute the damped force for a single node.
    ///
    /// # Arguments
    /// - `position`: current position in Poincaré ball
    /// - `dynamics`: current velocity state
    ///
    /// # Returns
    /// The damped force vector in tangent space: F_damped = F_radial - λ·v
    pub fn compute_damped_force(
        &self,
        position: &[f64],
        dynamics: &NodeDynamics,
    ) -> Vec<f64> {
        // Compute radial homeostatic force (in Euclidean, approx tangent at origin)
        let radial_force = self.radial_field.compute_displacement(position);

        // Apply damping: F_damped = F - λ·v
        radial_force
            .iter()
            .zip(dynamics.velocity.iter())
            .map(|(&f, &v)| f - self.config.damping_coeff * v)
            .collect()
    }

    /// Apply one integration step for a single node.
    ///
    /// This is the core of the damped homeostasis engine:
    /// 1. Compute damped force in tangent space
    /// 2. Update velocity: v' = v + η·F_damped
    /// 3. Clip velocity to max_velocity
    /// 4. Project to Poincaré ball via exp_map₀
    /// 5. Clamp to max_radius
    ///
    /// # Arguments
    /// - `position`: current position in Poincaré ball (‖x‖ < 1)
    /// - `dynamics`: mutable velocity state (updated in-place)
    ///
    /// # Returns
    /// New position in the Poincaré ball.
    pub fn step(
        &self,
        position: &[f64],
        dynamics: &mut NodeDynamics,
    ) -> Vec<f64> {
        let dim = position.len();

        // Ensure velocity has correct dimension
        if dynamics.velocity.len() != dim {
            dynamics.velocity = vec![0.0; dim];
        }

        // 1. Compute damped force
        let damped_force = self.compute_damped_force(position, dynamics);

        // 2. Update velocity in tangent space: v' = v + η·F
        for (v, &f) in dynamics.velocity.iter_mut().zip(damped_force.iter()) {
            *v += self.config.step_size * f;
        }

        // 3. Clip velocity magnitude to prevent runaway
        let vel_norm: f64 = dynamics.velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
        if vel_norm > self.config.max_velocity {
            let scale = self.config.max_velocity / vel_norm;
            for v in dynamics.velocity.iter_mut() {
                *v *= scale;
            }
        }

        // 4. Map the current position to tangent space, add velocity, reproject
        //    Use log₀(x) to get tangent vector, add velocity, then exp₀
        let tangent = log_map_zero(position).unwrap_or_else(|_| {
            // If position is invalid, start from rest
            vec![0.0; dim]
        });

        let new_tangent: Vec<f64> = tangent
            .iter()
            .zip(dynamics.velocity.iter())
            .map(|(&t, &v)| t + v)
            .collect();

        // 5. Reproject via exponential map: exp₀(tangent + velocity)
        let new_position = exp_map_zero(&new_tangent);

        // 6. Safety clamp: ensure ‖x‖ ≤ max_radius
        project_to_ball(&new_position, self.config.max_radius)
    }

    /// Apply one homeostasis tick to a batch of nodes.
    ///
    /// # Arguments
    /// - `positions`: current positions in the Poincaré ball (one per node)
    /// - `dynamics`: mutable velocity states (one per node, updated in-place)
    ///
    /// # Returns
    /// `DampingReport` with new positions and energy diagnostics.
    pub fn tick(
        &self,
        positions: &[Vec<f64>],
        dynamics: &mut [NodeDynamics],
    ) -> DampingReport {
        assert_eq!(positions.len(), dynamics.len());

        let mut new_positions = Vec::with_capacity(positions.len());
        let mut total_kinetic = 0.0;
        let mut total_displacement = 0.0;
        let mut clamped_count = 0usize;

        for (pos, dyn_state) in positions.iter().zip(dynamics.iter_mut()) {
            let _old_norm: f64 = pos.iter().map(|x| x * x).sum::<f64>().sqrt();
            let new_pos = self.step(pos, dyn_state);
            let new_norm: f64 = new_pos.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Track if we hit the clamp
            if new_norm >= self.config.max_radius - 1e-6 {
                clamped_count += 1;
            }

            // Euclidean displacement (for diagnostics)
            let disp: f64 = pos
                .iter()
                .zip(new_pos.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();

            total_displacement += disp;
            total_kinetic += dyn_state.kinetic_energy();
            new_positions.push(new_pos);
        }

        let n = positions.len().max(1) as f64;

        DampingReport {
            new_positions,
            mean_displacement: total_displacement / n,
            mean_kinetic_energy: total_kinetic / n,
            clamped_nodes: clamped_count,
            total_nodes: positions.len(),
            is_settled: total_kinetic / n < 1e-8,
        }
    }

    pub fn config(&self) -> &DampingConfig {
        &self.config
    }
}

/// Result of one homeostasis tick across all nodes.
#[derive(Debug, Clone)]
pub struct DampingReport {
    /// New positions after damped integration.
    pub new_positions: Vec<Vec<f64>>,

    /// Mean Euclidean displacement this tick.
    pub mean_displacement: f64,

    /// Mean kinetic energy ½‖v‖² across all nodes.
    /// Near zero = system has settled.
    pub mean_kinetic_energy: f64,

    /// Number of nodes that hit the boundary clamp.
    pub clamped_nodes: usize,

    /// Total nodes processed.
    pub total_nodes: usize,

    /// Whether the system has effectively settled (kinetic ≈ 0).
    pub is_settled: bool,
}

impl std::fmt::Display for DampingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Damping[nodes={}, disp={:.6}, KE={:.6}, clamped={}, {}]",
            self.total_nodes,
            self.mean_displacement,
            self.mean_kinetic_energy,
            self.clamped_nodes,
            if self.is_settled { "SETTLED" } else { "active" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_point_passes() {
        let guard = HomeostasisGuard::default();
        assert!(guard.check(0.5).is_ok());
    }

    #[test]
    fn test_too_close_to_center_fails() {
        let guard = HomeostasisGuard::default();
        assert!(guard.check(0.01).is_err());
    }

    #[test]
    fn test_correct_pushes_outward() {
        let guard = HomeostasisGuard::new(0.1);
        let mut point = vec![0.01, 0.01, 0.0];
        guard.correct(&mut point);
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 0.1).abs() < 1e-10,
            "norm should be min_radius after correction: {norm}"
        );
    }

    #[test]
    fn test_correct_pulls_inward() {
        let guard = HomeostasisGuard::new(0.05);
        let mut point = vec![0.8, 0.7, 0.0];
        let orig_norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(orig_norm > 0.95);
        guard.correct(&mut point);
        let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 0.95).abs() < 1e-10,
            "norm should be max_radius after correction: {norm}"
        );
    }

    #[test]
    fn test_correct_preserves_direction() {
        let guard = HomeostasisGuard::new(0.2);
        let mut point = vec![0.05, 0.0, 0.0];
        guard.correct(&mut point);
        // Direction should be preserved: only x-axis is non-zero
        assert!((point[1]).abs() < 1e-15);
        assert!((point[2]).abs() < 1e-15);
        assert!((point[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_correct_origin() {
        let guard = HomeostasisGuard::new(0.1);
        let mut point = vec![0.0, 0.0, 0.0];
        guard.correct(&mut point);
        assert!((point[0] - 0.1).abs() < 1e-15);
    }

    #[test]
    fn test_normal_point_unchanged() {
        let guard = HomeostasisGuard::default();
        let mut point = vec![0.3, 0.2, 0.1];
        let original = point.clone();
        guard.correct(&mut point);
        for (a, b) in point.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-15, "Point should be unchanged");
        }
    }

    // ── RadialField tests ──

    #[test]
    fn test_radial_field_no_force_in_zone() {
        let field = RadialField::default();
        let f = field.force_at(0.5); // Well inside the zone
        assert!((f - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_radial_field_repulsion_near_center() {
        let field = RadialField::default();
        let f = field.force_at(0.01); // Very close to center
        assert!(f > 0.0, "Should repel outward: {f}");
        let f2 = field.force_at(0.03);
        assert!(f > f2, "Closer to center should have stronger repulsion");
    }

    #[test]
    fn test_radial_field_attraction_near_boundary() {
        let field = RadialField::default();
        let f = field.force_at(0.98); // Near boundary
        assert!(f < 0.0, "Should attract inward: {f}");
    }

    #[test]
    fn test_radial_field_displacement_direction() {
        let field = RadialField::new(0.1, 0.9, 0.2);
        let point = vec![0.02, 0.0, 0.0]; // Inside min_radius
        let disp = field.compute_displacement(&point);
        // Displacement should point in same direction as point (radial outward)
        assert!(disp[0] > 0.0, "Should push outward: {:?}", disp);
        assert!(disp[1].abs() < 1e-15);
        assert!(disp[2].abs() < 1e-15);
    }

    #[test]
    fn test_radial_field_apply_moves_outward() {
        let field = RadialField::new(0.2, 0.9, 0.5);
        let mut point = vec![0.05, 0.0, 0.0]; // Inside min_radius
        let orig_norm = 0.05_f64;
        field.apply(&mut point);
        let new_norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            new_norm > orig_norm,
            "Should move outward: {orig_norm} → {new_norm}"
        );
    }

    #[test]
    fn test_radial_field_potential_zero_in_zone() {
        let field = RadialField::default();
        assert!((field.potential(0.5) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_radial_field_potential_positive_outside() {
        let field = RadialField::default();
        assert!(field.potential(0.01) > 0.0);
        assert!(field.potential(0.98) > 0.0);
    }

    // ── DampedHomeostasis tests ──

    #[test]
    fn test_damped_step_stays_in_ball() {
        let engine = DampedHomeostasis::with_defaults();
        let pos = vec![0.8, 0.3, 0.0]; // Near boundary
        let mut dyn_state = NodeDynamics::new(3);

        // Run many steps — position must always stay inside the ball
        let mut current = pos;
        for _ in 0..100 {
            current = engine.step(&current, &mut dyn_state);
            let norm: f64 = current.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                norm <= engine.config.max_radius + 1e-6,
                "Position escaped ball: norm={norm}"
            );
        }
    }

    #[test]
    fn test_damped_pushes_toward_zone() {
        let engine = DampedHomeostasis::with_defaults();
        // Start very close to center (outside min_radius zone)
        let pos = vec![0.005, 0.0, 0.0];
        let mut dyn_state = NodeDynamics::new(3);

        let new_pos = engine.step(&pos, &mut dyn_state);
        let old_norm: f64 = pos.iter().map(|x| x * x).sum::<f64>().sqrt();
        let new_norm: f64 = new_pos.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Should move outward (toward homeostatic zone)
        assert!(
            new_norm > old_norm,
            "Should push outward: {old_norm} → {new_norm}"
        );
    }

    #[test]
    fn test_damped_velocity_decays() {
        let engine = DampedHomeostasis::with_defaults();
        let pos = vec![0.5, 0.0, 0.0]; // In homeostatic zone
        let mut dyn_state = NodeDynamics::new(3);
        // Give it an artificial velocity
        dyn_state.velocity = vec![0.05, 0.0, 0.0];

        // After many steps with no force (in zone), velocity should decay.
        // Effective per-step factor ≈ 1 - λ·dt = 1 - 0.3·0.01 = 0.997,
        // so ~550 steps needed for 0.05 → <0.01 (0.997^550 ≈ 0.19).
        let mut current = pos;
        for _ in 0..600 {
            current = engine.step(&current, &mut dyn_state);
        }

        let vel_norm: f64 = dyn_state.velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            vel_norm < 0.01,
            "Velocity should decay due to damping after 600 steps: {vel_norm}"
        );
    }

    #[test]
    fn test_damped_batch_tick() {
        let engine = DampedHomeostasis::with_defaults();
        let positions = vec![
            vec![0.005, 0.0, 0.0], // Too close to center
            vec![0.5, 0.3, 0.0],   // In zone
            vec![0.9, 0.3, 0.0],   // Near boundary
        ];
        let mut dynamics: Vec<NodeDynamics> = (0..3).map(|_| NodeDynamics::new(3)).collect();

        let report = engine.tick(&positions, &mut dynamics);

        assert_eq!(report.total_nodes, 3);
        assert_eq!(report.new_positions.len(), 3);

        // All positions must be inside the ball
        for pos in &report.new_positions {
            let norm: f64 = pos.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm <= 0.95 + 1e-6, "Position escaped: norm={norm}");
        }
    }

    #[test]
    fn test_damped_settling() {
        let engine = DampedHomeostasis::with_defaults();
        // Point in zone, no velocity → should settle immediately
        let positions = vec![vec![0.5, 0.0, 0.0]];
        let mut dynamics = vec![NodeDynamics::new(3)];

        let report = engine.tick(&positions, &mut dynamics);
        assert!(
            report.mean_kinetic_energy < 1e-6,
            "Should be near settled: KE={}",
            report.mean_kinetic_energy
        );
    }

    #[test]
    fn test_damped_velocity_clipping() {
        let engine = DampedHomeostasis::with_defaults();
        let pos = vec![0.5, 0.0, 0.0];
        let mut dyn_state = NodeDynamics::new(3);
        // Give extreme velocity
        dyn_state.velocity = vec![10.0, 0.0, 0.0];

        let _new_pos = engine.step(&pos, &mut dyn_state);

        let vel_norm: f64 = dyn_state.velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            vel_norm <= engine.config.max_velocity + 1e-6,
            "Velocity should be clipped: {vel_norm}"
        );
    }

    #[test]
    fn test_damping_report_display() {
        let report = DampingReport {
            new_positions: vec![],
            mean_displacement: 0.001,
            mean_kinetic_energy: 1e-9,
            clamped_nodes: 0,
            total_nodes: 10,
            is_settled: true,
        };
        let s = format!("{report}");
        assert!(s.contains("SETTLED"));
        assert!(s.contains("nodes=10"));
    }
}
