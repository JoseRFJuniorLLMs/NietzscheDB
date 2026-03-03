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
}
