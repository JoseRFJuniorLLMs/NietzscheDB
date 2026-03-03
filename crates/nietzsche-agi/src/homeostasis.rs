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
}
