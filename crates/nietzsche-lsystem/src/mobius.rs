//! Möbius addition for the Poincaré ball model of hyperbolic space.
//!
//! The **Möbius addition** u ⊕ v is the group operation of the Poincaré
//! ball (Gyrovector space). It is used to move a point `v` relative to
//! anchor point `u` while staying in the hyperbolic manifold.
//!
//! ## Formula
//! ```text
//! u ⊕ v = [ (1 + 2⟨u,v⟩ + ‖v‖²) · u  +  (1 - ‖u‖²) · v ]
//!         ─────────────────────────────────────────────────
//!                  1 + 2⟨u,v⟩ + ‖u‖²·‖v‖²
//! ```
//!
//! ## Child / Sibling spawning
//!
//! - **SpawnChild**: displace `depth_offset` units radially outward from the
//!   parent (toward the Poincaré boundary → more specific/episodic concept).
//! - **SpawnSibling**: displace `distance` units at `angle` radians from the
//!   parent's radial direction (lateral association at the same depth level).

use rand::Rng;

// ─────────────────────────────────────────────
// Core Möbius addition
// ─────────────────────────────────────────────

/// Compute `u ⊕ v` in the Poincaré ball of any dimension.
///
/// # Panics
/// Panics (debug) if `u` and `v` have different lengths.
pub fn mobius_add(u: &[f64], v: &[f64]) -> Vec<f64> {
    debug_assert_eq!(u.len(), v.len(), "dimension mismatch in Möbius add");

    let dot_uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x * x).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x * x).sum();

    let denom  = 1.0 + 2.0 * dot_uv + norm_u_sq * norm_v_sq;
    let coeff_u = (1.0 + 2.0 * dot_uv + norm_v_sq) / denom;
    let coeff_v = (1.0 - norm_u_sq) / denom;

    u.iter().zip(v.iter())
        .map(|(ui, vi)| coeff_u * ui + coeff_v * vi)
        .collect()
}

/// Project a coordinate vector back into the open unit ball if it has
/// drifted to the boundary (‖result‖ ≥ 0.999).
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
// Child spawning
// ─────────────────────────────────────────────

/// Compute the embedding of a **child** node spawned from `parent`.
///
/// The child is displaced `depth_offset` units radially outward — deeper
/// in the Poincaré ball (higher specificity).
///
/// If the parent is at the origin, a random outward direction is chosen.
pub fn spawn_child(parent: &[f64], depth_offset: f64, rng: &mut impl Rng) -> Vec<f64> {
    let direction = radial_direction(parent, rng);
    let v: Vec<f64> = direction.iter().map(|d| d * depth_offset).collect();
    let result = mobius_add(parent, &v);
    project_into_ball(result)
}

// ─────────────────────────────────────────────
// Sibling spawning
// ─────────────────────────────────────────────

/// Compute the embedding of a **sibling** node spawned from `parent`.
///
/// The sibling is displaced `distance` units at `angle` radians from the
/// parent's radial direction — a lateral association in hyperbolic space.
///
/// In dimensions ≥ 2 the rotation is performed in the plane spanned by
/// the radial direction and the first perpendicular basis vector. In 1-D
/// the sibling is placed on the opposite side.
pub fn spawn_sibling(parent: &[f64], angle: f64, distance: f64, rng: &mut impl Rng) -> Vec<f64> {
    let dim = parent.len();
    let radial = radial_direction(parent, rng);

    // Perpendicular direction in the radial plane
    let perp: Vec<f64> = if dim >= 2 {
        // Rotate radial by 90° in the first two coordinate axes
        let mut p = vec![0.0_f64; dim];
        p[0] = -radial[1];
        p[1] =  radial[0];
        let norm: f64 = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            p.iter().map(|x| x / norm).collect()
        } else {
            radial.clone()
        }
    } else {
        // 1-D: flip direction
        radial.iter().map(|x| -x).collect()
    };

    // Displacement: cos(angle) * radial + sin(angle) * perp, scaled by distance
    let v: Vec<f64> = radial.iter().zip(perp.iter())
        .map(|(r, p)| distance * (angle.cos() * r + angle.sin() * p))
        .collect();

    let result = mobius_add(parent, &v);
    project_into_ball(result)
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

/// Unit vector pointing radially outward from the origin.
///
/// If the parent is near the origin, a uniformly random direction is used.
fn radial_direction(parent: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let norm: f64 = parent.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm > 1e-8 {
        parent.iter().map(|x| x / norm).collect()
    } else {
        // Random unit vector (Box-Muller / normalised Gaussian coordinates)
        let coords: Vec<f64> = (0..parent.len())
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
            .collect();
        let n: f64 = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-12 {
            coords.iter().map(|x| x / n).collect()
        } else {
            let mut d = vec![0.0; parent.len()];
            d[0] = 1.0;
            d
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mobius_add_identity_at_origin() {
        // 0 ⊕ v = v
        let zero = vec![0.0, 0.0];
        let v    = vec![0.3, 0.1];
        let result = mobius_add(&zero, &v);
        assert!((result[0] - 0.3).abs() < 1e-12);
        assert!((result[1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn mobius_add_stays_in_ball() {
        let u = vec![0.5, 0.3];
        let v = vec![0.1, 0.2];
        let w = mobius_add(&u, &v);
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "result ‖w‖ = {norm:.6} must be < 1.0");
    }

    #[test]
    fn spawn_child_is_deeper() {
        let mut rng = rand::thread_rng();
        let parent = vec![0.3, 0.0];
        let child  = spawn_child(&parent, 0.1, &mut rng);
        let child_norm: f64 = child.iter().map(|x| x * x).sum::<f64>().sqrt();
        let parent_norm: f64 = parent.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Child should be deeper (larger norm)
        assert!(child_norm > parent_norm, "child norm {child_norm:.4} should exceed parent {parent_norm:.4}");
    }

    #[test]
    fn spawn_child_stays_in_ball() {
        let mut rng = rand::thread_rng();
        // Near boundary — ensure projection keeps it valid
        for _ in 0..20 {
            let parent = vec![0.8, 0.0];
            let child  = spawn_child(&parent, 0.3, &mut rng);
            let norm: f64 = child.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "child outside ball: norm = {norm}");
        }
    }

    #[test]
    fn spawn_sibling_stays_in_ball() {
        let mut rng = rand::thread_rng();
        let parent = vec![0.4, 0.2];
        for angle in [0.0_f64, 0.5, 1.0, 1.57, 3.14] {
            let sib = spawn_sibling(&parent, angle, 0.15, &mut rng);
            let norm: f64 = sib.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "sibling outside ball at angle={angle}: norm={norm}");
        }
    }

    #[test]
    fn project_into_ball_clamps_boundary() {
        let outside = vec![0.9, 0.9]; // norm ≈ 1.27
        let projected = project_into_ball(outside);
        let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "projection still outside: norm={norm}");
    }
}
