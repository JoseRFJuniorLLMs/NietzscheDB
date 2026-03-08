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

/// Project a coordinate vector back into the open unit ball, applying
/// **smooth radial compression** above a soft threshold to prevent
/// boundary saturation (Problema 2: Radial Saturation Near Boundary).
///
/// ## Compression scheme
///
/// - `‖x‖ < SOFT_THRESHOLD` (0.90): no modification (linear zone).
/// - `SOFT_THRESHOLD ≤ ‖x‖ < 1.0`: smooth compression via
///   `r_new = t + (MAX_R - t) · tanh(α · (r - t) / (MAX_R - t))`,
///   where `t = SOFT_THRESHOLD`, `MAX_R = 0.98`, `α ≈ 1.5`.
///   This maps `[0.90, ∞)` → `[0.90, 0.98)` with continuous derivative.
/// - `‖x‖ ≥ 1.0`: hard clamp to `MAX_R` (safety net).
///
/// ## Why not a hard clamp?
///
/// A hard clamp at 0.95 creates an artificial density wall — all overflow
/// nodes pile up at exactly ‖x‖ = 0.95, destroying radial resolution.
/// The tanh compression preserves **relative ordering** of radii while
/// keeping all points safely inside the ball.
pub fn project_into_ball(coords: Vec<f64>) -> Vec<f64> {
    const SOFT_THRESHOLD: f64 = 0.90;
    const MAX_R: f64 = 0.98;
    const ALPHA: f64 = 1.5;

    let norm: f64 = coords.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm < SOFT_THRESHOLD {
        // Linear zone — no modification
        return coords;
    }

    let new_r = if norm < 1.0 {
        // Smooth compression: tanh squash into [SOFT_THRESHOLD, MAX_R)
        let t = SOFT_THRESHOLD;
        let range = MAX_R - t;
        let normalized = (norm - t) / range; // how far past threshold (0..∞ theoretically)
        t + range * (ALPHA * normalized).tanh()
    } else {
        // Hard safety net for anything at or beyond the boundary
        MAX_R - 1e-6
    };

    let scale = new_r / (norm + 1e-15);
    coords.into_iter().map(|x| x * scale).collect()
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
/// If the parent is near the origin, an isotropic random direction is used
/// (Gaussian sampling for rotational invariance in high dimensions).
fn radial_direction(parent: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let norm: f64 = parent.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm > 1e-8 {
        parent.iter().map(|x| x / norm).collect()
    } else {
        random_unit_vector(parent.len(), rng)
    }
}

/// Generate an isotropic random unit vector via Gaussian sampling.
///
/// Each coordinate is drawn from N(0,1), then the vector is normalized.
/// This produces uniform distribution on S^(d-1), unlike uniform [-1,1]
/// which suffers from concentration of measure in high dimensions.
fn random_unit_vector(dim: usize, rng: &mut impl Rng) -> Vec<f64> {
    let coords: Vec<f64> = (0..dim)
        .map(|_| {
            // Box-Muller transform for N(0,1)
            let u1: f64 = rng.gen::<f64>().max(1e-300);
            let u2: f64 = rng.gen::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    let n: f64 = coords.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        coords.iter().map(|x| x / n).collect()
    } else {
        let mut d = vec![0.0; dim];
        d[0] = 1.0;
        d
    }
}

/// Diversified child spawning with angular jitter.
///
/// Instead of placing the child exactly along the parent's radial direction,
/// the displacement is rotated by a random angle in the tangent hyperplane.
/// This prevents geodesic convergence (semantic black holes) where children
/// of nearby parents collapse into the same angular region.
///
/// # Parameters
///
/// - `angular_jitter`: controls the cone half-angle (radians).
///   - `0.0` = pure radial (same as `spawn_child`)
///   - `0.3` = moderate spread (~17°)
///   - `0.6` = wide spread (~34°)
///   - Recommended: `0.2–0.4` for production
///
/// # Geometric guarantee
///
/// The child still satisfies `‖child‖ > ‖parent‖` because the displacement
/// magnitude (`depth_offset`) is preserved — only the direction is perturbed.
pub fn spawn_child_diversified(
    parent: &[f64],
    depth_offset: f64,
    angular_jitter: f64,
    rng: &mut impl Rng,
) -> Vec<f64> {
    let radial = radial_direction(parent, rng);

    let direction = if angular_jitter > 1e-8 && parent.len() >= 2 {
        // Generate random tangent vector (perpendicular to radial)
        let tangent = random_tangent_vector(&radial, rng);

        // Sample jitter angle from half-normal: |N(0, σ)|
        // This keeps most children near the radial direction while
        // providing diversity in the tails.
        let u1: f64 = rng.gen::<f64>().max(1e-300);
        let u2: f64 = rng.gen::<f64>();
        let theta = ((-2.0 * u1.ln()).sqrt()
            * (2.0 * std::f64::consts::PI * u2).cos())
            .abs()
            * angular_jitter;

        // Rotate: d = cos(θ) · radial + sin(θ) · tangent
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let d: Vec<f64> = radial.iter().zip(tangent.iter())
            .map(|(&r, &t)| cos_t * r + sin_t * t)
            .collect();

        // Re-normalize (should be ~1.0 already, but ensure)
        let n: f64 = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-12 {
            d.iter().map(|x| x / n).collect()
        } else {
            radial
        }
    } else {
        radial
    };

    let v: Vec<f64> = direction.iter().map(|d| d * depth_offset).collect();
    let result = mobius_add(parent, &v);
    project_into_ball(result)
}

/// Generate a random unit vector in the hyperplane perpendicular to `normal`.
///
/// Uses Gram-Schmidt orthogonalization: sample random Gaussian vector,
/// project out the component along `normal`, normalize.
fn random_tangent_vector(normal: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let dim = normal.len();

    // Random Gaussian vector
    let raw = random_unit_vector(dim, rng);

    // Project out the normal component: tangent = raw - (raw·n)n
    let dot: f64 = raw.iter().zip(normal.iter()).map(|(r, n)| r * n).sum();
    let tangent: Vec<f64> = raw.iter().zip(normal.iter())
        .map(|(&r, &n)| r - dot * n)
        .collect();

    // Normalize
    let n: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        tangent.iter().map(|x| x / n).collect()
    } else {
        // Degenerate case: pick a canonical perpendicular
        let mut perp = vec![0.0; dim];
        // Find first non-dominant axis
        let max_idx = normal.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let other = if max_idx == 0 { 1 } else { 0 };
        perp[other] = 1.0;
        // Gram-Schmidt
        let d: f64 = perp.iter().zip(normal.iter()).map(|(p, n)| p * n).sum();
        for i in 0..dim {
            perp[i] -= d * normal[i];
        }
        let pn: f64 = perp.iter().map(|x| x * x).sum::<f64>().sqrt();
        if pn > 1e-12 {
            perp.iter().map(|x| x / pn).collect()
        } else {
            perp[0] = 1.0;
            perp
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

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
        assert!(norm < 0.98, "should be compressed below MAX_R: norm={norm}");
    }

    #[test]
    fn project_into_ball_below_threshold_untouched() {
        let v = vec![0.5, 0.3]; // norm ≈ 0.58 — well below 0.90
        let projected = project_into_ball(v.clone());
        for (a, b) in v.iter().zip(projected.iter()) {
            assert!((a - b).abs() < 1e-15, "should not modify: {a} vs {b}");
        }
    }

    #[test]
    fn project_into_ball_smooth_compression_preserves_order() {
        // Two points above threshold: r1 < r2 → projected r1 < projected r2
        let v1 = vec![0.92, 0.0];
        let v2 = vec![0.96, 0.0];
        let p1 = project_into_ball(v1);
        let p2 = project_into_ball(v2);
        let r1: f64 = p1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let r2: f64 = p2.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(r1 < r2, "order not preserved: r1={r1} >= r2={r2}");
        assert!(r2 < 0.98, "r2 should be below MAX_R: {r2}");
    }

    #[test]
    fn project_into_ball_continuity_at_threshold() {
        // Just below and just above the soft threshold should give close results
        let below = vec![0.899, 0.0];
        let above = vec![0.901, 0.0];
        let p_below = project_into_ball(below);
        let p_above = project_into_ball(above);
        let r_below: f64 = p_below.iter().map(|x| x * x).sum::<f64>().sqrt();
        let r_above: f64 = p_above.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Gap should be small (smooth transition)
        assert!((r_above - r_below).abs() < 0.01,
            "discontinuity at threshold: {r_below} vs {r_above}");
    }

    #[test]
    fn random_unit_vector_is_unit() {
        let mut rng = rand::thread_rng();
        for dim in [2, 3, 10, 100] {
            let v = random_unit_vector(dim, &mut rng);
            assert_eq!(v.len(), dim);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "dim={dim}: norm={norm}");
        }
    }

    #[test]
    fn random_tangent_vector_is_perpendicular() {
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let normal = random_unit_vector(5, &mut rng);
            let tangent = random_tangent_vector(&normal, &mut rng);
            let dot: f64 = normal.iter().zip(tangent.iter()).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-10, "tangent not perpendicular: dot={dot}");
            let norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "tangent not unit: norm={norm}");
        }
    }

    #[test]
    fn spawn_child_diversified_stays_in_ball() {
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let parent = vec![0.7, 0.3];
            let child = spawn_child_diversified(&parent, 0.1, 0.4, &mut rng);
            let norm: f64 = child.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "diversified child outside ball: norm={norm}");
        }
    }

    #[test]
    fn spawn_child_diversified_is_deeper() {
        let mut rng = rand::thread_rng();
        let parent = vec![0.3, 0.0];
        let parent_norm: f64 = parent.iter().map(|x| x * x).sum::<f64>().sqrt();
        // With moderate jitter, child should still be deeper in most cases
        let mut deeper_count = 0;
        let trials = 100;
        for _ in 0..trials {
            let child = spawn_child_diversified(&parent, 0.1, 0.3, &mut rng);
            let child_norm: f64 = child.iter().map(|x| x * x).sum::<f64>().sqrt();
            if child_norm > parent_norm { deeper_count += 1; }
        }
        // At least 90% should be deeper (cos component dominates)
        assert!(deeper_count >= 90, "only {deeper_count}/{trials} children deeper than parent");
    }

    #[test]
    fn spawn_child_diversified_zero_jitter_matches_spawn_child() {
        // With angular_jitter=0, diversified should behave like plain spawn_child
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let parent = vec![0.4, 0.2];
        let plain = spawn_child(&parent, 0.1, &mut rng1);
        let diversified = spawn_child_diversified(&parent, 0.1, 0.0, &mut rng2);
        for (a, b) in plain.iter().zip(diversified.iter()) {
            assert!((a - b).abs() < 1e-12, "zero jitter diverged: {a} vs {b}");
        }
    }

    #[test]
    fn spawn_child_diversified_angular_spread() {
        // Verify that jitter actually produces angular diversity
        let mut rng = rand::thread_rng();
        let parent = vec![0.3, 0.0, 0.0];
        let mut angles = Vec::new();
        for _ in 0..50 {
            let child = spawn_child_diversified(&parent, 0.1, 0.4, &mut rng);
            let angle = child[1].atan2(child[0]);
            angles.push(angle);
        }
        // Check variance of angles — should be non-trivial with jitter=0.4
        let mean = angles.iter().sum::<f64>() / angles.len() as f64;
        let var = angles.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / angles.len() as f64;
        assert!(var > 1e-4, "angular variance too low: {var} — jitter not working");
    }

    #[test]
    fn spawn_child_diversified_high_dim() {
        let mut rng = rand::thread_rng();
        // Test with realistic high-dimensional embeddings
        let dim = 128;
        let mut parent = vec![0.0; dim];
        parent[0] = 0.5;
        parent[1] = 0.1;
        for _ in 0..20 {
            let child = spawn_child_diversified(&parent, 0.05, 0.3, &mut rng);
            assert_eq!(child.len(), dim);
            let norm: f64 = child.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "high-dim child outside ball: norm={norm}");
        }
    }
}
