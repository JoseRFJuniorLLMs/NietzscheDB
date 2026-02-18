//! Chebyshev polynomial approximation of the graph heat kernel `e^{−tL̃}`.
//!
//! ## Heat kernel on graphs
//!
//! The heat diffusion equation on a graph is:
//!
//! ```text
//! ∂f/∂t = −L̃ f     →     f(t) = e^{−tL̃} f(0)
//! ```
//!
//! The operator `e^{−tL̃}` is approximated by the truncated Chebyshev series
//! (Defferrard et al., NeurIPS 2016):
//!
//! ```text
//! K_t f ≈ Σ_{k=0}^{K}  c_k(t) · T_k(L̂) · f
//! ```
//!
//! where:
//! - `L̂ = 2/λ_max · L̃ − I` — scaled Laplacian (eigenvalues ∈ `[−1, 1]`)
//! - `T_k` — Chebyshev polynomial of the first kind, computed by recurrence
//! - `c_k(t) = e^{−t} · (2 − δ_{k0}) · I_k(t)` — modified Bessel coefficients
//!
//! ## Chebyshev recurrence (O(K × |E|) per call)
//!
//! ```text
//! T_0(L̂)·f = f
//! T_1(L̂)·f = L̂·f
//! T_k(L̂)·f = 2·L̂·T_{k-1}(L̂)·f − T_{k-2}(L̂)·f
//! ```
//!
//! No matrix multiplication needed — only repeated Laplacian-vector products.

use crate::laplacian::HyperbolicLaplacian;

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

/// Default number of Chebyshev terms (accuracy vs. cost trade-off).
pub const K_DEFAULT: usize = 10;

/// Safe upper bound on the largest eigenvalue of the normalized Laplacian.
/// The normalized Laplacian always has λ_max ≤ 2.
pub const LAMBDA_MAX: f64 = 2.0;

// ─────────────────────────────────────────────
// Modified Bessel function I_k(t)
// ─────────────────────────────────────────────

/// Compute `[I_0(t), I_1(t), ..., I_{k_max}(t)]` via the series:
///
/// ```text
/// I_k(t) = Σ_{m=0}^∞  (t/2)^{2m+k} / (m! · (m+k)!)
/// ```
///
/// Returns a vector of length `k_max + 1`.
/// For `t ≈ 0`: `I_0(0) = 1`, `I_k(0) = 0` for `k > 0`.
pub fn modified_bessel_i(t: f64, k_max: usize) -> Vec<f64> {
    if t.abs() < 1e-8 {
        let mut v = vec![0.0f64; k_max + 1];
        v[0] = 1.0;
        return v;
    }

    let half_t = t / 2.0;
    let mut result = vec![0.0f64; k_max + 1];

    for k in 0..=k_max {
        // First term of the series: (t/2)^k / k!
        let mut term = 1.0f64;
        for j in 1..=(k as u32) {
            term *= half_t / j as f64;
        }
        let mut sum = term;

        // Subsequent terms: multiply by (t/2)^2 / (m · (m+k))
        for m in 1..80usize {
            term *= (half_t * half_t) / (m as f64 * (m + k) as f64);
            let prev = sum;
            sum += term;
            if (sum - prev).abs() <= 1e-15 * sum.abs().max(1e-300) {
                break;
            }
        }

        result[k] = sum;
    }

    result
}

// ─────────────────────────────────────────────
// Chebyshev coefficients
// ─────────────────────────────────────────────

/// Compute Chebyshev coefficients for the heat kernel at time `t`:
///
/// ```text
/// c_k(t) = e^{−t} · (2 − δ_{k0}) · I_k(t)
/// ```
///
/// Returns a vector of length `k_max + 1`.
pub fn chebyshev_coefficients(t: f64, k_max: usize) -> Vec<f64> {
    let bessel = modified_bessel_i(t, k_max);
    let et = (-t).exp();

    bessel.iter().enumerate()
        .map(|(k, ik)| {
            let factor = if k == 0 { 1.0 } else { 2.0 };
            et * factor * ik
        })
        .collect()
}

// ─────────────────────────────────────────────
// Heat kernel application
// ─────────────────────────────────────────────

/// Apply `e^{−tL̃}` to signal `x` using `k_max` Chebyshev terms.
///
/// # Arguments
/// * `lap`        — the hyperbolic Laplacian
/// * `x`          — input signal (one value per node, `x.len() == lap.n`)
/// * `t`          — diffusion time (scale ≥ 0)
/// * `k_max`      — number of Chebyshev terms (use [`K_DEFAULT`])
/// * `lambda_max` — largest eigenvalue bound (use [`LAMBDA_MAX`])
///
/// # Returns
/// Signal after diffusion — same length as `x`.
pub fn apply_heat_kernel(
    lap:        &HyperbolicLaplacian,
    x:          &[f64],
    t:          f64,
    k_max:      usize,
    lambda_max: f64,
) -> Vec<f64> {
    let n = lap.n;
    if n == 0 || x.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(x.len(), n);

    let coeffs = chebyshev_coefficients(t, k_max);

    // ── Chebyshev recurrence ──────────────────────────────────────────────
    // T_0(L̂)·x = x
    // T_1(L̂)·x = L̂·x = apply_scaled(x)
    // T_k(L̂)·x = 2·L̂·T_{k-1} − T_{k-2}

    let mut t_prev: Vec<f64> = x.to_vec();                          // T_0
    let mut t_curr: Vec<f64> = lap.apply_scaled(x, lambda_max);    // T_1

    let mut result: Vec<f64> = vec![0.0; n];

    // k = 0 contribution
    for i in 0..n {
        result[i] += coeffs[0] * t_prev[i];
    }

    // k = 1 contribution (if requested)
    if k_max >= 1 {
        for i in 0..n {
            result[i] += coeffs[1] * t_curr[i];
        }
    }

    // k = 2 .. k_max
    for k in 2..=k_max {
        let lt_curr  = lap.apply_scaled(&t_curr, lambda_max);
        let t_next: Vec<f64> = lt_curr.iter().zip(t_prev.iter())
            .map(|(lt, tp)| 2.0 * lt - tp)
            .collect();

        for i in 0..n {
            result[i] += coeffs[k] * t_next[i];
        }

        t_prev = t_curr;
        t_curr = t_next;
    }

    result
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bessel function tests ──────────────────

    #[test]
    fn bessel_i0_at_zero_is_one() {
        let b = modified_bessel_i(0.0, 4);
        assert!((b[0] - 1.0).abs() < 1e-10, "I_0(0) = {}", b[0]);
        for k in 1..=4 {
            assert!(b[k].abs() < 1e-10, "I_{k}(0) should be 0, got {}", b[k]);
        }
    }

    #[test]
    fn bessel_i0_known_value() {
        // I_0(1.0) ≈ 1.2660658777520082
        let b = modified_bessel_i(1.0, 2);
        assert!((b[0] - 1.2660658777520082).abs() < 1e-8, "I_0(1) = {}", b[0]);
    }

    #[test]
    fn bessel_i1_known_value() {
        // I_1(1.0) ≈ 0.5651591039924851
        let b = modified_bessel_i(1.0, 2);
        assert!((b[1] - 0.5651591039924851).abs() < 1e-8, "I_1(1) = {}", b[1]);
    }

    #[test]
    fn bessel_i2_known_value() {
        // I_2(1.0) ≈ 0.1357476697670383
        let b = modified_bessel_i(1.0, 3);
        assert!((b[2] - 0.1357476697670383).abs() < 1e-7, "I_2(1) = {}", b[2]);
    }

    #[test]
    fn bessel_values_are_positive() {
        let b = modified_bessel_i(2.0, 8);
        for (k, &v) in b.iter().enumerate() {
            assert!(v >= 0.0, "I_{k}(2) = {v} is negative");
        }
    }

    // ── Chebyshev coefficient tests ────────────

    #[test]
    fn chebyshev_coefficients_all_non_negative() {
        for &t in &[0.1_f64, 1.0, 5.0, 10.0] {
            let c = chebyshev_coefficients(t, K_DEFAULT);
            for (k, &ck) in c.iter().enumerate() {
                assert!(ck >= 0.0, "c_{k}({t}) = {ck} is negative");
            }
        }
    }

    #[test]
    fn chebyshev_c0_equals_bessel_times_exp() {
        let t = 1.5;
        let c = chebyshev_coefficients(t, 2);
        let i0 = modified_bessel_i(t, 0)[0];
        let expected = (-t).exp() * i0;
        assert!((c[0] - expected).abs() < 1e-10, "c_0 mismatch: {} vs {}", c[0], expected);
    }

    #[test]
    fn chebyshev_coefficients_decay_with_k() {
        // For fixed t, c_k should generally decrease for large k
        let c = chebyshev_coefficients(1.0, 12);
        // c[10] should be much smaller than c[0]
        assert!(c[10] < c[0], "coefficients should decay: c[10]={} c[0]={}", c[10], c[0]);
    }

    // ── Heat kernel application tests ──────────

    #[test]
    fn apply_heat_kernel_empty_input_returns_empty() {
        use nietzsche_graph::{MockVectorStore, db::NietzscheDB};
        use tempfile::TempDir;
        let dir = TempDir::new().unwrap();
        let db  = NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap();
        let lap = crate::laplacian::HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        let result = apply_heat_kernel(&lap, &[], 1.0, K_DEFAULT, LAMBDA_MAX);
        assert!(result.is_empty());
    }

    #[test]
    fn apply_heat_kernel_preserves_total_signal() {
        // For a connected graph with uniform signal, the total activation
        // should be approximately preserved (heat equation conserves mass).
        use nietzsche_graph::{Edge, EdgeType, MockVectorStore, Node, PoincareVector, db::NietzscheDB};
        use uuid::Uuid;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let mut db = NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap();

        let nodes: Vec<Node> = (0..4)
            .map(|i| Node::new(
                Uuid::new_v4(),
                PoincareVector::new(vec![0.1 * (i as f64 + 1.0), 0.0]),
                serde_json::json!({}),
            ))
            .collect();
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
        for n in nodes { db.insert_node(n).unwrap(); }
        for i in 0..3 {
            db.insert_edge(Edge::new(ids[i], ids[i+1], EdgeType::Association, 1.0)).unwrap();
        }

        let lap = crate::laplacian::HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        let x: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0]; // impulse on node 0
        let y = apply_heat_kernel(&lap, &x, 1.0, K_DEFAULT, LAMBDA_MAX);

        // All outputs should be finite
        for (i, v) in y.iter().enumerate() {
            assert!(v.is_finite(), "y[{i}] = {v} is not finite");
        }

        // Total signal approximately preserved (Chebyshev is not exact but close)
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        assert!((sum_x - sum_y).abs() < 0.5,
            "signal sum changed too much: {sum_x:.4} → {sum_y:.4}");
    }
}
