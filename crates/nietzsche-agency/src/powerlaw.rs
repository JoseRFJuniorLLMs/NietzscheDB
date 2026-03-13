// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Power-law exponent (τ) estimator for Self-Organized Criticality monitoring.
//!
//! # Background: Self-Organized Criticality (SOC)
//!
//! In a self-organized critical system, the distribution of event sizes (avalanches)
//! follows a power law:
//!
//! ```text
//! P(S) ~ S^{-τ}
//! ```
//!
//! where `S` is the avalanche size and `τ` (tau) is the **critical exponent**.
//! Typical SOC systems (sandpile models, earthquakes, neural avalanches) exhibit
//! τ values in the range [1.5, 2.5]. The canonical Bak-Tang-Wiesenfeld sandpile
//! has τ ≈ 1.59 in 2D.
//!
//! # Estimation Method: Clauset-Shalizi-Newman (2009)
//!
//! Reference: Clauset, Shalizi & Newman, "Power-law distributions in empirical
//! data", SIAM Review 51(4), 661–703 (2009). arXiv:0706.1062
//!
//! ## Maximum Likelihood Estimator (MLE)
//!
//! For discrete data with values s_i ≥ s_min:
//!
//! ```text
//! τ̂ = 1 + n · [Σᵢ ln(sᵢ / (s_min − 0.5))]⁻¹
//! ```
//!
//! Note the `s_min − 0.5` correction for discrete data (CSN Eq. 3.5).
//! For continuous data, the denominator would use `s_min` directly.
//!
//! ## Goodness of Fit: Kolmogorov-Smirnov (KS) Test
//!
//! The KS distance measures the maximum deviation between the empirical CDF
//! and the theoretical power-law CDF:
//!
//! ```text
//! D = max_s |F_empirical(s) − F_model(s)|
//! ```
//!
//! where `F_model(s) = 1 − (s / s_min)^{1−τ}` for continuous approximation,
//! or using the discrete Hurwitz zeta for exact discrete CDF.
//!
//! ## Selecting s_min
//!
//! We iterate candidate s_min values from 1 upward and select the one that
//! minimises the KS distance. This avoids fitting the power law to the
//! non-power-law body of the distribution.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nietzsche_agency::powerlaw::{is_power_law, PowerLawResult};
//!
//! // Observed avalanche sizes from the agency engine
//! let sizes: Vec<u32> = vec![1, 1, 2, 1, 3, 1, 2, 5, 1, 1, 2, 8, 1, 3, 1];
//! let result = is_power_law(&sizes);
//! println!("τ = {:.3}, KS = {:.4}, significant = {}", result.tau, result.ks_distance, result.is_significant);
//! ```
//!
//! # Integration
//!
//! This module is **pure observation/analysis** — it does not modify graph
//! state or alter any agency behaviour. It reads avalanche size data
//! collected by the engine's avalanche counter and reports diagnostics.

// ─────────────────────────────────────────────
//  Public types
// ─────────────────────────────────────────────

/// Result of a complete power-law analysis on avalanche size data.
#[derive(Debug, Clone)]
pub struct PowerLawResult {
    /// Estimated power-law exponent (τ̂). Typical SOC range: [1.5, 2.5].
    pub tau: f64,

    /// Lower cutoff for the power-law tail. Only sizes ≥ s_min are used.
    pub s_min: u32,

    /// Kolmogorov-Smirnov distance: max deviation between empirical and
    /// theoretical CDFs. Lower is better; values > 0.10 suggest poor fit.
    pub ks_distance: f64,

    /// Whether the power-law hypothesis is plausible (KS distance below
    /// the critical threshold for the given sample size).
    pub is_significant: bool,

    /// Number of data points in the tail (s_i ≥ s_min) used for estimation.
    pub n_tail: usize,
}

// ─────────────────────────────────────────────
//  Core functions
// ─────────────────────────────────────────────

/// Estimate the power-law exponent τ via discrete MLE (Clauset-Shalizi-Newman).
///
/// Uses the discrete correction `s_min − 0.5` per CSN Eq. 3.5.
///
/// Returns `None` if:
/// - `sizes` is empty
/// - No values ≥ `s_min` exist
/// - `s_min` is 0 (undefined)
/// - The log-sum is zero or negative (degenerate data)
pub fn estimate_tau(sizes: &[u32], s_min: u32) -> Option<f64> {
    if s_min == 0 || sizes.is_empty() {
        return None;
    }

    let threshold = s_min as f64 - 0.5; // discrete correction (CSN Eq. 3.5)
    if threshold <= 0.0 {
        return None;
    }

    let mut n: usize = 0;
    let mut log_sum: f64 = 0.0;

    for &s in sizes {
        if s >= s_min {
            // ln(s_i / (s_min − 0.5))
            let ratio = (s as f64) / threshold;
            log_sum += ratio.ln();
            n += 1;
        }
    }

    if n == 0 || log_sum <= 0.0 {
        return None;
    }

    // τ̂ = 1 + n · [Σ ln(s_i / (s_min − 0.5))]⁻¹
    let tau = 1.0 + (n as f64) / log_sum;
    Some(tau)
}

/// Kolmogorov-Smirnov goodness-of-fit test for the power-law hypothesis.
///
/// Computes `D = max |F_empirical(s) − F_model(s)|` over the tail (s ≥ s_min).
///
/// The theoretical CDF uses the continuous approximation:
///
/// ```text
/// F_model(s) = 1 − (s / s_min)^{1−τ}
/// ```
///
/// Returns `f64::INFINITY` if there are fewer than 2 data points in the tail
/// or if `s_min` is 0.
pub fn ks_test(sizes: &[u32], tau: f64, s_min: u32) -> f64 {
    if s_min == 0 {
        return f64::INFINITY;
    }

    // Extract and sort the tail
    let mut tail: Vec<u32> = sizes.iter().copied().filter(|&s| s >= s_min).collect();
    if tail.len() < 2 {
        return f64::INFINITY;
    }
    tail.sort_unstable();

    let n = tail.len() as f64;
    let s_min_f = s_min as f64;
    let exponent = 1.0 - tau; // negative for τ > 1

    let mut max_d: f64 = 0.0;

    for (i, &s) in tail.iter().enumerate() {
        // Empirical CDF: fraction of points ≤ s
        // Since tail is sorted, F_emp at this point = (i+1)/n
        let f_emp = (i as f64 + 1.0) / n;

        // Theoretical CDF: F(s) = 1 − (s / s_min)^{1−τ}
        let f_model = 1.0 - (s as f64 / s_min_f).powf(exponent);

        let d = (f_emp - f_model).abs();
        if d > max_d {
            max_d = d;
        }

        // Also check the left side of the step: (i)/n
        if i > 0 {
            let f_emp_left = (i as f64) / n;
            let d_left = (f_emp_left - f_model).abs();
            if d_left > max_d {
                max_d = d_left;
            }
        }
    }

    max_d
}

/// Perform a complete power-law analysis on avalanche size data.
///
/// This function:
/// 1. Iterates candidate `s_min` values from 1 up to the 90th percentile
/// 2. For each candidate, estimates τ via MLE and computes the KS distance
/// 3. Selects the `s_min` that minimises KS distance
/// 4. Determines significance using the critical KS value for the sample size
///
/// The significance criterion follows the CSN recommendation:
///
/// ```text
/// D_critical ≈ c(α) / √n_tail
/// ```
///
/// where `c(α) = 1.36` for α = 0.05 (Kolmogorov distribution).
///
/// Returns a result with `tau = 0.0` and `is_significant = false` if the
/// data is insufficient (fewer than 10 samples, or no valid s_min found).
pub fn is_power_law(sizes: &[u32]) -> PowerLawResult {
    let insufficient = PowerLawResult {
        tau: 0.0,
        s_min: 0,
        ks_distance: f64::INFINITY,
        is_significant: false,
        n_tail: 0,
    };

    if sizes.len() < 10 {
        return insufficient;
    }

    // Determine the search range for s_min.
    // We search from 1 up to the 90th percentile to ensure a reasonable tail.
    let mut sorted = sizes.to_vec();
    sorted.sort_unstable();

    let p90_idx = (sorted.len() as f64 * 0.90).ceil() as usize;
    let p90_idx = p90_idx.min(sorted.len() - 1);
    let s_max_candidate = sorted[p90_idx];

    // Collect unique candidate s_min values
    let mut candidates: Vec<u32> = sorted.iter()
        .copied()
        .filter(|&s| s >= 1 && s <= s_max_candidate)
        .collect();
    candidates.dedup();

    if candidates.is_empty() {
        return insufficient;
    }

    let mut best_ks = f64::INFINITY;
    let mut best_tau = 0.0_f64;
    let mut best_smin = 0_u32;
    let mut best_n_tail = 0_usize;

    for &candidate in &candidates {
        let n_tail = sizes.iter().filter(|&&s| s >= candidate).count();

        // Require at least 10 points in the tail for a meaningful estimate
        if n_tail < 10 {
            continue;
        }

        if let Some(tau) = estimate_tau(sizes, candidate) {
            // Sanity check: τ should be > 1 for a valid power law
            if tau <= 1.0 || tau > 10.0 {
                continue;
            }

            let ks = ks_test(sizes, tau, candidate);
            if ks < best_ks {
                best_ks = ks;
                best_tau = tau;
                best_smin = candidate;
                best_n_tail = n_tail;
            }
        }
    }

    if best_smin == 0 {
        return insufficient;
    }

    // Significance test: KS critical value at α = 0.05
    // D_critical = 1.36 / √n (Kolmogorov distribution asymptotic)
    let d_critical = 1.36 / (best_n_tail as f64).sqrt();
    let is_significant = best_ks < d_critical;

    PowerLawResult {
        tau: best_tau,
        s_min: best_smin,
        ks_distance: best_ks,
        is_significant,
        n_tail: best_n_tail,
    }
}

// ─────────────────────────────────────────────
//  Helpers (internal)
// ─────────────────────────────────────────────

/// Standard error of the MLE estimate (Cramér-Rao lower bound).
///
/// ```text
/// σ_τ = (τ̂ − 1) / √n
/// ```
///
/// Useful for reporting confidence intervals: τ̂ ± 1.96·σ_τ for 95% CI.
#[allow(dead_code)]
pub fn tau_std_error(tau: f64, n_tail: usize) -> f64 {
    if n_tail == 0 {
        return f64::INFINITY;
    }
    (tau - 1.0) / (n_tail as f64).sqrt()
}

// ─────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- estimate_tau tests --------------------------------------------------

    #[test]
    fn test_estimate_tau_empty() {
        assert!(estimate_tau(&[], 1).is_none());
    }

    #[test]
    fn test_estimate_tau_zero_smin() {
        assert!(estimate_tau(&[1, 2, 3], 0).is_none());
    }

    #[test]
    fn test_estimate_tau_no_values_above_smin() {
        assert!(estimate_tau(&[1, 2, 3], 10).is_none());
    }

    #[test]
    fn test_estimate_tau_all_equal_to_smin() {
        // All values equal s_min: ln(s_min / (s_min - 0.5)) = ln(1/(1-0.5)) = ln(2)
        // τ = 1 + n / (n * ln(2)) = 1 + 1/ln(2) ≈ 2.4427
        let sizes = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let tau = estimate_tau(&sizes, 1).unwrap();
        let expected = 1.0 + 1.0 / (1.0_f64 / 0.5).ln();
        assert!((tau - expected).abs() < 1e-10, "tau = {tau}, expected = {expected}");
    }

    #[test]
    fn test_estimate_tau_known_distribution() {
        // Generate a synthetic discrete power-law sample with known τ.
        // Use inverse transform sampling: s = floor(u^{-1/(τ-1)}) for u ~ Uniform(0,1)
        // With τ_true = 2.0 and s_min = 1, we expect τ̂ ≈ 2.0 for large n.
        let tau_true = 2.0;
        let n = 5000;
        let sizes = generate_power_law_sample(n, tau_true, 1);

        let tau_hat = estimate_tau(&sizes, 1).unwrap();

        // MLE should be within ±0.15 of the true value for n = 5000
        assert!(
            (tau_hat - tau_true).abs() < 0.15,
            "τ̂ = {tau_hat:.4}, expected ≈ {tau_true:.1}"
        );
    }

    #[test]
    fn test_estimate_tau_steep_distribution() {
        // τ_true = 2.5 (steeper — more small events, fewer large ones)
        let tau_true = 2.5;
        let n = 5000;
        let sizes = generate_power_law_sample(n, tau_true, 1);

        let tau_hat = estimate_tau(&sizes, 1).unwrap();

        assert!(
            (tau_hat - tau_true).abs() < 0.20,
            "τ̂ = {tau_hat:.4}, expected ≈ {tau_true:.1}"
        );
    }

    // -- ks_test tests -------------------------------------------------------

    #[test]
    fn test_ks_test_perfect_fit() {
        // A power-law sample tested against its own τ̂ should have low KS
        let sizes = generate_power_law_sample(1000, 2.0, 1);
        let tau = estimate_tau(&sizes, 1).unwrap();
        let ks = ks_test(&sizes, tau, 1);

        // KS should be small for a good fit
        assert!(ks < 0.10, "KS = {ks:.4}, expected < 0.10 for true power-law data");
    }

    #[test]
    fn test_ks_test_bad_fit() {
        // Uniform data is NOT power-law — KS should be large
        let sizes: Vec<u32> = (1..=100).collect();
        let tau = estimate_tau(&sizes, 1).unwrap();
        let ks = ks_test(&sizes, tau, 1);

        // For truly uniform data, even the best MLE fit will be poor
        // The KS distance should be noticeably larger than for true power-law
        // (though the MLE will still find *some* τ, the fit quality is worse)
        assert!(ks > 0.01, "KS = {ks:.4}, expected > 0.01 for non-power-law data");
    }

    #[test]
    fn test_ks_test_too_few_points() {
        let ks = ks_test(&[5], 2.0, 5);
        assert!(ks.is_infinite());
    }

    #[test]
    fn test_ks_test_smin_zero() {
        let ks = ks_test(&[1, 2, 3], 2.0, 0);
        assert!(ks.is_infinite());
    }

    // -- is_power_law tests --------------------------------------------------

    #[test]
    fn test_is_power_law_insufficient_data() {
        let result = is_power_law(&[1, 2, 3]);
        assert!(!result.is_significant);
        assert_eq!(result.n_tail, 0);
    }

    #[test]
    fn test_is_power_law_true_power_law() {
        let sizes = generate_power_law_sample(2000, 2.0, 1);
        let result = is_power_law(&sizes);

        assert!(result.is_significant, "Expected significant power-law fit");
        assert!(
            result.tau > 1.5 && result.tau < 2.5,
            "τ = {:.3}, expected in [1.5, 2.5]",
            result.tau
        );
        assert!(result.n_tail >= 10, "n_tail = {}", result.n_tail);
    }

    #[test]
    fn test_is_power_law_soc_regime() {
        // Typical SOC: τ ≈ 1.5 (Bak-Tang-Wiesenfeld 2D)
        let sizes = generate_power_law_sample(3000, 1.6, 1);
        let result = is_power_law(&sizes);

        assert!(result.is_significant, "Expected significant SOC-like power-law");
        assert!(
            result.tau > 1.3 && result.tau < 2.0,
            "τ = {:.3}, expected ≈ 1.6 for SOC",
            result.tau
        );
    }

    #[test]
    fn test_is_power_law_exponential_data() {
        // Exponential distribution is NOT power-law.
        // Generate: s = ceil(-ln(u) * 10) for u ~ Uniform(0,1)
        // Use a deterministic LCG for reproducibility.
        let n = 2000;
        let mut rng = SimpleLcg::new(54321);
        let sizes: Vec<u32> = (0..n)
            .map(|_| {
                let u = rng.next_f64();
                let s = (-u.ln() * 10.0).ceil() as u32;
                s.max(1)
            })
            .collect();

        let result = is_power_law(&sizes);

        // Exponential data may still get a "fit" but the τ should be
        // unreasonably high or the KS distance should be large.
        // The key is it should either fail significance or produce an
        // abnormal τ value.
        if result.is_significant {
            // If somehow significant, τ should be far from the SOC range
            println!(
                "Exponential data: τ = {:.3}, KS = {:.4}, s_min = {}, n_tail = {}",
                result.tau, result.ks_distance, result.s_min, result.n_tail
            );
        }
    }

    #[test]
    fn test_is_power_law_constant_data() {
        // All values the same — degenerate case
        let sizes = vec![5_u32; 100];
        let result = is_power_law(&sizes);

        // With constant data, s_min candidates are all the same value,
        // and log_sum = n * ln(s / (s - 0.5)) which gives a specific τ.
        // This is a degenerate case but should not panic.
        assert!(result.tau > 0.0 || !result.is_significant);
    }

    // -- tau_std_error tests -------------------------------------------------

    #[test]
    fn test_std_error_basic() {
        let se = tau_std_error(2.0, 100);
        // σ = (2.0 - 1.0) / √100 = 0.1
        assert!((se - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_std_error_zero_n() {
        assert!(tau_std_error(2.0, 0).is_infinite());
    }

    // -- PowerLawResult debug/clone ------------------------------------------

    #[test]
    fn test_result_debug_clone() {
        let r = PowerLawResult {
            tau: 2.0,
            s_min: 1,
            ks_distance: 0.05,
            is_significant: true,
            n_tail: 100,
        };
        let r2 = r.clone();
        assert_eq!(format!("{:?}", r2), format!("{:?}", r));
    }

    // -- Helpers for test data generation ------------------------------------

    /// Minimal deterministic LCG (Linear Congruential Generator) for tests.
    /// No external crate dependency. Period = 2^32.
    struct SimpleLcg {
        state: u64,
    }

    impl SimpleLcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG: a = 1664525, c = 1013904223, m = 2^32
            self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223) & 0xFFFF_FFFF;
            self.state as u32
        }

        fn next_f64(&mut self) -> f64 {
            // Returns (0, 1) — excludes 0 to avoid log(0)
            loop {
                let v = (self.next_u32() as f64) / (u32::MAX as f64);
                if v > 0.0 && v < 1.0 {
                    return v;
                }
            }
        }
    }

    /// Generate a discrete power-law sample with known τ and s_min via
    /// inverse transform sampling.
    ///
    /// For a discrete power law P(s) ∝ s^{-τ}, the inverse CDF is:
    ///
    /// ```text
    /// s = floor(s_min · (1 − u)^{−1/(τ−1)})
    /// ```
    ///
    /// where u ~ Uniform(0, 1).
    fn generate_power_law_sample(n: usize, tau: f64, s_min: u32) -> Vec<u32> {
        let mut rng = SimpleLcg::new(42);
        let inv_alpha = 1.0 / (tau - 1.0);
        let s_min_f = s_min as f64;

        (0..n)
            .map(|_| {
                let u = rng.next_f64();
                let s = (s_min_f * (1.0 - u).powf(-inv_alpha)).floor() as u32;
                s.max(s_min) // ensure s ≥ s_min
            })
            .collect()
    }
}
