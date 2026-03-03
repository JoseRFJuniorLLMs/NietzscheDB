//! # Criticality — formal edge-of-chaos regime detection
//!
//! This module formalizes the "edge of chaos" as a **measurable region** in the
//! system's state space, not a metaphor. It operates on three observable
//! dimensions simultaneously:
//!
//! ## The Three Observables
//!
//! | Observable | Symbol | Meaning |
//! |------------|--------|---------|
//! | Spectral connectivity | λ₂ | Fiedler eigenvalue of graph Laplacian |
//! | Energy variance | σ²_E | Variance of node energies across the graph |
//! | Spectral velocity | dλ₂/dt | Rate of change of connectivity |
//!
//! ## Regime Classification
//!
//! ```text
//! ┌──────────┬────────────────────────────────────────────────────────┐
//! │  Rigid   │ λ₂ < λ_min  OR  σ²_E ≈ 0  OR  |d²λ₂/dt²| >> 0     │
//! │          │ System is frozen or crystallized — no innovation      │
//! ├──────────┼────────────────────────────────────────────────────────┤
//! │ Critical │ λ_min < λ₂ < λ_max  AND  σ²_E ≈ σ²_crit            │
//! │          │ AND  0 < dλ₂/dt < κ  AND  |d²λ₂/dt²| ≈ 0           │
//! │          │ Small perturbations → local reconfiguration           │
//! │          │ No perturbation → global collapse. This is the target │
//! ├──────────┼────────────────────────────────────────────────────────┤
//! │Turbulent │ λ₂ > λ_max  OR  σ²_E >> σ²_crit                     │
//! │          │ System is chaotic — no structural coherence           │
//! └──────────┴────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Scalable Band Limits
//!
//! ```text
//! λ_min = c₁ / √N         (connectivity floor scales with graph size)
//! λ_max = c₂ · log(N)     (connectivity ceiling grows logarithmically)
//! ```
//!
//! ## Adaptive Damping
//!
//! ```text
//! λ_damp(t) = λ₀ + k · (σ²_E - σ²_crit) / σ²_crit
//! ```
//!
//! - Energy oscillates too much → damping increases
//! - Energy too stable → damping decreases
//! - Energy ≈ critical → system breathes
//!
//! ## Antifragility Index
//!
//! ```text
//! A = ΔH(G) / ΔE_input
//! ```
//!
//! - A = 0: rigid (perturbation has no effect)
//! - 0 < A < 1: antifragile (controlled growth from perturbation)
//! - A > 1: unstable (perturbation amplified — system breaking)

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the criticality detector.
#[derive(Debug, Clone)]
pub struct CriticalityConfig {
    /// Coefficient c₁ for λ_min = c₁ / √N.
    /// Default: 0.1
    pub c1: f64,

    /// Coefficient c₂ for λ_max = c₂ · log(N).
    /// Default: 0.5
    pub c2: f64,

    /// Critical energy variance σ²_crit.
    /// The "sweet spot" where the system is optimally responsive.
    /// Default: 0.05
    pub sigma_crit_sq: f64,

    /// Tolerance for energy variance comparison.
    /// |σ²_E - σ²_crit| ≤ tolerance × σ²_crit → considered "critical".
    /// Default: 0.5 (50% relative tolerance)
    pub variance_tolerance: f64,

    /// Maximum acceptable spectral velocity κ = dλ₂/dt.
    /// Above this, connectivity is changing too fast.
    /// Default: 0.1
    pub max_spectral_velocity: f64,

    /// Base damping coefficient λ₀.
    /// Default: 0.3
    pub base_damping: f64,

    /// Damping feedback gain k.
    /// Controls how strongly variance deviations adjust damping.
    /// Default: 0.5
    pub damping_gain: f64,

    /// Minimum damping (prevents underdamped oscillation).
    /// Default: 0.05
    pub min_damping: f64,

    /// Maximum damping (prevents overdamped death).
    /// Default: 0.95
    pub max_damping: f64,
}

impl Default for CriticalityConfig {
    fn default() -> Self {
        Self {
            c1: 0.1,
            c2: 0.5,
            sigma_crit_sq: 0.05,
            variance_tolerance: 0.5,
            max_spectral_velocity: 0.1,
            base_damping: 0.3,
            damping_gain: 0.5,
            min_damping: 0.05,
            max_damping: 0.95,
        }
    }
}

// ─────────────────────────────────────────────
// RegimeState
// ─────────────────────────────────────────────

/// The three possible dynamical regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegimeState {
    /// Frozen/crystallized. λ₂ too low or σ²_E ≈ 0.
    /// The system is stable but cannot innovate.
    Rigid,

    /// Edge of chaos. All observables in the critical band.
    /// Small perturbations → local reconfiguration (antifragile).
    Critical,

    /// Chaotic. λ₂ too high or σ²_E >> σ²_crit.
    /// The system is unstable and structurally incoherent.
    Turbulent,
}

impl std::fmt::Display for RegimeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegimeState::Rigid => write!(f, "RIGID"),
            RegimeState::Critical => write!(f, "CRITICAL"),
            RegimeState::Turbulent => write!(f, "TURBULENT"),
        }
    }
}

// ─────────────────────────────────────────────
// CriticalityMetrics
// ─────────────────────────────────────────────

/// Complete diagnostics of the system's dynamical regime.
///
/// This is the formal "dashboard" for the edge-of-chaos state.
/// Every field is measurable and auditable.
#[derive(Debug, Clone)]
pub struct CriticalityMetrics {
    /// Current Fiedler eigenvalue λ₂.
    pub lambda2: f64,

    /// Spectral velocity: dλ₂/dt (finite difference from last 2 measurements).
    pub lambda_velocity: f64,

    /// Spectral acceleration: d²λ₂/dt² (second-order finite difference).
    pub lambda_acceleration: f64,

    /// Energy variance σ²_E across all nodes.
    pub energy_variance: f64,

    /// Antifragility index A = ΔH(G) / ΔE_input.
    /// In [0, 1] = antifragile. > 1 = unstable. = 0 = rigid.
    pub antifragility_index: f64,

    /// Computed adaptive damping coefficient.
    pub adaptive_damping: f64,

    /// Current scalable band: λ_min for this graph size.
    pub lambda_min: f64,

    /// Current scalable band: λ_max for this graph size.
    pub lambda_max: f64,

    /// Classified regime.
    pub regime: RegimeState,

    /// Number of nodes N used for band computation.
    pub node_count: usize,
}

impl std::fmt::Display for CriticalityMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Criticality[{} | λ₂={:.4} ∈ [{:.3},{:.3}] | dλ/dt={:.4} | d²λ/dt²={:.4} | σ²={:.4} | A={:.3} | damp={:.3}]",
            self.regime,
            self.lambda2,
            self.lambda_min,
            self.lambda_max,
            self.lambda_velocity,
            self.lambda_acceleration,
            self.energy_variance,
            self.antifragility_index,
            self.adaptive_damping,
        )
    }
}

// ─────────────────────────────────────────────
// CriticalityDetector
// ─────────────────────────────────────────────

/// Formal edge-of-chaos regime detector.
///
/// Given spectral and energy measurements, classifies the system's
/// dynamical regime and computes adaptive damping.
///
/// **Pure computation** — no graph access.
pub struct CriticalityDetector {
    config: CriticalityConfig,
}

impl CriticalityDetector {
    pub fn new(config: CriticalityConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(CriticalityConfig::default())
    }

    /// Compute the scalable λ₂ band for a given graph size.
    ///
    /// - `λ_min = c₁ / √N` (floor — minimum viable connectivity)
    /// - `λ_max = c₂ · log(N)` (ceiling — maximum before rigidity)
    ///
    /// For N=1, returns (c₁, c₂·log(2)) as minimum.
    pub fn compute_band(&self, n_nodes: usize) -> (f64, f64) {
        let n = (n_nodes.max(2)) as f64;
        let lambda_min = self.config.c1 / n.sqrt();
        let lambda_max = self.config.c2 * n.ln();
        (lambda_min, lambda_max.max(lambda_min + 0.01))
    }

    /// Compute adaptive damping coefficient.
    ///
    /// ```text
    /// λ(t) = λ₀ + k · (σ²_E - σ²_crit) / σ²_crit
    /// ```
    ///
    /// - High variance → more damping (calm the system)
    /// - Low variance → less damping (let it breathe)
    /// - Critical variance → base damping (equilibrium)
    pub fn compute_adaptive_damping(&self, energy_variance: f64) -> f64 {
        if self.config.sigma_crit_sq < 1e-15 {
            return self.config.base_damping;
        }

        let deviation = (energy_variance - self.config.sigma_crit_sq) / self.config.sigma_crit_sq;
        let damping = self.config.base_damping + self.config.damping_gain * deviation;

        damping.clamp(self.config.min_damping, self.config.max_damping)
    }

    /// Compute the antifragility index.
    ///
    /// ```text
    /// A = ΔH(G) / ΔE_input
    /// ```
    ///
    /// Where:
    /// - `ΔH(G)` is the change in graph structural entropy
    /// - `ΔE_input` is the energy injected into the system this cycle
    ///
    /// Returns 0 if no energy was input (no perturbation).
    pub fn compute_antifragility(
        &self,
        entropy_delta: f64,
        energy_input: f64,
    ) -> f64 {
        if energy_input.abs() < 1e-15 {
            return 0.0;
        }
        (entropy_delta / energy_input).clamp(-2.0, 2.0)
    }

    /// Classify the system's dynamical regime.
    ///
    /// # Arguments
    /// - `lambda2`: current Fiedler eigenvalue
    /// - `lambda_velocity`: dλ₂/dt (finite difference)
    /// - `lambda_acceleration`: d²λ₂/dt² (second-order)
    /// - `energy_variance`: σ²_E across all node energies
    /// - `n_nodes`: number of nodes in the graph
    ///
    /// # Returns
    /// `RegimeState` classification.
    pub fn classify_regime(
        &self,
        lambda2: f64,
        lambda_velocity: f64,
        energy_variance: f64,
        n_nodes: usize,
    ) -> RegimeState {
        let (lambda_min, lambda_max) = self.compute_band(n_nodes);

        // Check Turbulent first (most dangerous)
        if lambda2 > lambda_max {
            return RegimeState::Turbulent;
        }
        if energy_variance > self.config.sigma_crit_sq * (1.0 + self.config.variance_tolerance) * 3.0 {
            return RegimeState::Turbulent;
        }

        // Check Rigid
        if lambda2 < lambda_min {
            return RegimeState::Rigid;
        }
        if energy_variance < self.config.sigma_crit_sq * 0.01 {
            // σ²_E ≈ 0 → system is dead
            return RegimeState::Rigid;
        }

        // Check Critical (the target zone)
        let variance_ratio = (energy_variance - self.config.sigma_crit_sq).abs()
            / self.config.sigma_crit_sq;
        let variance_ok = variance_ratio <= self.config.variance_tolerance;
        let velocity_ok = lambda_velocity.abs() <= self.config.max_spectral_velocity;

        if variance_ok && velocity_ok {
            RegimeState::Critical
        } else if lambda_velocity.abs() > self.config.max_spectral_velocity * 2.0 {
            // Excessive spectral velocity → turbulent transition
            RegimeState::Turbulent
        } else {
            // In band but not at critical variance → rigid (conservative default)
            RegimeState::Rigid
        }
    }

    /// Full criticality analysis.
    ///
    /// # Arguments
    /// - `lambda2_history`: last 3+ λ₂ measurements (most recent last)
    /// - `energy_variance`: σ²_E across all node energies
    /// - `entropy_delta`: ΔH(G) from last cycle
    /// - `energy_input`: total energy injected this cycle
    /// - `n_nodes`: number of nodes in the graph
    ///
    /// # Returns
    /// Complete `CriticalityMetrics` with regime classification.
    pub fn analyze(
        &self,
        lambda2_history: &[f64],
        energy_variance: f64,
        entropy_delta: f64,
        energy_input: f64,
        n_nodes: usize,
    ) -> CriticalityMetrics {
        let lambda2 = lambda2_history.last().copied().unwrap_or(0.0);

        // Compute finite differences
        let lambda_velocity = if lambda2_history.len() >= 2 {
            let n = lambda2_history.len();
            lambda2_history[n - 1] - lambda2_history[n - 2]
        } else {
            0.0
        };

        let lambda_acceleration = if lambda2_history.len() >= 3 {
            let n = lambda2_history.len();
            // d²λ/dt² ≈ λ(t) - 2·λ(t-1) + λ(t-2)
            lambda2_history[n - 1] - 2.0 * lambda2_history[n - 2] + lambda2_history[n - 3]
        } else {
            0.0
        };

        let (lambda_min, lambda_max) = self.compute_band(n_nodes);
        let adaptive_damping = self.compute_adaptive_damping(energy_variance);
        let antifragility_index = self.compute_antifragility(entropy_delta, energy_input);
        let regime = self.classify_regime(lambda2, lambda_velocity, energy_variance, n_nodes);

        CriticalityMetrics {
            lambda2,
            lambda_velocity,
            lambda_acceleration,
            energy_variance,
            antifragility_index,
            adaptive_damping,
            lambda_min,
            lambda_max,
            regime,
            node_count: n_nodes,
        }
    }

    pub fn config(&self) -> &CriticalityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalable_band_small_graph() {
        let detector = CriticalityDetector::with_defaults();
        let (min, max) = detector.compute_band(10);
        // λ_min = 0.1 / √10 ≈ 0.0316
        assert!(min > 0.03 && min < 0.04, "λ_min for N=10: {min}");
        // λ_max = 0.5 · ln(10) ≈ 1.15
        assert!(max > 1.0 && max < 1.3, "λ_max for N=10: {max}");
    }

    #[test]
    fn test_scalable_band_large_graph() {
        let detector = CriticalityDetector::with_defaults();
        let (min_small, max_small) = detector.compute_band(100);
        let (min_large, max_large) = detector.compute_band(10000);

        // Larger graph → smaller λ_min (easier to satisfy)
        assert!(
            min_large < min_small,
            "Larger graph should have smaller λ_min: {min_large} vs {min_small}"
        );
        // Larger graph → larger λ_max (more room)
        assert!(
            max_large > max_small,
            "Larger graph should have larger λ_max: {max_large} vs {max_small}"
        );
    }

    #[test]
    fn test_adaptive_damping_high_variance() {
        let detector = CriticalityDetector::with_defaults();
        // σ²_E = 0.2 >> σ²_crit = 0.05 → damping should increase
        let damp = detector.compute_adaptive_damping(0.2);
        assert!(
            damp > detector.config.base_damping,
            "High variance should increase damping: {damp}"
        );
    }

    #[test]
    fn test_adaptive_damping_low_variance() {
        let detector = CriticalityDetector::with_defaults();
        // σ²_E = 0.01 < σ²_crit = 0.05 → damping should decrease
        let damp = detector.compute_adaptive_damping(0.01);
        assert!(
            damp < detector.config.base_damping,
            "Low variance should decrease damping: {damp}"
        );
    }

    #[test]
    fn test_adaptive_damping_critical_variance() {
        let detector = CriticalityDetector::with_defaults();
        // σ²_E ≈ σ²_crit → damping ≈ base
        let damp = detector.compute_adaptive_damping(0.05);
        assert!(
            (damp - detector.config.base_damping).abs() < 0.01,
            "Critical variance should give base damping: {damp}"
        );
    }

    #[test]
    fn test_adaptive_damping_clamped() {
        let detector = CriticalityDetector::with_defaults();
        // Extreme variance → damping clamped
        let damp = detector.compute_adaptive_damping(100.0);
        assert!(
            damp <= detector.config.max_damping,
            "Damping should be clamped: {damp}"
        );

        let damp2 = detector.compute_adaptive_damping(0.0);
        assert!(
            damp2 >= detector.config.min_damping,
            "Damping should be clamped: {damp2}"
        );
    }

    #[test]
    fn test_antifragility_controlled() {
        let detector = CriticalityDetector::with_defaults();
        // Small entropy increase for moderate energy input → A ∈ (0, 1)
        let a = detector.compute_antifragility(0.05, 0.1);
        assert!(
            a > 0.0 && a < 1.0,
            "Controlled growth should give A ∈ (0,1): {a}"
        );
    }

    #[test]
    fn test_antifragility_rigid() {
        let detector = CriticalityDetector::with_defaults();
        // No entropy change → A = 0
        let a = detector.compute_antifragility(0.0, 0.1);
        assert!((a - 0.0).abs() < 1e-10, "Rigid system: A = {a}");
    }

    #[test]
    fn test_antifragility_unstable() {
        let detector = CriticalityDetector::with_defaults();
        // Entropy increase > energy input → A > 1
        let a = detector.compute_antifragility(0.2, 0.1);
        assert!(a > 1.0, "Unstable system: A = {a}");
    }

    #[test]
    fn test_antifragility_no_input() {
        let detector = CriticalityDetector::with_defaults();
        let a = detector.compute_antifragility(0.1, 0.0);
        assert!((a - 0.0).abs() < 1e-10, "No input → A = 0: {a}");
    }

    #[test]
    fn test_regime_rigid_low_lambda() {
        let detector = CriticalityDetector::with_defaults();
        // λ₂ = 0.001 (below λ_min for N=100)
        let regime = detector.classify_regime(0.001, 0.0, 0.05, 100);
        assert_eq!(regime, RegimeState::Rigid);
    }

    #[test]
    fn test_regime_rigid_dead_variance() {
        let detector = CriticalityDetector::with_defaults();
        // σ²_E ≈ 0 → system is dead, even if λ₂ is fine
        let regime = detector.classify_regime(0.5, 0.0, 0.0001, 100);
        assert_eq!(regime, RegimeState::Rigid);
    }

    #[test]
    fn test_regime_turbulent_high_lambda() {
        let detector = CriticalityDetector::with_defaults();
        // λ₂ = 10.0 >> λ_max for N=100
        let regime = detector.classify_regime(10.0, 0.0, 0.05, 100);
        assert_eq!(regime, RegimeState::Turbulent);
    }

    #[test]
    fn test_regime_turbulent_high_variance() {
        let detector = CriticalityDetector::with_defaults();
        // σ²_E = 5.0 >> σ²_crit
        let regime = detector.classify_regime(0.5, 0.0, 5.0, 100);
        assert_eq!(regime, RegimeState::Turbulent);
    }

    #[test]
    fn test_regime_critical() {
        let detector = CriticalityDetector::with_defaults();
        // λ₂ in band, σ²_E ≈ σ²_crit, low velocity
        // λ_min(100) = 0.1/10 = 0.01, λ_max(100) = 0.5*ln(100) ≈ 2.3
        let regime = detector.classify_regime(0.5, 0.02, 0.05, 100);
        assert_eq!(
            regime,
            RegimeState::Critical,
            "Should be critical with λ₂=0.5, σ²=0.05"
        );
    }

    #[test]
    fn test_regime_turbulent_high_velocity() {
        let detector = CriticalityDetector::with_defaults();
        // λ₂ in band, but velocity too high (> 2× threshold)
        let regime = detector.classify_regime(0.5, 0.25, 0.05, 100);
        assert_eq!(regime, RegimeState::Turbulent);
    }

    #[test]
    fn test_full_analysis() {
        let detector = CriticalityDetector::with_defaults();
        let history = vec![0.45, 0.48, 0.50];
        let metrics = detector.analyze(
            &history,
            0.05,  // energy variance at critical
            0.03,  // small entropy increase
            0.1,   // moderate energy input
            100,   // 100 nodes
        );

        assert_eq!(metrics.regime, RegimeState::Critical);
        assert!((metrics.lambda2 - 0.50).abs() < 1e-10);
        assert!((metrics.lambda_velocity - 0.02).abs() < 1e-10);
        assert!(metrics.antifragility_index > 0.0 && metrics.antifragility_index < 1.0);
        assert!((metrics.adaptive_damping - detector.config.base_damping).abs() < 0.01);
    }

    #[test]
    fn test_analysis_empty_history() {
        let detector = CriticalityDetector::with_defaults();
        let metrics = detector.analyze(&[], 0.05, 0.0, 0.0, 10);
        assert_eq!(metrics.lambda2, 0.0);
        assert_eq!(metrics.lambda_velocity, 0.0);
        assert_eq!(metrics.lambda_acceleration, 0.0);
    }

    #[test]
    fn test_metrics_display() {
        let metrics = CriticalityMetrics {
            lambda2: 0.5,
            lambda_velocity: 0.02,
            lambda_acceleration: -0.001,
            energy_variance: 0.05,
            antifragility_index: 0.3,
            adaptive_damping: 0.3,
            lambda_min: 0.01,
            lambda_max: 2.3,
            regime: RegimeState::Critical,
            node_count: 100,
        };
        let s = format!("{metrics}");
        assert!(s.contains("CRITICAL"));
        assert!(s.contains("λ₂="));
    }

    #[test]
    fn test_regime_display() {
        assert_eq!(format!("{}", RegimeState::Rigid), "RIGID");
        assert_eq!(format!("{}", RegimeState::Critical), "CRITICAL");
        assert_eq!(format!("{}", RegimeState::Turbulent), "TURBULENT");
    }
}
