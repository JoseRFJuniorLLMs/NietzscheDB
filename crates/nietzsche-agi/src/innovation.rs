// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Innovation — Acceptance function Φ(τ) for knowledge integration
//!
//! The [`InnovationEvaluator`] decides the fate of each inference using
//! a weighted acceptance function that balances three forces:
//!
//! ```text
//! Φ(τ) = α·S + β·D - γ·R
//! ```
//!
//! | Symbol | Name | Meaning | Default Weight |
//! |--------|------|---------|----------------|
//! | S | Stability | E(τ) from the stability evaluator | α = 0.50 |
//! | D | Discovery | D(τ) from the discovery field | β = 0.35 |
//! | R | Rupture | Rupture risk indicator | γ = 0.60 |
//!
//! ## Decision Thresholds
//!
//! | Decision | Condition | Action |
//! |----------|-----------|--------|
//! | **Accept** | Φ(τ) ≥ accept_threshold | Direct insertion with full weight |
//! | **Sandbox** | sandbox_threshold ≤ Φ(τ) < accept_threshold | Quarantine for spectral testing |
//! | **Reject** | Φ(τ) < sandbox_threshold | Discard — not worth the risk |
//!
//! ## Design Philosophy
//!
//! The acceptance function creates a **metabolic balance**:
//! - Pure stability (high S, low D) → conservative reasoning, accepted but not innovative
//! - Pure discovery (high D, low S) → speculative, sent to sandbox for testing
//! - High rupture (high R) → penalized regardless of other scores
//! - The "sweet spot" is moderate stability + moderate discovery + low rupture

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the acceptance function Φ(τ).
///
/// Calibrated defaults: α=0.50, β=0.35, γ=0.60
#[derive(Debug, Clone)]
pub struct InnovationConfig {
    /// Weight for stability component (S).
    /// Higher α → more conservative system.
    /// Default: 0.50
    pub alpha: f64,

    /// Weight for discovery component (D).
    /// Higher β → more innovative system.
    /// Default: 0.35
    pub beta: f64,

    /// Weight for rupture penalty (R).
    /// Higher γ → stricter rejection of broken paths.
    /// Default: 0.60
    pub gamma: f64,

    /// Minimum Φ(τ) for direct acceptance.
    /// Default: 0.50
    pub accept_threshold: f64,

    /// Minimum Φ(τ) for sandbox (quarantine).
    /// Below this, the inference is rejected.
    /// Default: 0.20
    pub sandbox_threshold: f64,
}

impl Default for InnovationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.50,
            beta: 0.35,
            gamma: 0.60,
            accept_threshold: 0.50,
            sandbox_threshold: 0.20,
        }
    }
}

// ─────────────────────────────────────────────
// AcceptanceDecision
// ─────────────────────────────────────────────

/// The three possible fates for an inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceptanceDecision {
    /// Φ(τ) ≥ accept_threshold: structurally sound and innovative enough.
    /// Insert directly into the manifold with full weight.
    Accept,

    /// sandbox_threshold ≤ Φ(τ) < accept_threshold: promising but uncertain.
    /// Send to quarantine for spectral testing before promotion.
    Sandbox,

    /// Φ(τ) < sandbox_threshold: too risky or too broken.
    /// Discard entirely — not worth the metabolic cost.
    Reject,
}

impl AcceptanceDecision {
    /// Returns true if the inference will be stored (either directly or in sandbox).
    pub fn is_stored(&self) -> bool {
        !matches!(self, AcceptanceDecision::Reject)
    }

    /// Returns true if the inference needs quarantine testing.
    pub fn needs_sandbox(&self) -> bool {
        matches!(self, AcceptanceDecision::Sandbox)
    }
}

impl std::fmt::Display for AcceptanceDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptanceDecision::Accept => write!(f, "ACCEPT"),
            AcceptanceDecision::Sandbox => write!(f, "SANDBOX"),
            AcceptanceDecision::Reject => write!(f, "REJECT"),
        }
    }
}

// ─────────────────────────────────────────────
// InnovationReport — detailed evaluation result
// ─────────────────────────────────────────────

/// Detailed result of the acceptance function evaluation.
#[derive(Debug, Clone)]
pub struct InnovationReport {
    /// Final acceptance score Φ(τ).
    pub phi: f64,

    /// Stability component: α·S
    pub stability_component: f64,

    /// Discovery component: β·D
    pub discovery_component: f64,

    /// Rupture penalty component: γ·R
    pub rupture_component: f64,

    /// The decision based on Φ(τ) and thresholds.
    pub decision: AcceptanceDecision,

    /// Input values (for transparency)
    pub stability_input: f64,
    pub discovery_input: f64,
    pub rupture_input: f64,
}

impl std::fmt::Display for InnovationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Φ(τ)={:.4} [αS={:.3}, βD={:.3}, γR={:.3}] → {}",
            self.phi,
            self.stability_component,
            self.discovery_component,
            self.rupture_component,
            self.decision,
        )
    }
}

// ─────────────────────────────────────────────
// InnovationEvaluator
// ─────────────────────────────────────────────

/// Evaluates inferences using the acceptance function Φ(τ) = αS + βD - γR.
///
/// This is the metabolic gate of the AGI system: it decides what knowledge
/// is accepted, what is tested, and what is rejected.
///
/// **Pure computation** — no graph access.
pub struct InnovationEvaluator {
    config: InnovationConfig,
}

impl InnovationEvaluator {
    pub fn new(config: InnovationConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(InnovationConfig::default())
    }

    /// Evaluate an inference and determine its fate.
    ///
    /// # Arguments
    /// - `stability`: S ∈ [0, 1] — from StabilityReport.energy (E(τ))
    /// - `discovery`: D ∈ [0, 1] — from DiscoveryReport.discovery_score
    /// - `rupture`: R ∈ [0, 1] — rupture risk indicator.
    ///   Recommended: `1.0 - rationale.fidelity` or `1.0 - min_hop_gcs / threshold`
    ///
    /// # Returns
    /// An [`InnovationReport`] with Φ(τ), sub-components, and decision.
    pub fn evaluate(
        &self,
        stability: f64,
        discovery: f64,
        rupture: f64,
    ) -> InnovationReport {
        let s = stability.clamp(0.0, 1.0);
        let d = discovery.clamp(0.0, 1.0);
        let r = rupture.clamp(0.0, 1.0);

        let stability_component = self.config.alpha * s;
        let discovery_component = self.config.beta * d;
        let rupture_component = self.config.gamma * r;

        // Φ(τ) = αS + βD - γR
        let phi = stability_component + discovery_component - rupture_component;

        let decision = if phi >= self.config.accept_threshold {
            AcceptanceDecision::Accept
        } else if phi >= self.config.sandbox_threshold {
            AcceptanceDecision::Sandbox
        } else {
            AcceptanceDecision::Reject
        };

        InnovationReport {
            phi,
            stability_component,
            discovery_component,
            rupture_component,
            decision,
            stability_input: s,
            discovery_input: d,
            rupture_input: r,
        }
    }

    /// Quick check: would this inference be accepted, sandboxed, or rejected?
    pub fn decide(
        &self,
        stability: f64,
        discovery: f64,
        rupture: f64,
    ) -> AcceptanceDecision {
        self.evaluate(stability, discovery, rupture).decision
    }

    /// Compute Φ(τ) from a Rationale and DiscoveryReport.
    ///
    /// Convenience method that extracts S, D, R from existing structures.
    /// - S = rationale.fidelity (or energy_seal if sealed)
    /// - D = discovery_score
    /// - R = 1.0 - min_hop_gcs (rupture risk from weakest link)
    pub fn evaluate_from_scores(
        &self,
        fidelity: f64,
        energy_seal: Option<f64>,
        discovery_score: f64,
        min_hop_gcs: f64,
    ) -> InnovationReport {
        let stability = energy_seal.unwrap_or(fidelity);
        let rupture = (1.0 - min_hop_gcs).clamp(0.0, 1.0);
        self.evaluate(stability, discovery_score, rupture)
    }

    pub fn config(&self) -> &InnovationConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// NavigabilityScore — second-order innovation metric
// ─────────────────────────────────────────────

/// Configuration for the navigability innovation score.
///
/// The score I = λ₂^α · E(τ)^β · [H_path^γ · (1 - H_path)]
/// creates a parabolic peak at H_path ≈ 0.5 (controlled innovation):
/// - H → 0: dogma (redundant paths, no novelty)
/// - H → 1: delirium (random noise, no structure)
/// - H ≈ 0.5: innovation sweet spot
#[derive(Debug, Clone)]
pub struct NavigabilityConfig {
    /// Exponent for spectral connectivity λ₂.
    /// Higher α → connectivity is a harder requirement.
    /// Default: 0.5
    pub alpha: f64,

    /// Exponent for trajectory energy E(τ).
    /// Higher β → stability is more important.
    /// Default: 0.3
    pub beta: f64,

    /// Exponent for path entropy H_path.
    /// Higher γ → the system rewards navigability diversity more.
    /// Default: 1.2
    pub gamma: f64,
}

impl Default for NavigabilityConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.3,
            gamma: 1.2,
        }
    }
}

/// Innovation grade derived from the navigability score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InnovationGrade {
    /// I ≥ 0.10 — structurally sound with productive diversity.
    /// The system is generating novel inferences without breaking coherence.
    HighFidelity,

    /// 0.05 ≤ I < 0.10 — promising territory.
    /// WeakBridges are being explored; some may promote.
    Promising,

    /// 0.01 ≤ I < 0.05 — speculative territory.
    /// Metaphoric drift is dominant; keep exploring but don't commit.
    Speculative,

    /// I < 0.01 — either dogmatic (H≈0) or delirious (H≈1).
    /// The system needs intervention (curiosity engine or consolidation).
    Rupture,
}

impl std::fmt::Display for InnovationGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InnovationGrade::HighFidelity => write!(f, "HighFidelity"),
            InnovationGrade::Promising => write!(f, "Promising"),
            InnovationGrade::Speculative => write!(f, "Speculative"),
            InnovationGrade::Rupture => write!(f, "Rupture"),
        }
    }
}

/// Report from navigability innovation evaluation.
#[derive(Debug, Clone)]
pub struct NavigabilityReport {
    /// Raw score I = λ₂^α · E^β · H^γ · (1-H).
    pub score: f64,

    /// Grade classification derived from score thresholds.
    pub grade: InnovationGrade,

    /// The parabolic term H^γ · (1-H) ∈ [0, ~0.37] for γ=1.2.
    /// Maximum at H ≈ γ/(γ+1) ≈ 0.545.
    pub entropy_parabola: f64,

    /// Input values for transparency.
    pub lambda2: f64,
    pub energy: f64,
    pub h_path: f64,
}

impl std::fmt::Display for NavigabilityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "I={:.6} [{}] (λ₂={:.3}, E={:.3}, H={:.3}, H·(1-H)={:.4})",
            self.score, self.grade, self.lambda2, self.energy, self.h_path, self.entropy_parabola,
        )
    }
}

/// Evaluates the **navigability innovation score** I(τ) from the metabolic state.
///
/// This is the second-order metric that couples the three metabolic observables
/// [λ₂, E(τ), H_path] into a single measure of "controlled innovation":
///
/// ```text
/// I = λ₂^α · E(τ)^β · [H_path^γ · (1 - H_path)]
/// ```
///
/// The term H^γ·(1-H) is a parabola with peak at H = γ/(γ+1).
/// This mathematically defines the **Point of Controlled Innovation**:
/// the system is neither dogmatic (H→0) nor delirious (H→1).
///
/// ## Integration
///
/// The navigability score feeds into the MetabolicSleepManager:
/// - **HighFidelity** → Cruising sleep (×1.0)
/// - **Promising** → Cruising sleep (×1.0)
/// - **Speculative** → CuriosityEngine may activate
/// - **Rupture** → CuriosityEngine forced activation (stagnation or delirium)
pub struct NavigabilityEvaluator {
    config: NavigabilityConfig,
}

impl NavigabilityEvaluator {
    pub fn new(config: NavigabilityConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(NavigabilityConfig::default())
    }

    /// Compute the navigability innovation score.
    ///
    /// # Arguments
    /// - `lambda2`: Fiedler eigenvalue λ₂ (spectral connectivity)
    /// - `energy`: E(τ) trajectory energy ∈ [0, 1]
    /// - `h_path`: H_path (navigability entropy, unnormalized)
    ///
    /// # Note on H_path normalization
    /// H_path from PathEntropyEstimator is Shannon entropy (not necessarily in [0,1]).
    /// We normalize it: h_norm = min(h_path / ln(n_unique), 1.0).
    /// If n_unique is not available, pass h_path already normalized to [0,1].
    pub fn evaluate(&self, lambda2: f64, energy: f64, h_path: f64) -> NavigabilityReport {
        let l2 = lambda2.max(0.0);
        let e = energy.clamp(0.0, 1.0);
        let h = h_path.clamp(0.0, 1.0);

        // Parabolic entropy term: H^γ · (1 - H)
        let entropy_parabola = h.powf(self.config.gamma) * (1.0 - h);

        // Full score: I = λ₂^α · E^β · H^γ·(1-H)
        let score = l2.powf(self.config.alpha)
            * e.powf(self.config.beta)
            * entropy_parabola;

        let grade = if score >= 0.10 {
            InnovationGrade::HighFidelity
        } else if score >= 0.05 {
            InnovationGrade::Promising
        } else if score >= 0.01 {
            InnovationGrade::Speculative
        } else {
            InnovationGrade::Rupture
        };

        NavigabilityReport {
            score,
            grade,
            entropy_parabola,
            lambda2: l2,
            energy: e,
            h_path: h,
        }
    }

    /// Quick classification without full report.
    pub fn grade(&self, lambda2: f64, energy: f64, h_path: f64) -> InnovationGrade {
        self.evaluate(lambda2, energy, h_path).grade
    }

    /// Returns the H_path value that maximizes the parabolic term.
    ///
    /// H_peak = γ / (γ + 1)
    ///
    /// For default γ=1.2: H_peak ≈ 0.545
    pub fn optimal_h_path(&self) -> f64 {
        self.config.gamma / (self.config.gamma + 1.0)
    }

    pub fn config(&self) -> &NavigabilityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_inference_accepted() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.9, 0.5, 0.0);
        // Φ = 0.50*0.9 + 0.35*0.5 - 0.60*0.0 = 0.45 + 0.175 = 0.625
        assert_eq!(report.decision, AcceptanceDecision::Accept);
        assert!(report.phi > 0.5);
    }

    #[test]
    fn test_innovative_but_unstable_sandbox() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.4, 0.8, 0.1);
        // Φ = 0.50*0.4 + 0.35*0.8 - 0.60*0.1 = 0.20 + 0.28 - 0.06 = 0.42
        assert_eq!(report.decision, AcceptanceDecision::Sandbox);
        assert!(report.phi >= 0.2 && report.phi < 0.5);
    }

    #[test]
    fn test_broken_path_rejected() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.2, 0.1, 0.9);
        // Φ = 0.50*0.2 + 0.35*0.1 - 0.60*0.9 = 0.10 + 0.035 - 0.54 = -0.405
        assert_eq!(report.decision, AcceptanceDecision::Reject);
        assert!(report.phi < 0.2);
    }

    #[test]
    fn test_high_discovery_low_stability() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.3, 1.0, 0.2);
        // Φ = 0.50*0.3 + 0.35*1.0 - 0.60*0.2 = 0.15 + 0.35 - 0.12 = 0.38
        assert_eq!(report.decision, AcceptanceDecision::Sandbox);
    }

    #[test]
    fn test_high_stability_no_discovery() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.95, 0.0, 0.0);
        // Φ = 0.50*0.95 + 0.35*0 - 0.60*0 = 0.475
        // Just below accept_threshold (0.50) → Sandbox
        assert_eq!(report.decision, AcceptanceDecision::Sandbox);
    }

    #[test]
    fn test_sweet_spot() {
        let eval = InnovationEvaluator::with_defaults();
        // Moderate stability + moderate discovery + low rupture
        let report = eval.evaluate(0.7, 0.6, 0.05);
        // Φ = 0.50*0.7 + 0.35*0.6 - 0.60*0.05 = 0.35 + 0.21 - 0.03 = 0.53
        assert_eq!(report.decision, AcceptanceDecision::Accept);
        assert!(report.phi > 0.5);
    }

    #[test]
    fn test_rupture_penalty_dominates() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.8, 0.8, 1.0);
        // Φ = 0.50*0.8 + 0.35*0.8 - 0.60*1.0 = 0.40 + 0.28 - 0.60 = 0.08
        assert_eq!(report.decision, AcceptanceDecision::Reject);
    }

    #[test]
    fn test_decide_shortcut() {
        let eval = InnovationEvaluator::with_defaults();
        assert_eq!(eval.decide(0.9, 0.5, 0.0), AcceptanceDecision::Accept);
        assert_eq!(eval.decide(0.2, 0.1, 0.9), AcceptanceDecision::Reject);
    }

    #[test]
    fn test_evaluate_from_scores() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate_from_scores(
            0.75,        // fidelity
            Some(0.82),  // energy_seal overrides
            0.5,         // discovery
            0.9,         // min_hop_gcs → R = 0.1
        );
        // S = 0.82 (sealed), D = 0.5, R = 0.1
        // Φ = 0.50*0.82 + 0.35*0.5 - 0.60*0.1 = 0.41 + 0.175 - 0.06 = 0.525
        assert_eq!(report.decision, AcceptanceDecision::Accept);
    }

    #[test]
    fn test_custom_config() {
        // Ultra-conservative: high stability weight, low discovery
        let config = InnovationConfig {
            alpha: 0.80,
            beta: 0.10,
            gamma: 0.70,
            accept_threshold: 0.60,
            sandbox_threshold: 0.30,
        };
        let eval = InnovationEvaluator::new(config);
        let report = eval.evaluate(0.5, 0.9, 0.1);
        // Φ = 0.80*0.5 + 0.10*0.9 - 0.70*0.1 = 0.40 + 0.09 - 0.07 = 0.42
        assert_eq!(report.decision, AcceptanceDecision::Sandbox);
    }

    #[test]
    fn test_report_display() {
        let eval = InnovationEvaluator::with_defaults();
        let report = eval.evaluate(0.7, 0.6, 0.1);
        let s = format!("{report}");
        assert!(s.contains("Φ(τ)="));
        assert!(s.contains("ACCEPT") || s.contains("SANDBOX") || s.contains("REJECT"));
    }

    #[test]
    fn test_decision_properties() {
        assert!(AcceptanceDecision::Accept.is_stored());
        assert!(AcceptanceDecision::Sandbox.is_stored());
        assert!(!AcceptanceDecision::Reject.is_stored());

        assert!(!AcceptanceDecision::Accept.needs_sandbox());
        assert!(AcceptanceDecision::Sandbox.needs_sandbox());
        assert!(!AcceptanceDecision::Reject.needs_sandbox());
    }

    // ── NavigabilityEvaluator tests ──

    #[test]
    fn test_navigability_sweet_spot() {
        let eval = NavigabilityEvaluator::with_defaults();
        // Good connectivity, good energy, H ≈ 0.5 (sweet spot)
        let report = eval.evaluate(0.5, 0.8, 0.5);
        assert!(
            report.score > 0.05,
            "Sweet spot should have meaningful score: {}",
            report.score
        );
        assert!(
            report.grade == InnovationGrade::Promising || report.grade == InnovationGrade::HighFidelity,
            "Sweet spot should be at least Promising: {}",
            report.grade
        );
    }

    #[test]
    fn test_navigability_dogma() {
        let eval = NavigabilityEvaluator::with_defaults();
        // H = 0 → dogma → H^γ·(1-H) = 0
        let report = eval.evaluate(0.5, 0.8, 0.0);
        assert_eq!(report.score, 0.0, "H=0 should give zero score");
        assert_eq!(report.grade, InnovationGrade::Rupture);
    }

    #[test]
    fn test_navigability_delirium() {
        let eval = NavigabilityEvaluator::with_defaults();
        // H = 1 → delirium → H^γ·(1-H) = 1^γ·0 = 0
        let report = eval.evaluate(0.5, 0.8, 1.0);
        assert_eq!(report.score, 0.0, "H=1 should give zero score");
        assert_eq!(report.grade, InnovationGrade::Rupture);
    }

    #[test]
    fn test_navigability_parabola_peak() {
        let eval = NavigabilityEvaluator::with_defaults();
        // The peak of H^γ·(1-H) is at H = γ/(γ+1)
        let h_peak = eval.optimal_h_path();
        assert!((h_peak - 1.2 / 2.2).abs() < 1e-10);

        // Score at peak should be higher than at H=0.1 or H=0.9
        let at_peak = eval.evaluate(0.5, 0.8, h_peak);
        let at_low = eval.evaluate(0.5, 0.8, 0.1);
        let at_high = eval.evaluate(0.5, 0.8, 0.9);

        assert!(
            at_peak.score > at_low.score,
            "Peak ({:.4}) should beat low ({:.4})",
            at_peak.score, at_low.score
        );
        assert!(
            at_peak.score > at_high.score,
            "Peak ({:.4}) should beat high ({:.4})",
            at_peak.score, at_high.score
        );
    }

    #[test]
    fn test_navigability_disconnected_graph() {
        let eval = NavigabilityEvaluator::with_defaults();
        // λ₂ = 0 → disconnected → score = 0
        let report = eval.evaluate(0.0, 0.8, 0.5);
        assert_eq!(report.score, 0.0, "Disconnected graph should give zero: {}", report.score);
        assert_eq!(report.grade, InnovationGrade::Rupture);
    }

    #[test]
    fn test_navigability_zero_energy() {
        let eval = NavigabilityEvaluator::with_defaults();
        // E = 0 → structurally broken → score = 0
        let report = eval.evaluate(0.5, 0.0, 0.5);
        assert_eq!(report.score, 0.0, "Zero energy should give zero: {}", report.score);
    }

    #[test]
    fn test_navigability_grade_ordering() {
        let eval = NavigabilityEvaluator::with_defaults();

        // Strong system → high fidelity
        let strong = eval.evaluate(2.0, 0.95, 0.55);
        // Weak system → speculative or rupture
        let weak = eval.evaluate(0.01, 0.3, 0.5);

        assert!(
            strong.score > weak.score,
            "Strong ({:.4}) should beat weak ({:.4})",
            strong.score, weak.score
        );
    }

    #[test]
    fn test_navigability_report_display() {
        let eval = NavigabilityEvaluator::with_defaults();
        let report = eval.evaluate(0.5, 0.8, 0.5);
        let s = format!("{report}");
        assert!(s.contains("I="));
        assert!(
            s.contains("HighFidelity") || s.contains("Promising")
                || s.contains("Speculative") || s.contains("Rupture")
        );
    }
}
