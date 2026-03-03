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
}
