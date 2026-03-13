// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Certification — Epistemological quality levels for inferences
//!
//! Maps the continuous energy E(τ) from the [`stability`](crate::stability) module
//! into discrete certification levels that classify the reliability of an inference.
//!
//! ## Levels
//!
//! | Level | E(τ) Range | Meaning |
//! |-------|-----------|---------|
//! | **StableInference** | ≥ 0.75 | Structurally sound, geodesically coherent, causally valid |
//! | **WeakBridge** | [0.50, 0.75) | Reasonable but with geometric or causal imperfections |
//! | **MetaphoricDrift** | [0.25, 0.50) | Culturally suggestive but structurally weak |
//! | **LogicalRupture** | < 0.25 | Geometrically broken — the inference is rejected |
//!
//! ## Design
//!
//! The certification is a **post-hoc stamp** applied after the stability evaluator
//! computes E(τ). It is stored in a [`CertificationSeal`] alongside the energy
//! value and timestamp, creating an immutable quality record.
//!
//! This prevents the system from accepting "cultural paths" (MetaphoricDrift)
//! as "structural truths" (StableInference).

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// CertificationLevel — the 4 epistemological tiers
// ─────────────────────────────────────────────

/// Epistemological certification level for an inference.
///
/// Determined by the energy E(τ) computed by the stability evaluator.
/// The levels form a strict hierarchy: only `StableInference` represents
/// a fully validated, structurally sound reasoning path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CertificationLevel {
    /// **Logical Rupture** — E(τ) < 0.25
    ///
    /// The trajectory is geometrically broken. At least one hop violates
    /// the geodesic constraint severely. This inference is **REJECTED**.
    ///
    /// The system must NOT use this inference for further reasoning.
    LogicalRupture = 0,

    /// **Metaphoric Drift** — 0.25 ≤ E(τ) < 0.50
    ///
    /// The trajectory has some geometric validity but is structurally weak.
    /// It may represent a cultural or metaphorical connection rather than
    /// a genuine structural relationship.
    ///
    /// Useful for exploration but not for building upon.
    MetaphoricDrift = 1,

    /// **Weak Bridge** — 0.50 ≤ E(τ) < 0.75
    ///
    /// The trajectory is geometrically reasonable but has imperfections:
    /// some acausal edges, minor geodesic deviations, or uneven quality.
    ///
    /// Can be used for reasoning but should be flagged as provisional.
    WeakBridge = 2,

    /// **Stable Inference** — E(τ) ≥ 0.75
    ///
    /// The trajectory is structurally sound: geodesically coherent,
    /// causally valid, and consistent across all hops.
    ///
    /// This is the gold standard — the inference can be trusted for
    /// synthesis, feedback loops, and knowledge crystallization.
    StableInference = 3,
}

impl CertificationLevel {
    /// Returns true if this level represents a usable (non-rejected) inference.
    pub fn is_usable(&self) -> bool {
        !matches!(self, CertificationLevel::LogicalRupture)
    }

    /// Returns true if this level is strong enough for synthesis operations.
    pub fn allows_synthesis(&self) -> bool {
        matches!(
            self,
            CertificationLevel::StableInference | CertificationLevel::WeakBridge
        )
    }

    /// Returns true if this inference should be flagged as provisional.
    pub fn is_provisional(&self) -> bool {
        matches!(
            self,
            CertificationLevel::WeakBridge | CertificationLevel::MetaphoricDrift
        )
    }

    /// Returns a human-readable description of this level.
    pub fn description(&self) -> &'static str {
        match self {
            CertificationLevel::StableInference => "Structurally sound, geodesically coherent",
            CertificationLevel::WeakBridge => "Reasonable but geometrically imperfect",
            CertificationLevel::MetaphoricDrift => "Culturally suggestive, structurally weak",
            CertificationLevel::LogicalRupture => "Geometrically broken — REJECTED",
        }
    }

    /// Returns the minimum energy threshold for this level.
    pub fn min_energy(&self) -> f64 {
        match self {
            CertificationLevel::StableInference => 0.75,
            CertificationLevel::WeakBridge => 0.50,
            CertificationLevel::MetaphoricDrift => 0.25,
            CertificationLevel::LogicalRupture => 0.0,
        }
    }
}

impl std::fmt::Display for CertificationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CertificationLevel::StableInference => write!(f, "StableInference"),
            CertificationLevel::WeakBridge => write!(f, "WeakBridge"),
            CertificationLevel::MetaphoricDrift => write!(f, "MetaphoricDrift"),
            CertificationLevel::LogicalRupture => write!(f, "LogicalRupture"),
        }
    }
}

// ─────────────────────────────────────────────
// CertificationConfig — customizable thresholds
// ─────────────────────────────────────────────

/// Configurable thresholds for certification levels.
///
/// The default thresholds match the standard definition:
/// - StableInference: ≥ 0.75
/// - WeakBridge: ≥ 0.50
/// - MetaphoricDrift: ≥ 0.25
/// - LogicalRupture: < 0.25
#[derive(Debug, Clone)]
pub struct CertificationConfig {
    /// Minimum E(τ) for StableInference.
    /// Default: 0.75
    pub stable_threshold: f64,

    /// Minimum E(τ) for WeakBridge.
    /// Default: 0.50
    pub weak_threshold: f64,

    /// Minimum E(τ) for MetaphoricDrift.
    /// Default: 0.25
    pub drift_threshold: f64,
}

impl Default for CertificationConfig {
    fn default() -> Self {
        Self {
            stable_threshold: 0.75,
            weak_threshold: 0.50,
            drift_threshold: 0.25,
        }
    }
}

// ─────────────────────────────────────────────
// CertificationSeal — immutable quality stamp
// ─────────────────────────────────────────────

/// An immutable quality stamp attached to an inference.
///
/// Once sealed, the certification cannot be changed. It serves as a
/// permanent record of the inference's structural quality at the time
/// of evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationSeal {
    /// The certification level.
    pub level: CertificationLevel,

    /// The exact energy value E(τ) that produced this certification.
    pub energy: f64,

    /// Unix timestamp when the seal was created.
    pub sealed_at: i64,
}

impl CertificationSeal {
    /// Create a new seal from an energy value.
    pub fn new(energy: f64, config: &CertificationConfig) -> Self {
        let level = certify(energy, config);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Self {
            level,
            energy,
            sealed_at: now,
        }
    }

    /// Create a seal with a specific timestamp (for testing or replay).
    pub fn with_timestamp(energy: f64, config: &CertificationConfig, timestamp: i64) -> Self {
        Self {
            level: certify(energy, config),
            energy,
            sealed_at: timestamp,
        }
    }
}

impl std::fmt::Display for CertificationSeal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] E(τ)={:.4} — {}",
            self.level,
            self.energy,
            self.level.description()
        )
    }
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Classify an energy value into a certification level.
///
/// This is the core function: given E(τ), determine which tier
/// the inference belongs to.
pub fn certify(energy: f64, config: &CertificationConfig) -> CertificationLevel {
    if energy >= config.stable_threshold {
        CertificationLevel::StableInference
    } else if energy >= config.weak_threshold {
        CertificationLevel::WeakBridge
    } else if energy >= config.drift_threshold {
        CertificationLevel::MetaphoricDrift
    } else {
        CertificationLevel::LogicalRupture
    }
}

/// Convenience: classify using default thresholds.
pub fn certify_default(energy: f64) -> CertificationLevel {
    certify(energy, &CertificationConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stable_inference() {
        let config = CertificationConfig::default();
        assert_eq!(certify(0.75, &config), CertificationLevel::StableInference);
        assert_eq!(certify(0.99, &config), CertificationLevel::StableInference);
        assert_eq!(certify(1.00, &config), CertificationLevel::StableInference);
    }

    #[test]
    fn test_weak_bridge() {
        let config = CertificationConfig::default();
        assert_eq!(certify(0.50, &config), CertificationLevel::WeakBridge);
        assert_eq!(certify(0.74, &config), CertificationLevel::WeakBridge);
    }

    #[test]
    fn test_metaphoric_drift() {
        let config = CertificationConfig::default();
        assert_eq!(certify(0.25, &config), CertificationLevel::MetaphoricDrift);
        assert_eq!(certify(0.49, &config), CertificationLevel::MetaphoricDrift);
    }

    #[test]
    fn test_logical_rupture() {
        let config = CertificationConfig::default();
        assert_eq!(certify(0.24, &config), CertificationLevel::LogicalRupture);
        assert_eq!(certify(0.0, &config), CertificationLevel::LogicalRupture);
    }

    #[test]
    fn test_boundary_values() {
        let config = CertificationConfig::default();
        // Exact boundaries → higher tier wins
        assert_eq!(certify(0.75, &config), CertificationLevel::StableInference);
        assert_eq!(certify(0.50, &config), CertificationLevel::WeakBridge);
        assert_eq!(certify(0.25, &config), CertificationLevel::MetaphoricDrift);
    }

    #[test]
    fn test_certification_seal() {
        let config = CertificationConfig::default();
        let seal = CertificationSeal::new(0.82, &config);
        assert_eq!(seal.level, CertificationLevel::StableInference);
        assert!((seal.energy - 0.82).abs() < 1e-10);
        assert!(seal.sealed_at > 0);
    }

    #[test]
    fn test_level_properties() {
        assert!(CertificationLevel::StableInference.is_usable());
        assert!(CertificationLevel::StableInference.allows_synthesis());
        assert!(!CertificationLevel::StableInference.is_provisional());

        assert!(CertificationLevel::WeakBridge.is_usable());
        assert!(CertificationLevel::WeakBridge.allows_synthesis());
        assert!(CertificationLevel::WeakBridge.is_provisional());

        assert!(CertificationLevel::MetaphoricDrift.is_usable());
        assert!(!CertificationLevel::MetaphoricDrift.allows_synthesis());
        assert!(CertificationLevel::MetaphoricDrift.is_provisional());

        assert!(!CertificationLevel::LogicalRupture.is_usable());
        assert!(!CertificationLevel::LogicalRupture.allows_synthesis());
    }

    #[test]
    fn test_level_ordering() {
        // LogicalRupture < MetaphoricDrift < WeakBridge < StableInference
        assert!(CertificationLevel::LogicalRupture < CertificationLevel::MetaphoricDrift);
        assert!(CertificationLevel::MetaphoricDrift < CertificationLevel::WeakBridge);
        assert!(CertificationLevel::WeakBridge < CertificationLevel::StableInference);
    }

    #[test]
    fn test_custom_thresholds() {
        let strict = CertificationConfig {
            stable_threshold: 0.90,
            weak_threshold: 0.70,
            drift_threshold: 0.40,
        };
        // 0.80 would be StableInference with defaults, but WeakBridge with strict
        assert_eq!(certify(0.80, &strict), CertificationLevel::WeakBridge);
        assert_eq!(certify(0.90, &strict), CertificationLevel::StableInference);
    }

    #[test]
    fn test_seal_display() {
        let config = CertificationConfig::default();
        let seal = CertificationSeal::with_timestamp(0.62, &config, 1000);
        let s = format!("{seal}");
        assert!(s.contains("WeakBridge"));
        assert!(s.contains("0.6200"));
    }

    #[test]
    fn test_certify_default() {
        assert_eq!(certify_default(0.80), CertificationLevel::StableInference);
        assert_eq!(certify_default(0.10), CertificationLevel::LogicalRupture);
    }
}
