//! Hard Bounds — The Constitution of the Forgetting Engine.
//!
//! Adaptive parameters (θ, θ_e, ε) can adjust within fixed bounds,
//! but NO Zaratustra cycle can rewrite the bounds themselves.
//! Only a human operator with admin access can modify bounds.
//!
//! This prevents "death by progressive permissiveness" — the most
//! insidious failure mode where thresholds gradually relax until
//! the forgetting engine stops functioning.

use serde::{Deserialize, Serialize};

/// Fixed deployment-time bounds that NO adaptive cycle can override.
///
/// These are the "iron laws" of the forgetting constitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardBounds {
    /// Vitality threshold θ: nodes with V(n) < θ are candidates for deletion.
    pub vitality_theta_min: f32,
    pub vitality_theta_max: f32,

    /// Energy threshold θ_e: nodes with e(n) < θ_e are weak.
    pub energy_theta_min: f32,
    pub energy_theta_max: f32,

    /// Ricci collapse threshold ε: if ΔRicci < -ε, abort deletion.
    pub ricci_epsilon_min: f32,
    pub ricci_epsilon_max: f32,

    /// Minimum universe size: if |S_t| < this, HALT all deletions.
    pub min_universe_size: usize,

    /// Maximum deletion rate per cycle (fraction of total nodes).
    pub max_deletion_rate: f32,

    /// Toxicity threshold for immediate deletion (without triple condition).
    pub toxicity_threshold_min: f32,
    pub toxicity_threshold_max: f32,

    /// Sacred vitality threshold: V(n) above this = automatically sacred.
    pub sacred_vitality_min: f32,
    pub sacred_vitality_max: f32,

    /// Causal edge count above which a node is automatically sacred.
    pub sacred_causal_threshold: usize,
}

impl Default for HardBounds {
    fn default() -> Self {
        Self {
            vitality_theta_min: 0.10,
            vitality_theta_max: 0.40,
            energy_theta_min: 0.02,
            energy_theta_max: 0.20,
            ricci_epsilon_min: 0.01,
            ricci_epsilon_max: 0.30,
            min_universe_size: 100,
            max_deletion_rate: 0.10,  // max 10% per cycle
            toxicity_threshold_min: 0.70,
            toxicity_threshold_max: 0.95,
            sacred_vitality_min: 0.75,
            sacred_vitality_max: 0.95,
            sacred_causal_threshold: 3,
        }
    }
}

/// Adaptive configuration for the Nezhmetdinov Forgetting Engine.
///
/// All values are clamped within [`HardBounds`] on construction and mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NezhmetdinovConfig {
    /// The iron laws — immutable during Zaratustra cycles.
    pub bounds: HardBounds,

    /// Current adaptive vitality threshold θ.
    pub vitality_threshold: f32,
    /// Current adaptive energy threshold θ_e.
    pub energy_threshold: f32,
    /// Current Ricci collapse threshold ε (positive value, compared as ΔRicci < -ε).
    pub ricci_epsilon: f32,
    /// Current toxicity threshold.
    pub toxicity_threshold: f32,
    /// Current sacred vitality threshold.
    pub sacred_vitality_threshold: f32,

    /// Enable/disable the forgetting engine.
    pub enabled: bool,
    /// Maximum nodes to scan per tick (embedding loads are expensive).
    pub max_scan_per_tick: usize,
}

impl Default for NezhmetdinovConfig {
    fn default() -> Self {
        let bounds = HardBounds::default();
        Self {
            vitality_threshold: 0.25,
            energy_threshold: 0.10,
            ricci_epsilon: 0.15,
            toxicity_threshold: 0.80,
            sacred_vitality_threshold: 0.80,
            enabled: true,
            max_scan_per_tick: 5000,
            bounds,
        }
    }
}

impl NezhmetdinovConfig {
    /// Create with explicit bounds and initial thresholds, clamping everything.
    pub fn new(bounds: HardBounds, theta: f32, theta_e: f32, epsilon: f32) -> Self {
        let mut config = Self {
            bounds,
            vitality_threshold: theta,
            energy_threshold: theta_e,
            ricci_epsilon: epsilon,
            ..Self::default()
        };
        config.enforce_bounds();
        config
    }

    /// Enforce all hard bounds on adaptive parameters.
    /// Called after any mutation to prevent drift.
    pub fn enforce_bounds(&mut self) {
        self.vitality_threshold = self.vitality_threshold
            .clamp(self.bounds.vitality_theta_min, self.bounds.vitality_theta_max);
        self.energy_threshold = self.energy_threshold
            .clamp(self.bounds.energy_theta_min, self.bounds.energy_theta_max);
        self.ricci_epsilon = self.ricci_epsilon
            .clamp(self.bounds.ricci_epsilon_min, self.bounds.ricci_epsilon_max);
        self.toxicity_threshold = self.toxicity_threshold
            .clamp(self.bounds.toxicity_threshold_min, self.bounds.toxicity_threshold_max);
        self.sacred_vitality_threshold = self.sacred_vitality_threshold
            .clamp(self.bounds.sacred_vitality_min, self.bounds.sacred_vitality_max);
    }

    /// Attempt to adapt vitality threshold. Returns the clamped value actually set.
    pub fn adapt_vitality_threshold(&mut self, new_theta: f32) -> f32 {
        self.vitality_threshold = new_theta.clamp(
            self.bounds.vitality_theta_min,
            self.bounds.vitality_theta_max,
        );
        self.vitality_threshold
    }

    /// Attempt to adapt energy threshold. Returns the clamped value actually set.
    pub fn adapt_energy_threshold(&mut self, new_theta_e: f32) -> f32 {
        self.energy_threshold = new_theta_e.clamp(
            self.bounds.energy_theta_min,
            self.bounds.energy_theta_max,
        );
        self.energy_threshold
    }

    /// Attempt to adapt Ricci epsilon. Returns the clamped value actually set.
    pub fn adapt_ricci_epsilon(&mut self, new_epsilon: f32) -> f32 {
        self.ricci_epsilon = new_epsilon.clamp(
            self.bounds.ricci_epsilon_min,
            self.bounds.ricci_epsilon_max,
        );
        self.ricci_epsilon
    }

    /// Check if a deletion would violate the minimum universe size.
    pub fn would_violate_min_universe(&self, current_size: usize, deletions: usize) -> bool {
        current_size.saturating_sub(deletions) < self.bounds.min_universe_size
    }

    /// Check if a deletion count would exceed the maximum deletion rate.
    pub fn would_exceed_deletion_rate(&self, current_size: usize, deletions: usize) -> bool {
        if current_size == 0 { return true; }
        (deletions as f32 / current_size as f32) > self.bounds.max_deletion_rate
    }

    /// Build from environment variables with default bounds.
    pub fn from_env() -> Self {
        fn env_f32(key: &str, default: f32) -> f32 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_bool(key: &str, default: bool) -> bool {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }

        let mut config = Self {
            vitality_threshold: env_f32("NEZHMETDINOV_VITALITY_THETA", 0.25),
            energy_threshold: env_f32("NEZHMETDINOV_ENERGY_THETA", 0.10),
            ricci_epsilon: env_f32("NEZHMETDINOV_RICCI_EPSILON", 0.15),
            toxicity_threshold: env_f32("NEZHMETDINOV_TOXICITY_THRESHOLD", 0.80),
            sacred_vitality_threshold: env_f32("NEZHMETDINOV_SACRED_THRESHOLD", 0.80),
            enabled: env_bool("NEZHMETDINOV_ENABLED", true),
            max_scan_per_tick: env_usize("NEZHMETDINOV_MAX_SCAN", 5000),
            bounds: HardBounds::default(),
        };
        config.enforce_bounds();
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_within_bounds() {
        let config = NezhmetdinovConfig::default();
        assert!(config.vitality_threshold >= config.bounds.vitality_theta_min);
        assert!(config.vitality_threshold <= config.bounds.vitality_theta_max);
        assert!(config.energy_threshold >= config.bounds.energy_theta_min);
        assert!(config.energy_threshold <= config.bounds.energy_theta_max);
    }

    #[test]
    fn adaptation_is_clamped() {
        let mut config = NezhmetdinovConfig::default();

        // Try to set vitality threshold way too low
        let actual = config.adapt_vitality_threshold(0.001);
        assert_eq!(actual, config.bounds.vitality_theta_min);

        // Try to set it way too high
        let actual = config.adapt_vitality_threshold(0.999);
        assert_eq!(actual, config.bounds.vitality_theta_max);
    }

    #[test]
    fn min_universe_protection() {
        let config = NezhmetdinovConfig::default();
        assert!(config.would_violate_min_universe(150, 100));
        assert!(!config.would_violate_min_universe(1000, 100));
    }

    #[test]
    fn deletion_rate_protection() {
        let config = NezhmetdinovConfig::default();
        // 10% max rate, 1000 nodes
        assert!(!config.would_exceed_deletion_rate(1000, 99));
        assert!(config.would_exceed_deletion_rate(1000, 101));
    }

    #[test]
    fn enforce_bounds_clamps_all() {
        let mut config = NezhmetdinovConfig::default();
        config.vitality_threshold = 99.0;
        config.energy_threshold = -1.0;
        config.ricci_epsilon = 100.0;
        config.enforce_bounds();

        assert_eq!(config.vitality_threshold, config.bounds.vitality_theta_max);
        assert_eq!(config.energy_threshold, config.bounds.energy_theta_min);
        assert_eq!(config.ricci_epsilon, config.bounds.ricci_epsilon_max);
    }
}
