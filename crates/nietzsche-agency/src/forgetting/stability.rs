//! Stability Monitor — Detection of three pathological attractors.
//!
//! The Forgetting Engine must avoid three collapse modes:
//!
//! 1. **Elitist Collapse**: Everything becomes elite, nothing new emerges.
//!    Detected by: mean vitality > 0.9 AND vitality variance < 0.01.
//!
//! 2. **Minimalist Collapse**: Destruction outpaces regeneration.
//!    Detected by: |S_t| < MIN_UNIVERSE_SIZE.
//!
//! 3. **Stationary Collapse**: The system stops forgetting and stagnates.
//!    Detected by: deletion rate ≈ 0 for N consecutive cycles.
//!
//! ## Defense Mechanism
//!
//! When a collapse is detected, the system injects controlled chaos
//! (thermal perturbation) to break out of the attractor.

use serde::{Deserialize, Serialize};

/// Types of pathological collapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CollapseType {
    /// Everything becomes elite, cognitive monoculture.
    Elitist,
    /// Destruction exceeds regeneration, universe shrinks below minimum.
    Minimalist,
    /// System stops forgetting, stagnation.
    Stationary,
}

impl CollapseType {
    pub fn label(&self) -> &'static str {
        match self {
            CollapseType::Elitist => "ELITIST_COLLAPSE",
            CollapseType::Minimalist => "MINIMALIST_COLLAPSE",
            CollapseType::Stationary => "STATIONARY_COLLAPSE",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            CollapseType::Elitist => "Mean vitality too high with low variance — cognitive monoculture",
            CollapseType::Minimalist => "Universe size below minimum — destruction exceeds regeneration",
            CollapseType::Stationary => "No deletions for consecutive cycles — system stagnated",
        }
    }
}

/// Collapse detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollapseAlert {
    pub collapse_type: CollapseType,
    pub cycle: u64,
    pub severity: f32, // 0.0 to 1.0
    pub details: String,
    /// Recommended thermal perturbation magnitude.
    pub recommended_perturbation: f32,
}

/// Thermal perturbation parameters for breaking out of collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPerturbation {
    /// Magnitude of energy noise to inject (0.0 to 1.0).
    pub energy_noise: f32,
    /// Fraction of nodes to perturb (0.0 to 1.0).
    pub perturbation_rate: f32,
    /// Whether to temporarily lower the vitality threshold.
    pub lower_threshold: bool,
    /// Amount to lower threshold by (if lower_threshold is true).
    pub threshold_reduction: f32,
}

impl ThermalPerturbation {
    /// Create perturbation for elitist collapse: increase chaos.
    pub fn for_elitist() -> Self {
        Self {
            energy_noise: 0.3,
            perturbation_rate: 0.1,
            lower_threshold: true,
            threshold_reduction: 0.05,
        }
    }

    /// Create perturbation for minimalist collapse: reduce destruction.
    pub fn for_minimalist() -> Self {
        Self {
            energy_noise: 0.1,
            perturbation_rate: 0.0, // Don't add more chaos
            lower_threshold: false,
            threshold_reduction: 0.0,
        }
    }

    /// Create perturbation for stationary collapse: force activity.
    pub fn for_stationary() -> Self {
        Self {
            energy_noise: 0.2,
            perturbation_rate: 0.05,
            lower_threshold: true,
            threshold_reduction: 0.1, // Aggressive threshold reduction
        }
    }
}

/// Configuration for stability monitoring.
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Mean vitality above which elitist collapse is suspected.
    pub elitist_vitality_threshold: f32,
    /// Vitality variance below which elitist collapse is suspected.
    pub elitist_variance_threshold: f32,
    /// Minimum universe size (minimalist collapse).
    pub min_universe_size: usize,
    /// Consecutive zero-deletion cycles before stationary collapse.
    pub stationary_cycle_limit: usize,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            elitist_vitality_threshold: 0.90,
            elitist_variance_threshold: 0.01,
            min_universe_size: 100,
            stationary_cycle_limit: 10,
        }
    }
}

/// Stability monitor that tracks system health across cycles.
#[derive(Debug, Clone)]
pub struct StabilityMonitor {
    pub config: StabilityConfig,
    /// Count of consecutive cycles with zero deletions.
    zero_deletion_streak: usize,
    /// History of alerts.
    pub alerts: Vec<CollapseAlert>,
}

impl StabilityMonitor {
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            zero_deletion_streak: 0,
            alerts: Vec::new(),
        }
    }

    /// Check all three collapse conditions.
    ///
    /// Returns a list of active collapse alerts (empty if healthy).
    pub fn check(
        &mut self,
        cycle: u64,
        total_nodes: usize,
        deletions_this_cycle: usize,
        mean_vitality: f32,
        vitality_variance: f32,
    ) -> Vec<CollapseAlert> {
        let mut alerts = Vec::new();

        // 1. Elitist Collapse
        if mean_vitality > self.config.elitist_vitality_threshold
            && vitality_variance < self.config.elitist_variance_threshold
        {
            let severity = (mean_vitality - self.config.elitist_vitality_threshold) * 10.0;
            alerts.push(CollapseAlert {
                collapse_type: CollapseType::Elitist,
                cycle,
                severity: severity.clamp(0.0, 1.0),
                details: format!(
                    "V_mean={:.3} > {:.3}, Var(V)={:.4} < {:.4}",
                    mean_vitality, self.config.elitist_vitality_threshold,
                    vitality_variance, self.config.elitist_variance_threshold,
                ),
                recommended_perturbation: 0.3,
            });
        }

        // 2. Minimalist Collapse
        if total_nodes < self.config.min_universe_size {
            let severity = 1.0 - (total_nodes as f32 / self.config.min_universe_size as f32);
            alerts.push(CollapseAlert {
                collapse_type: CollapseType::Minimalist,
                cycle,
                severity: severity.clamp(0.0, 1.0),
                details: format!(
                    "|S_t|={} < MIN={}",
                    total_nodes, self.config.min_universe_size,
                ),
                recommended_perturbation: 0.0, // Don't perturb, just stop deleting
            });
        }

        // 3. Stationary Collapse
        if deletions_this_cycle == 0 {
            self.zero_deletion_streak += 1;
        } else {
            self.zero_deletion_streak = 0;
        }

        if self.zero_deletion_streak >= self.config.stationary_cycle_limit {
            let severity = (self.zero_deletion_streak as f32 / self.config.stationary_cycle_limit as f32)
                .min(1.0);
            alerts.push(CollapseAlert {
                collapse_type: CollapseType::Stationary,
                cycle,
                severity,
                details: format!(
                    "{} consecutive cycles with 0 deletions (limit={})",
                    self.zero_deletion_streak, self.config.stationary_cycle_limit,
                ),
                recommended_perturbation: 0.2,
            });
        }

        self.alerts.extend(alerts.clone());
        alerts
    }

    /// Generate a thermal perturbation response for a collapse alert.
    pub fn generate_perturbation(alert: &CollapseAlert) -> ThermalPerturbation {
        match alert.collapse_type {
            CollapseType::Elitist => ThermalPerturbation::for_elitist(),
            CollapseType::Minimalist => ThermalPerturbation::for_minimalist(),
            CollapseType::Stationary => ThermalPerturbation::for_stationary(),
        }
    }

    /// Check if any collapse is currently active.
    pub fn is_collapsing(&self) -> bool {
        !self.alerts.is_empty() && self.alerts.last()
            .map(|a| a.severity > 0.5)
            .unwrap_or(false)
    }

    /// Reset the zero-deletion streak (e.g., after perturbation).
    pub fn reset_stationary_counter(&mut self) {
        self.zero_deletion_streak = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_collapse_healthy_system() {
        let mut monitor = StabilityMonitor::new(StabilityConfig::default());
        let alerts = monitor.check(1, 5000, 100, 0.65, 0.05);
        assert!(alerts.is_empty());
    }

    #[test]
    fn detects_elitist_collapse() {
        let mut monitor = StabilityMonitor::new(StabilityConfig::default());
        let alerts = monitor.check(1, 5000, 10, 0.95, 0.005);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].collapse_type, CollapseType::Elitist);
    }

    #[test]
    fn detects_minimalist_collapse() {
        let mut monitor = StabilityMonitor::new(StabilityConfig::default());
        let alerts = monitor.check(1, 50, 10, 0.5, 0.05);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].collapse_type, CollapseType::Minimalist);
    }

    #[test]
    fn detects_stationary_collapse() {
        let config = StabilityConfig {
            stationary_cycle_limit: 5,
            ..Default::default()
        };
        let mut monitor = StabilityMonitor::new(config);

        // 4 cycles with no deletions: no alert yet
        for i in 0..4 {
            let alerts = monitor.check(i as u64, 5000, 0, 0.5, 0.05);
            assert!(alerts.is_empty(), "should not alert at cycle {}", i);
        }

        // 5th cycle: stationary collapse detected
        let alerts = monitor.check(5, 5000, 0, 0.5, 0.05);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].collapse_type, CollapseType::Stationary);
    }

    #[test]
    fn deletion_resets_stationary_counter() {
        let config = StabilityConfig {
            stationary_cycle_limit: 3,
            ..Default::default()
        };
        let mut monitor = StabilityMonitor::new(config);

        // 2 cycles no deletion
        monitor.check(1, 5000, 0, 0.5, 0.05);
        monitor.check(2, 5000, 0, 0.5, 0.05);

        // 1 deletion resets
        monitor.check(3, 5000, 1, 0.5, 0.05);

        // 2 more no deletion: still no alert (counter was reset)
        let alerts = monitor.check(5, 5000, 0, 0.5, 0.05);
        assert!(alerts.is_empty());
    }

    #[test]
    fn perturbation_types() {
        let elitist = ThermalPerturbation::for_elitist();
        assert!(elitist.lower_threshold);
        assert!(elitist.energy_noise > 0.0);

        let minimalist = ThermalPerturbation::for_minimalist();
        assert!(!minimalist.lower_threshold);
        assert_eq!(minimalist.perturbation_rate, 0.0);

        let stationary = ThermalPerturbation::for_stationary();
        assert!(stationary.lower_threshold);
        assert!(stationary.threshold_reduction > elitist.threshold_reduction);
    }
}
