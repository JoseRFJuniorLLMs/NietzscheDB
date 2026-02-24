//! Anti-Gaming Normalization — Goodhart's Law Protection.
//!
//! Prevents the Forgetting Engine from gaming its own metrics:
//! - Creating artificial voids to inflate TGC
//! - Deleting nodes just to boost mean vitality
//! - Oscillating thresholds to generate "activity" metrics
//!
//! The key insight: if TGC is suspiciously stable and high while
//! the system is actively deleting, something is wrong.

use serde::{Deserialize, Serialize};

/// Anti-gaming detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiGamingReport {
    /// Cycle number.
    pub cycle: u64,
    /// Whether gaming is suspected.
    pub gaming_suspected: bool,
    /// Specific violations detected.
    pub violations: Vec<GamingViolation>,
    /// Normalized TGC (after anti-gaming adjustment).
    pub normalized_tgc: f32,
    /// Raw TGC before normalization.
    pub raw_tgc: f32,
}

/// Types of gaming violations detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GamingViolation {
    /// TGC suspiciously stable (variance near zero) over many cycles.
    SuspiciousStability {
        tgc_variance: f32,
        window_size: usize,
    },
    /// Deletion rate unnaturally constant (no adaptation happening).
    ConstantDeletionRate {
        rate_variance: f32,
        mean_rate: f32,
    },
    /// Void generation exactly matches deletion count (too perfect).
    PerfectVoidBalance {
        deletions: usize,
        generations: usize,
    },
    /// Threshold oscillating rapidly (gaming the bounds).
    ThresholdOscillation {
        threshold_changes: usize,
        window: usize,
    },
    /// Mean vitality artificially inflated by selective deletion.
    ArtificialVitalityInflation {
        vitality_before_deletion: f32,
        vitality_after_deletion: f32,
        ratio: f32,
    },
}

/// Anti-Gaming monitor.
#[derive(Debug, Clone)]
pub struct AntiGamingMonitor {
    /// History of TGC values for variance analysis.
    tgc_history: Vec<f32>,
    /// History of deletion rates.
    deletion_rate_history: Vec<f32>,
    /// History of threshold values (for oscillation detection).
    threshold_history: Vec<f32>,
    /// Minimum TGC variance before suspicion (below = too stable).
    pub min_tgc_variance: f32,
    /// Maximum "perfect balance" tolerance.
    pub balance_tolerance: f32,
    /// Window size for rolling analysis.
    pub analysis_window: usize,
}

impl Default for AntiGamingMonitor {
    fn default() -> Self {
        Self {
            tgc_history: Vec::new(),
            deletion_rate_history: Vec::new(),
            threshold_history: Vec::new(),
            min_tgc_variance: 0.001,
            balance_tolerance: 0.05,
            analysis_window: 20,
        }
    }
}

impl AntiGamingMonitor {
    pub fn new(min_variance: f32, window: usize) -> Self {
        Self {
            min_tgc_variance: min_variance,
            analysis_window: window,
            ..Self::default()
        }
    }

    /// Analyze the current cycle for gaming behavior.
    ///
    /// Returns an AntiGamingReport with any detected violations.
    pub fn analyze(
        &mut self,
        cycle: u64,
        raw_tgc: f32,
        deletion_count: usize,
        total_nodes: usize,
        generation_count: usize,
        current_threshold: f32,
        vitality_before: f32,
        vitality_after: f32,
    ) -> AntiGamingReport {
        // Record history
        self.tgc_history.push(raw_tgc);
        let deletion_rate = if total_nodes > 0 {
            deletion_count as f32 / total_nodes as f32
        } else {
            0.0
        };
        self.deletion_rate_history.push(deletion_rate);
        self.threshold_history.push(current_threshold);

        let mut violations = Vec::new();

        // Check 1: Suspicious TGC stability
        if self.tgc_history.len() >= self.analysis_window {
            let window = &self.tgc_history[self.tgc_history.len() - self.analysis_window..];
            let var = variance(window);
            if var < self.min_tgc_variance && raw_tgc > 0.1 {
                violations.push(GamingViolation::SuspiciousStability {
                    tgc_variance: var,
                    window_size: self.analysis_window,
                });
            }
        }

        // Check 2: Constant deletion rate
        if self.deletion_rate_history.len() >= self.analysis_window {
            let window = &self.deletion_rate_history[self.deletion_rate_history.len() - self.analysis_window..];
            let var = variance(window);
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            if var < 0.0001 && mean > 0.01 {
                violations.push(GamingViolation::ConstantDeletionRate {
                    rate_variance: var,
                    mean_rate: mean,
                });
            }
        }

        // Check 3: Perfect void-deletion balance
        if deletion_count > 0 && generation_count > 0 {
            let ratio = generation_count as f32 / deletion_count as f32;
            if (ratio - 1.0).abs() < self.balance_tolerance {
                violations.push(GamingViolation::PerfectVoidBalance {
                    deletions: deletion_count,
                    generations: generation_count,
                });
            }
        }

        // Check 4: Threshold oscillation
        if self.threshold_history.len() >= 10 {
            let window = &self.threshold_history[self.threshold_history.len() - 10..];
            let changes = window.windows(2)
                .filter(|w| (w[0] - w[1]).abs() > 0.01)
                .count();
            if changes >= 7 { // 70%+ of recent cycles had threshold changes
                violations.push(GamingViolation::ThresholdOscillation {
                    threshold_changes: changes,
                    window: 10,
                });
            }
        }

        // Check 5: Artificial vitality inflation
        if deletion_count > 0 && vitality_before > 0.0 {
            let ratio = vitality_after / vitality_before;
            // If vitality jumped more than 20% in a single cycle via deletion alone
            if ratio > 1.20 {
                violations.push(GamingViolation::ArtificialVitalityInflation {
                    vitality_before_deletion: vitality_before,
                    vitality_after_deletion: vitality_after,
                    ratio,
                });
            }
        }

        let gaming_suspected = !violations.is_empty();

        // Normalize TGC: penalize if gaming is suspected
        let penalty = if gaming_suspected {
            0.5_f32.powi(violations.len() as i32) // 50% penalty per violation
        } else {
            1.0
        };
        let normalized_tgc = raw_tgc * penalty;

        AntiGamingReport {
            cycle,
            gaming_suspected,
            violations,
            normalized_tgc,
            raw_tgc,
        }
    }

    /// Reset the monitor (e.g., after admin intervention).
    pub fn reset(&mut self) {
        self.tgc_history.clear();
        self.deletion_rate_history.clear();
        self.threshold_history.clear();
    }
}

/// Compute variance of a slice.
fn variance(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_gaming_detected_normal_operation() {
        let mut monitor = AntiGamingMonitor::default();
        let report = monitor.analyze(1, 0.3, 10, 1000, 3, 0.25, 0.5, 0.52);
        assert!(!report.gaming_suspected);
        assert_eq!(report.normalized_tgc, report.raw_tgc);
    }

    #[test]
    fn detects_suspicious_stability() {
        let mut monitor = AntiGamingMonitor::new(0.001, 10);
        // Feed 20 cycles with identical TGC
        for i in 0..20 {
            let report = monitor.analyze(i, 0.5, 5, 100, 2, 0.25, 0.5, 0.52);
            if i >= 9 {
                assert!(
                    report.violations.iter().any(|v| matches!(v, GamingViolation::SuspiciousStability { .. })),
                    "should detect suspicious stability at cycle {}", i
                );
            }
        }
    }

    #[test]
    fn detects_perfect_balance() {
        let mut monitor = AntiGamingMonitor::default();
        let report = monitor.analyze(1, 0.3, 10, 1000, 10, 0.25, 0.5, 0.52);
        assert!(report.violations.iter().any(|v| matches!(v, GamingViolation::PerfectVoidBalance { .. })));
    }

    #[test]
    fn detects_vitality_inflation() {
        let mut monitor = AntiGamingMonitor::default();
        // Before=0.5, After=0.8 → 60% jump
        let report = monitor.analyze(1, 0.3, 50, 1000, 5, 0.25, 0.5, 0.8);
        assert!(report.violations.iter().any(|v| matches!(v, GamingViolation::ArtificialVitalityInflation { .. })));
    }

    #[test]
    fn penalty_reduces_tgc() {
        let mut monitor = AntiGamingMonitor::default();
        let report = monitor.analyze(1, 0.5, 10, 1000, 10, 0.25, 0.4, 0.7);
        if report.gaming_suspected {
            assert!(report.normalized_tgc < report.raw_tgc);
        }
    }
}
