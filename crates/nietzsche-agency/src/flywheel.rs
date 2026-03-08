//! Phase XXIV — Cognitive Flywheel: unified autonomous pipeline.
//!
//! The Cognitive Flywheel connects all agency subsystems into a
//! self-reinforcing feedback loop:
//!
//! ```text
//! Observe → Learn → Compress → Heal → Grow → Observe...
//! ```
//!
//! Each tick of the flywheel:
//!
//! 1. **Observe**: World Model collects environmental signals
//! 2. **Learn**: Learning Engine identifies patterns and hotspots
//! 3. **Diagnose**: Thermodynamics + Health determine system state
//! 4. **Compress**: Knowledge Compression proposes merges
//! 5. **Heal**: Self-Healing repairs pathologies
//! 6. **Adapt**: ECAN + Gravity redistribute attention and energy
//! 7. **Grow**: L-System + Desire Engine fill knowledge gaps
//!
//! The Flywheel produces a `FlywheelReport` summarizing one complete
//! cognitive cycle and recommends which subsystems need tuning.
//!
//! ## Design
//!
//! Pure analysis module — reads outputs from all other subsystems
//! and produces a unified recommendation. No mutations.

use serde::{Deserialize, Serialize};

use crate::config::AgencyConfig;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Cognitive Flywheel.
#[derive(Debug, Clone)]
pub struct FlywheelConfig {
    /// Whether the flywheel is enabled.
    pub enabled: bool,
    /// Tick interval for flywheel analysis (default: 10).
    pub interval: u64,
    /// Momentum decay factor (0.0–1.0, default: 0.95).
    pub momentum_decay: f64,
    /// Minimum momentum to consider the system "spinning" (default: 0.3).
    pub min_momentum: f64,
}

impl Default for FlywheelConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 10,
            momentum_decay: 0.95,
            min_momentum: 0.3,
        }
    }
}

/// Build a `FlywheelConfig` from the global `AgencyConfig`.
pub fn build_flywheel_config(config: &AgencyConfig) -> FlywheelConfig {
    FlywheelConfig {
        enabled: config.flywheel_enabled,
        interval: config.flywheel_interval,
        momentum_decay: config.flywheel_momentum_decay,
        min_momentum: config.flywheel_min_momentum,
    }
}

// ─────────────────────────────────────────────
// Subsystem status
// ─────────────────────────────────────────────

/// Status of a subsystem in the flywheel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubsystemStatus {
    /// Operating normally.
    Healthy,
    /// Elevated activity, within bounds.
    Active,
    /// Needs attention — degraded performance.
    Degraded,
    /// Disabled or not producing data.
    Inactive,
}

impl std::fmt::Display for SubsystemStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Active => write!(f, "active"),
            Self::Degraded => write!(f, "degraded"),
            Self::Inactive => write!(f, "inactive"),
        }
    }
}

/// Input signals from all subsystems for flywheel analysis.
#[derive(Debug, Clone, Default)]
pub struct FlywheelInput {
    /// Was a health report produced?
    pub has_health: bool,
    /// Health: mean energy.
    pub mean_energy: f32,
    /// Health: gap count.
    pub gap_count: usize,
    /// Thermodynamics: temperature.
    pub temperature: Option<f64>,
    /// Thermodynamics: phase name.
    pub phase: Option<String>,
    /// ECAN: attention flow.
    pub attention_flow: Option<f32>,
    /// Hebbian: edges potentiated.
    pub hebbian_potentiated: Option<usize>,
    /// Gravity: mean force.
    pub gravity_mean_force: Option<f64>,
    /// Healing: total issues found.
    pub healing_issues: Option<usize>,
    /// Learning: access hotspot count.
    pub learning_hotspots: Option<usize>,
    /// Compression: estimated reduction.
    pub compression_reduction: Option<usize>,
    /// Sharding: imbalance ratio.
    pub sharding_imbalance: Option<f64>,
    /// World model: anomaly count.
    pub world_anomalies: Option<usize>,
    /// Tick duration (ms).
    pub tick_duration_ms: u64,
    /// Number of intents generated.
    pub intent_count: usize,
}

// ─────────────────────────────────────────────
// Flywheel state
// ─────────────────────────────────────────────

/// In-memory flywheel state tracking momentum and cycle count.
#[derive(Debug, Clone)]
pub struct FlywheelState {
    /// Current momentum (0.0–1.0) — how well the feedback loop is running.
    pub momentum: f64,
    /// Total cognitive cycles completed.
    pub cycles: u64,
    /// Consecutive healthy cycles.
    pub healthy_streak: u64,
    /// Last tick analyzed.
    pub last_tick: u64,
}

impl FlywheelState {
    pub fn new() -> Self {
        Self {
            momentum: 0.0,
            cycles: 0,
            healthy_streak: 0,
            last_tick: 0,
        }
    }
}

impl Default for FlywheelState {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Flywheel report
// ─────────────────────────────────────────────

/// Report from one cognitive flywheel cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlywheelReport {
    /// Current momentum (0.0–1.0).
    pub momentum: f64,
    /// Total cognitive cycles.
    pub cycles: u64,
    /// Consecutive healthy cycles.
    pub healthy_streak: u64,
    /// Overall system health assessment.
    pub overall_status: String,
    /// Per-subsystem status.
    pub subsystems: Vec<SubsystemEntry>,
    /// Recommendations for the operator.
    pub recommendations: Vec<String>,
    /// Is the flywheel "spinning" (momentum > min)?
    pub is_spinning: bool,
    /// Energy efficiency (useful work / tick duration).
    pub efficiency: f64,
}

/// Status entry for a subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsystemEntry {
    pub name: String,
    pub status: SubsystemStatus,
    pub detail: String,
}

// ─────────────────────────────────────────────
// Analysis
// ─────────────────────────────────────────────

/// Run the flywheel analysis — assess system health and generate recommendations.
pub fn run_flywheel_cycle(
    state: &mut FlywheelState,
    input: &FlywheelInput,
    config: &FlywheelConfig,
    tick: u64,
) -> FlywheelReport {
    state.cycles += 1;
    state.last_tick = tick;

    let mut subsystems = Vec::new();
    let mut recommendations = Vec::new();
    let mut health_score = 0.0f64;
    let mut total_checks = 0usize;

    // ── Assess each subsystem ────────────────

    // Health
    if input.has_health {
        let status = if input.mean_energy > 0.3 && input.gap_count < 5 {
            SubsystemStatus::Healthy
        } else if input.gap_count > 20 {
            SubsystemStatus::Degraded
        } else {
            SubsystemStatus::Active
        };
        subsystems.push(SubsystemEntry {
            name: "health".into(),
            status,
            detail: format!("energy={:.2}, gaps={}", input.mean_energy, input.gap_count),
        });
        health_score += if status == SubsystemStatus::Healthy { 1.0 } else { 0.5 };
        total_checks += 1;

        if input.gap_count > 10 {
            recommendations.push("Consider triggering L-System growth to fill knowledge gaps".into());
        }
    }

    // Thermodynamics
    match (&input.temperature, &input.phase) {
        (Some(t), Some(phase)) => {
            let status = if *t > 0.85 {
                recommendations.push("Temperature critical — system in gas phase, reduce mutations".into());
                SubsystemStatus::Degraded
            } else if *t < 0.15 {
                recommendations.push("Temperature too low — system frozen, consider energy injection".into());
                SubsystemStatus::Degraded
            } else {
                SubsystemStatus::Healthy
            };
            subsystems.push(SubsystemEntry {
                name: "thermodynamics".into(),
                status,
                detail: format!("T={:.2}, phase={}", t, phase),
            });
            health_score += if status == SubsystemStatus::Healthy { 1.0 } else { 0.3 };
            total_checks += 1;
        }
        _ => {
            subsystems.push(SubsystemEntry {
                name: "thermodynamics".into(),
                status: SubsystemStatus::Inactive,
                detail: "no data".into(),
            });
        }
    }

    // ECAN
    if let Some(flow) = input.attention_flow {
        let status = if flow > 0.0 { SubsystemStatus::Active } else { SubsystemStatus::Inactive };
        subsystems.push(SubsystemEntry {
            name: "attention".into(),
            status,
            detail: format!("flow={:.2}", flow),
        });
        health_score += if flow > 0.0 { 0.8 } else { 0.2 };
        total_checks += 1;
    }

    // Hebbian
    if let Some(pot) = input.hebbian_potentiated {
        subsystems.push(SubsystemEntry {
            name: "hebbian".into(),
            status: if pot > 0 { SubsystemStatus::Active } else { SubsystemStatus::Healthy },
            detail: format!("potentiated={}", pot),
        });
        health_score += 0.8;
        total_checks += 1;
    }

    // Healing
    if let Some(issues) = input.healing_issues {
        let status = if issues == 0 {
            SubsystemStatus::Healthy
        } else if issues > 50 {
            recommendations.push(format!("Self-healing found {} issues — graph may need maintenance", issues));
            SubsystemStatus::Degraded
        } else {
            SubsystemStatus::Active
        };
        subsystems.push(SubsystemEntry {
            name: "healing".into(),
            status,
            detail: format!("issues={}", issues),
        });
        health_score += if status == SubsystemStatus::Healthy { 1.0 } else { 0.5 };
        total_checks += 1;
    }

    // Learning
    if let Some(hotspots) = input.learning_hotspots {
        subsystems.push(SubsystemEntry {
            name: "learning".into(),
            status: SubsystemStatus::Active,
            detail: format!("hotspots={}", hotspots),
        });
        health_score += 0.8;
        total_checks += 1;
    }

    // Compression
    if let Some(reduction) = input.compression_reduction {
        if reduction > 100 {
            recommendations.push(format!("Knowledge compression can reduce {} nodes — consider running merges", reduction));
        }
        subsystems.push(SubsystemEntry {
            name: "compression".into(),
            status: SubsystemStatus::Active,
            detail: format!("reduction={}", reduction),
        });
        health_score += 0.8;
        total_checks += 1;
    }

    // Sharding
    if let Some(imbalance) = input.sharding_imbalance {
        let status = if imbalance > 5.0 {
            recommendations.push("Severe shard imbalance — some regions are overloaded".into());
            SubsystemStatus::Degraded
        } else if imbalance > 3.0 {
            SubsystemStatus::Active
        } else {
            SubsystemStatus::Healthy
        };
        subsystems.push(SubsystemEntry {
            name: "sharding".into(),
            status,
            detail: format!("imbalance={:.1}x", imbalance),
        });
        health_score += if status == SubsystemStatus::Healthy { 1.0 } else { 0.5 };
        total_checks += 1;
    }

    // World model
    if let Some(anomalies) = input.world_anomalies {
        if anomalies > 0 {
            recommendations.push(format!("{} environmental anomalies detected — investigate access patterns", anomalies));
        }
        subsystems.push(SubsystemEntry {
            name: "world_model".into(),
            status: if anomalies == 0 { SubsystemStatus::Healthy } else { SubsystemStatus::Active },
            detail: format!("anomalies={}", anomalies),
        });
        health_score += if anomalies == 0 { 1.0 } else { 0.6 };
        total_checks += 1;
    }

    // ── Compute momentum ────────────────────

    let normalized_health = if total_checks > 0 {
        health_score / total_checks as f64
    } else {
        0.5
    };

    // Momentum = exponential moving average of health
    state.momentum = state.momentum * config.momentum_decay
        + normalized_health * (1.0 - config.momentum_decay);

    let is_spinning = state.momentum >= config.min_momentum;

    // Track healthy streak
    let all_healthy = subsystems.iter()
        .all(|s| s.status == SubsystemStatus::Healthy || s.status == SubsystemStatus::Active || s.status == SubsystemStatus::Inactive);
    if all_healthy {
        state.healthy_streak += 1;
    } else {
        state.healthy_streak = 0;
    }

    // Overall status
    let overall_status = if state.momentum > 0.8 {
        "excellent"
    } else if state.momentum > 0.5 {
        "good"
    } else if state.momentum > 0.3 {
        "fair"
    } else {
        "poor"
    }.to_string();

    // Efficiency = intents per ms of tick duration
    let efficiency = if input.tick_duration_ms > 0 {
        input.intent_count as f64 / input.tick_duration_ms as f64
    } else {
        0.0
    };

    // Performance recommendation
    if input.tick_duration_ms > 5000 {
        recommendations.push(format!(
            "Tick duration {}ms is too high — consider reducing max_scan values",
            input.tick_duration_ms,
        ));
    }

    FlywheelReport {
        momentum: state.momentum,
        cycles: state.cycles,
        healthy_streak: state.healthy_streak,
        overall_status,
        subsystems,
        recommendations,
        is_spinning,
        efficiency,
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = FlywheelConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.interval, 10);
        assert!((cfg.momentum_decay - 0.95).abs() < 1e-6);
    }

    #[test]
    fn healthy_system_builds_momentum() {
        let config = FlywheelConfig::default();
        let mut state = FlywheelState::new();

        for tick in 0..20 {
            let input = FlywheelInput {
                has_health: true,
                mean_energy: 0.5,
                gap_count: 2,
                temperature: Some(0.5),
                phase: Some("Liquid".into()),
                attention_flow: Some(1.0),
                ..Default::default()
            };
            run_flywheel_cycle(&mut state, &input, &config, tick);
        }

        assert!(state.momentum > 0.5);
        assert!(state.healthy_streak > 0);
    }

    #[test]
    fn degraded_system_reduces_momentum() {
        let config = FlywheelConfig::default();
        let mut state = FlywheelState::new();
        state.momentum = 0.8;

        let input = FlywheelInput {
            has_health: true,
            mean_energy: 0.1,
            gap_count: 50,
            temperature: Some(0.95),
            phase: Some("Gas".into()),
            healing_issues: Some(100),
            ..Default::default()
        };

        let report = run_flywheel_cycle(&mut state, &input, &config, 1);
        assert!(!report.recommendations.is_empty());
        assert!(report.overall_status != "excellent");
    }

    #[test]
    fn report_serializes() {
        let report = FlywheelReport {
            momentum: 0.75,
            cycles: 100,
            healthy_streak: 42,
            overall_status: "good".into(),
            subsystems: vec![SubsystemEntry {
                name: "health".into(),
                status: SubsystemStatus::Healthy,
                detail: "energy=0.5".into(),
            }],
            recommendations: vec!["all good".into()],
            is_spinning: true,
            efficiency: 0.5,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"momentum\":0.75"));
        assert!(json.contains("\"is_spinning\":true"));
    }

    #[test]
    fn spinning_detection() {
        let config = FlywheelConfig {
            min_momentum: 0.3,
            ..Default::default()
        };
        let mut state = FlywheelState::new();
        state.momentum = 0.5;

        let input = FlywheelInput {
            has_health: true,
            mean_energy: 0.5,
            gap_count: 0,
            ..Default::default()
        };

        let report = run_flywheel_cycle(&mut state, &input, &config, 1);
        assert!(report.is_spinning);
    }
}
