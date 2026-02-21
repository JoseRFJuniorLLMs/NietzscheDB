//! Configuration for the Zaratustra engine.

use serde::{Deserialize, Serialize};

/// Configuration for a single Zaratustra cycle.
///
/// A *cycle* runs three sequential phases:
/// 1. **Will to Power** — energy propagation
/// 2. **Eternal Recurrence** — echo/snapshot creation
/// 3. **Übermensch** — statistics on the elite tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZaratustraConfig {
    // ── Will to Power ─────────────────────────────────────────────────────────

    /// Propagation coefficient α ∈ (0, 1].
    ///
    /// Each node receives α × (mean energy of its out-neighbours) added to its
    /// own energy.  Higher α → faster propagation, risk of runaway inflation.
    /// Default: `0.1`
    pub alpha: f32,

    /// Temporal decay coefficient δ ∈ [0, 1).
    ///
    /// After propagation, every node's energy is multiplied by `(1 − δ)`.
    /// This keeps the system bounded and drains stagnant nodes.
    /// Default: `0.02`
    pub decay: f32,

    /// Cap applied after propagation to prevent energy from exceeding 1.0.
    /// Default: `1.0`
    pub energy_cap: f32,

    // ── Eternal Recurrence ────────────────────────────────────────────────────

    /// Minimum energy level for a node to qualify for an echo (snapshot).
    ///
    /// Only nodes whose energy is ≥ this value after propagation are snapshotted.
    /// Default: `0.7`
    pub echo_threshold: f32,

    /// Maximum number of echoes stored per node in the JSON content field.
    ///
    /// When the limit is reached the oldest echo is evicted (ring-buffer semantics).
    /// Default: `5`
    pub max_echoes_per_node: usize,

    // ── Übermensch ────────────────────────────────────────────────────────────

    /// Fraction of nodes (by energy rank) considered "elite" / Übermensch tier.
    ///
    /// Used only for reporting — no data is moved between tiers yet.
    /// Default: `0.1` (top 10 %)
    pub ubermensch_top_fraction: f32,

    // ── Iteration ─────────────────────────────────────────────────────────────

    /// How many Will-to-Power propagation steps to run per cycle.
    /// Default: `3`
    pub propagation_steps: usize,
}

impl Default for ZaratustraConfig {
    fn default() -> Self {
        Self {
            alpha:                  0.10,
            decay:                  0.02,
            energy_cap:             1.0,
            echo_threshold:         0.70,
            max_echoes_per_node:    5,
            ubermensch_top_fraction: 0.10,
            propagation_steps:      3,
        }
    }
}

impl ZaratustraConfig {
    /// Load from environment variables, falling back to defaults.
    ///
    /// | Variable                           | Default |
    /// |------------------------------------|---------|
    /// | `ZARATUSTRA_ALPHA`                 | `0.10`  |
    /// | `ZARATUSTRA_DECAY`                 | `0.02`  |
    /// | `ZARATUSTRA_ECHO_THRESHOLD`        | `0.70`  |
    /// | `ZARATUSTRA_MAX_ECHOES`            | `5`     |
    /// | `ZARATUSTRA_UBERMENSCH_FRACTION`   | `0.10`  |
    /// | `ZARATUSTRA_PROPAGATION_STEPS`     | `3`     |
    pub fn from_env() -> Self {
        fn env_f32(key: &str, default: f32) -> f32 {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }
        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        }

        Self {
            alpha:                  env_f32("ZARATUSTRA_ALPHA",              0.10),
            decay:                  env_f32("ZARATUSTRA_DECAY",              0.02),
            energy_cap:             1.0,
            echo_threshold:         env_f32("ZARATUSTRA_ECHO_THRESHOLD",     0.70),
            max_echoes_per_node:    env_usize("ZARATUSTRA_MAX_ECHOES",       5),
            ubermensch_top_fraction: env_f32("ZARATUSTRA_UBERMENSCH_FRACTION", 0.10),
            propagation_steps:      env_usize("ZARATUSTRA_PROPAGATION_STEPS", 3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_alpha() {
        let cfg = ZaratustraConfig::default();
        assert!((cfg.alpha - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn default_decay() {
        let cfg = ZaratustraConfig::default();
        assert!((cfg.decay - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn default_energy_cap() {
        let cfg = ZaratustraConfig::default();
        assert!((cfg.energy_cap - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn default_echo_threshold() {
        let cfg = ZaratustraConfig::default();
        assert!((cfg.echo_threshold - 0.70).abs() < f32::EPSILON);
    }

    #[test]
    fn default_max_echoes_per_node() {
        let cfg = ZaratustraConfig::default();
        assert_eq!(cfg.max_echoes_per_node, 5);
    }

    #[test]
    fn default_ubermensch_top_fraction() {
        let cfg = ZaratustraConfig::default();
        assert!((cfg.ubermensch_top_fraction - 0.10).abs() < f32::EPSILON);
    }

    #[test]
    fn default_propagation_steps() {
        let cfg = ZaratustraConfig::default();
        assert_eq!(cfg.propagation_steps, 3);
    }

    /// All env-var tests are combined into a single test function to avoid
    /// race conditions from parallel test execution (env vars are process-global).
    #[test]
    fn from_env_all_scenarios() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        // ── Scenario 1: defaults when nothing is set ─────────────────────────
        std::env::remove_var("ZARATUSTRA_ALPHA");
        std::env::remove_var("ZARATUSTRA_DECAY");
        std::env::remove_var("ZARATUSTRA_ECHO_THRESHOLD");
        std::env::remove_var("ZARATUSTRA_MAX_ECHOES");
        std::env::remove_var("ZARATUSTRA_UBERMENSCH_FRACTION");
        std::env::remove_var("ZARATUSTRA_PROPAGATION_STEPS");

        let cfg = ZaratustraConfig::from_env();
        let def = ZaratustraConfig::default();

        assert!((cfg.alpha - def.alpha).abs() < f32::EPSILON);
        assert!((cfg.decay - def.decay).abs() < f32::EPSILON);
        assert!((cfg.echo_threshold - def.echo_threshold).abs() < f32::EPSILON);
        assert_eq!(cfg.max_echoes_per_node, def.max_echoes_per_node);
        assert!((cfg.ubermensch_top_fraction - def.ubermensch_top_fraction).abs() < f32::EPSILON);
        assert_eq!(cfg.propagation_steps, def.propagation_steps);
        // energy_cap is always 1.0 (not configurable via env)
        assert!((cfg.energy_cap - 1.0).abs() < f32::EPSILON);

        // ── Scenario 2: custom values ────────────────────────────────────────
        std::env::set_var("ZARATUSTRA_ALPHA", "0.55");
        std::env::set_var("ZARATUSTRA_DECAY", "0.08");
        std::env::set_var("ZARATUSTRA_ECHO_THRESHOLD", "0.50");
        std::env::set_var("ZARATUSTRA_MAX_ECHOES", "10");
        std::env::set_var("ZARATUSTRA_UBERMENSCH_FRACTION", "0.25");
        std::env::set_var("ZARATUSTRA_PROPAGATION_STEPS", "7");

        let cfg = ZaratustraConfig::from_env();
        assert!((cfg.alpha - 0.55).abs() < 1e-5);
        assert!((cfg.decay - 0.08).abs() < 1e-5);
        assert!((cfg.echo_threshold - 0.50).abs() < 1e-5);
        assert_eq!(cfg.max_echoes_per_node, 10);
        assert!((cfg.ubermensch_top_fraction - 0.25).abs() < 1e-5);
        assert_eq!(cfg.propagation_steps, 7);

        // ── Scenario 3: invalid values fall back to defaults ─────────────────
        std::env::set_var("ZARATUSTRA_ALPHA", "not_a_float");
        std::env::set_var("ZARATUSTRA_MAX_ECHOES", "abc");

        let cfg = ZaratustraConfig::from_env();
        assert!((cfg.alpha - 0.10).abs() < f32::EPSILON, "invalid float should fall back to default");
        assert_eq!(cfg.max_echoes_per_node, 5, "invalid usize should fall back to default");

        // ── Cleanup ──────────────────────────────────────────────────────────
        std::env::remove_var("ZARATUSTRA_ALPHA");
        std::env::remove_var("ZARATUSTRA_DECAY");
        std::env::remove_var("ZARATUSTRA_ECHO_THRESHOLD");
        std::env::remove_var("ZARATUSTRA_MAX_ECHOES");
        std::env::remove_var("ZARATUSTRA_UBERMENSCH_FRACTION");
        std::env::remove_var("ZARATUSTRA_PROPAGATION_STEPS");
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = ZaratustraConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let parsed: ZaratustraConfig = serde_json::from_str(&json).expect("deserialize");
        assert!((parsed.alpha - cfg.alpha).abs() < f32::EPSILON);
        assert_eq!(parsed.propagation_steps, cfg.propagation_steps);
    }
}
