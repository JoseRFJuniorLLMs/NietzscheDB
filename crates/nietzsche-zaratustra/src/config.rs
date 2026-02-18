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
