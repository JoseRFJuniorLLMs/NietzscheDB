/// Configuration for the Wiederkehr daemon engine.
#[derive(Debug, Clone)]
pub struct DaemonEngineConfig {
    /// How often the engine wakes up to tick (seconds).
    pub tick_secs: u64,
    /// Energy drained from each daemon per tick.
    pub decay_per_tick: f64,
    /// Daemons below this energy are reaped.
    pub min_energy: f64,
    /// Maximum nodes to evaluate per daemon per tick.
    pub max_nodes_per_tick: usize,
}

impl Default for DaemonEngineConfig {
    fn default() -> Self {
        Self {
            tick_secs:          30,
            decay_per_tick:     0.01,
            min_energy:         0.01,
            max_nodes_per_tick: 10_000,
        }
    }
}

impl DaemonEngineConfig {
    /// Build config from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("DAEMON_TICK_SECS") {
            if let Ok(n) = v.parse() { cfg.tick_secs = n; }
        }
        if let Ok(v) = std::env::var("DAEMON_DECAY_PER_TICK") {
            if let Ok(f) = v.parse() { cfg.decay_per_tick = f; }
        }
        if let Ok(v) = std::env::var("DAEMON_MIN_ENERGY") {
            if let Ok(f) = v.parse() { cfg.min_energy = f; }
        }
        if let Ok(v) = std::env::var("DAEMON_MAX_NODES_PER_TICK") {
            if let Ok(n) = v.parse() { cfg.max_nodes_per_tick = n; }
        }
        cfg
    }
}
