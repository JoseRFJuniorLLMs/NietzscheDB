/// Configuration for the agency engine and its daemons.
#[derive(Debug, Clone)]
pub struct AgencyConfig {
    /// Seconds between agency engine ticks (0 = disabled).
    pub tick_secs: u64,

    // -- EntropyDaemon --
    /// Hausdorff variance threshold above which an entropy spike is emitted.
    pub entropy_variance_threshold: f32,
    /// Number of angular regions for entropy partitioning.
    pub entropy_region_count: usize,

    // -- CoherenceDaemon --
    /// Maximum Jaccard overlap(t=0.1, t=10.0) before a coherence drop alert.
    pub coherence_overlap_threshold: f64,

    // -- GapDaemon --
    /// Number of angular sectors for gap detection.
    pub gap_sector_count: usize,
    /// Number of radial depth bands.
    pub gap_depth_bins: usize,
    /// Minimum relative density per sector (fraction of expected uniform).
    pub gap_min_density: f64,

    // -- Observer --
    /// Number of ticks between full health reports.
    pub observer_report_interval: usize,
    /// Mean energy below which the observer emits a wake-up.
    pub observer_wake_energy_threshold: f32,
    /// Hausdorff lower bound for wake-up.
    pub observer_wake_hausdorff_lo: f32,
    /// Hausdorff upper bound for wake-up.
    pub observer_wake_hausdorff_hi: f32,

    // -- Reactor --
    /// Minimum ticks between repeated sleep/lsystem intents (cooldown).
    pub reactor_cooldown_ticks: u64,

    // -- Counterfactual --
    /// Max BFS hops for impact propagation.
    pub counterfactual_max_hops: usize,

    // -- Desire → Action --
    /// Minimum desire priority to trigger an automatic dream (0.0–1.0).
    pub desire_dream_threshold: f32,
    /// BFS depth for agency-triggered dreams.
    pub desire_dream_depth: usize,

    // -- NiilistaGcDaemon --
    /// Poincaré distance below which two nodes are considered redundant.
    pub niilista_distance_threshold: f32,
    /// Minimum cluster size to emit a SemanticRedundancy event.
    pub niilista_min_group_size: usize,
    /// Maximum nodes to scan per tick (embedding loads are expensive).
    pub niilista_max_scan: usize,

    // -- Evolution --
    /// Minimum ticks between evolution suggestions.
    pub evolution_cooldown_ticks: u64,

    // -- Circuit Breaker --
    /// Maximum number of reflexive actions that can fire in a single tick.
    pub circuit_breaker_max_actions: usize,
    /// Maximum global energy sum allowed before blocking reflexes.
    pub circuit_breaker_energy_sum_threshold: f32,

    // -- LTD Daemon --
    /// Weight reduction per caregiver correction event (default: 0.05 = 5%).
    /// None uses the daemon's built-in default.
    pub ltd_rate: Option<f64>,
    /// Minimum correction count before triggering LTD.
    /// None = any non-zero correction count triggers LTD.
    pub ltd_correction_threshold: Option<u64>,
}

impl Default for AgencyConfig {
    fn default() -> Self {
        Self {
            tick_secs: 60,
            entropy_variance_threshold: 0.25,
            entropy_region_count: 8,
            coherence_overlap_threshold: 0.70,
            gap_sector_count: 16,
            gap_depth_bins: 5,
            gap_min_density: 0.1,
            observer_report_interval: 5,
            observer_wake_energy_threshold: 0.3,
            observer_wake_hausdorff_lo: 0.5,
            observer_wake_hausdorff_hi: 1.9,
            reactor_cooldown_ticks: 3,
            counterfactual_max_hops: 3,
            desire_dream_threshold: 0.6,
            desire_dream_depth: 5,
            niilista_distance_threshold: 0.01,
            niilista_min_group_size: 2,
            niilista_max_scan: 200,
            evolution_cooldown_ticks: 5,
            circuit_breaker_max_actions: 20,
            circuit_breaker_energy_sum_threshold: 50.0,
            ltd_rate: None,
            ltd_correction_threshold: None,
        }
    }
}

impl AgencyConfig {
    /// Build from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        fn env_u64(key: &str, default: u64) -> u64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_usize(key: &str, default: usize) -> usize {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_f32(key: &str, default: f32) -> f32 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }
        fn env_f64(key: &str, default: f64) -> f64 {
            std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        }

        Self {
            tick_secs:                      env_u64("AGENCY_TICK_SECS", 60),
            entropy_variance_threshold:     env_f32("AGENCY_ENTROPY_THRESHOLD", 0.25),
            entropy_region_count:           env_usize("AGENCY_ENTROPY_REGIONS", 8),
            coherence_overlap_threshold:    env_f64("AGENCY_COHERENCE_THRESHOLD", 0.70),
            gap_sector_count:               env_usize("AGENCY_GAP_SECTORS", 16),
            gap_depth_bins:                 env_usize("AGENCY_GAP_DEPTH_BINS", 5),
            gap_min_density:                env_f64("AGENCY_GAP_MIN_DENSITY", 0.1),
            observer_report_interval:       env_usize("AGENCY_OBSERVER_INTERVAL", 5),
            observer_wake_energy_threshold: env_f32("AGENCY_OBSERVER_ENERGY_WAKE", 0.3),
            observer_wake_hausdorff_lo:     env_f32("AGENCY_OBSERVER_HAUSDORFF_LO", 0.5),
            observer_wake_hausdorff_hi:     env_f32("AGENCY_OBSERVER_HAUSDORFF_HI", 1.9),
            reactor_cooldown_ticks:         env_u64("AGENCY_REACTOR_COOLDOWN", 3),
            counterfactual_max_hops:        env_usize("AGENCY_CF_MAX_HOPS", 3),
            desire_dream_threshold:         env_f32("AGENCY_DESIRE_DREAM_THRESHOLD", 0.6),
            desire_dream_depth:             env_usize("AGENCY_DESIRE_DREAM_DEPTH", 5),
            niilista_distance_threshold:    env_f32("AGENCY_NIILISTA_DISTANCE", 0.01),
            niilista_min_group_size:        env_usize("AGENCY_NIILISTA_MIN_GROUP", 2),
            niilista_max_scan:              env_usize("AGENCY_NIILISTA_MAX_SCAN", 200),
            evolution_cooldown_ticks:       env_u64("AGENCY_EVOLUTION_COOLDOWN", 5),
            circuit_breaker_max_actions:    env_usize("AGENCY_CIRCUIT_BREAKER_MAX", 20),
            circuit_breaker_energy_sum_threshold: env_f32("AGENCY_CIRCUIT_BREAKER_ENERGY", 50.0),
            ltd_rate: std::env::var("AGENCY_LTD_RATE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok()),
            ltd_correction_threshold: std::env::var("AGENCY_LTD_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<u64>().ok()),
        }
    }
}
