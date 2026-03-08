use crate::quantum::QuantumConfig;

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
    /// Model name for GNN inference (e.g. "threshold_v1").
    pub gnn_model_name: String,
    /// Model name for MCTS search (e.g. "policy_v1").
    pub mcts_model_name: String,
    pub ppo_model_name: String,

    // -- CentroidGuardian (Phase IX) --
    /// Temporal damping factor α ∈ (0, 1].
    /// Controls how quickly the centroid follows the new Fréchet mean.
    /// 0.05 = slow/stable (production), 1.0 = no damping.
    pub centroid_alpha: f64,
    /// Maximum allowed Poincaré distance between C_t and C_{t-1}.
    /// If drift exceeds this, maturity promotions are frozen.
    pub centroid_drift_threshold: f64,
    /// Maximum nodes to scan for centroid computation per tick.
    /// Caps the I/O cost of loading embeddings (~12-24 KB each).
    pub centroid_max_scan: usize,
    /// Dimensionality of embeddings for the CentroidGuardian.
    /// Must match the collection's vector dimension (e.g. 3072 for Gemini).
    pub centroid_dim: usize,

    // -- Maturity / AxiomRegistry (Phase IX) --
    /// Maturity score above which Active → Mature promotion occurs.
    pub maturity_promote_threshold: f64,
    /// Maturity score below which Mature → Active demotion occurs (hysteresis).
    pub maturity_demote_threshold: f64,
    /// Maximum nodes to evaluate for maturity per tick (caps cost).
    pub maturity_max_eval: usize,
    /// Minimum Poincaré distance between axiom embeddings to avoid deduplication.
    /// If a new axiom candidate is closer than ε to an existing axiom, skip registration.
    pub axiom_dedup_epsilon: f64,
    /// Epoch interval for era snapshots (0 = disabled).
    pub axiom_era_snapshot_interval: u64,

    // -- HyperbolicHealthMonitor (Phase X) --
    /// Tick interval for health checks (0 = disabled).
    pub hyp_health_interval: u64,
    /// Maximum embeddings to sample per health check.
    pub hyp_health_max_sample: usize,
    /// mean_r threshold for boundary crowding alert.
    pub hyp_health_crowding_r: f64,
    /// Number of radial bins for the density histogram.
    pub hyp_health_bins: usize,
    /// Maximum history snapshots to retain.
    pub hyp_health_history_len: usize,

    // -- Quantum Entanglement --
    /// Configurable thresholds for quantum entanglement-based edge collapse.
    /// Controls when fidelity between Bloch states forces edge materialisation.
    pub quantum: QuantumConfig,

    // -- Hebbian LTP (Phase XII.5) --
    /// Whether Hebbian LTP is enabled (default: true).
    pub hebbian_enabled: bool,
    /// Weight increase per unit of LTP trace (default: 0.02).
    pub hebbian_ltp_rate: f32,
    /// Trace decay factor per tick (default: 0.9).
    pub hebbian_trace_decay: f32,
    /// Maximum edge weight from potentiation (default: 5.0).
    pub hebbian_max_weight: f32,

    // -- Cognitive Thermodynamics (Phase XIII) --
    /// Tick interval for thermodynamic analysis (0 = disabled, default: 5).
    pub thermo_interval: u64,
    /// Cold phase boundary (T < t_cold = solid, default: 0.15).
    pub thermo_t_cold: f64,
    /// Hot phase boundary (T > t_hot = gas, default: 0.85).
    pub thermo_t_hot: f64,
    /// Thermal conductivity for heat flow (default: 0.05).
    pub thermo_conductivity: f64,
    /// Maximum heat flow per edge per tick (default: 0.02).
    pub thermo_max_heat_flow: f32,
    /// Whether heat flow (energy redistribution) is enabled (default: true).
    pub thermo_enable_heat_flow: bool,

    // -- ECAN: Economic Attention Network (Phase XII) --
    /// Tick interval for ECAN cycles (1 = every tick, 0 = disabled).
    pub ecan_interval: u64,
    /// Maximum nodes to scan per ECAN cycle.
    pub ecan_max_scan: usize,
    /// Budget multiplier for attention allocation.
    pub ecan_budget_scale: f32,
    /// Fraction of budget a node demands per cycle.
    pub ecan_demand_fraction: f32,
    /// Minimum energy to participate in attention economy.
    pub ecan_energy_floor: f32,
    /// Max bids per source node.
    pub ecan_max_bids: usize,
    /// Energy boost per unit of attention received.
    pub ecan_energy_gain: f32,
    /// Curiosity threshold for exploration bids.
    pub ecan_curiosity_threshold: f32,
    /// Maximum exploration ratio (caps explore budget).
    pub ecan_max_explore_ratio: f32,
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
            centroid_alpha: 0.05,
            centroid_drift_threshold: 0.15,
            centroid_max_scan: 10_000,
            centroid_dim: 3072,
            maturity_promote_threshold: 0.6,
            maturity_demote_threshold: 0.3,
            maturity_max_eval: 5_000,
            axiom_dedup_epsilon: 0.05,
            axiom_era_snapshot_interval: 10,
            hyp_health_interval: 100,
            hyp_health_max_sample: 2_000,
            hyp_health_crowding_r: 0.96,
            hyp_health_bins: 50,
            hyp_health_history_len: 20,
            ltd_rate: None,
            ltd_correction_threshold: None,
            gnn_model_name: "gnn_v1".to_string(),
            mcts_model_name: "mcts_v1".to_string(),
            ppo_model_name: "ppo_growth_v1".to_string(),
            quantum: QuantumConfig::default(),
            hebbian_enabled: true,
            hebbian_ltp_rate: 0.02,
            hebbian_trace_decay: 0.9,
            hebbian_max_weight: 5.0,
            thermo_interval: 5,
            thermo_t_cold: 0.15,
            thermo_t_hot: 0.85,
            thermo_conductivity: 0.05,
            thermo_max_heat_flow: 0.02,
            thermo_enable_heat_flow: true,
            ecan_interval: 1,
            ecan_max_scan: 10_000,
            ecan_budget_scale: 1.0,
            ecan_demand_fraction: 0.5,
            ecan_energy_floor: 0.05,
            ecan_max_bids: 5,
            ecan_energy_gain: 0.1,
            ecan_curiosity_threshold: 0.1,
            ecan_max_explore_ratio: 0.6,
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
            centroid_alpha:          env_f64("AGENCY_CENTROID_ALPHA", 0.05),
            centroid_drift_threshold: env_f64("AGENCY_CENTROID_DRIFT_THRESHOLD", 0.15),
            centroid_max_scan:       env_usize("AGENCY_CENTROID_MAX_SCAN", 10_000),
            centroid_dim:            env_usize("AGENCY_CENTROID_DIM", 3072),
            maturity_promote_threshold: env_f64("AGENCY_MATURITY_PROMOTE", 0.6),
            maturity_demote_threshold:  env_f64("AGENCY_MATURITY_DEMOTE", 0.3),
            maturity_max_eval:          env_usize("AGENCY_MATURITY_MAX_EVAL", 5_000),
            axiom_dedup_epsilon:        env_f64("AGENCY_AXIOM_DEDUP_EPSILON", 0.05),
            axiom_era_snapshot_interval: env_u64("AGENCY_AXIOM_ERA_INTERVAL", 10),
            hyp_health_interval:    env_u64("AGENCY_HYP_HEALTH_INTERVAL", 100),
            hyp_health_max_sample:  env_usize("AGENCY_HYP_HEALTH_MAX_SAMPLE", 2_000),
            hyp_health_crowding_r:  env_f64("AGENCY_HYP_HEALTH_CROWDING_R", 0.96),
            hyp_health_bins:        env_usize("AGENCY_HYP_HEALTH_BINS", 50),
            hyp_health_history_len: env_usize("AGENCY_HYP_HEALTH_HISTORY", 20),
            ltd_rate: std::env::var("AGENCY_LTD_RATE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok()),
            ltd_correction_threshold: std::env::var("AGENCY_LTD_THRESHOLD")
                .ok()
                .and_then(|v| v.parse::<u64>().ok()),
            gnn_model_name: std::env::var("AGENCY_GNN_MODEL").unwrap_or_else(|_| "gnn_v1".into()),
            mcts_model_name: std::env::var("AGENCY_MCTS_MODEL").unwrap_or_else(|_| "mcts_v1".into()),
            ppo_model_name: std::env::var("AGENCY_PPO_MODEL").unwrap_or_else(|_| "ppo_growth_v1".into()),
            quantum: QuantumConfig::from_env(),
            hebbian_enabled: std::env::var("AGENCY_HEBBIAN_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            hebbian_ltp_rate:       env_f32("AGENCY_HEBBIAN_LTP_RATE", 0.02),
            hebbian_trace_decay:    env_f32("AGENCY_HEBBIAN_TRACE_DECAY", 0.9),
            hebbian_max_weight:     env_f32("AGENCY_HEBBIAN_MAX_WEIGHT", 5.0),
            thermo_interval:        env_u64("AGENCY_THERMO_INTERVAL", 5),
            thermo_t_cold:          env_f64("AGENCY_THERMO_T_COLD", 0.15),
            thermo_t_hot:           env_f64("AGENCY_THERMO_T_HOT", 0.85),
            thermo_conductivity:    env_f64("AGENCY_THERMO_CONDUCTIVITY", 0.05),
            thermo_max_heat_flow:   env_f32("AGENCY_THERMO_MAX_HEAT_FLOW", 0.02),
            thermo_enable_heat_flow: std::env::var("AGENCY_THERMO_HEAT_FLOW")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            ecan_interval:          env_u64("AGENCY_ECAN_INTERVAL", 1),
            ecan_max_scan:          env_usize("AGENCY_ECAN_MAX_SCAN", 10_000),
            ecan_budget_scale:      env_f32("AGENCY_ECAN_BUDGET_SCALE", 1.0),
            ecan_demand_fraction:   env_f32("AGENCY_ECAN_DEMAND_FRACTION", 0.5),
            ecan_energy_floor:      env_f32("AGENCY_ECAN_ENERGY_FLOOR", 0.05),
            ecan_max_bids:          env_usize("AGENCY_ECAN_MAX_BIDS", 5),
            ecan_energy_gain:       env_f32("AGENCY_ECAN_ENERGY_GAIN", 0.1),
            ecan_curiosity_threshold: env_f32("AGENCY_ECAN_CURIOSITY_THRESHOLD", 0.1),
            ecan_max_explore_ratio: env_f32("AGENCY_ECAN_MAX_EXPLORE_RATIO", 0.6),
        }
    }
}
