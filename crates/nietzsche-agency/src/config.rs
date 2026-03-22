// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
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

    // -- Semantic Gravity (Phase XIV) --
    /// Whether gravity field computation is enabled (default: true).
    pub gravity_enabled: bool,
    /// Gravitational coupling constant (default: 1.0).
    pub gravity_g_constant: f64,
    /// Minimum mass to qualify as a gravity well (default: 0.5).
    pub gravity_well_threshold: f64,
    /// Maximum node pairs to evaluate (default: 5000).
    pub gravity_max_pairs: usize,
    /// Tick interval for gravity computation (default: 3).
    pub gravity_interval: u64,
    /// Whether gravity pulls actually redistribute energy (default: false).
    pub gravity_apply_pulls: bool,

    // -- Shatter Protocol (Phase XVI) --
    /// Whether the shatter protocol is enabled (default: true).
    pub shatter_enabled: bool,
    /// Degree threshold (in + out) to trigger shattering (default: 500).
    pub shatter_threshold: usize,
    /// Maximum avatar shards per super-node (default: 8).
    pub shatter_max_avatars: usize,
    /// Tick interval for shatter scans (default: 5).
    pub shatter_interval: u64,

    // -- Self-Healing Graph (Phase XIX) --
    /// Whether self-healing is enabled (default: true).
    pub healing_enabled: bool,
    /// Tick interval between healing scans (default: 10).
    pub healing_interval: u64,
    /// Maximum nodes to scan per healing tick (default: 5000).
    pub healing_max_scan: usize,
    /// Embedding norm threshold for boundary drift (default: 0.995).
    pub healing_norm_threshold: f64,
    /// Minimum orphan age in seconds before cleanup (default: 3600).
    pub healing_orphan_min_age: i64,
    /// Phantom ratio above which ghost cleanup triggers (default: 0.15).
    pub healing_ghost_ratio: f64,
    /// Whether neural anomaly detection via ONNX is enabled (default: false).
    /// Requires the `anomaly_detector` model to be loaded in the REGISTRY.
    /// If true but model is not available, falls back to heuristic-only.
    pub healing_neural_enabled: bool,
    /// Anomaly score threshold above which a node is flagged (default: 0.7).
    /// The anomaly_detector outputs a combined score in [0, 1].
    pub healing_neural_threshold: f64,
    /// Maximum nodes to run through neural anomaly detection per tick (default: 200).
    /// Neural inference is more expensive than heuristic checks.
    pub healing_neural_max_infer: usize,

    // -- DirtySet: Adaptive Sampling (Phase XV) --
    /// Whether adaptive sampling is enabled (default: true).
    pub dirty_enabled: bool,
    /// If dirty fraction >= this ratio, do full scan (default: 0.3).
    pub dirty_full_scan_ratio: f64,
    /// Number of stability samples from clean nodes (default: 50).
    pub dirty_stability_sample: usize,

    // -- Graph Learning Engine (Phase XX) --
    /// Whether the learning engine is enabled (default: true).
    pub learning_enabled: bool,
    /// Tick interval for learning analysis (default: 5).
    pub learning_interval: u64,
    /// Rolling window size for pattern detection (default: 50).
    pub learning_window_size: usize,
    /// Maximum entries tracked per category (default: 1000).
    pub learning_max_tracked: usize,
    /// Minimum access count to qualify as hotspot (default: 10).
    pub learning_hotspot_threshold: u64,
    /// Growth rate above which a sector is "booming" (default: 5.0).
    pub learning_growth_rate_threshold: f64,

    // -- Knowledge Compression (Phase XXI) --
    /// Whether compression is enabled (default: true).
    pub compression_enabled: bool,
    /// Tick interval between compression scans (default: 20).
    pub compression_interval: u64,
    /// Maximum nodes to scan per compression tick (default: 2000).
    pub compression_max_scan: usize,
    /// Poincaré distance for near-duplicate detection (default: 0.005).
    pub compression_duplicate_epsilon: f64,
    /// Energy threshold for stale nodes (default: 0.05).
    pub compression_stale_energy: f32,
    /// Max degree for stale cluster candidates (default: 5).
    pub compression_stale_max_degree: usize,
    /// Minimum cluster size for compression (default: 3).
    pub compression_min_cluster: usize,
    /// Maximum merges proposed per tick (default: 50).
    pub compression_max_merges: usize,

    // -- Hyperbolic Sharding (Phase XXII) --
    /// Whether sharding analysis is enabled (default: true).
    pub sharding_enabled: bool,
    /// Tick interval for sharding scans (default: 30).
    pub sharding_interval: u64,
    /// Maximum nodes to sample for shard analysis (default: 5000).
    pub sharding_max_sample: usize,
    /// Number of radial bands (default: 4).
    pub sharding_radial_bands: usize,
    /// Number of angular sectors per band (default: 8).
    pub sharding_angular_sectors: usize,
    /// Imbalance ratio for rebalance suggestion (default: 3.0).
    pub sharding_imbalance_threshold: f64,

    // -- World Model (Phase XXIII) --
    /// Whether the world model is enabled (default: true).
    pub world_model_enabled: bool,
    /// Tick interval for world model snapshots (default: 10).
    pub world_model_snapshot_interval: u64,
    /// Rolling history length (default: 100).
    pub world_model_history_length: usize,
    /// Anomaly detection sensitivity in std devs (default: 3.0).
    pub world_model_anomaly_sensitivity: f64,

    // -- Hyperbolic Contrastive Training (Phase XVIII) --
    /// Whether hyperbolic training is enabled (default: true).
    pub hyp_training_enabled: bool,
    /// Tick interval for training runs (default: 50).
    pub hyp_training_interval: u64,
    /// Learning rate for Riemannian SGD (default: 0.01).
    pub hyp_training_lr: f64,
    /// Number of negative samples per positive edge (default: 10).
    pub hyp_training_num_negatives: usize,
    /// Margin for negative samples (default: 0.1).
    pub hyp_training_margin: f64,
    /// Max norm for ball projection (default: 0.95).
    pub hyp_training_max_norm: f64,
    /// Training epochs per tick (default: 5).
    pub hyp_training_epochs: usize,
    /// Convergence threshold (default: 1e-4).
    pub hyp_training_convergence: f64,
    /// Burn-in epochs with reduced LR (default: 2).
    pub hyp_training_burn_in: usize,
    /// Max edges sampled per epoch (default: 5000).
    pub hyp_training_max_edges: usize,

    // -- Cognitive Flywheel (Phase XXIV) --
    /// Whether the flywheel is enabled (default: true).
    pub flywheel_enabled: bool,
    /// Tick interval for flywheel analysis (default: 10).
    pub flywheel_interval: u64,
    /// Momentum decay factor (default: 0.95).
    pub flywheel_momentum_decay: f64,
    /// Minimum momentum for "spinning" state (default: 0.3).
    pub flywheel_min_momentum: f64,

    // -- Temporal Edge Decay (Phase B1) --
    /// Whether temporal decay is enabled (default: true).
    pub temporal_decay_enabled: bool,
    /// Tick interval for decay scans (default: 10).
    pub temporal_decay_interval: u64,
    /// Base decay rate λ (default: 1e-7, ~80 day half-life).
    pub temporal_decay_lambda: f64,
    /// Minimum effective weight before edge is marked for pruning (default: 0.01).
    pub temporal_decay_prune_threshold: f32,
    /// Maximum edges to scan per tick (default: 5000).
    pub temporal_decay_max_scan: usize,
    /// Whether to actually emit prune intents (default: false — report only).
    pub temporal_decay_enable_pruning: bool,

    // -- Autonomous Graph Growth (Phase C) --
    /// Whether autonomous graph growth is enabled (default: true).
    pub growth_enabled: bool,
    /// Tick interval for growth scans (default: 20).
    pub growth_interval: u64,
    /// Maximum candidates to evaluate per tick (default: 100).
    pub growth_max_candidates: usize,
    /// Minimum embedding similarity to propose edge (default: 0.7 in Poincaré distance < threshold).
    pub growth_distance_threshold: f64,
    /// Maximum new edges to propose per tick (default: 50).
    pub growth_max_new_edges: usize,
    /// Minimum energy for a node to be a growth source (default: 0.1).
    pub growth_min_energy: f32,
    /// Maximum degree for a node to receive new edges (default: 100).
    pub growth_max_target_degree: usize,
    /// Whether to use the neural edge_predictor ONNX model for scoring candidates (default: false).
    /// Requires the model to be loaded in `nietzsche_neural::REGISTRY` as "edge_predictor".
    pub growth_neural_enabled: bool,
    /// Minimum predicted probability from the neural edge_predictor to accept an edge (default: 0.5).
    pub growth_neural_threshold: f32,

    // -- Cognitive Layer (Phase E) --
    /// Whether the cognitive layer is enabled (default: true).
    pub cognitive_enabled: bool,
    /// Tick interval for cognitive scans (default: 30).
    pub cognitive_interval: u64,
    /// Maximum nodes to sample for clustering (default: 2000).
    pub cognitive_max_sample: usize,
    /// Poincaré distance threshold for cluster membership (default: 0.3).
    pub cognitive_cluster_radius: f64,
    /// Minimum cluster size to propose concept node (default: 5).
    pub cognitive_min_cluster: usize,
    /// Maximum concept nodes to propose per tick (default: 10).
    pub cognitive_max_concepts: usize,
    /// Whether to use the neural cluster_scorer model for keep/split/merge
    /// decisions instead of heuristic thresholds (default: false).
    /// Requires the `cluster_scorer` ONNX model to be loaded in the registry.
    /// Falls back to heuristics if the model is not available.
    pub cognitive_neural_enabled: bool,

    // -- Phase 28: Phantom Reaper --
    /// Whether phantom reaping is enabled (default: true).
    pub reap_phantoms_enabled: bool,
    /// Tick interval for phantom reap scans (default: 15).
    pub reap_phantoms_interval: u64,
    /// Maximum nodes to scan per tick (default: 5000).
    pub reap_phantoms_max_scan: usize,
    /// Minimum node age in seconds before eligible for reaping (default: 300 = 5 min).
    pub reap_phantoms_min_age: i64,

    // -- Phase 27: Epistemic Evolution --
    /// Whether Phase 27 evolution is enabled (default: true).
    pub evolution_27_enabled: bool,
    /// Tick interval for evolution analysis (default: 40).
    pub evolution_27_interval: u64,
    /// Maximum nodes to evaluate per tick (default: 500).
    pub evolution_27_max_eval: usize,
    /// Minimum composite score — nodes below this are candidates (default: 0.4).
    pub evolution_27_quality_floor: f32,
    /// Maximum mutations to propose per tick (default: 5).
    pub evolution_27_max_proposals: usize,
    /// Minimum energy for participation (default: 0.05).
    pub evolution_27_min_energy: f32,
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
            gravity_enabled: true,
            gravity_g_constant: 1.0,
            gravity_well_threshold: 0.5,
            gravity_max_pairs: 5000,
            gravity_interval: 3,
            gravity_apply_pulls: false,
            shatter_enabled: true,
            shatter_threshold: 500,
            shatter_max_avatars: 8,
            shatter_interval: 5,
            healing_enabled: true,
            healing_interval: 10,
            healing_max_scan: 5000,
            healing_norm_threshold: 0.995,
            healing_orphan_min_age: 3600,
            healing_ghost_ratio: 0.15,
            healing_neural_enabled: false,
            healing_neural_threshold: 0.7,
            healing_neural_max_infer: 200,
            dirty_enabled: true,
            dirty_full_scan_ratio: 0.3,
            dirty_stability_sample: 50,
            // Phase XX
            learning_enabled: true,
            learning_interval: 5,
            learning_window_size: 50,
            learning_max_tracked: 1000,
            learning_hotspot_threshold: 10,
            learning_growth_rate_threshold: 5.0,
            // Phase XXI
            compression_enabled: true,
            compression_interval: 20,
            compression_max_scan: 2000,
            compression_duplicate_epsilon: 0.005,
            compression_stale_energy: 0.05,
            compression_stale_max_degree: 5,
            compression_min_cluster: 3,
            compression_max_merges: 50,
            // Phase XXII
            sharding_enabled: true,
            sharding_interval: 30,
            sharding_max_sample: 5000,
            sharding_radial_bands: 4,
            sharding_angular_sectors: 8,
            sharding_imbalance_threshold: 3.0,
            // Phase XXIII
            world_model_enabled: true,
            world_model_snapshot_interval: 10,
            world_model_history_length: 100,
            world_model_anomaly_sensitivity: 3.0,
            // Phase XVIII
            hyp_training_enabled: true,
            hyp_training_interval: 50,
            hyp_training_lr: 0.01,
            hyp_training_num_negatives: 10,
            hyp_training_margin: 0.1,
            hyp_training_max_norm: 0.95,
            hyp_training_epochs: 5,
            hyp_training_convergence: 1e-4,
            hyp_training_burn_in: 2,
            hyp_training_max_edges: 5_000,
            // Phase XXIV
            flywheel_enabled: true,
            flywheel_interval: 10,
            flywheel_momentum_decay: 0.95,
            flywheel_min_momentum: 0.3,
            // Phase B1 — Temporal Edge Decay
            temporal_decay_enabled: true,
            temporal_decay_interval: 10,
            temporal_decay_lambda: 1e-7,
            temporal_decay_prune_threshold: 0.01,
            temporal_decay_max_scan: 5_000,
            temporal_decay_enable_pruning: false,
            // Phase C — Autonomous Graph Growth
            growth_enabled: true,
            growth_interval: 20,
            growth_max_candidates: 100,
            growth_distance_threshold: 1.5,
            growth_max_new_edges: 50,
            growth_min_energy: 0.1,
            growth_max_target_degree: 100,
            growth_neural_enabled: false,
            growth_neural_threshold: 0.5,
            // Phase E — Cognitive Layer
            cognitive_enabled: true,
            cognitive_interval: 30,
            cognitive_max_sample: 2_000,
            cognitive_cluster_radius: 0.3,
            cognitive_min_cluster: 5,
            cognitive_max_concepts: 10,
            cognitive_neural_enabled: false,
            // Phase 28 — Phantom Reaper
            reap_phantoms_enabled: true,
            reap_phantoms_interval: 15,
            reap_phantoms_max_scan: 5_000,
            reap_phantoms_min_age: 300,
            // Phase 27 — Epistemic Evolution
            evolution_27_enabled: true,
            evolution_27_interval: 40,
            evolution_27_max_eval: 500,
            evolution_27_quality_floor: 0.4,
            evolution_27_max_proposals: 5,
            evolution_27_min_energy: 0.05,
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
            gravity_enabled: std::env::var("AGENCY_GRAVITY_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            gravity_g_constant:     env_f64("AGENCY_GRAVITY_G_CONSTANT", 1.0),
            gravity_well_threshold: env_f64("AGENCY_GRAVITY_WELL_THRESHOLD", 0.5),
            gravity_max_pairs:      env_usize("AGENCY_GRAVITY_MAX_PAIRS", 5000),
            gravity_interval:       env_u64("AGENCY_GRAVITY_INTERVAL", 3),
            gravity_apply_pulls: std::env::var("AGENCY_GRAVITY_APPLY_PULLS")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            healing_enabled: std::env::var("AGENCY_HEALING_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            healing_interval:       env_u64("AGENCY_HEALING_INTERVAL", 10),
            healing_max_scan:       env_usize("AGENCY_HEALING_MAX_SCAN", 5000),
            healing_norm_threshold: env_f64("AGENCY_HEALING_NORM_THRESHOLD", 0.995),
            healing_orphan_min_age: env_u64("AGENCY_HEALING_ORPHAN_AGE", 3600) as i64,
            healing_ghost_ratio:    env_f64("AGENCY_HEALING_GHOST_RATIO", 0.15),
            healing_neural_enabled: std::env::var("AGENCY_HEALING_NEURAL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            healing_neural_threshold: env_f64("AGENCY_HEALING_NEURAL_THRESHOLD", 0.7),
            healing_neural_max_infer: env_usize("AGENCY_HEALING_NEURAL_MAX_INFER", 200),
            shatter_enabled: std::env::var("AGENCY_SHATTER_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            shatter_threshold:  env_usize("AGENCY_SHATTER_THRESHOLD", 500),
            shatter_max_avatars: env_usize("AGENCY_SHATTER_MAX_AVATARS", 8),
            shatter_interval:   env_u64("AGENCY_SHATTER_INTERVAL", 5),
            dirty_enabled: std::env::var("AGENCY_DIRTY_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            dirty_full_scan_ratio:  env_f64("AGENCY_DIRTY_FULL_SCAN_RATIO", 0.3),
            dirty_stability_sample: env_usize("AGENCY_DIRTY_STABILITY_SAMPLE", 50),
            // Phase XX
            learning_enabled: std::env::var("AGENCY_LEARNING_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            learning_interval:      env_u64("AGENCY_LEARNING_INTERVAL", 5),
            learning_window_size:   env_usize("AGENCY_LEARNING_WINDOW_SIZE", 50),
            learning_max_tracked:   env_usize("AGENCY_LEARNING_MAX_TRACKED", 1000),
            learning_hotspot_threshold: env_u64("AGENCY_LEARNING_HOTSPOT_THRESHOLD", 10),
            learning_growth_rate_threshold: env_f64("AGENCY_LEARNING_GROWTH_RATE", 5.0),
            // Phase XXI
            compression_enabled: std::env::var("AGENCY_COMPRESSION_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            compression_interval:    env_u64("AGENCY_COMPRESSION_INTERVAL", 20),
            compression_max_scan:    env_usize("AGENCY_COMPRESSION_MAX_SCAN", 2000),
            compression_duplicate_epsilon: env_f64("AGENCY_COMPRESSION_EPSILON", 0.005),
            compression_stale_energy: env_f32("AGENCY_COMPRESSION_STALE_ENERGY", 0.05),
            compression_stale_max_degree: env_usize("AGENCY_COMPRESSION_STALE_DEGREE", 5),
            compression_min_cluster: env_usize("AGENCY_COMPRESSION_MIN_CLUSTER", 3),
            compression_max_merges:  env_usize("AGENCY_COMPRESSION_MAX_MERGES", 50),
            // Phase XXII
            sharding_enabled: std::env::var("AGENCY_SHARDING_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            sharding_interval:      env_u64("AGENCY_SHARDING_INTERVAL", 30),
            sharding_max_sample:    env_usize("AGENCY_SHARDING_MAX_SAMPLE", 5000),
            sharding_radial_bands:  env_usize("AGENCY_SHARDING_RADIAL_BANDS", 4),
            sharding_angular_sectors: env_usize("AGENCY_SHARDING_ANGULAR_SECTORS", 8),
            sharding_imbalance_threshold: env_f64("AGENCY_SHARDING_IMBALANCE", 3.0),
            // Phase XXIII
            world_model_enabled: std::env::var("AGENCY_WORLD_MODEL_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            world_model_snapshot_interval: env_u64("AGENCY_WORLD_MODEL_INTERVAL", 10),
            world_model_history_length: env_usize("AGENCY_WORLD_MODEL_HISTORY", 100),
            world_model_anomaly_sensitivity: env_f64("AGENCY_WORLD_MODEL_ANOMALY", 3.0),
            // Phase XVIII
            hyp_training_enabled: std::env::var("AGENCY_HYP_TRAINING_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            hyp_training_interval:       env_u64("AGENCY_HYP_TRAINING_INTERVAL", 50),
            hyp_training_lr:             env_f64("AGENCY_HYP_TRAINING_LR", 0.01),
            hyp_training_num_negatives:  env_usize("AGENCY_HYP_TRAINING_NUM_NEG", 10),
            hyp_training_margin:         env_f64("AGENCY_HYP_TRAINING_MARGIN", 0.1),
            hyp_training_max_norm:       env_f64("AGENCY_HYP_TRAINING_MAX_NORM", 0.95),
            hyp_training_epochs:         env_usize("AGENCY_HYP_TRAINING_EPOCHS", 5),
            hyp_training_convergence:    env_f64("AGENCY_HYP_TRAINING_CONVERGENCE", 1e-4),
            hyp_training_burn_in:        env_usize("AGENCY_HYP_TRAINING_BURN_IN", 2),
            hyp_training_max_edges:      env_usize("AGENCY_HYP_TRAINING_MAX_EDGES", 5_000),
            // Phase XXIV
            flywheel_enabled: std::env::var("AGENCY_FLYWHEEL_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            flywheel_interval:       env_u64("AGENCY_FLYWHEEL_INTERVAL", 10),
            flywheel_momentum_decay: env_f64("AGENCY_FLYWHEEL_MOMENTUM_DECAY", 0.95),
            flywheel_min_momentum:   env_f64("AGENCY_FLYWHEEL_MIN_MOMENTUM", 0.3),
            // Phase B1 — Temporal Edge Decay
            temporal_decay_enabled: std::env::var("AGENCY_TEMPORAL_DECAY_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            temporal_decay_interval:       env_u64("AGENCY_TEMPORAL_DECAY_INTERVAL", 10),
            temporal_decay_lambda:         env_f64("AGENCY_TEMPORAL_DECAY_LAMBDA", 1e-7),
            temporal_decay_prune_threshold: env_f32("AGENCY_TEMPORAL_DECAY_PRUNE", 0.01),
            temporal_decay_max_scan:       env_usize("AGENCY_TEMPORAL_DECAY_MAX_SCAN", 5_000),
            temporal_decay_enable_pruning: std::env::var("AGENCY_TEMPORAL_DECAY_PRUNING")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            // Phase C — Autonomous Graph Growth
            growth_enabled: std::env::var("AGENCY_GROWTH_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            growth_interval:          env_u64("AGENCY_GROWTH_INTERVAL", 20),
            growth_max_candidates:    env_usize("AGENCY_GROWTH_MAX_CANDIDATES", 100),
            growth_distance_threshold: env_f64("AGENCY_GROWTH_DISTANCE_THRESHOLD", 1.5),
            growth_max_new_edges:     env_usize("AGENCY_GROWTH_MAX_NEW_EDGES", 50),
            growth_min_energy:        env_f32("AGENCY_GROWTH_MIN_ENERGY", 0.1),
            growth_max_target_degree: env_usize("AGENCY_GROWTH_MAX_TARGET_DEGREE", 100),
            growth_neural_enabled: std::env::var("AGENCY_GROWTH_NEURAL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            growth_neural_threshold: env_f32("AGENCY_GROWTH_NEURAL_THRESHOLD", 0.5),
            // Phase E — Cognitive Layer
            cognitive_enabled: std::env::var("AGENCY_COGNITIVE_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            cognitive_interval:       env_u64("AGENCY_COGNITIVE_INTERVAL", 30),
            cognitive_max_sample:     env_usize("AGENCY_COGNITIVE_MAX_SAMPLE", 2_000),
            cognitive_cluster_radius: env_f64("AGENCY_COGNITIVE_CLUSTER_RADIUS", 0.3),
            cognitive_min_cluster:    env_usize("AGENCY_COGNITIVE_MIN_CLUSTER", 5),
            cognitive_max_concepts:   env_usize("AGENCY_COGNITIVE_MAX_CONCEPTS", 10),
            cognitive_neural_enabled: std::env::var("AGENCY_COGNITIVE_NEURAL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            // Phase 28 — Phantom Reaper
            reap_phantoms_enabled: std::env::var("AGENCY_REAP_PHANTOMS_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            reap_phantoms_interval:  env_u64("AGENCY_REAP_PHANTOMS_INTERVAL", 15),
            reap_phantoms_max_scan:  env_usize("AGENCY_REAP_PHANTOMS_MAX_SCAN", 5_000),
            reap_phantoms_min_age:   env_u64("AGENCY_REAP_PHANTOMS_MIN_AGE", 300) as i64,
            // Phase 27 — Epistemic Evolution
            evolution_27_enabled: std::env::var("AGENCY_EVOLUTION_27_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            evolution_27_interval:       env_u64("AGENCY_EVOLUTION_27_INTERVAL", 40),
            evolution_27_max_eval:       env_usize("AGENCY_EVOLUTION_27_MAX_EVAL", 500),
            evolution_27_quality_floor:  env_f32("AGENCY_EVOLUTION_27_QUALITY_FLOOR", 0.4),
            evolution_27_max_proposals:  env_usize("AGENCY_EVOLUTION_27_MAX_PROPOSALS", 5),
            evolution_27_min_energy:     env_f32("AGENCY_EVOLUTION_27_MIN_ENERGY", 0.05),
        }
    }
}
