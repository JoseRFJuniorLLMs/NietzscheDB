//! `nietzsche-agency` — Autonomous agency engine for NietzscheDB.
//!
//! Transforms NietzscheDB from a reactive knowledge graph into a
//! **self-monitoring subconscious** with:
//!
//! - **Daemons** that detect entropy spikes, coherence drops, and knowledge gaps
//! - **MetaObserver** that aggregates health metrics and triggers wake-ups
//! - **Reactor** that converts events into executable intents (sleep, L-System, etc.)
//! - **Motor de Desejo** that transforms knowledge gaps into desire signals
//! - **Observer Identity** — a meta-node in the graph representing self-awareness
//! - **Counterfactual Engine** that simulates "what-if" scenarios
//! - **Store** for persisting HealthReports and DesireSignals to RocksDB CF_META
//!
//! ## Architecture
//!
//! ```text
//!                     ┌─────────────┐
//!                     │ AgencyEngine │
//!                     └──────┬──────┘
//!                            │ tick()
//!     ┌─────────┬────────┬───┴───┬──────────┬──────────┬───────────┐
//!     │         │        │       │          │          │           │
//!     ▼         ▼        ▼       ▼          ▼          ▼           ▼
//!  Entropy  Coherence  Gap   Observer   Reactor    Desire     Counterfactual
//!  Daemon   Daemon    Daemon  Identity             Engine     Engine
//!     │         │        │       ▲          │
//!     └─────────┴────────┘       │          │
//!                  │ publish()   │ subscribe()
//!                  ▼             │          │
//!            AgencyEventBus ─────┘          │
//!                                           ▼
//!                                    AgencyIntents
//!                               (sleep, lsystem, gap signals)
//! ```

pub mod axiom_registry;
pub mod centroid_guardian;
pub mod hyperbolic_health;
pub mod maturity;
pub mod code_as_data;
pub mod circuit_breaker;
pub mod config;
pub mod counterfactual;
pub mod daemons;
pub mod desire;
pub mod dialectic;
pub mod engine;
pub mod error;
pub mod event_bus;
pub mod evolution;
pub mod forgetting;
pub mod identity;
pub mod observer;
pub mod quantum;
pub mod reactor;
pub mod store;

// Phase XII — ECAN (Economic Attention Network) + Curiosity Engine
pub mod attention_economy;
pub mod curiosity_engine;
pub mod attention_cycle;

// Phase XII.5 — Hebbian LTP (structural plasticity)
pub mod hebbian;

// Phase XIII — Cognitive Thermodynamics
pub mod thermodynamics;

// Phase XIV — Semantic Gravity
pub mod gravity;

// Cognitive Dashboard (unified observability)
pub mod cognitive_dashboard;

// Phase XV — DirtySet (Temporal Adaptive Sampling)
pub mod dirty_set;

// Phase XV.1 — ObservationBridge (live metrics for visualization)
pub mod observation;

// Phase XVI — Shatter Protocol (super-node splitting)
pub mod shatter;

pub use centroid_guardian::{CentroidGuardian, CentroidUpdate, GuardianError};
pub use config::AgencyConfig;
pub use counterfactual::{CounterfactualEngine, CounterfactualOp, CounterfactualResult};
pub use daemons::{AgencyDaemon, DaemonReport, NiilistaGcDaemon, LTDDaemon, NezhmetdinovDaemon};
pub use desire::{DesireEngine, DesireSignal, list_pending_desires, fulfill_desire};
pub use engine::{AgencyEngine, AgencyTickReport};
pub use error::AgencyError;
pub use event_bus::{AgencyEvent, AgencyEventBus, SectorId, WakeUpReason};
pub use evolution::{EvolutionStrategy, EvolvedRule, EvolvedRuleType, RuleEvolution, EvolutionState};
pub use forgetting::{
    NezhmetdinovConfig, HardBounds, VitalityWeights, Verdict, ForgetteringJudgment,
    MikhailThallReport, DeletionLedger, VoidTracker, TgcCalculator, EliteDriftTracker,
    StabilityMonitor, AntiGamingMonitor, nezhmetdinov_vitality,
};
pub use identity::ObserverIdentity;
pub use observer::{HealthReport, EnergyPercentiles, MetaObserver};
pub use quantum::{BlochState, QuantumConfig, QuantumGate, poincare_to_bloch, bloch_to_poincare, batch_poincare_to_bloch, entanglement_proxy};
pub use reactor::{AgencyIntent, AgencyReactor};
pub use code_as_data::{ActionNode, ActionScanReport, create_action_node, scan_activatable_actions, record_firing, tick_cooldowns};
pub use dialectic::{DialecticConfig, DialecticReport, Contradiction, Synthesis, run_dialectic_cycle};
pub use store::{put_health_report, list_health_reports, get_latest_health_report, prune_health_reports};

// Phase X — Hyperbolic Health Monitor
pub use hyperbolic_health::{HyperbolicHealth, HyperbolicHealthMonitor, HyperbolicDiagnosis};

// Phase IX — Self Geométrico
pub use axiom_registry::{AxiomRegistry, AxiomRecord, EraSnapshot};
pub use maturity::{MaturityConfig, MaturityScore, NodeClass, NodeMaturityInput, evaluate_maturity, evaluate_batch};

// Phase XII — ECAN
pub use attention_economy::{AttentionState, AttentionBid, AttentionConfig, AttentionReport};
pub use curiosity_engine::{CuriosityState, CuriosityConfig};
pub use attention_cycle::{EcanConfig, EcanCycle, run_ecan_cycle};

// Phase XII.5 — Hebbian LTP
pub use hebbian::{HebbianConfig, HebbianDelta, HebbianReport, HebbianState, run_hebbian_tick};

// Phase XIII — Cognitive Thermodynamics
pub use thermodynamics::{
    ThermodynamicsConfig, ThermodynamicReport, ThermodynamicState,
    PhaseState, HeatFlow, cognitive_temperature, shannon_entropy,
    helmholtz_free_energy, classify_phase, exploration_modifier,
    run_thermodynamic_cycle,
};

// Phase XIV — Semantic Gravity
pub use gravity::{
    GravityConfig, GravityNode, GravityForce, GravityPull, GravityReport,
    GravityWell, GravityState, gravitational_force, compute_gravity_field,
    run_gravity_tick,
};

// Cognitive Dashboard
pub use cognitive_dashboard::{
    CognitiveDashboard, HealthSnapshot, AttentionSnapshot, HebbianSnapshot,
    ThermoSnapshot, MaturitySnapshot, GravitySnapshot, GravityAttraction,
};

// Phase XV — DirtySet
pub use dirty_set::{DirtySet, DirtySetConfig, ScanDecision};

// Phase XV.1 — ObservationBridge
pub use observation::{
    ObservationFrame, SystemGauges, NodeVisual, GravityLine,
    NodeObservation, temperature_to_rgb,
};

// Phase XVI — Shatter Protocol
pub use shatter::{
    ShatterConfig, ShatterCandidate, ShatterPlan, ShatterReport,
    AvatarPlan, ShatterDaemon, build_shatter_config, run_shatter_scan,
    scan_super_nodes, build_shatter_plan,
};
