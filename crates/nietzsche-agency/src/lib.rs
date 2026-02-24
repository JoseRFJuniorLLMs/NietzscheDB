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
