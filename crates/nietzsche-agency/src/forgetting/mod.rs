//! Forgetting Engine — The Nezhmetdinov Active Forgetting System.
//!
//! A four-layer architecture for intelligent data deletion:
//!
//! ## CAMADA 1 — O JULGAMENTO LOCAL (vitality + judgment + bounds + ricci)
//! Evaluates each node with V(n) = σ(w₁e + w₂H − w₃ξ + w₄π + w₅κ − w₆τ)
//! and applies the Triple Condition with Ricci curvature veto.
//!
//! ## CAMADA 2 — REGISTRO HISTÓRICO (ledger)
//! Merkle Tree-based deletion receipts for tamper-proof auditing.
//!
//! ## CAMADA 3 — METABOLISMO GENERATIVO (void_tracker)
//! Tracks Poincaré coordinates of deleted nodes as seeds for generation.
//!
//! ## CAMADA 4 — SAÚDE GLOBAL (tgc + elite_drift + anti_gaming + stability)
//! Four vital signs: TGC, Var(V), Elite Drift, Anti-Gaming.

pub mod anti_gaming;
pub mod bounds;
pub mod causal_immunity;
pub mod elite_drift;
pub mod friction;
pub mod judgment;
pub mod ledger;
pub mod ricci;
pub mod stability;
pub mod telemetry;
pub mod tgc;
pub mod vitality;
pub mod vitality_variance;
pub mod void_tracker;
pub mod zaratustra_cycle;

// ── Re-exports ──────────────────────────────────────────────────

// CAMADA 1: Julgamento Local
pub use vitality::{VitalityWeights, VitalityInput, nezhmetdinov_vitality, nezhmetdinov_vitality_batch, sigmoid};
pub use judgment::{Verdict, ForgetteringJudgment, MikhailThallReport};
pub use bounds::{HardBounds, NezhmetdinovConfig};
pub use ricci::{RicciShield, RicciSimulation};

// CAMADA 2: Registro Histórico
pub use ledger::{DeletionLedger, DeletionReceipt, StructuralHash, InclusionProof};

// CAMADA 3: Metabolismo Generativo
pub use void_tracker::{VoidCoordinate, VoidTracker};

// CAMADA 4: Saúde Global
pub use tgc::{TgcCalculator, TgcSnapshot};
pub use elite_drift::{EliteDriftTracker, EliteCentroid};
pub use anti_gaming::{AntiGamingMonitor, AntiGamingReport, GamingViolation};
pub use stability::{StabilityMonitor, StabilityConfig, CollapseType, CollapseAlert, ThermalPerturbation};

// Telemetry
pub use telemetry::{CycleTelemetry, TelemetryWriter, format_cycle_summary};

// Extended modules
pub use causal_immunity::{CausalStatus, CausalAnalysis};
pub use friction::{FrictionCalculator, FrictionScore, CycleFriction};
pub use vitality_variance::{VitalityVarianceTracker, VitalityVarianceSnapshot, CognitiveHealthClass};
pub use zaratustra_cycle::{ZaratustraCycle, ZaratustraCycleReport};
