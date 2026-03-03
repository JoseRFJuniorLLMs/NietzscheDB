//! # nietzsche-agi
//!
//! AGI inference layer for NietzscheDB.
//!
//! This crate sits on top of the existing multi-manifold geometry engine and adds
//! **explicit reasoning with verifiable trajectories** — the missing "consciousness"
//! layer that connects geodesic computation → GCS validation → rule engine →
//! rationale → synthesis → feedback loop.
//!
//! # Architecture (4 Layers)
//!
//! ## Layer 1 — Representation & Storage
//! - [`representation::SynthesisNode`] — AGI node wrapper with inference metadata
//! - [`rationale::Rationale`] — proof object accompanying every inference
//! - [`rationale::InferenceType`] — classification of reasoning patterns
//!
//! ## Layer 2 — Verifiable Semantic Navigation
//! - [`trajectory::GeodesicTrajectory`] — validated path through the Poincaré ball
//! - [`trajectory::GeodesicCoherenceScore`] — per-hop quality metric
//! - [`trajectory::validate_trajectory`] — full GCS validation pipeline
//!
//! ## Layer 3 — Explicit Inference
//! - [`inference_engine::InferenceEngine`] — rule engine that classifies trajectories
//! - [`synthesis::FrechetSynthesizer`] — dialectical synthesis via Fréchet mean
//! - [`dialectic::DialecticDetector`] — cross-cluster tension detection
//!
//! ## Layer 4 — Dynamic Update
//! - [`feedback_loop::FeedbackLoop`] — re-insertion of conclusions into the graph
//! - [`homeostasis::HomeostasisGuard`] — radial repulsion preventing center collapse
//! - [`homeostasis::RadialField`] — smooth gradient-based homeostasis (Phase V)
//! - [`relevance_decay::RelevanceDecay`] — frequency-based weight adjustment
//! - [`evolution::EvolutionScheduler`] — autonomous evolution orchestrator
//!
//! ## Layer 5 — Stability Motor (Phase V)
//! - [`stability::StabilityEvaluator`] — continuous energy function E(τ)
//! - [`certification::CertificationLevel`] — 4-tier epistemological classification
//! - [`certification::CertificationSeal`] — immutable quality stamp
//! - [`spectral::SpectralMonitor`] — λ₂ Laplacian connectivity monitoring
//! - [`spectral::DriftTracker`] — λ₂ evolution tracking (Evolutionary Model)
//!
//! ## Layer 6 — Metabolic Equilibrium (Phase VI)
//! - [`discovery::DiscoveryField`] — D(τ) = |∇E(τ)| + θ_cluster (productive friction)
//! - [`innovation::InnovationEvaluator`] — Φ(τ) = αS + βD - γR (acceptance function)
//! - [`innovation::AcceptanceDecision`] — Accept / Sandbox / Reject
//! - [`sandbox::SandboxEvaluator`] — quarantine + spectral promotion lifecycle
//!
//! ## Layer 7 — Criticality & Metabolic Dynamics (Phase VI+)
//! - [`homeostasis::DampedHomeostasis`] — damped dynamics in tangent space via exp_map
//! - [`criticality::CriticalityDetector`] — formal edge-of-chaos regime detection
//! - [`criticality::RegimeState`] — Rigid / Critical / Turbulent classification
//! - [`spectral::AdaptiveBand`] — mobile λ₂ target band with ε-adaptation
//! - [`evolution::MetabolicInput`] — pre-computed snapshot for metabolic cycle
//! - [`evolution::CycleReport`] — complete auditable cycle output
//!
//! # Design Principles
//!
//! 1. **Every inference carries a Rationale** — no black-box reasoning
//! 2. **GCS validates every hop** — broken geodesics are rejected (LogicalRupture)
//! 3. **Synthesis preserves manifold structure** — Fréchet mean, not Euclidean average
//! 4. **Homeostasis prevents collapse** — radial field + damped dynamics via exp_map
//! 5. **All operations use f64 internally** — promoted from f32 storage for precision
//! 6. **Every rationale carries an energy seal** — E(τ) certifies epistemological quality
//! 7. **Spectral health monitors structural integrity** — λ₂ detects fragmentation
//! 8. **Innovation is metabolized, not suppressed** — Φ(τ) balances stability vs discovery
//! 9. **Sandbox quarantine tests before committing** — no untested knowledge enters the manifold
//! 10. **Criticality is formally defined** — edge-of-chaos = spectral band ∩ energy variance ∩ bounded velocity
//! 11. **Damping is adaptive** — λ(t) = λ₀ + k·(σ²-σ²_crit)/σ²_crit prevents oscillation AND death

pub mod representation;
pub mod rationale;
pub mod trajectory;
pub mod inference_engine;
pub mod synthesis;
pub mod dialectic;
pub mod feedback_loop;
pub mod homeostasis;
pub mod relevance_decay;
pub mod evolution;
pub mod stability;
pub mod certification;
pub mod spectral;
pub mod discovery;
pub mod innovation;
pub mod sandbox;
pub mod criticality;
pub mod error;

// ── Public re-exports ──

pub use error::AgiError;
pub use rationale::{InferenceType, Rationale};
pub use representation::SynthesisNode;
pub use trajectory::{GeodesicTrajectory, GeodesicCoherenceScore, validate_trajectory};
pub use inference_engine::InferenceEngine;
pub use synthesis::FrechetSynthesizer;
pub use dialectic::DialecticDetector;
pub use feedback_loop::FeedbackLoop;
pub use homeostasis::{HomeostasisGuard, RadialField, DampedHomeostasis, DampingReport};
pub use relevance_decay::RelevanceDecay;
pub use evolution::EvolutionScheduler;

// ── Phase V re-exports ──
pub use stability::{StabilityEvaluator, StabilityReport};
pub use certification::{CertificationLevel, CertificationSeal, certify, certify_default};
pub use spectral::{SpectralMonitor, SpectralHealth, ConnectivityClass, DriftTracker, AdaptiveBand};

// ── Phase VI re-exports ──
pub use discovery::{DiscoveryField, DiscoveryReport};
pub use innovation::{InnovationEvaluator, InnovationReport, AcceptanceDecision};
pub use sandbox::{SandboxEvaluator, SandboxEntry, SandboxVerdict};

// ── Phase VI+ re-exports (Criticality & Metabolic Dynamics) ──
pub use criticality::{CriticalityDetector, CriticalityMetrics, RegimeState};
pub use evolution::{MetabolicInput, CycleReport};
