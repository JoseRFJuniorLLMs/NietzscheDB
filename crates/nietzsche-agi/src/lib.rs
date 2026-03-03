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
//! - [`relevance_decay::RelevanceDecay`] — frequency-based weight adjustment
//! - [`evolution::EvolutionScheduler`] — autonomous evolution orchestrator
//!
//! # Design Principles
//!
//! 1. **Every inference carries a Rationale** — no black-box reasoning
//! 2. **GCS validates every hop** — broken geodesics are rejected (LogicalRupture)
//! 3. **Synthesis preserves manifold structure** — Fréchet mean, not Euclidean average
//! 4. **Homeostasis prevents collapse** — synthesis nodes can't converge at origin
//! 5. **All operations use f64 internally** — promoted from f32 storage for precision

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
pub use homeostasis::HomeostasisGuard;
pub use relevance_decay::RelevanceDecay;
pub use evolution::EvolutionScheduler;
