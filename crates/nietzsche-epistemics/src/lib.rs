//! `nietzsche-epistemics` — Epistemic quality metrics for NietzscheDB.
//!
//! Provides quantitative measures of knowledge graph quality:
//!
//! - **Hierarchy consistency**: edges respect Poincaré ball depth ordering
//! - **Coherence**: connected nodes are semantically proportionate
//! - **Coverage**: embedding space utilization
//! - **Redundancy**: path diversity between connected components
//! - **Novelty**: new mutations add genuinely new information
//!
//! Used by NietzscheLab (Phase 27 — Evolution) to score mutations
//! and decide accept/reject in the epistemic evolution loop.

pub mod hierarchy;
pub mod coherence;
pub mod coverage;
pub mod redundancy;
pub mod novelty;
pub mod scorer;

pub use scorer::{
    EpistemicScore, EpistemicDelta, ScoreWeights,
    evaluate_subgraph, evaluate_mutation,
};
