//! # nietzsche-experiment
//!
//! Link Prediction experiment for validating Topological Generative Capacity (TGC).
//!
//! ## Hypothesis
//!
//! If the dialectical tension / L-System growth genuinely improves the topological
//! organisation of the hyperbolic embedding space, then:
//!
//! - **Normal mode** (TGC active): AUC should climb over cycles.
//! - **TGC off**: AUC should stagnate (no topological refinement).
//! - **TGC inverted** (penalize expansion): AUC should collapse — possibly below 0.5.
//!
//! ## Hard Negative Sampling
//!
//! Random negatives in hyperbolic space are trivially separable (exponential volume
//! growth pushes random pairs to the boundary). To defeat this, we generate:
//!
//! 1. **Random negatives**: baseline global separability check.
//! 2. **Hard negatives** (2-hop non-neighbors): test local topological coherence.
//!
//! ## AUC computation
//!
//! Wilcoxon-Mann-Whitney U statistic, O(N log N) via rank-sum.

pub mod auc;
pub mod sampling;
pub mod telemetry;
