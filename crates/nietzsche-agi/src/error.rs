// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! AGI error types.

use nietzsche_graph::GraphError;
use nietzsche_hyp_ops::error::HypError;

/// Unified error type for the AGI inference layer.
#[derive(Debug, thiserror::Error)]
pub enum AgiError {
    /// A required node was not found in the graph.
    #[error("node not found: {0}")]
    NodeNotFound(uuid::Uuid),

    /// The embedding for a node was missing (meta exists but no vector).
    #[error("embedding missing for node {0}")]
    EmbeddingMissing(uuid::Uuid),

    /// The trajectory failed GCS validation.
    #[error("trajectory rejected: GCS {gcs:.4} < threshold {threshold:.4} at hop {hop}")]
    TrajectoryRejected {
        gcs: f64,
        threshold: f64,
        hop: usize,
    },

    /// A geodesic coherence check found the path is broken.
    #[error("logical rupture: {reason}")]
    LogicalRupture { reason: String },

    /// Homeostasis check: synthesis point too close to the origin.
    #[error("homeostasis violation: ‖x‖ = {norm:.6} < min_radius {min_radius:.6}")]
    HomeostasisViolation { norm: f64, min_radius: f64 },

    /// Dimension mismatch between vectors.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Empty input where at least one element is required.
    #[error("empty input: {context}")]
    EmptyInput { context: String },

    /// Propagated from nietzsche-graph.
    #[error("graph: {0}")]
    Graph(#[from] GraphError),

    /// Propagated from nietzsche-hyp-ops.
    #[error("hyperbolic: {0}")]
    Hyperbolic(#[from] HypError),
}

pub type AgiResult<T> = Result<T, AgiError>;
