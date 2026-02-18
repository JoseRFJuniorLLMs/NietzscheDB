//! Error types for hyperbolic operations.

/// Errors that can occur during hyperbolic geometry operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HypError {
    /// A vector was outside the open unit ball (‖x‖ ≥ 1.0).
    #[error("vector outside Poincaré ball: ‖x‖ = {norm:.6} ≥ 1.0")]
    OutsideBall { norm: f64 },

    /// An operation received an empty input where at least one element is required.
    #[error("empty input: at least one point is required")]
    EmptyInput,

    /// Two inputs had incompatible dimensions.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}
