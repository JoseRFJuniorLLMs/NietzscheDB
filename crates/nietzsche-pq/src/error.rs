use thiserror::Error;

#[derive(Debug, Error)]
pub enum PQError {
    #[error("dimension {dim} not divisible by {m} sub-vectors")]
    DimensionMismatch { dim: usize, m: usize },
    #[error("not enough training vectors: need at least {need}, got {got}")]
    InsufficientTraining { need: usize, got: usize },
    #[error("codebook not trained")]
    NotTrained,
    #[error("serialization error: {0}")]
    Serialization(String),
}
