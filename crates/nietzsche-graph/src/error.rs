use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("node not found: {0}")]
    NodeNotFound(uuid::Uuid),

    #[error("edge not found: {0}")]
    EdgeNotFound(uuid::Uuid),

    #[error("embedding invariant violated: ‖x‖ must be < 1.0, got {0:.6}")]
    InvalidEmbedding(f64),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("storage error: {0}")]
    Storage(String),

    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<bincode::Error> for GraphError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}
