use thiserror::Error;

#[derive(Debug, Error)]
pub enum DreamError {
    #[error("dream not found: {0}")]
    NotFound(String),

    #[error("dream already applied: {0}")]
    AlreadyApplied(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),

    #[error("neural error: {0}")]
    Neural(#[from] nietzsche_neural::NeuralError),

    #[error("ort error: {0}")]
    Ort(#[from] ort::Error),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("internal error: {0}")]
    Internal(String),
}
