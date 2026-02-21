use thiserror::Error;

#[derive(Debug, Error)]
pub enum NamedVectorError {
    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("named vector '{name}' not found for node {node_id}")]
    NotFound { node_id: uuid::Uuid, name: String },
    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}
