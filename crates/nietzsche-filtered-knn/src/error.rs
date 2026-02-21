//! Error types for the filtered KNN module.

use thiserror::Error;

/// Errors that can occur during filtered KNN operations.
#[derive(Debug, Error)]
pub enum FilteredKnnError {
    /// An error originating from the graph storage layer.
    #[error("graph storage error: {0}")]
    Storage(#[from] nietzsche_graph::GraphError),

    /// The query vector dimension does not match the indexed vectors.
    #[error("dimension mismatch: query has {query} dims, storage has {storage} dims")]
    DimensionMismatch { query: usize, storage: usize },

    /// The k parameter is zero, which is invalid for KNN search.
    #[error("k must be > 0")]
    InvalidK,

    /// A node referenced in the filter could not be found.
    #[error("node not found: {0}")]
    NodeNotFound(uuid::Uuid),

    /// A JSON path expression failed to resolve.
    #[error("invalid JSON path: {0}")]
    InvalidJsonPath(String),

    /// Serialization / deserialization error.
    #[error("serde error: {0}")]
    Serde(String),
}
