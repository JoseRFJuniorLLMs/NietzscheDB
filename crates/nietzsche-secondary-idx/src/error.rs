use thiserror::Error;

/// Errors returned by the secondary index subsystem.
#[derive(Debug, Error)]
pub enum SecondaryIdxError {
    /// The index already exists.
    #[error("index already exists: {0}")]
    IndexAlreadyExists(String),

    /// The referenced index does not exist.
    #[error("index not found: {0}")]
    IndexNotFound(String),

    /// The field path does not resolve to an indexable value.
    #[error("field path not found in node content: {0}")]
    FieldNotFound(String),

    /// The value type does not match the index type.
    #[error("type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    /// A storage-level error from the underlying GraphStorage / RocksDB layer.
    #[error("storage error: {0}")]
    Storage(String),

    /// Serialization / deserialization failure.
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<nietzsche_graph::GraphError> for SecondaryIdxError {
    fn from(e: nietzsche_graph::GraphError) -> Self {
        Self::Storage(e.to_string())
    }
}

impl From<serde_json::Error> for SecondaryIdxError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

impl From<bincode::Error> for SecondaryIdxError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}
