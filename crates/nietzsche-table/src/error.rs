use thiserror::Error;

/// Errors that can occur during table operations.
#[derive(Debug, Error)]
pub enum TableError {
    /// An error originating from the underlying SQLite database.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// The provided schema is invalid (e.g., no columns, duplicate names).
    #[error("Invalid schema: {0}")]
    InvalidSchema(String),

    /// The requested table does not exist.
    #[error("Table not found: {0}")]
    TableNotFound(String),

    /// An error occurred during JSON serialization or deserialization.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}
