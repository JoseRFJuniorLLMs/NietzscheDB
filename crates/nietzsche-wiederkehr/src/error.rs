use thiserror::Error;

#[derive(Debug, Error)]
pub enum DaemonError {
    #[error("daemon not found: {0}")]
    NotFound(String),

    #[error("daemon already exists: {0}")]
    AlreadyExists(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("evaluation error: {0}")]
    Eval(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
