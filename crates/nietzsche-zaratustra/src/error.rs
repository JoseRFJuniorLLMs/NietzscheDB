//! Error type for the Zaratustra engine.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ZaratustraError {
    #[error("graph operation failed: {0}")]
    Graph(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
