// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
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
