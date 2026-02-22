// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Error types for the Swartz SQL layer.

use nietzsche_graph::GraphError;

#[derive(Debug, thiserror::Error)]
pub enum SwartzError {
    #[error("GlueSQL error: {0}")]
    Glue(String),

    #[error("Storage error: {0}")]
    Storage(#[from] GraphError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),
}

impl From<serde_json::Error> for SwartzError {
    fn from(e: serde_json::Error) -> Self {
        SwartzError::Serialization(e.to_string())
    }
}
