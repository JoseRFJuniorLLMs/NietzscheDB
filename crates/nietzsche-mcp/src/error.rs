//! MCP error types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),

    #[error("query error: {0}")]
    Query(#[from] nietzsche_query::QueryError),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("tool not found: {0}")]
    ToolNotFound(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
