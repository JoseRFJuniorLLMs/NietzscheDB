use thiserror::Error;

/// Structured NQL error types.
#[derive(Debug, Error)]
pub enum QueryError {
    // ── Parse errors ──────────────────────────────────────

    #[error("NQL parse error: {0}")]
    Parse(String),

    // ── Execution errors ──────────────────────────────────

    #[error("NQL execution error: {0}")]
    Execution(String),

    #[error("unknown alias '{alias}' — not bound by MATCH pattern")]
    UnknownAlias { alias: String },

    #[error("unknown node field '{field}' — available: id, energy, depth, hausdorff_local, lsystem_generation, created_at, node_type")]
    UnknownField { field: String },

    #[error("parameter ${name} not provided")]
    ParamNotFound { name: String },

    #[error("parameter ${name}: expected {expected}, got {got}")]
    ParamTypeMismatch { name: String, expected: &'static str, got: String },

    #[error("type mismatch in {context}: cannot compare {left_type} with {right_type}")]
    TypeMismatch { context: String, left_type: &'static str, right_type: &'static str },

    #[error("{op} requires string operands")]
    StringOpTypeMismatch { op: String },

    // ── Storage errors ────────────────────────────────────

    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),
}

impl From<pest::error::Error<crate::parser::Rule>> for QueryError {
    fn from(e: pest::error::Error<crate::parser::Rule>) -> Self {
        Self::Parse(e.to_string())
    }
}
