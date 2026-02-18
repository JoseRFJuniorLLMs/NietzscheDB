use thiserror::Error;

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("NQL parse error: {0}")]
    Parse(String),

    #[error("NQL execution error: {0}")]
    Execution(String),

    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),
}

impl From<pest::error::Error<crate::parser::Rule>> for QueryError {
    fn from(e: pest::error::Error<crate::parser::Rule>) -> Self {
        Self::Parse(e.to_string())
    }
}
