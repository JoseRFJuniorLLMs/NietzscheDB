use thiserror::Error;

#[derive(Debug, Error)]
pub enum NarrativeError {
    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),

    #[error("no data available for narrative generation")]
    NoData,

    #[error("narrative generation error: {0}")]
    Generation(String),

    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}
