pub mod tree;
pub mod advisor;

pub use tree::{MctsTree, MctsNode, MctsConfig};
pub use advisor::{MctsAdvisor, AdvisorIntent};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MctsError {
    #[error("Graph error: {0}")]
    GraphError(#[from] nietzsche_graph::GraphError),
    #[error("Neural error: {0}")]
    NeuralError(#[from] nietzsche_neural::NeuralError),
    #[error("GNN error: {0}")]
    GnnError(#[from] nietzsche_gnn::GnnError),
    #[error("MCTS error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, MctsError>;
