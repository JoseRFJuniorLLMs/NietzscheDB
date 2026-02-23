pub mod sampler;
pub mod engine;

pub use sampler::{NeighborSampler, SampledSubgraph};
pub use engine::{GnnEngine, GnnPrediction};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum GnnError {
    #[error("Graph error: {0}")]
    GraphError(#[from] nietzsche_graph::GraphError),
    #[error("Neural error: {0}")]
    NeuralError(#[from] nietzsche_neural::NeuralError),
    #[error("ORT error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Sampler error: {0}")]
    SamplerError(String),
}

pub type Result<T> = std::result::Result<T, GnnError>;
