pub mod sampler;
pub mod engine;
pub mod edge_predictor;

pub use sampler::{NeighborSampler, SampledSubgraph};
pub use engine::{GnnEngine, GnnPrediction};
pub use edge_predictor::EdgePredictorNet;

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
