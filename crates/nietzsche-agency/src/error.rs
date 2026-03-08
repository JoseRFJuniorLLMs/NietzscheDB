use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgencyError {
    #[error("graph error: {0}")]
    Graph(#[from] nietzsche_graph::GraphError),

    #[error("diffusion error: {0}")]
    Diffusion(#[from] nietzsche_pregel::DiffusionError),

    #[error("centroid guardian: {0}")]
    Guardian(#[from] crate::centroid_guardian::GuardianError),

    #[error("agency: {0}")]
    Internal(String),
}
