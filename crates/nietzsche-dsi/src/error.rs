use thiserror::Error;

#[derive(Debug, Error)]
pub enum DsiError {
    #[error("Graph error: {0}")]
    Graph(#[from] nietzsche_graph::error::GraphError),
    
    #[error("VQ-VAE error: {0}")]
    VqVae(#[from] nietzsche_vqvae::VqError),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Semantic ID mismatch: {0}")]
    IdMismatch(String),
}
