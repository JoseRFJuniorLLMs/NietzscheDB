// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VqError {
    #[error("Neural error: {0}")]
    Neural(#[from] nietzsche_neural::NeuralError),
    
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Codebook error: {0}")]
    Codebook(String),
}
