// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlError {
    #[error("Neural error: {0}")]
    Neural(#[from] nietzsche_neural::NeuralError),
    
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Environment error: {0}")]
    Env(String),
    
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}
