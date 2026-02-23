use std::path::PathBuf;
use dashmap::DashMap;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};

#[derive(Error, Debug)]
pub enum NeuralError {
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Tensor error: {0}")]
    TensorError(String),
}

pub type Result<T> = std::result::Result<T, NeuralError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub path: PathBuf,
    pub version: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

pub struct ModelRegistry {
    sessions: DashMap<String, Arc<Mutex<Session>>>,
    metadata: DashMap<String, ModelMetadata>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
            metadata: DashMap::new(),
        }
    }

    pub fn load_model(&self, meta: ModelMetadata) -> Result<()> {
        let session = Session::builder()?
            .commit_from_file(&meta.path)?;
        
        self.sessions.insert(meta.name.clone(), Arc::new(Mutex::new(session)));
        self.metadata.insert(meta.name.clone(), meta);
        Ok(())
    }

    pub fn get_session(&self, name: &str) -> Result<Arc<Mutex<Session>>> {
        self.sessions.get(name)
            .map(|s| Arc::clone(s.value()))
            .ok_or_else(|| NeuralError::ModelNotFound(name.to_string()))
    }

    pub fn unload_model(&self, name: &str) {
        self.sessions.remove(name);
        self.metadata.remove(name);
    }

    pub fn list_models(&self) -> Vec<ModelMetadata> {
        self.metadata.iter().map(|kv| kv.value().clone()).collect()
    }
}

pub mod tensor;
pub static REGISTRY: Lazy<ModelRegistry> = Lazy::new(ModelRegistry::new);
