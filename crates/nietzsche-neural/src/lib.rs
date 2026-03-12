use std::path::PathBuf;
use dashmap::DashMap;
use ort::execution_providers::CUDAExecutionProvider;
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
        let cuda_ep = CUDAExecutionProvider::default();
        let session = Session::builder()?
            .with_execution_providers([cuda_ep.build()])?
            .commit_from_file(&meta.path)?;

        #[cfg(feature = "cuda")]
        tracing::info!(model = %meta.name, "Model loaded with CUDA execution provider (GPU)");
        #[cfg(not(feature = "cuda"))]
        tracing::warn!(model = %meta.name, "Model loaded WITHOUT CUDA (CPU fallback) — build with feature 'cuda' for GPU");

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

    /// Check whether a model is loaded in the registry.
    pub fn has_model(&self, name: &str) -> bool {
        self.sessions.contains_key(name)
    }

    /// Run inference on a model with raw f32 data and return flattened output.
    ///
    /// This is a convenience wrapper that avoids requiring callers to depend
    /// on `ort` or `ndarray` directly.  Provide the model name, input shape
    /// (e.g. `[1, 320]`), and a flat `Vec<f32>` whose length equals the
    /// product of the shape dimensions.
    ///
    /// Returns the first output tensor as a flattened `Vec<f32>`.
    pub fn infer_f32(
        &self,
        model_name: &str,
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let session_arc = self.get_session(model_name)?;
        let mut session = session_arc
            .lock()
            .map_err(|e| NeuralError::TensorError(format!("session lock poisoned: {e}")))?;

        let input_value = ort::value::Value::from_array((input_shape, input_data))
            .map_err(NeuralError::OrtError)?;

        let outputs = session
            .run(ort::inputs![input_value])
            .map_err(NeuralError::OrtError)?;

        let first_output = outputs
            .iter()
            .next()
            .map(|(_, v)| v)
            .ok_or_else(|| NeuralError::TensorError("model produced no outputs".into()))?;

        let (_shape, data) = first_output
            .try_extract_tensor::<f32>()
            .map_err(NeuralError::OrtError)?;

        Ok(data.to_vec())
    }
}

pub mod tensor;
pub mod cluster_scorer;
pub static REGISTRY: Lazy<ModelRegistry> = Lazy::new(ModelRegistry::new);
