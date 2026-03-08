//! Neural Edge Predictor — link prediction for graph topology optimization.
//!
//! ONNX model: `models/edge_predictor.onnx`
//! Input:  [B, 256] (128D node_a ⊕ 128D node_b)
//! Output: [B, 1]   (edge probability, sigmoid)
//!
//! Used by the L-System to decide where to create new connections instead
//! of random edge creation.

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Neural link predictor: given two node embeddings, predict edge probability.
pub struct EdgePredictorNet {
    model_name: String,
}

impl EdgePredictorNet {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("edge_predictor.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "edge_predictor".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 256],
                output_shape: vec![1, 1],
            });
        }
        Self {
            model_name: "edge_predictor".into(),
        }
    }

    /// Predict the probability that an edge should exist between two nodes.
    ///
    /// - `embedding_a`: 128D embedding of node A
    /// - `embedding_b`: 128D embedding of node B
    ///
    /// Returns probability in [0, 1].
    pub fn predict(&self, embedding_a: &[f32], embedding_b: &[f32]) -> Result<f32, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("edge_predictor not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let mut input_data = Vec::with_capacity(256);
        input_data.extend_from_slice(embedding_a);
        input_data.extend_from_slice(embedding_b);

        // Pad if embeddings are smaller than 128D
        while input_data.len() < 256 {
            input_data.push(0.0);
        }

        let input_array = ndarray::Array2::from_shape_vec((1, 256), input_data)
            .map_err(|e| format!("tensor: {e}"))?;
        let input_value = ort::value::Tensor::from_array(input_array)
            .map_err(|e| format!("value: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => input_value])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        Ok(output.1.iter().next().copied().unwrap_or(0.0))
    }

    /// Batch predict edge probabilities for multiple node pairs.
    ///
    /// `pairs`: list of (embedding_a, embedding_b) tuples.
    pub fn predict_batch(
        &self,
        pairs: &[(&[f32], &[f32])],
    ) -> Result<Vec<f32>, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("edge_predictor not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let batch_size = pairs.len();
        let mut input_data = Vec::with_capacity(batch_size * 256);
        for (a, b) in pairs {
            let mut pair = Vec::with_capacity(256);
            pair.extend_from_slice(a);
            pair.extend_from_slice(b);
            while pair.len() < 256 {
                pair.push(0.0);
            }
            input_data.extend_from_slice(&pair);
        }

        let input_array =
            ndarray::Array2::from_shape_vec((batch_size, 256), input_data)
                .map_err(|e| format!("tensor: {e}"))?;
        let input_value = ort::value::Tensor::from_array(input_array)
            .map_err(|e| format!("value: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => input_value])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        Ok(output.1.iter().copied().collect())
    }
}
