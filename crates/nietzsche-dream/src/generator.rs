// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Neural Dream Generator — generates new node embeddings from seed + noise.
//!
//! ONNX model: `models/dream_generator.onnx`
//! Input:  [B, 192] (128D seed embedding + 64D noise)
//! Output: [B, 128] (generated embedding, Euclidean)

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Neural dream generator that creates new embeddings by diffusing from a seed.
pub struct DreamGeneratorNet {
    model_name: String,
}

impl DreamGeneratorNet {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("dream_generator.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "dream_generator".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 192],
                output_shape: vec![1, 128],
            });
        }
        Self {
            model_name: "dream_generator".into(),
        }
    }

    /// Generate a new embedding from a seed embedding and noise vector.
    ///
    /// - `seed`: 128D anchor embedding (the node being "dreamed from")
    /// - `noise`: 64D noise controlling creativity (low = nearby, high = far)
    ///
    /// Returns 128D generated embedding (Euclidean, needs exp_map_zero for Poincaré).
    pub fn generate(&self, seed: &[f32], noise: &[f32]) -> Result<Vec<f32>, String> {
        if seed.len() != 128 {
            return Err(format!("seed must be 128D, got {}", seed.len()));
        }
        if noise.len() != 64 {
            return Err(format!("noise must be 64D, got {}", noise.len()));
        }

        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("dream_generator not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        // Concatenate seed + noise → [1, 192]
        let mut input_data = Vec::with_capacity(192);
        input_data.extend_from_slice(seed);
        input_data.extend_from_slice(noise);

        let input_tensor = ndarray::Array2::from_shape_vec((1, 192), input_data)
            .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        Ok(output.1.iter().copied().collect())
    }

    /// Generate with a specified creativity level (0.0 = deterministic, 1.0 = wild).
    pub fn dream(&self, seed: &[f32], creativity: f32) -> Result<Vec<f32>, String> {
        let noise: Vec<f32> = (0..64)
            .map(|_| rand::random::<f32>() * creativity * 2.0 - creativity)
            .collect();
        self.generate(seed, &noise)
    }
}
