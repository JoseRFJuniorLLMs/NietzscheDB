// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Neural Value Network for MCTS — evaluates graph state quality.
//!
//! ONNX model: `models/value_network.onnx`
//! Input:  [B, 64] (Krylov subspace projection of graph state)
//! Output: [B, 1]  (value score: 0=bad, 1=excellent)
//!
//! Used by MctsAdvisor to score leaf nodes during tree search,
//! replacing random rollouts with learned evaluation.

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Neural value network for MCTS state evaluation.
pub struct ValueNetworkInference {
    model_name: String,
}

impl ValueNetworkInference {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("value_network.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "value_network".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 64],
                output_shape: vec![1, 1],
            });
        }
        Self {
            model_name: "value_network".into(),
        }
    }

    /// Evaluate a graph state and return a value score in [0, 1].
    ///
    /// `state`: 64D Krylov subspace projection of graph health.
    pub fn evaluate(&self, state: &[f32]) -> Result<f32, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("value_network not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let mut input_data = state.to_vec();
        input_data.resize(64, 0.0);

        let input_tensor = ndarray::Array2::from_shape_vec((1, 64), input_data)
            .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        Ok(output.1.iter().next().copied().unwrap_or(0.5))
    }

    /// Batch evaluate multiple graph states.
    pub fn evaluate_batch(&self, states: &[&[f32]]) -> Result<Vec<f32>, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("value_network not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let batch_size = states.len();
        let mut input_data = Vec::with_capacity(batch_size * 64);
        for s in states {
            let mut state = s.to_vec();
            state.resize(64, 0.0);
            input_data.extend_from_slice(&state);
        }

        let input_tensor =
            ndarray::Array2::from_shape_vec((batch_size, 64), input_data)
                .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        Ok(output.1.iter().copied().collect())
    }
}
