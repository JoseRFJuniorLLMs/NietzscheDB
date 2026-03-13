// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Neural Anomaly Detector — autoencoder for detecting degenerative graph patterns.
//!
//! ONNX model: `models/anomaly_detector.onnx`
//! Input:  [B, 64]  (graph health state)
//! Output: [B, 65]  (reconstructed 64D + 1D anomaly score)
//!
//! Used by Wiederkehr daemon engine to flag unhealthy graph regions.

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Result of anomaly detection on a graph state.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Learned anomaly score in [0, 1]. Higher = more anomalous.
    pub anomaly_score: f32,
    /// Reconstruction error (MSE between input and reconstructed).
    pub reconstruction_error: f32,
    /// Combined score: learned + reconstruction-based.
    pub combined_score: f32,
    /// Whether this state is flagged as anomalous (combined > threshold).
    pub is_anomalous: bool,
}

/// Neural anomaly detector for graph health monitoring.
pub struct AnomalyDetectorNet {
    model_name: String,
    threshold: f32,
}

impl AnomalyDetectorNet {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("anomaly_detector.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "anomaly_detector".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 64],
                output_shape: vec![1, 65],
            });
        }
        Self {
            model_name: "anomaly_detector".into(),
            threshold: 0.5,
        }
    }

    /// Set the anomaly detection threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect anomalies in a graph health state.
    ///
    /// `state`: 64D graph health vector (same format as PPO/ValueNetwork state).
    pub fn detect(&self, state: &[f32]) -> Result<AnomalyResult, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("anomaly_detector not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let mut input_data = state.to_vec();
        input_data.resize(64, 0.0);

        let input_tensor = ndarray::Array2::from_shape_vec((1, 64), input_data.clone())
            .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let raw: Vec<f32> = output.1.iter().copied().collect();

        // First 64 values = reconstruction, last 1 = anomaly score
        let reconstructed = &raw[..64.min(raw.len())];
        let anomaly_score = raw.get(64).copied().unwrap_or(0.0);

        // Compute reconstruction error
        let reconstruction_error: f32 = input_data
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 64.0;

        // Combined: learned + reconstruction-based
        let recon_sigmoid = 1.0 / (1.0 + (-reconstruction_error * 10.0).exp());
        let combined_score = anomaly_score * 0.5 + recon_sigmoid * 0.5;

        Ok(AnomalyResult {
            anomaly_score,
            reconstruction_error,
            combined_score,
            is_anomalous: combined_score > self.threshold,
        })
    }

    /// Batch detect anomalies.
    pub fn detect_batch(&self, states: &[&[f32]]) -> Result<Vec<AnomalyResult>, String> {
        states.iter().map(|s| self.detect(s)).collect()
    }
}
