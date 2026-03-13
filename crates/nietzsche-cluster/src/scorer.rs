// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Neural Cluster Scorer — evaluates cluster health and recommends actions.
//!
//! ONNX model: `models/cluster_scorer.onnx`
//! Input:  [B, 261] (128D centroid + 128D variance + 5 scalar stats)
//! Output: [B, 3]   (softmax probabilities: [keep, split, merge])

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Recommended action for a cluster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterAction {
    Keep,
    Split,
    Merge,
}

impl ClusterAction {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => ClusterAction::Keep,
            1 => ClusterAction::Split,
            2 => ClusterAction::Merge,
            _ => ClusterAction::Keep,
        }
    }
}

/// Result of cluster scoring.
#[derive(Debug, Clone)]
pub struct ClusterScore {
    pub action: ClusterAction,
    pub keep_prob: f32,
    pub split_prob: f32,
    pub merge_prob: f32,
}

/// Neural cluster scorer for evaluating cluster health.
pub struct ClusterScorerNet {
    model_name: String,
}

impl ClusterScorerNet {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("cluster_scorer.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "cluster_scorer".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 261],
                output_shape: vec![1, 3],
            });
        }
        Self {
            model_name: "cluster_scorer".into(),
        }
    }

    /// Score a cluster given its centroid, variance, and scalar statistics.
    ///
    /// - `centroid`: 128D cluster centroid embedding
    /// - `variance`: 128D per-dimension variance
    /// - `stats`: 5 scalar stats [size_norm, density, avg_weight, diameter_norm, coherence]
    pub fn score(
        &self,
        centroid: &[f32],
        variance: &[f32],
        stats: &[f32; 5],
    ) -> Result<ClusterScore, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("cluster_scorer not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        // Build input: centroid (128) + variance (128) + stats (5) = 261
        let mut input_data = Vec::with_capacity(261);
        input_data.extend_from_slice(centroid);
        input_data.extend_from_slice(variance);
        input_data.extend_from_slice(stats);

        // Pad if centroid/variance are smaller than 128
        while input_data.len() < 261 {
            input_data.push(0.0);
        }

        let input_tensor = ndarray::Array2::from_shape_vec((1, 261), input_data)
            .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let probs: Vec<f32> = output.1.iter().copied().collect();
        let (max_idx, _) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));

        Ok(ClusterScore {
            action: ClusterAction::from_index(max_idx),
            keep_prob: probs.get(0).copied().unwrap_or(0.0),
            split_prob: probs.get(1).copied().unwrap_or(0.0),
            merge_prob: probs.get(2).copied().unwrap_or(0.0),
        })
    }
}
