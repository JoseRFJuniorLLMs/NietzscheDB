use uuid::Uuid;
use ndarray::{Array2};
use nietzsche_neural::REGISTRY;
use nietzsche_neural::tensor::to_ndarray;
use ort::value::Value;
use crate::{Result, SampledSubgraph, GnnError};

pub struct GnnPrediction {
    pub node_id: Uuid,
    /// Predicted energy or next-state probability
    pub score: f32,
    /// Predicted embedding shift (delta in Poincare space)
    pub embedding_delta: Vec<f32>,
}

pub struct GnnEngine {
    model_name: String,
}

impl GnnEngine {
    pub fn new(model_name: &str) -> Self {
        Self { model_name: model_name.to_string() }
    }

    /// Returns true if the underlying ONNX model is available in the registry.
    pub fn is_loaded(&self) -> bool {
        REGISTRY.get_session(&self.model_name).is_ok()
    }

    pub async fn predict(&self, subgraph: &SampledSubgraph) -> Result<Vec<GnnPrediction>> {
        let session = REGISTRY.get_session(&self.model_name)
            .map_err(|e| GnnError::NeuralError(e))?;

        // Prepare node features tensor [batch, dim]
        let dim = if let Some(first) = subgraph.nodes.first() {
            first.embedding.coords.len()
        } else {
            return Ok(vec![]);
        };

        let mut features = Array2::<f32>::zeros((subgraph.nodes.len(), dim));
        for (idx, node) in subgraph.nodes.iter().enumerate() {
            let coords = node.embedding.coords.iter().map(|&x| x as f32);
            for (d, val) in coords.enumerate() {
                features[[idx, d]] = val;
            }
        }

        // Prepare adjacency tensor [num_edges, 2] or sparse indices
        // For simplicity in this base version, we pass indices as a tensor if the model expects it.
        // Most ONNX GNNs expect: edge_index [2, num_edges]
        let mut edge_index = Array2::<i64>::zeros((2, subgraph.edges.len()));
        for (i, (src, dst, _)) in subgraph.edges.iter().enumerate() {
            edge_index[[0, i]] = *src as i64;
            edge_index[[1, i]] = *dst as i64;
        }

        // Run inference
        let mut session_guard = session.lock().map_err(|_| GnnError::SamplerError("Session lock poisoned".to_string()))?;
        let outputs = session_guard.run(vec![
            ("x", Value::from_array(features).map_err(GnnError::OrtError)?.into_dyn()),
            ("edge_index", Value::from_array(edge_index).map_err(GnnError::OrtError)?.into_dyn()),
        ])?;
        
        // Extract results (assuming first output contains scores)
        let scores_val = outputs.values().next()
            .ok_or_else(|| GnnError::SamplerError("Model produced no outputs".to_string()))?;
        let scores_tensor = to_ndarray(&scores_val)?;
        
        let mut predictions = Vec::with_capacity(subgraph.nodes.len());
        for (idx, node) in subgraph.nodes.iter().enumerate() {
            // Assume scores_tensor is [N, 1] or [N]
            let score = if scores_tensor.ndim() == 1 {
                scores_tensor.get(idx).cloned().unwrap_or(0.0)
            } else if scores_tensor.ndim() == 2 {
                scores_tensor[[idx, 0]]
            } else {
                0.0
            };

            // If the model produces more than 1 output, the second one might be embedding_delta
            let mut embedding_delta = vec![];
            if outputs.len() > 1 {
                if let Some(delta_val) = outputs.values().nth(1) {
                    if let Ok(delta_tensor) = to_ndarray(&delta_val) {
                        // Assume delta_tensor is [N, dim]
                        if delta_tensor.ndim() == 2 && delta_tensor.shape()[0] == subgraph.nodes.len() {
                            let dim = delta_tensor.shape()[1];
                            embedding_delta = (0..dim).map(|d| delta_tensor[[idx, d]]).collect();
                        }
                    }
                }
            }

            predictions.push(GnnPrediction {
                node_id: node.id,
                score,
                embedding_delta,
            });
        }

        Ok(predictions)
    }
}
