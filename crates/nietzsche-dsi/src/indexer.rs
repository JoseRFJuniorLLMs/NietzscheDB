// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! DSI Indexer — encodes node embeddings into hierarchical VQ codes via the `vqvae` ONNX model.
//!
//! The `vqvae` model:
//!   Input:  [1, 3072]  (continuous embedding, zero-padded from collection dim)
//!   Output: [1, 3076]  (3072 reconstruction floats + 4 VQ code indices)
//!
//! The last 4 values of the output are the hierarchical VQ codes (one per level).
//! These codes form a [`SemanticId`] that is stored in RocksDB for prefix-scan retrieval.

use crate::{Result, DsiError, SemanticId};
use nietzsche_graph::GraphStorage;
use nietzsche_neural::REGISTRY;
use uuid::Uuid;

/// Expected input dimension for the vqvae ONNX model.
const VQVAE_INPUT_DIM: usize = 3072;
/// Number of reconstruction floats in the output (before the VQ code indices).
const VQVAE_RECONSTRUCTION_DIM: usize = 3072;
/// Number of hierarchical VQ code levels produced by the model.
pub const VQVAE_NUM_LEVELS: usize = 4;
/// Total output dimension: reconstruction + code indices.
const VQVAE_OUTPUT_DIM: usize = VQVAE_RECONSTRUCTION_DIM + VQVAE_NUM_LEVELS; // 3076

/// Name of the vqvae model in the neural registry.
const VQVAE_MODEL_NAME: &str = "vqvae";

pub struct DsiIndexer {
    num_levels: usize,
}

impl DsiIndexer {
    /// Create a new DSI indexer.
    ///
    /// `num_levels` is the number of hierarchical VQ code levels to use (max 4).
    /// The `vqvae` ONNX model must be loaded in the neural [`REGISTRY`] before indexing.
    pub fn new(num_levels: usize) -> Self {
        Self {
            num_levels: num_levels.min(VQVAE_NUM_LEVELS).max(1),
        }
    }

    /// Check whether the vqvae model is loaded and available.
    pub fn is_available(&self) -> bool {
        REGISTRY.get_session(VQVAE_MODEL_NAME).is_ok()
    }

    /// Encode a raw embedding (any dimension) into `num_levels` VQ codes using the vqvae ONNX model.
    ///
    /// The embedding is zero-padded to 3072D before being fed to the model.
    /// Returns the VQ code indices (one per level).
    pub fn encode_embedding(&self, coords: &[f32]) -> Result<Vec<u16>> {
        let session = REGISTRY
            .get_session(VQVAE_MODEL_NAME)
            .map_err(|e| DsiError::Internal(format!("vqvae model not loaded: {e}")))?;

        let mut guard = session
            .lock()
            .map_err(|e| DsiError::Internal(format!("vqvae session lock: {e}")))?;

        // Zero-pad the embedding to VQVAE_INPUT_DIM (3072).
        let mut input_data = vec![0.0f32; VQVAE_INPUT_DIM];
        let copy_len = coords.len().min(VQVAE_INPUT_DIM);
        input_data[..copy_len].copy_from_slice(&coords[..copy_len]);

        let input_tensor =
            ndarray::Array2::from_shape_vec((1, VQVAE_INPUT_DIM), input_data)
                .map_err(|e| DsiError::Internal(format!("vqvae input tensor: {e}")))?;

        let outputs = guard
            .run(
                ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor)
                    .map_err(|e| DsiError::Internal(format!("vqvae tensor alloc: {e}")))?],
            )
            .map_err(|e| DsiError::Internal(format!("vqvae inference: {e}")))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DsiError::Internal(format!("vqvae extract: {e}")))?;

        let raw: Vec<f32> = output.1.iter().copied().collect();

        if raw.len() < VQVAE_OUTPUT_DIM {
            return Err(DsiError::Internal(format!(
                "vqvae output too short: expected {VQVAE_OUTPUT_DIM}, got {}",
                raw.len()
            )));
        }

        // The last VQVAE_NUM_LEVELS values are the VQ code indices.
        let code_start = VQVAE_RECONSTRUCTION_DIM;
        let codes: Vec<u16> = raw[code_start..code_start + self.num_levels]
            .iter()
            .map(|&v| v.round() as u16)
            .collect();

        Ok(codes)
    }

    /// Generate a hierarchical [`SemanticId`] for a node and persist it to storage.
    ///
    /// Reads the node's Poincare embedding from `storage`, encodes it via the vqvae ONNX model,
    /// and writes the resulting semantic ID to both the forward and reverse DSI indices.
    pub fn index_node(
        &self,
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Result<SemanticId> {
        let embedding = storage
            .get_embedding(node_id)?
            .ok_or_else(|| nietzsche_graph::error::GraphError::NodeNotFound(*node_id))?;

        let codes = self.encode_embedding(&embedding.coords)?;
        let id = SemanticId::new(codes);

        // Persist to RocksDB (forward: node_id -> semantic_id, reverse: semantic_id -> node_id).
        let semantic_bytes = id.to_prefix_bytes();
        storage.put_dsi_id(node_id, &semantic_bytes)?;

        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexer_creation() {
        let indexer = DsiIndexer::new(4);
        assert_eq!(indexer.num_levels, 4);

        // Clamped to max 4
        let indexer = DsiIndexer::new(10);
        assert_eq!(indexer.num_levels, 4);

        // Clamped to min 1
        let indexer = DsiIndexer::new(0);
        assert_eq!(indexer.num_levels, 1);
    }
}
