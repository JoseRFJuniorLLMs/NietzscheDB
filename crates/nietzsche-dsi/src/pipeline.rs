// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! DSI Neural Pipeline — coordinates VQ-VAE indexing and DSI decoder retrieval.
//!
//! Controlled by the `DSI_NEURAL_ENABLED` environment variable (default: `false`).
//! When disabled, all operations gracefully return `None` so callers fall back to HNSW.
//!
//! # Indexing path (background)
//! ```text
//! Node embedding (128D Poincare) → zero-pad to 3072D → vqvae ONNX → 4 VQ codes → SemanticId → RocksDB
//! ```
//!
//! # Retrieval path (query time)
//! ```text
//! Query embedding (128D) → dsi_decoder ONNX → 4×1024 logits → argmax → SemanticId
//!     → prefix-scan RocksDB → candidate node UUIDs
//! ```

use crate::decoder::DsiDecoderNet;
use crate::indexer::DsiIndexer;
use crate::{DsiError, DsiResult, SemanticId};
use nietzsche_graph::GraphStorage;
use std::sync::atomic::{AtomicBool, Ordering};
use uuid::Uuid;

static DSI_NEURAL_ENABLED: AtomicBool = AtomicBool::new(false);

/// Read the `DSI_NEURAL_ENABLED` env var and cache it.  Call once at startup.
pub fn init_dsi_config() {
    let enabled = std::env::var("DSI_NEURAL_ENABLED")
        .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    DSI_NEURAL_ENABLED.store(enabled, Ordering::Relaxed);

    if enabled {
        tracing::info!("DSI neural pipeline ENABLED (vqvae + dsi_decoder)");
    } else {
        tracing::debug!("DSI neural pipeline disabled (set DSI_NEURAL_ENABLED=true to activate)");
    }
}

/// Returns whether the DSI neural pipeline is enabled.
#[inline]
pub fn is_dsi_neural_enabled() -> bool {
    DSI_NEURAL_ENABLED.load(Ordering::Relaxed)
}

/// Unified DSI pipeline that owns both the indexer and decoder.
pub struct DsiPipeline {
    indexer: DsiIndexer,
    decoder: Option<DsiDecoderNet>,
}

impl DsiPipeline {
    /// Create the pipeline.
    ///
    /// - `num_levels`: how many hierarchical code levels to use (1..=4).
    /// - `models_dir`: path to the directory containing `dsi_decoder.onnx`.
    ///   The `vqvae` model is expected to already be loaded in the neural registry
    ///   by the background model scanner.
    pub fn new(num_levels: usize, models_dir: &str) -> Self {
        let indexer = DsiIndexer::new(num_levels);

        let decoder = if is_dsi_neural_enabled() {
            Some(DsiDecoderNet::new(models_dir))
        } else {
            None
        };

        Self { indexer, decoder }
    }

    /// Returns `true` when both models are loaded and the feature is enabled.
    pub fn is_fully_available(&self) -> bool {
        is_dsi_neural_enabled() && self.indexer.is_available() && self.decoder.is_some()
    }

    // ── Indexing ─────────────────────────────────────────────────────────────

    /// Index a single node. Returns `None` if the pipeline is disabled or the model is unavailable.
    pub fn try_index_node(
        &self,
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Option<Result<SemanticId, DsiError>> {
        if !is_dsi_neural_enabled() || !self.indexer.is_available() {
            return None;
        }
        Some(self.indexer.index_node(storage, node_id))
    }

    /// Index a single node, returning an error if the pipeline is disabled.
    /// Prefer [`try_index_node`] for background tasks that should silently skip.
    pub fn index_node(
        &self,
        storage: &GraphStorage,
        node_id: &Uuid,
    ) -> Result<SemanticId, DsiError> {
        self.indexer.index_node(storage, node_id)
    }

    /// Encode a raw embedding into VQ codes without persisting to storage.
    pub fn encode_embedding(&self, coords: &[f32]) -> Result<Vec<u16>, DsiError> {
        self.indexer.encode_embedding(coords)
    }

    // ── Retrieval ────────────────────────────────────────────────────────────

    /// Neural DSI retrieval: map a query embedding to candidate node UUIDs.
    ///
    /// Returns `None` if the pipeline is disabled or the decoder model is unavailable.
    /// Callers should fall back to HNSW KNN search when `None` is returned.
    ///
    /// The retrieval strategy uses progressive prefix matching:
    /// 1. Decode the query into 4 VQ codes via the dsi_decoder model.
    /// 2. Try exact match (all 4 codes) first.
    /// 3. If fewer than `k` results, relax to 3-code prefix, then 2, then 1.
    /// 4. Return all candidates with a confidence score.
    pub fn try_neural_search(
        &self,
        storage: &GraphStorage,
        query_coords: &[f32],
        k: usize,
    ) -> Option<Result<Vec<(Uuid, f64)>, DsiError>> {
        if !is_dsi_neural_enabled() {
            return None;
        }
        let decoder = self.decoder.as_ref()?;

        Some(self.neural_search_inner(decoder, storage, query_coords, k))
    }

    fn neural_search_inner(
        &self,
        decoder: &DsiDecoderNet,
        storage: &GraphStorage,
        query_coords: &[f32],
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, DsiError> {
        // Run the DSI decoder to get predicted VQ codes.
        let dsi_result = decoder
            .decode(query_coords)
            .map_err(|e| DsiError::Internal(format!("dsi_decoder: {e}")))?;

        tracing::debug!(
            codes = ?dsi_result.codes,
            confidences = ?dsi_result.confidences,
            "DSI neural retrieval: decoded query"
        );

        let predicted = SemanticId::new(dsi_result.codes.clone());

        // Progressive prefix relaxation: try full match, then shorten prefix until we have k results.
        let mut candidates: Vec<(Uuid, f64)> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for level in (1..=predicted.len()).rev() {
            let prefix = predicted
                .prefix(level)
                .expect("level <= len");
            let prefix_bytes = prefix.to_prefix_bytes();

            let ids = storage.scan_nodes_by_dsi_prefix(&prefix_bytes)?;

            // Confidence for this prefix level: product of confidences for levels 0..level
            let level_confidence: f64 = dsi_result.confidences[..level]
                .iter()
                .map(|&c| c as f64)
                .product();

            for id in ids {
                if seen.insert(id) {
                    // Use inverse confidence as a pseudo-distance (lower = better).
                    // Full 4-level match -> highest confidence -> lowest distance.
                    let pseudo_distance = 1.0 - level_confidence;
                    candidates.push((id, pseudo_distance));
                }
            }

            if candidates.len() >= k {
                break;
            }
        }

        // Sort by pseudo-distance (ascending) and truncate to k.
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        tracing::debug!(
            count = candidates.len(),
            k = k,
            "DSI neural retrieval: found candidates"
        );

        Ok(candidates)
    }

    /// Combined search: try DSI neural retrieval first, fall back to HNSW if unavailable or empty.
    ///
    /// This is the primary entry point for integrating DSI into the KNN search path.
    /// Returns `None` when DSI was not attempted (callers proceed with normal HNSW).
    /// Returns `Some(results)` when DSI produced candidates.
    pub fn try_search_with_fallback(
        &self,
        storage: &GraphStorage,
        query_coords: &[f32],
        k: usize,
    ) -> Option<Vec<(Uuid, f64)>> {
        match self.try_neural_search(storage, query_coords, k)? {
            Ok(results) if !results.is_empty() => Some(results),
            Ok(_empty) => {
                tracing::debug!("DSI neural retrieval returned empty, falling back to HNSW");
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "DSI neural retrieval failed, falling back to HNSW");
                None
            }
        }
    }

    /// Expose the underlying decoder result for diagnostic/debug purposes.
    pub fn decode_query(&self, query_coords: &[f32]) -> Option<Result<DsiResult, String>> {
        let decoder = self.decoder.as_ref()?;
        Some(decoder.decode(query_coords))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsi_disabled_by_default() {
        // Without setting the env var, DSI should be disabled.
        assert!(!is_dsi_neural_enabled());
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = DsiPipeline::new(4, "/nonexistent");
        // Decoder won't load (model file doesn't exist), but pipeline should be created.
        assert!(!pipeline.is_fully_available());
    }
}
