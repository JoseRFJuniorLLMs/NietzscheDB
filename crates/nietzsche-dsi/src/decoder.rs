//! Neural DSI Decoder — maps query embeddings to hierarchical document IDs.
//!
//! ONNX model: `models/dsi_decoder.onnx`
//! Input:  [B, 128]        (query embedding)
//! Output: [B, 4, 1024]    (logits per hierarchy level)
//!
//! At inference: argmax each level → 4 VQ code indices → node address.

use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;

/// Result of DSI neural retrieval.
#[derive(Debug, Clone)]
pub struct DsiResult {
    /// Hierarchical code sequence (one index per level).
    pub codes: Vec<u16>,
    /// Confidence per level (max softmax probability).
    pub confidences: Vec<f32>,
}

/// Neural DSI decoder for O(1) document retrieval.
pub struct DsiDecoderNet {
    model_name: String,
    num_levels: usize,
    codebook_size: usize,
}

impl DsiDecoderNet {
    pub fn new(models_dir: &str) -> Self {
        let path = PathBuf::from(models_dir).join("dsi_decoder.onnx");
        if path.exists() {
            let _ = REGISTRY.load_model(ModelMetadata {
                name: "dsi_decoder".into(),
                path,
                version: "1.0".into(),
                input_shape: vec![1, 128],
                output_shape: vec![1, 4, 1024],
            });
        }
        Self {
            model_name: "dsi_decoder".into(),
            num_levels: 4,
            codebook_size: 1024,
        }
    }

    /// Decode a query embedding into hierarchical VQ codes.
    ///
    /// `query`: 128D query embedding (Krylov-compressed).
    pub fn decode(&self, query: &[f32]) -> Result<DsiResult, String> {
        let session = REGISTRY
            .get_session(&self.model_name)
            .map_err(|e| format!("dsi_decoder not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let mut input_data = query.to_vec();
        input_data.resize(128, 0.0);

        let input_tensor = ndarray::Array2::from_shape_vec((1, 128), input_data)
            .map_err(|e| format!("tensor: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        // Output shape: [1, num_levels, codebook_size]
        let raw: Vec<f32> = output.1.iter().copied().collect();

        let mut codes = Vec::with_capacity(self.num_levels);
        let mut confidences = Vec::with_capacity(self.num_levels);

        for level in 0..self.num_levels {
            let start = level * self.codebook_size;
            let end = start + self.codebook_size;
            let logits = &raw[start..end.min(raw.len())];

            // Argmax
            let (best_idx, &best_val) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((0, &0.0));

            // Softmax confidence for the best code
            let max_val = best_val;
            let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
            let confidence = 1.0 / exp_sum;

            codes.push(best_idx as u16);
            confidences.push(confidence);
        }

        Ok(DsiResult { codes, confidences })
    }
}
