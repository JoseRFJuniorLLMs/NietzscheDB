// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Neural encoder implementations backed by ONNX models.
//!
//! These replace the passthrough encoders with learned feature extractors:
//! - `ImageNeuralEncoder`: CNN 64x64 RGB → 128D latent → Poincaré
//! - `AudioNeuralEncoder`: Conv mel-spectrogram [1,64,32] → 128D latent → Poincaré

use crate::types::LatentVector;
use crate::encoder::SensoryEncoder;
use nietzsche_neural::{ModelMetadata, REGISTRY};
use std::path::PathBuf;
use std::sync::Mutex;
use once_cell::sync::Lazy;

static INIT: Lazy<Mutex<bool>> = Lazy::new(|| Mutex::new(false));

fn ensure_models_loaded(models_dir: &str) {
    let mut loaded = INIT.lock().unwrap();
    if *loaded {
        return;
    }

    let image_path = PathBuf::from(models_dir).join("image_encoder.onnx");
    if image_path.exists() {
        let _ = REGISTRY.load_model(ModelMetadata {
            name: "image_encoder".into(),
            path: image_path,
            version: "1.0".into(),
            input_shape: vec![1, 3, 64, 64],
            output_shape: vec![1, 128],
        });
    }

    let audio_path = PathBuf::from(models_dir).join("audio_encoder.onnx");
    if audio_path.exists() {
        let _ = REGISTRY.load_model(ModelMetadata {
            name: "audio_encoder".into(),
            path: audio_path,
            version: "1.0".into(),
            input_shape: vec![1, 1, 64, 32],
            output_shape: vec![1, 128],
        });
    }

    *loaded = true;
}

// ─────────────────────────────────────────────
// ImageNeuralEncoder
// ─────────────────────────────────────────────

/// CNN-based image encoder: 64x64 RGB → 128D latent → Poincaré ball.
///
/// ONNX model: `models/image_encoder.onnx`
/// Input:  [B, 3, 64, 64] (normalized RGB)
/// Output: [B, 128] (Euclidean, projected to Poincaré via exp_map_zero)
pub struct ImageNeuralEncoder {
    models_dir: String,
}

impl ImageNeuralEncoder {
    pub fn new(models_dir: &str) -> Self {
        ensure_models_loaded(models_dir);
        Self { models_dir: models_dir.to_string() }
    }

    /// Encode a flattened 64x64x3 RGB image (12288 f32 values) into a latent vector.
    /// The input should be in CHW format (channel-first), normalized to [0, 1].
    pub fn encode_image(&self, image_chw: &[f32]) -> Result<LatentVector, String> {
        let session = REGISTRY
            .get_session("image_encoder")
            .map_err(|e| format!("image_encoder model not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let input_tensor = ndarray::Array4::from_shape_vec(
            (1, 3, 64, 64),
            image_chw.to_vec(),
        ).map_err(|e| format!("tensor shape: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let euclidean: Vec<f32> = output.1.iter().copied().collect();

        // Project to Poincaré ball
        let f64_vec: Vec<f64> = euclidean.iter().map(|&x| x as f64).collect();
        let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
        let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();

        Ok(LatentVector::new(data))
    }
}

impl SensoryEncoder for ImageNeuralEncoder {
    fn encode(&self, euclidean_latent: &[f32]) -> LatentVector {
        // If called via trait with raw image data, try neural encoding
        // Falls back to exp_map_zero if model not available
        match self.encode_image(euclidean_latent) {
            Ok(latent) => latent,
            Err(_) => {
                let f64_vec: Vec<f64> = euclidean_latent.iter().map(|&x| x as f64).collect();
                let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
                let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();
                LatentVector::new(data)
            }
        }
    }

    fn modality(&self) -> &'static str {
        "image"
    }
}

// ─────────────────────────────────────────────
// AudioNeuralEncoder
// ─────────────────────────────────────────────

/// Conv-based audio encoder: mel spectrogram [1,64,32] → 128D latent → Poincaré.
///
/// ONNX model: `models/audio_encoder.onnx`
/// Input:  [B, 1, 64, 32] (mel spectrogram)
/// Output: [B, 128] (Euclidean, projected to Poincaré via exp_map_zero)
pub struct AudioNeuralEncoder {
    models_dir: String,
}

impl AudioNeuralEncoder {
    pub fn new(models_dir: &str) -> Self {
        ensure_models_loaded(models_dir);
        Self { models_dir: models_dir.to_string() }
    }

    /// Encode a flattened mel spectrogram (64*32 = 2048 f32 values).
    pub fn encode_mel(&self, mel_spectrogram: &[f32]) -> Result<LatentVector, String> {
        let session = REGISTRY
            .get_session("audio_encoder")
            .map_err(|e| format!("audio_encoder model not loaded: {e}"))?;

        let mut guard = session.lock().map_err(|e| format!("lock: {e}"))?;

        let input_tensor = ndarray::Array4::from_shape_vec(
            (1, 1, 64, 32),
            mel_spectrogram.to_vec(),
        ).map_err(|e| format!("tensor shape: {e}"))?;

        let outputs = guard
            .run(ort::inputs!["input" => ort::value::Tensor::from_array(input_tensor).map_err(|e| format!("tensor: {e}"))?])
            .map_err(|e| format!("inference: {e}"))?;

        let output = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;

        let euclidean: Vec<f32> = output.1.iter().copied().collect();

        let f64_vec: Vec<f64> = euclidean.iter().map(|&x| x as f64).collect();
        let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
        let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();

        Ok(LatentVector::new(data))
    }
}

impl SensoryEncoder for AudioNeuralEncoder {
    fn encode(&self, euclidean_latent: &[f32]) -> LatentVector {
        match self.encode_mel(euclidean_latent) {
            Ok(latent) => latent,
            Err(_) => {
                let f64_vec: Vec<f64> = euclidean_latent.iter().map(|&x| x as f64).collect();
                let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
                let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();
                LatentVector::new(data)
            }
        }
    }

    fn modality(&self) -> &'static str {
        "audio"
    }
}
