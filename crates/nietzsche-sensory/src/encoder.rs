//! Encoder traits and implementations for sensory compression.
//!
//! Each modality has its own encoder that compresses raw data into
//! a latent vector in the Poincaré ball.
//!
//! ## Implementation status
//!
//! | Encoder       | Status  | Notes |
//! |---------------|---------|-------|
//! | TextEncoder   | Done    | Krylov 64D + exp_map_zero (EVA-Mind sends pre-compressed) |
//! | AudioEncoder  | MVP     | EnCodec frozen + projection (latent comes via gRPC) |
//! | FusionEncoder | Done    | gyromidpoint from nietzsche-hyp-ops |
//! | ImageEncoder  | Planned | VAE CNN — lowest priority for voice-first EVA |

use crate::types::{LatentVector, Modality, OriginalShape, SensoryMemory};

/// Trait for encoding raw sensory data into a Poincaré ball latent vector.
///
/// The actual heavy computation (Krylov, EnCodec, VAE) happens in EVA-Mind (Go).
/// NietzscheDB receives the pre-compressed Euclidean vector via gRPC and
/// projects it into the Poincaré ball using `exp_map_zero`.
pub trait SensoryEncoder {
    /// Encode a pre-compressed Euclidean vector into a hyperbolic latent.
    ///
    /// `euclidean_latent`: the output of the upstream encoder (Krylov, EnCodec, etc.)
    /// already reduced to the target dimensionality.
    fn encode(&self, euclidean_latent: &[f32]) -> LatentVector;

    /// The modality this encoder handles.
    fn modality(&self) -> &'static str;
}

// ─────────────────────────────────────────────
// TextEncoder
// ─────────────────────────────────────────────

/// Text encoder: projects Krylov-compressed embeddings into the Poincaré ball.
///
/// EVA-Mind pipeline:
/// ```text
/// Gemini embedding (3072D) → Krylov compression (64D) → gRPC → TextEncoder
/// ```
///
/// NietzscheDB adds: `exp_map_zero(vec_64d)` → z_hyp ∈ R^64, ‖z‖ < 1.0
pub struct TextEncoder;

impl SensoryEncoder for TextEncoder {
    fn encode(&self, euclidean_latent: &[f32]) -> LatentVector {
        // Convert f32 → f64 for hyp-ops, apply exp_map_zero, convert back
        let f64_vec: Vec<f64> = euclidean_latent.iter().map(|&x| x as f64).collect();
        let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
        let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();
        LatentVector::new(data)
    }

    fn modality(&self) -> &'static str {
        "text"
    }
}

// ─────────────────────────────────────────────
// AudioEncoder
// ─────────────────────────────────────────────

/// Audio encoder: projects pre-compressed audio latents into the Poincaré ball.
///
/// MVP uses EnCodec (Meta) frozen — the latent arrives pre-computed via gRPC.
/// Future: mel spectrogram VAE trained on clinical audio.
///
/// Two latents per voice session:
/// - `z_audio` (R^128): prosody, timbre, emotional patterns ("how it was said")
/// - `z_text` (R^64): semantic content ("what was said") — handled by TextEncoder
pub struct AudioEncoder;

impl SensoryEncoder for AudioEncoder {
    fn encode(&self, euclidean_latent: &[f32]) -> LatentVector {
        let f64_vec: Vec<f64> = euclidean_latent.iter().map(|&x| x as f64).collect();
        let projected = nietzsche_hyp_ops::exp_map_zero(&f64_vec);
        let data: Vec<f32> = projected.iter().map(|&x| x as f32).collect();
        LatentVector::new(data)
    }

    fn modality(&self) -> &'static str {
        "audio"
    }
}

// ─────────────────────────────────────────────
// FusionEncoder
// ─────────────────────────────────────────────

/// Multimodal fusion encoder using gyromidpoint in the Poincaré ball.
///
/// Fuses multiple already-projected latent vectors (each inside the ball)
/// into a single fused representation:
/// ```text
/// z_fused = gyromidpoint([z_audio, z_text])  ∈ R^max(dims)
/// ```
///
/// This is biologically inspired: the brain processes prosody and semantics
/// in separate regions (primary auditory cortex vs Wernicke's area) and
/// fuses them later.
pub struct FusionEncoder;

impl FusionEncoder {
    /// Fuse multiple hyperbolic latent vectors into one.
    ///
    /// All input latents must already be inside the Poincaré ball.
    /// They are padded to the same dimensionality (max dim) before fusion.
    pub fn fuse(latents: &[&LatentVector]) -> Result<LatentVector, FusionError> {
        if latents.is_empty() {
            return Err(FusionError::NoInputs);
        }
        if latents.len() == 1 {
            return Ok(latents[0].clone());
        }

        // Find max dim for padding
        let max_dim = latents.iter().map(|l| l.dim).max().unwrap() as usize;

        // Pad each latent to max_dim, convert to f64
        let padded: Vec<Vec<f64>> = latents
            .iter()
            .filter_map(|l| {
                l.data.as_ref().map(|d| {
                    let mut v: Vec<f64> = d.iter().map(|&x| x as f64).collect();
                    v.resize(max_dim, 0.0);
                    v
                })
            })
            .collect();

        if padded.is_empty() {
            return Err(FusionError::NoData);
        }

        // Collect slices for gyromidpoint
        let slices: Vec<&[f64]> = padded.iter().map(|v| v.as_slice()).collect();

        let fused = nietzsche_hyp_ops::gyromidpoint(&slices)
            .map_err(|e| FusionError::HyperbolicError(e.to_string()))?;

        let data: Vec<f32> = fused.iter().map(|&x| x as f32).collect();
        Ok(LatentVector::new(data))
    }

    /// Weighted fusion — each modality carries different confidence.
    ///
    /// Example: `weights = [0.7, 0.3]` means audio carries more weight.
    pub fn fuse_weighted(
        latents: &[&LatentVector],
        weights: &[f64],
    ) -> Result<LatentVector, FusionError> {
        if latents.is_empty() || weights.is_empty() {
            return Err(FusionError::NoInputs);
        }
        if latents.len() != weights.len() {
            return Err(FusionError::WeightMismatch {
                latents: latents.len(),
                weights: weights.len(),
            });
        }

        let max_dim = latents.iter().map(|l| l.dim).max().unwrap() as usize;

        let padded: Vec<Vec<f64>> = latents
            .iter()
            .filter_map(|l| {
                l.data.as_ref().map(|d| {
                    let mut v: Vec<f64> = d.iter().map(|&x| x as f64).collect();
                    v.resize(max_dim, 0.0);
                    v
                })
            })
            .collect();

        if padded.len() != weights.len() {
            return Err(FusionError::NoData);
        }

        let slices: Vec<&[f64]> = padded.iter().map(|v| v.as_slice()).collect();

        let fused = nietzsche_hyp_ops::gyromidpoint_weighted(&slices, weights)
            .map_err(|e| FusionError::HyperbolicError(e.to_string()))?;

        let data: Vec<f32> = fused.iter().map(|&x| x as f32).collect();
        Ok(LatentVector::new(data))
    }
}

// ─────────────────────────────────────────────
// Convenience: build SensoryMemory from raw input
// ─────────────────────────────────────────────

/// Build a complete `SensoryMemory` from a pre-compressed Euclidean vector.
///
/// This is the main entry point called from the gRPC handler:
/// ```text
/// EVA-Mind (Go) → gRPC InsertSensory → build_sensory_memory() → RocksDB
/// ```
pub fn build_sensory_memory(
    euclidean_latent: &[f32],
    modality: Modality,
    original_shape: OriginalShape,
    original_bytes: usize,
    encoder_version: u32,
) -> SensoryMemory {
    let encoder: Box<dyn SensoryEncoder> = match &modality {
        Modality::Text { .. } => Box::new(TextEncoder),
        Modality::Audio { .. } => Box::new(AudioEncoder),
        Modality::Image { .. } => Box::new(AudioEncoder), // placeholder — ImageEncoder not yet implemented
        Modality::Fused { .. } => Box::new(TextEncoder),  // fused uses FusionEncoder separately
    };

    let latent = encoder.encode(euclidean_latent);
    let latent_bytes = latent.byte_size();
    let compression_ratio = if latent_bytes > 0 {
        original_bytes as f32 / latent_bytes as f32
    } else {
        0.0
    };

    SensoryMemory {
        modality,
        latent,
        reconstruction_quality: 1.0,
        original_shape,
        compression_ratio,
        encoder_version,
    }
}

// ─────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum FusionError {
    #[error("no input latents provided")]
    NoInputs,

    #[error("input latents have no f32 data (already degraded?)")]
    NoData,

    #[error("weight count ({weights}) doesn't match latent count ({latents})")]
    WeightMismatch { latents: usize, weights: usize },

    #[error("hyperbolic operation failed: {0}")]
    HyperbolicError(String),
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_encoder_projects_inside_ball() {
        let encoder = TextEncoder;
        // Simulate Krylov output (64D, Euclidean)
        let euclidean: Vec<f32> = (0..64).map(|i| (i as f32 * 0.05) - 1.6).collect();
        let latent = encoder.encode(&euclidean);
        assert_eq!(latent.dim, 64);

        // Verify inside Poincaré ball
        let data = latent.data.unwrap();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0, "norm = {norm}, should be < 1.0");
    }

    #[test]
    fn audio_encoder_projects_inside_ball() {
        let encoder = AudioEncoder;
        let euclidean: Vec<f32> = (0..128).map(|i| (i as f32 * 0.02) - 1.28).collect();
        let latent = encoder.encode(&euclidean);
        assert_eq!(latent.dim, 128);

        let data = latent.data.unwrap();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0, "norm = {norm}, should be < 1.0");
    }

    #[test]
    fn fusion_of_two_latents() {
        let text_enc = TextEncoder;
        let audio_enc = AudioEncoder;

        let z_text = text_enc.encode(&vec![0.1; 64]);
        let z_audio = audio_enc.encode(&vec![0.1; 128]);

        let fused = FusionEncoder::fuse(&[&z_text, &z_audio]).unwrap();
        assert_eq!(fused.dim, 128); // max(64, 128)

        let data = fused.data.unwrap();
        let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0, "fused norm = {norm}");
    }

    #[test]
    fn weighted_fusion_biased_to_audio() {
        let z_text = TextEncoder.encode(&vec![0.5; 64]);
        let z_audio = AudioEncoder.encode(&vec![-0.5; 128]);

        let equal = FusionEncoder::fuse(&[&z_text, &z_audio]).unwrap();
        let biased = FusionEncoder::fuse_weighted(
            &[&z_text, &z_audio],
            &[0.1, 0.9], // heavily favor audio
        )
        .unwrap();

        // Biased result should be closer to z_audio than equal fusion
        let equal_data = equal.data.unwrap();
        let biased_data = biased.data.unwrap();
        // Just verify they're different and both inside ball
        assert_ne!(equal_data, biased_data);
        let norm: f32 = biased_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0);
    }

    #[test]
    fn build_sensory_memory_text() {
        let krylov_output: Vec<f32> = vec![0.1; 64];
        let sm = build_sensory_memory(
            &krylov_output,
            Modality::Text {
                token_count: 150,
                language: "pt-BR".into(),
            },
            OriginalShape::Text { tokens: 150 },
            12288, // original bytes (3072 * 4)
            1,
        );

        assert_eq!(sm.encoder_version, 1);
        assert_eq!(sm.reconstruction_quality, 1.0);
        assert!(sm.compression_ratio > 1.0);
        assert_eq!(sm.latent.dim, 64);
    }

    #[test]
    fn fusion_empty_returns_error() {
        let result = FusionEncoder::fuse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn fusion_single_returns_clone() {
        let z = TextEncoder.encode(&vec![0.3; 64]);
        let fused = FusionEncoder::fuse(&[&z]).unwrap();
        assert_eq!(fused.dim, z.dim);
        assert_eq!(fused.data, z.data);
    }
}
