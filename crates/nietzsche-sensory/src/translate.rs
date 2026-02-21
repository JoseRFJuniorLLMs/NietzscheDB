//! Synesthesia — cross-modal projection via hyperbolic parallel transport.
//!
//! Translates a node's sensory representation from one modality to another
//! using the log/exp map pattern on the Poincaré ball:
//!
//! ```text
//! source_latent ──log_map──► tangent ──W_modal──► rotated ──exp_map──► target_latent
//! ```
//!
//! The radius (depth) is preserved while the "angle" (modality direction) changes.

use crate::types::{LatentVector, Modality, SensoryMemory};

/// Result of a cross-modal translation.
#[derive(Debug, Clone)]
pub struct TranslationResult {
    /// The translated sensory memory in the target modality.
    pub translated: SensoryMemory,
    /// Estimated quality loss from the translation (0.0 = lossless).
    pub quality_loss: f32,
}

/// Translate a sensory memory from one modality to another.
///
/// Uses hyperbolic parallel transport: log_map at origin → linear rotation
/// via modality matrix → exp_map back to the ball. The radius is preserved,
/// ensuring depth (hierarchical position) is maintained.
pub fn translate_modality(
    source: &SensoryMemory,
    target_modality: Modality,
) -> TranslationResult {
    let source_coords = source.latent.as_f32().unwrap_or_default();
    let dim = source_coords.len();

    // Step 1: log_map from Poincaré ball to tangent space at origin
    let tangent = log_map_origin(&source_coords);

    // Step 2: Apply cross-modal rotation matrix (simplified: permute + scale)
    let rotated = apply_modal_transform(&tangent, &source.modality, &target_modality);

    // Step 3: exp_map back to Poincaré ball
    let target_coords = exp_map_origin(&rotated);

    // Estimate quality loss (proportional to angular distance between modalities)
    let quality_loss = estimate_quality_loss(&source.modality, &target_modality);

    let translated = SensoryMemory {
        modality: target_modality,
        latent: LatentVector {
            data:       Some(target_coords),
            dim:        dim as u32,
            quant_level: source.latent.quant_level,
            quantized:  None,
        },
        reconstruction_quality: (source.reconstruction_quality - quality_loss).max(0.0),
        original_shape: source.original_shape.clone(),
        compression_ratio: source.compression_ratio,
        encoder_version: source.encoder_version,
    };

    TranslationResult { translated, quality_loss }
}

/// Log map at origin of the Poincaré ball: x → atanh(‖x‖) * x/‖x‖
fn log_map_origin(coords: &[f32]) -> Vec<f32> {
    let norm: f32 = coords.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return coords.to_vec();
    }
    let scale = norm.atanh() / norm;
    coords.iter().map(|x| x * scale).collect()
}

/// Exp map at origin of the Poincaré ball: v → tanh(‖v‖) * v/‖v‖
fn exp_map_origin(tangent: &[f32]) -> Vec<f32> {
    let norm: f32 = tangent.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return tangent.to_vec();
    }
    let scale = norm.tanh() / norm;
    tangent.iter().map(|x| x * scale).collect()
}

/// Apply a simplified cross-modal transformation.
///
/// In a full implementation, this would use a learned rotation matrix W
/// that maps between modality subspaces. Here we use a deterministic
/// permutation + sign flip that preserves the tangent vector norm.
fn apply_modal_transform(tangent: &[f32], from: &Modality, to: &Modality) -> Vec<f32> {
    let shift = modal_shift(from, to);
    let dim = tangent.len();
    let mut result = vec![0.0_f32; dim];
    for i in 0..dim {
        let j = (i + shift) % dim;
        // Alternate sign to create orthogonal-like rotation
        let sign = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
        result[j] = tangent[i] * sign;
    }
    result
}

/// Deterministic shift offset between modalities.
fn modal_shift(from: &Modality, to: &Modality) -> usize {
    let from_idx = modality_index(from);
    let to_idx = modality_index(to);
    ((to_idx as isize - from_idx as isize).unsigned_abs()) * 7 + 3
}

fn modality_index(m: &Modality) -> usize {
    match m {
        Modality::Text { .. }  => 0,
        Modality::Audio { .. } => 1,
        Modality::Image { .. } => 2,
        Modality::Fused { .. } => 3,
    }
}

fn estimate_quality_loss(from: &Modality, to: &Modality) -> f32 {
    if std::mem::discriminant(from) == std::mem::discriminant(to) {
        return 0.0;
    }
    // Cross-modal translation has inherent quality loss
    match (modality_index(from), modality_index(to)) {
        (0, 1) | (1, 0) => 0.15, // Text ↔ Audio
        (0, 2) | (2, 0) => 0.20, // Text ↔ Image
        (1, 2) | (2, 1) => 0.25, // Audio ↔ Image
        (3, _) | (_, 3) => 0.10, // Fused ↔ any
        _ => 0.20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OriginalShape, QuantLevel};

    fn make_sensory(modality: Modality, coords: Vec<f32>) -> SensoryMemory {
        let dim = coords.len() as u32;
        SensoryMemory {
            modality,
            latent: LatentVector {
                data: Some(coords),
                dim,
                quant_level: QuantLevel::F32,
                quantized: None,
            },
            reconstruction_quality: 0.9,
            original_shape: OriginalShape::Text { tokens: 64 },
            compression_ratio: 4.0,
            encoder_version: 1,
        }
    }

    #[test]
    fn translate_preserves_radius() {
        let source = make_sensory(
            Modality::Audio { sample_rate: 16000, duration_ms: 5000, channels: 1 },
            vec![0.3, 0.2, 0.1, 0.4, 0.05, 0.15, 0.25, 0.1],
        );
        let source_norm: f32 = source.latent.as_f32().unwrap().iter().map(|x| x * x).sum::<f32>().sqrt();

        let result = translate_modality(
            &source,
            Modality::Text { token_count: 100, language: "en".into() },
        );
        let target_norm: f32 = result.translated.latent.as_f32().unwrap().iter().map(|x| x * x).sum::<f32>().sqrt();

        // Radius should be approximately preserved (within numerical tolerance)
        assert!((source_norm - target_norm).abs() < 0.05,
            "radius changed too much: {source_norm} vs {target_norm}");
    }

    #[test]
    fn translate_reduces_quality() {
        let source = make_sensory(
            Modality::Audio { sample_rate: 16000, duration_ms: 5000, channels: 1 },
            vec![0.3; 8],
        );
        let result = translate_modality(
            &source,
            Modality::Text { token_count: 100, language: "en".into() },
        );
        assert!(result.quality_loss > 0.0);
        assert!(result.translated.reconstruction_quality < source.reconstruction_quality);
    }

    #[test]
    fn identity_translation() {
        let source = make_sensory(
            Modality::Text { token_count: 100, language: "en".into() },
            vec![0.2; 8],
        );
        let result = translate_modality(
            &source,
            Modality::Text { token_count: 200, language: "pt".into() },
        );
        assert!((result.quality_loss - 0.0).abs() < 1e-6);
    }
}
