//! Core data structures for sensory memory.
//!
//! These types are stored in the `sensory` RocksDB column family,
//! keyed by node UUID. They are **lazy-loaded** — graph traversals
//! never touch this data unless reconstruction is explicitly requested.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// SensoryMemory — the top-level struct per node
// ─────────────────────────────────────────────

/// Complete sensory memory attached to a single graph node.
///
/// Stored separately from the node itself (RocksDB CF `sensory`)
/// to avoid bloating graph traversals with large latent vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensoryMemory {
    /// Which modality this memory encodes.
    pub modality: Modality,

    /// The compressed latent vector — the memory itself.
    pub latent: LatentVector,

    /// Reconstruction quality estimate ∈ \[0.0, 1.0\].
    /// 1.0 = near-lossless, 0.0 = irrecoverable.
    /// Updated when `degrade()` is called.
    pub reconstruction_quality: f32,

    /// Original shape metadata needed by the decoder.
    pub original_shape: OriginalShape,

    /// Compression ratio (original_bytes / latent_bytes).
    pub compression_ratio: f32,

    /// Which encoder version produced this latent.
    /// Allows decoder hot-reload without invalidating stored latents.
    pub encoder_version: u32,
}

// ─────────────────────────────────────────────
// LatentVector — the actual compressed data
// ─────────────────────────────────────────────

/// A compressed latent vector that can be progressively quantized.
///
/// At full precision (`data` field, f32), this is the best possible
/// reconstruction. As energy decays, `quantized` replaces `data`
/// with lower-precision representations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentVector {
    /// Full-precision latent (f32). Present when energy ≥ 0.5.
    /// Set to `None` after aggressive quantization to save space.
    pub data: Option<Vec<f32>>,

    /// Quantized representation. Populated when energy drops below 0.7.
    /// Format depends on `quant_level`.
    pub quantized: Option<Vec<u8>>,

    /// Current quantization level.
    pub quant_level: QuantLevel,

    /// Dimensionality of the latent space.
    pub dim: u32,
}

/// Quantization precision level, ordered by degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantLevel {
    /// Full 32-bit floating point. Energy ≥ 0.7.
    F32 = 0,
    /// Half precision 16-bit. Energy ∈ [0.5, 0.7).
    F16 = 1,
    /// 8-bit integer quantization. Energy ∈ [0.3, 0.5).
    Int8 = 2,
    /// Product quantization (64 bytes). Energy ∈ [0.1, 0.3).
    PQ = 3,
    /// Irrecoverable — no reconstruction possible. Energy < 0.1.
    Gone = 4,
}

impl LatentVector {
    /// Create a new full-precision latent vector.
    pub fn new(data: Vec<f32>) -> Self {
        let dim = data.len() as u32;
        Self {
            data: Some(data),
            quantized: None,
            quant_level: QuantLevel::F32,
            dim,
        }
    }

    /// Get the f32 representation for decoder input.
    ///
    /// If the vector has been quantized, this dequantizes back to f32
    /// (with precision loss). Returns `None` if `quant_level == Gone`.
    pub fn as_f32(&self) -> Option<Vec<f32>> {
        match self.quant_level {
            QuantLevel::Gone => None,
            QuantLevel::F32 => self.data.clone(),
            QuantLevel::F16 => self.dequantize_f16(),
            QuantLevel::Int8 => self.dequantize_int8(),
            QuantLevel::PQ => self.dequantize_pq(),
        }
    }

    /// Degrade the latent vector based on current node energy.
    ///
    /// This implements progressive forgetting:
    /// - energy ≥ 0.7 → keep f32 (no change)
    /// - energy ∈ [0.5, 0.7) → quantize to f16
    /// - energy ∈ [0.3, 0.5) → quantize to int8
    /// - energy ∈ [0.1, 0.3) → product quantize to 64 bytes
    /// - energy < 0.1 → mark as gone (irrecoverable)
    ///
    /// Returns the new `reconstruction_quality` estimate.
    pub fn degrade(&mut self, energy: f32) -> f32 {
        let target_level = Self::level_for_energy(energy);

        // Never upgrade — only degrade
        if target_level <= self.quant_level {
            return self.quality_estimate();
        }

        match target_level {
            QuantLevel::F32 => { /* no-op */ }
            QuantLevel::F16 => self.quantize_to_f16(),
            QuantLevel::Int8 => self.quantize_to_int8(),
            QuantLevel::PQ => self.quantize_to_pq(),
            QuantLevel::Gone => {
                self.data = None;
                self.quantized = None;
                self.quant_level = QuantLevel::Gone;
            }
        }

        self.quality_estimate()
    }

    /// Determine the target quantization level for a given energy.
    pub fn level_for_energy(energy: f32) -> QuantLevel {
        if energy >= 0.7 {
            QuantLevel::F32
        } else if energy >= 0.5 {
            QuantLevel::F16
        } else if energy >= 0.3 {
            QuantLevel::Int8
        } else if energy >= 0.1 {
            QuantLevel::PQ
        } else {
            QuantLevel::Gone
        }
    }

    /// Estimate reconstruction quality based on quantization level.
    pub fn quality_estimate(&self) -> f32 {
        match self.quant_level {
            QuantLevel::F32 => 1.0,
            QuantLevel::F16 => 0.85,
            QuantLevel::Int8 => 0.60,
            QuantLevel::PQ => 0.30,
            QuantLevel::Gone => 0.0,
        }
    }

    /// Approximate byte size of the stored latent.
    pub fn byte_size(&self) -> usize {
        let data_size = self.data.as_ref().map_or(0, |d| d.len() * 4);
        let quant_size = self.quantized.as_ref().map_or(0, |q| q.len());
        data_size + quant_size
    }

    // ── Quantization internals ────────────────

    fn quantize_to_f16(&mut self) {
        if let Some(ref data) = self.data {
            // f16 approximation: store as 2 bytes per value (IEEE 754 half)
            // Simplified: store high 16 bits of f32
            let quantized: Vec<u8> = data.iter().flat_map(|&v| {
                let bits = v.to_bits();
                let half = f32_to_f16_bits(bits);
                half.to_le_bytes()
            }).collect();
            self.quantized = Some(quantized);
            self.quant_level = QuantLevel::F16;
            // Keep f32 data until int8 level for potential rollback
        }
    }

    fn quantize_to_int8(&mut self) {
        // Use f32 data if available, otherwise dequantize f16 first
        let f32_data = if let Some(ref data) = self.data {
            data.clone()
        } else if let Some(ref _q) = self.quantized {
            match self.dequantize_f16() {
                Some(d) => d,
                None => return,
            }
        } else {
            return;
        };

        // Symmetric int8: scale = max(|v|) / 127
        let max_abs = f32_data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs > 1e-10 { max_abs / 127.0 } else { 1.0 };

        let mut quantized = Vec::with_capacity(f32_data.len() + 4);
        // Store scale factor in first 4 bytes
        quantized.extend_from_slice(&scale.to_le_bytes());
        // Store quantized values
        for &v in &f32_data {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            quantized.push(q as u8);
        }

        self.quantized = Some(quantized);
        self.data = None; // Free f32 data
        self.quant_level = QuantLevel::Int8;
    }

    fn quantize_to_pq(&mut self) {
        // Product Quantization: extremely lossy, 64 bytes fixed output.
        // Simplified PQ: divide dim into 64 sub-vectors, keep 1 byte per sub.
        let f32_data = match self.as_f32() {
            Some(d) => d,
            None => return,
        };

        let dim = f32_data.len();
        let n_subvectors = 64.min(dim);
        let sub_size = dim / n_subvectors;

        let mut pq = Vec::with_capacity(n_subvectors + 4);
        // Store original dim for dequant
        pq.extend_from_slice(&(dim as u32).to_le_bytes());

        for i in 0..n_subvectors {
            let start = i * sub_size;
            let end = if i == n_subvectors - 1 { dim } else { start + sub_size };
            // Centroid of sub-vector → single byte
            let mean: f32 = f32_data[start..end].iter().sum::<f32>() / (end - start) as f32;
            // Map [-1, 1] → [0, 255]
            let byte = ((mean.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
            pq.push(byte);
        }

        self.quantized = Some(pq);
        self.data = None;
        self.quant_level = QuantLevel::PQ;
    }

    fn dequantize_f16(&self) -> Option<Vec<f32>> {
        let quantized = self.quantized.as_ref()?;
        if quantized.len() % 2 != 0 {
            return None;
        }
        let values: Vec<f32> = quantized.chunks_exact(2).map(|chunk| {
            let half = u16::from_le_bytes([chunk[0], chunk[1]]);
            f16_bits_to_f32(half)
        }).collect();
        Some(values)
    }

    fn dequantize_int8(&self) -> Option<Vec<f32>> {
        let quantized = self.quantized.as_ref()?;
        if quantized.len() < 5 {
            return None;
        }
        let scale = f32::from_le_bytes([quantized[0], quantized[1], quantized[2], quantized[3]]);
        let values: Vec<f32> = quantized[4..].iter().map(|&b| {
            (b as i8) as f32 * scale
        }).collect();
        Some(values)
    }

    fn dequantize_pq(&self) -> Option<Vec<f32>> {
        let quantized = self.quantized.as_ref()?;
        if quantized.len() < 5 {
            return None;
        }
        let orig_dim = u32::from_le_bytes([quantized[0], quantized[1], quantized[2], quantized[3]]) as usize;
        let pq_data = &quantized[4..];
        let n_subvectors = pq_data.len();
        if n_subvectors == 0 {
            return None;
        }
        let sub_size = orig_dim / n_subvectors;

        let mut values = Vec::with_capacity(orig_dim);
        for (i, &byte) in pq_data.iter().enumerate() {
            let mean = (byte as f32 / 127.5) - 1.0;
            let count = if i == n_subvectors - 1 {
                orig_dim - i * sub_size
            } else {
                sub_size
            };
            for _ in 0..count {
                values.push(mean);
            }
        }
        Some(values)
    }
}

// ─────────────────────────────────────────────
// Modality — what kind of sensory data
// ─────────────────────────────────────────────

/// The sensory modality of a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Modality {
    /// Image data (medical exam, photo, etc.)
    Image {
        width: u32,
        height: u32,
        channels: u8,
    },
    /// Audio data (voice session, prosody)
    Audio {
        sample_rate: u32,
        duration_ms: u32,
        channels: u8,
    },
    /// Text data (clinical note, transcript)
    Text {
        token_count: u32,
        language: String,
    },
    /// Fused multimodal memory
    Fused {
        /// The component modalities that were fused.
        components: Vec<Modality>,
    },
}

impl Modality {
    /// Default latent dimensionality for this modality.
    pub fn default_dim(&self) -> u32 {
        match self {
            Modality::Image { .. } => 256,
            Modality::Audio { .. } => 128,
            Modality::Text { .. } => 64,
            Modality::Fused { .. } => 256,
        }
    }

    /// Short label for logging and display.
    pub fn label(&self) -> &'static str {
        match self {
            Modality::Image { .. } => "image",
            Modality::Audio { .. } => "audio",
            Modality::Text { .. } => "text",
            Modality::Fused { .. } => "fused",
        }
    }
}

// ─────────────────────────────────────────────
// OriginalShape — metadata for decoder
// ─────────────────────────────────────────────

/// Shape metadata of the original data, needed by the decoder
/// to reconstruct an approximation of the right dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OriginalShape {
    Image {
        width: u32,
        height: u32,
        channels: u8,
    },
    Audio {
        samples: u32,
        sample_rate: u32,
    },
    Text {
        tokens: u32,
    },
}

// ─────────────────────────────────────────────
// IEEE 754 half-precision helpers
// ─────────────────────────────────────────────

/// Convert f32 bits to f16 bits (IEEE 754 half-precision).
fn f32_to_f16_bits(f: u32) -> u16 {
    let sign = (f >> 16) & 0x8000;
    let exponent = ((f >> 23) & 0xFF) as i32;
    let mantissa = f & 0x7F_FFFF;

    if exponent == 0xFF {
        // Inf / NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }

    let new_exp = exponent - 127 + 15;

    if new_exp >= 31 {
        // Overflow → Inf
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        // Underflow → zero (skip denormals for simplicity)
        return sign as u16;
    }

    let new_mantissa = mantissa >> 13;
    (sign | ((new_exp as u32) << 10) | new_mantissa) as u16
}

/// Convert f16 bits back to f32.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1F) as u32;
    let mantissa = (h & 0x3FF) as u32;

    if exponent == 0x1F {
        // Inf / NaN
        let f_bits = (sign << 31) | 0x7F80_0000 | (mantissa << 13);
        return f32::from_bits(f_bits);
    }
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized → normalize
        let mut m = mantissa;
        let mut e = 0_i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let new_exp = (127 - 15 - e) as u32;
        let new_mantissa = (m & 0x3FF) << 13;
        let f_bits = (sign << 31) | (new_exp << 23) | new_mantissa;
        return f32::from_bits(f_bits);
    }

    let new_exp = exponent + 127 - 15;
    let new_mantissa = mantissa << 13;
    let f_bits = (sign << 31) | (new_exp << 23) | new_mantissa;
    f32::from_bits(f_bits)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_latent(dim: usize) -> LatentVector {
        let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01) - 0.5).collect();
        LatentVector::new(data)
    }

    #[test]
    fn new_latent_is_f32() {
        let lv = sample_latent(128);
        assert_eq!(lv.quant_level, QuantLevel::F32);
        assert_eq!(lv.dim, 128);
        assert!(lv.data.is_some());
        assert!(lv.quantized.is_none());
    }

    #[test]
    fn quality_at_f32_is_one() {
        let lv = sample_latent(64);
        assert_eq!(lv.quality_estimate(), 1.0);
    }

    #[test]
    fn degrade_high_energy_no_change() {
        let mut lv = sample_latent(64);
        let q = lv.degrade(0.9);
        assert_eq!(q, 1.0);
        assert_eq!(lv.quant_level, QuantLevel::F32);
    }

    #[test]
    fn degrade_to_f16() {
        let mut lv = sample_latent(64);
        let q = lv.degrade(0.6);
        assert_eq!(lv.quant_level, QuantLevel::F16);
        assert!(q < 1.0);
        assert!(q > 0.5);
    }

    #[test]
    fn degrade_to_int8() {
        let mut lv = sample_latent(64);
        let q = lv.degrade(0.4);
        assert_eq!(lv.quant_level, QuantLevel::Int8);
        assert!(lv.data.is_none()); // f32 freed
        assert!(q < 0.85);
    }

    #[test]
    fn degrade_to_pq() {
        let mut lv = sample_latent(128);
        let q = lv.degrade(0.2);
        assert_eq!(lv.quant_level, QuantLevel::PQ);
        assert!(q < 0.6);
    }

    #[test]
    fn degrade_to_gone() {
        let mut lv = sample_latent(64);
        let q = lv.degrade(0.05);
        assert_eq!(lv.quant_level, QuantLevel::Gone);
        assert_eq!(q, 0.0);
        assert!(lv.data.is_none());
        assert!(lv.quantized.is_none());
    }

    #[test]
    fn never_upgrades() {
        let mut lv = sample_latent(64);
        lv.degrade(0.4); // → int8
        assert_eq!(lv.quant_level, QuantLevel::Int8);
        lv.degrade(0.9); // should NOT upgrade back to f32
        assert_eq!(lv.quant_level, QuantLevel::Int8);
    }

    #[test]
    fn f16_roundtrip_preserves_magnitude() {
        let mut lv = sample_latent(64);
        let original = lv.data.clone().unwrap();
        lv.degrade(0.6); // → f16
        let recovered = lv.as_f32().unwrap();
        assert_eq!(recovered.len(), original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            // f16 has ~3 decimal digits of precision
            assert!(
                (a - b).abs() < 0.01,
                "f16 roundtrip: {a} vs {b}"
            );
        }
    }

    #[test]
    fn int8_roundtrip_preserves_sign() {
        let mut lv = sample_latent(64);
        let original = lv.data.clone().unwrap();
        lv.degrade(0.4); // → int8
        let recovered = lv.as_f32().unwrap();
        assert_eq!(recovered.len(), original.len());
        for (a, b) in original.iter().zip(recovered.iter()) {
            // int8 is coarser but should preserve sign
            if a.abs() > 0.05 {
                assert_eq!(a.signum(), b.signum(), "sign flip at {a} → {b}");
            }
        }
    }

    #[test]
    fn pq_recovers_something() {
        let mut lv = sample_latent(128);
        lv.degrade(0.2); // → PQ
        let recovered = lv.as_f32();
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap().len(), 128);
    }

    #[test]
    fn gone_returns_none() {
        let mut lv = sample_latent(64);
        lv.degrade(0.05);
        assert!(lv.as_f32().is_none());
    }

    #[test]
    fn byte_size_decreases_with_degradation() {
        let mut lv = sample_latent(128);
        let size_f32 = lv.byte_size();

        let mut lv2 = sample_latent(128);
        lv2.degrade(0.4); // → int8
        let size_int8 = lv2.byte_size();

        let mut lv3 = sample_latent(128);
        lv3.degrade(0.2); // → PQ
        let size_pq = lv3.byte_size();

        assert!(size_f32 > size_int8, "f32={size_f32} > int8={size_int8}");
        assert!(size_int8 > size_pq, "int8={size_int8} > pq={size_pq}");
    }

    #[test]
    fn modality_default_dims() {
        assert_eq!(Modality::Audio { sample_rate: 16000, duration_ms: 5000, channels: 1 }.default_dim(), 128);
        assert_eq!(Modality::Text { token_count: 100, language: "pt".into() }.default_dim(), 64);
        assert_eq!(Modality::Image { width: 224, height: 224, channels: 3 }.default_dim(), 256);
    }

    #[test]
    fn sensory_memory_serialization_roundtrip() {
        let sm = SensoryMemory {
            modality: Modality::Audio {
                sample_rate: 16000,
                duration_ms: 5000,
                channels: 1,
            },
            latent: sample_latent(128),
            reconstruction_quality: 1.0,
            original_shape: OriginalShape::Audio {
                samples: 80000,
                sample_rate: 16000,
            },
            compression_ratio: 625.0,
            encoder_version: 1,
        };

        let bytes = bincode::serialize(&sm).unwrap();
        let recovered: SensoryMemory = bincode::deserialize(&bytes).unwrap();
        assert_eq!(recovered.latent.dim, 128);
        assert_eq!(recovered.encoder_version, 1);
    }
}
