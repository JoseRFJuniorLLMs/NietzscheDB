//! # nietzsche-sensory
//!
//! **Phase 11 — Sensory Compression Layer**
//!
//! The brain doesn't store experiences — it stores *reconstruction instructions*.
//! This crate implements that principle: raw sensory data (audio, text, image)
//! is compressed into latent vectors living in the Poincaré ball, and
//! progressively degraded as node energy decays.
//!
//! ## Architecture
//!
//! ```text
//! ENCODING                           STORAGE              DECODING
//! ────────                           ───────              ────────
//! Audio  (PCM)  → AudioEncoder  ─┐
//! Text   (str)  → TextEncoder   ─┤→ SensoryMemory ─→ Decoder(modality) → approx
//! Image  (H×W)  → ImageEncoder  ─┤   (LatentVector)
//! Multi         → FusionEncoder ─┘
//! ```
//!
//! ## Progressive degradation
//!
//! | Energy | Precision | Ratio |
//! |--------|-----------|-------|
//! | ≥ 0.7  | f32       | 1×    |
//! | ≥ 0.5  | f16       | 2×    |
//! | ≥ 0.3  | int8      | 4×    |
//! | ≥ 0.1  | PQ 64B    | 16×   |
//! | < 0.1  | None      | ∞     |

pub mod types;
pub mod storage;
pub mod encoder;

pub use types::*;
