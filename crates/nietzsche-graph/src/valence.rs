//! **Valence / Arousal** — emotional dimensions affecting heat diffusion.
//!
//! Human memory is inseparable from emotion. Traumatic or rewarding events
//! have a different "gravity" in hyperbolic space. Memories carry two
//! emotional axes:
//!
//! - **Valence** ∈ [-1.0, 1.0] — pleasure/displeasure dimension.
//!   Negative = punishing/traumatic. Positive = rewarding/pleasant.
//! - **Arousal** ∈ [0.0, 1.0] — intensity/activation dimension.
//!   High = emotionally intense. Low = calm/neutral.
//!
//! # Effect on Diffusion
//!
//! Heat (energy) propagates faster through emotionally charged memories:
//!
//! - **Arousal** amplifies the `energy_bias` in `diffusion_walk()`:
//!   ```text
//!   effective_bias = energy_bias * (1.0 + arousal)
//!   ```
//!   A node with arousal=1.0 doubles the temperature gradient, making
//!   the walk greedily attracted to high-energy emotional neighbors.
//!
//! - **Valence** modulates Laplacian edge weights in spectral diffusion:
//!   ```text
//!   valence_mod = 1.0 + |valence_u + valence_v| / 2.0
//!   w(u, v) = valence_mod / (1.0 + d_H(u, v))
//!   ```
//!   Edges between nodes of matching emotional polarity (both positive
//!   or both negative) propagate heat faster (emotional clustering).
//!   Opposite-polarity edges are unaffected (valence_mod ≈ 1.0).

use crate::model::NodeMeta;

// ─────────────────────────────────────────────
// Emotional gravity helpers
// ─────────────────────────────────────────────

/// Clamp valence to [-1.0, 1.0].
#[inline]
pub fn clamp_valence(v: f32) -> f32 {
    v.clamp(-1.0, 1.0)
}

/// Clamp arousal to [0.0, 1.0].
#[inline]
pub fn clamp_arousal(a: f32) -> f32 {
    a.clamp(0.0, 1.0)
}

/// Compute the effective energy bias for a node given its arousal.
///
/// `effective_bias = energy_bias * (1.0 + arousal)`
///
/// Arousal=0.0 → bias unchanged. Arousal=1.0 → bias doubled.
#[inline]
pub fn arousal_modulated_bias(energy_bias: f32, arousal: f32) -> f32 {
    energy_bias * (1.0 + clamp_arousal(arousal))
}

/// Compute the valence modifier for a Laplacian edge weight between two nodes.
///
/// Nodes with matching emotional polarity (both positive or both negative)
/// get a boost, increasing heat conductivity between them.
///
/// `valence_mod = 1.0 + |valence_u + valence_v| / 2.0`
///
/// - Both positive (0.8 + 0.6 → |1.4|/2 → 1.7): strong boost
/// - Both negative (-0.5 + -0.7 → |-1.2|/2 → 1.6): strong boost
/// - Opposite (+0.5 + -0.5 → |0|/2 → 1.0): no boost
/// - Both neutral (0 + 0 → 0 → 1.0): no boost
#[inline]
pub fn valence_edge_modifier(valence_u: f32, valence_v: f32) -> f64 {
    let sum = clamp_valence(valence_u) as f64 + clamp_valence(valence_v) as f64;
    1.0 + sum.abs() / 2.0
}

/// Compute the "emotional gravity" of a node — a combined intensity measure.
///
/// `gravity = arousal * (1.0 + |valence|)`
///
/// High arousal + strong valence → high gravity (emotionally significant).
/// Low arousal + neutral valence → low gravity (mundane fact).
#[inline]
pub fn emotional_gravity(meta: &NodeMeta) -> f32 {
    clamp_arousal(meta.arousal) * (1.0 + clamp_valence(meta.valence).abs())
}

/// Set valence and arousal on a NodeMeta, clamping to valid ranges.
pub fn set_emotion(meta: &mut NodeMeta, valence: f32, arousal: f32) {
    meta.valence = clamp_valence(valence);
    meta.arousal = clamp_arousal(arousal);
}

/// Decay arousal over time (emotions calm down).
///
/// `new_arousal = arousal * (1.0 - rate)`
///
/// Returns the new arousal value.
pub fn decay_arousal(meta: &mut NodeMeta, rate: f32) -> f32 {
    let rate = rate.clamp(0.0, 1.0);
    meta.arousal = (meta.arousal * (1.0 - rate)).max(0.0);
    meta.arousal
}

/// Reinforce emotion on a node (e.g., after retrieval in emotional context).
///
/// Shifts valence toward `target_valence` by `strength` and increases arousal.
pub fn reinforce_emotion(
    meta: &mut NodeMeta,
    target_valence: f32,
    strength: f32,
) {
    let strength = strength.clamp(0.0, 1.0);
    // Shift valence toward target
    meta.valence = clamp_valence(
        meta.valence + (clamp_valence(target_valence) - meta.valence) * strength,
    );
    // Boost arousal
    meta.arousal = clamp_arousal(meta.arousal + strength * 0.5);
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Node, PoincareVector};
    use uuid::Uuid;

    fn test_meta(valence: f32, arousal: f32) -> NodeMeta {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.1, 0.0]),
            serde_json::json!({}),
        );
        node.meta.valence = valence;
        node.meta.arousal = arousal;
        node.meta
    }

    #[test]
    fn clamp_valence_bounds() {
        assert_eq!(clamp_valence(-2.0), -1.0);
        assert_eq!(clamp_valence(2.0), 1.0);
        assert_eq!(clamp_valence(0.5), 0.5);
    }

    #[test]
    fn clamp_arousal_bounds() {
        assert_eq!(clamp_arousal(-0.5), 0.0);
        assert_eq!(clamp_arousal(1.5), 1.0);
        assert_eq!(clamp_arousal(0.7), 0.7);
    }

    #[test]
    fn arousal_modulates_bias() {
        // No arousal → bias unchanged
        assert!((arousal_modulated_bias(1.0, 0.0) - 1.0).abs() < 1e-6);
        // Full arousal → bias doubled
        assert!((arousal_modulated_bias(1.0, 1.0) - 2.0).abs() < 1e-6);
        // Mid arousal → bias * 1.5
        assert!((arousal_modulated_bias(2.0, 0.5) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn valence_modifier_same_polarity_boosts() {
        // Both positive
        let m = valence_edge_modifier(0.8, 0.6);
        assert!((m - 1.7).abs() < 1e-6, "both positive should boost, got {m}");

        // Both negative
        let m = valence_edge_modifier(-0.5, -0.7);
        assert!((m - 1.6).abs() < 1e-6, "both negative should boost, got {m}");
    }

    #[test]
    fn valence_modifier_opposite_no_boost() {
        let m = valence_edge_modifier(0.5, -0.5);
        assert!((m - 1.0).abs() < 1e-6, "opposite polarities should not boost, got {m}");
    }

    #[test]
    fn valence_modifier_neutral_no_boost() {
        let m = valence_edge_modifier(0.0, 0.0);
        assert!((m - 1.0).abs() < 1e-6, "neutral should not boost, got {m}");
    }

    #[test]
    fn emotional_gravity_high_arousal_strong_valence() {
        let meta = test_meta(0.8, 1.0);
        let g = emotional_gravity(&meta);
        // 1.0 * (1.0 + 0.8) = 1.8
        assert!((g - 1.8).abs() < 1e-6, "expected 1.8, got {g}");
    }

    #[test]
    fn emotional_gravity_low_arousal_neutral() {
        let meta = test_meta(0.0, 0.0);
        let g = emotional_gravity(&meta);
        assert_eq!(g, 0.0, "neutral calm node should have 0 gravity");
    }

    #[test]
    fn emotional_gravity_negative_valence() {
        let meta = test_meta(-0.9, 0.8);
        let g = emotional_gravity(&meta);
        // 0.8 * (1.0 + 0.9) = 1.52
        assert!((g - 1.52).abs() < 1e-6);
    }

    #[test]
    fn set_emotion_clamps() {
        let mut meta = test_meta(0.0, 0.0);
        set_emotion(&mut meta, -5.0, 3.0);
        assert_eq!(meta.valence, -1.0);
        assert_eq!(meta.arousal, 1.0);
    }

    #[test]
    fn decay_arousal_reduces() {
        let mut meta = test_meta(0.5, 0.8);
        let new = decay_arousal(&mut meta, 0.25);
        // 0.8 * 0.75 = 0.6
        assert!((new - 0.6).abs() < 1e-6);
        assert!((meta.arousal - 0.6).abs() < 1e-6);
    }

    #[test]
    fn decay_arousal_floors_at_zero() {
        let mut meta = test_meta(0.0, 0.1);
        decay_arousal(&mut meta, 1.0);
        assert_eq!(meta.arousal, 0.0);
    }

    #[test]
    fn reinforce_shifts_valence_and_boosts_arousal() {
        let mut meta = test_meta(0.0, 0.2);
        reinforce_emotion(&mut meta, 0.8, 0.5);

        // Valence shifts: 0.0 + (0.8 - 0.0) * 0.5 = 0.4
        assert!((meta.valence - 0.4).abs() < 1e-6);
        // Arousal: 0.2 + 0.5 * 0.5 = 0.45
        assert!((meta.arousal - 0.45).abs() < 1e-6);
    }

    #[test]
    fn reinforce_caps_at_bounds() {
        let mut meta = test_meta(0.9, 0.9);
        reinforce_emotion(&mut meta, 1.0, 1.0);

        // Valence: 0.9 + (1.0 - 0.9) * 1.0 = 1.0
        assert_eq!(meta.valence, 1.0);
        // Arousal: min(0.9 + 0.5, 1.0) = 1.0
        assert_eq!(meta.arousal, 1.0);
    }
}
