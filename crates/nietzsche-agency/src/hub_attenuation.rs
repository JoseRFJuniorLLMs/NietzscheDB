// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Hub Avalanche Attenuation — prevents hub domination in SOC cascades.
//!
//! # Problem
//!
//! In scale-free graphs, high-degree hubs accumulate disproportionate
//! energy because `semantic_mass = energy × ln(degree + 1)` amplifies
//! their budget.  This causes **hub avalanche domination**: all cascades
//! funnel through a handful of super-connectors, killing power-law
//! diversity and producing fake criticality.
//!
//! # Solution (3 mechanisms)
//!
//! ## 1. Angular Focus (Gaussian)
//!
//! Avalanches prefer to spread within the same **semantic domain**
//! (angular sector of the Poincaré ball) rather than jumping to
//! the nearest hub:
//!
//! ```text
//! angular_focus = exp(−Δθ² / σ²)
//! ```
//!
//! where `Δθ = |θ_source − θ_target|` (wrapped to [0, π]) and σ
//! controls the domain width.  Default σ = 1.0 ≈ 57° focus cone.
//!
//! ## 2. Hub Penalty
//!
//! Inverts the logarithmic amplification of semantic mass:
//!
//! ```text
//! hub_penalty = 1 / ln(degree + 2)
//! ```
//!
//! This exactly cancels the `ln(degree+1)` factor in `semantic_mass`,
//! making bid reception roughly degree-independent. Hubs still
//! **send** bids normally but **receive** less per incoming bid.
//!
//! ## 3. Refractory Period (Cinzas)
//!
//! After a node "fires" (wins a high-attention auction), it enters a
//! cooldown period where its incoming attention is attenuated:
//!
//! ```text
//! refractory_factor = if cooldown > 0 { refractory_leak } else { 1.0 }
//! ```
//!
//! Default: `refractory_leak = 0.1` for `refractory_ticks = 5` ticks.
//! This forces avalanches to find **new paths** instead of repeatedly
//! exciting the same hub.
//!
//! # Integration
//!
//! The attenuation is applied as a **multiplicative modifier** on bids
//! in `attention_cycle.rs`:
//!
//! ```text
//! effective_bid = raw_bid × angular_focus × hub_penalty × refractory_factor
//! ```
//!
//! All three mechanisms are independently gated by env vars and default
//! to OFF (safe deploy).
//!
//! # References
//!
//! - Bak, Tang & Wiesenfeld (1987) — Self-organized criticality
//! - Beggs & Plenz (2003) — Neuronal avalanches
//! - Friston (2010) — Free Energy Principle & functional connectivity
//! - User's "Hyperbolic Avalanche Focusing" insight (2026-03-12)

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for hub avalanche attenuation.
#[derive(Debug, Clone)]
pub struct HubAttenuationConfig {
    /// Master switch: enable hub attenuation (default: false).
    pub enabled: bool,

    // ── Angular Focus ──
    /// Enable angular focus attenuation (default: true when master is on).
    pub angular_enabled: bool,
    /// Gaussian width σ for angular focus (radians). Default: 1.0 ≈ 57°.
    /// Smaller = tighter focus within semantic domain.
    /// Larger = more permissive cross-domain flow.
    pub angular_sigma: f32,

    // ── Hub Penalty ──
    /// Enable hub degree penalty (default: true when master is on).
    pub hub_penalty_enabled: bool,

    // ── Refractory Period ──
    /// Enable refractory period for high-activity nodes (default: true).
    pub refractory_enabled: bool,
    /// Number of ticks a node stays in refractory state after firing.
    pub refractory_ticks: u32,
    /// Attention multiplier during refractory period [0, 1]. Default: 0.1.
    pub refractory_leak: f32,
    /// Minimum attention received in a single tick to trigger refractory.
    /// Only nodes that receive more than this threshold enter cooldown.
    pub refractory_threshold: f32,
}

impl Default for HubAttenuationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            angular_enabled: true,
            angular_sigma: 1.0,
            hub_penalty_enabled: true,
            refractory_enabled: true,
            refractory_ticks: 5,
            refractory_leak: 0.1,
            refractory_threshold: 0.5,
        }
    }
}

impl HubAttenuationConfig {
    /// Load from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("AGENCY_HUB_ATTENUATION_ENABLED")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        Self {
            enabled,
            angular_enabled: std::env::var("AGENCY_HUB_ANGULAR_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            angular_sigma: std::env::var("AGENCY_HUB_ANGULAR_SIGMA")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            hub_penalty_enabled: std::env::var("AGENCY_HUB_PENALTY_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            refractory_enabled: std::env::var("AGENCY_HUB_REFRACTORY_ENABLED")
                .map(|v| v != "0" && v.to_lowercase() != "false")
                .unwrap_or(true),
            refractory_ticks: std::env::var("AGENCY_HUB_REFRACTORY_TICKS")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(5),
            refractory_leak: std::env::var("AGENCY_HUB_REFRACTORY_LEAK")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(0.1),
            refractory_threshold: std::env::var("AGENCY_HUB_REFRACTORY_THRESHOLD")
                .ok().and_then(|s| s.parse().ok())
                .unwrap_or(0.5),
        }
    }
}

// ─────────────────────────────────────────────
// Angular Focus
// ─────────────────────────────────────────────

/// Compute angular distance between two nodes using depth-as-theta proxy.
///
/// Both `theta_a` and `theta_b` are in [0, 2π). The distance is the
/// shorter arc length wrapped to [0, π].
#[inline]
pub fn angular_distance(theta_a: f32, theta_b: f32) -> f32 {
    let diff = (theta_a - theta_b).abs();
    let tau = std::f32::consts::TAU;
    // Wrap to [0, π]
    if diff > tau / 2.0 {
        tau - diff
    } else {
        diff
    }
}

/// Gaussian angular focus: preference for same-domain propagation.
///
/// Returns a value in (0, 1]:
/// - 1.0 when source and target are in the same angular direction
/// - Decays as a Gaussian with width σ
///
/// ```text
/// focus = exp(−Δθ² / σ²)
/// ```
#[inline]
pub fn angular_focus(theta_source: f32, theta_target: f32, sigma: f32) -> f32 {
    let d_theta = angular_distance(theta_source, theta_target);
    let sigma_sq = sigma * sigma;
    (-d_theta * d_theta / sigma_sq).exp()
}

// ─────────────────────────────────────────────
// Hub Penalty
// ─────────────────────────────────────────────

/// Hub degree penalty: attenuates bid reception for high-degree nodes.
///
/// ```text
/// penalty = 1 / ln(degree + 2)
/// ```
///
/// This inversely cancels the `ln(degree+1)` amplification in semantic_mass.
/// For degree 0: penalty ≈ 1/ln(2) ≈ 1.44 (slight boost for isolated nodes)
/// For degree 10: penalty ≈ 1/ln(12) ≈ 0.40
/// For degree 100: penalty ≈ 1/ln(102) ≈ 0.22
/// For degree 1000: penalty ≈ 1/ln(1002) ≈ 0.14
#[inline]
pub fn hub_penalty(degree: usize) -> f32 {
    1.0 / ((degree as f32) + 2.0).ln()
}

// ─────────────────────────────────────────────
// Refractory Period Tracker
// ─────────────────────────────────────────────

/// Tracks refractory (cooldown) state for nodes after high-activity events.
///
/// When a node receives attention exceeding `threshold`, it enters a
/// refractory period of `duration` ticks where incoming attention is
/// multiplied by `leak` (typically 0.1).
///
/// This forces avalanches to explore new pathways instead of repeatedly
/// exciting the same hubs — analogous to neuronal refractory periods.
#[derive(Debug, Clone)]
pub struct RefractoryTracker {
    /// Remaining cooldown ticks per node. Nodes not in map are not refractory.
    cooldowns: HashMap<Uuid, u32>,
    /// Total nodes that have entered refractory state (lifetime counter).
    pub total_refractory_events: u64,
}

/// Serializable snapshot of refractory state for API/persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefractorySnapshot {
    pub active_refractory_nodes: usize,
    pub total_refractory_events: u64,
}

impl RefractoryTracker {
    pub fn new() -> Self {
        Self {
            cooldowns: HashMap::new(),
            total_refractory_events: 0,
        }
    }

    /// Mark a node as entering refractory state.
    pub fn trigger(&mut self, node_id: Uuid, duration: u32) {
        self.cooldowns.insert(node_id, duration);
        self.total_refractory_events += 1;
    }

    /// Advance one tick: decrement all cooldowns, remove expired entries.
    pub fn tick(&mut self) {
        self.cooldowns.retain(|_, remaining| {
            if *remaining > 0 {
                *remaining -= 1;
            }
            *remaining > 0
        });
    }

    /// Get the refractory factor for a node.
    ///
    /// Returns `leak` if the node is in refractory, `1.0` otherwise.
    #[inline]
    pub fn factor(&self, node_id: &Uuid, leak: f32) -> f32 {
        if self.cooldowns.contains_key(node_id) {
            leak
        } else {
            1.0
        }
    }

    /// Check if a node is currently in refractory state.
    #[inline]
    pub fn is_refractory(&self, node_id: &Uuid) -> bool {
        self.cooldowns.contains_key(node_id)
    }

    /// Number of nodes currently in refractory state.
    pub fn active_count(&self) -> usize {
        self.cooldowns.len()
    }

    /// Create a serializable snapshot.
    pub fn snapshot(&self) -> RefractorySnapshot {
        RefractorySnapshot {
            active_refractory_nodes: self.cooldowns.len(),
            total_refractory_events: self.total_refractory_events,
        }
    }

    /// Mark nodes that received high attention as refractory.
    ///
    /// Called after auction resolution with the received-attention map.
    pub fn mark_high_receivers(
        &mut self,
        received: &HashMap<Uuid, f32>,
        threshold: f32,
        duration: u32,
    ) {
        for (&node_id, &attention) in received {
            if attention > threshold && !self.cooldowns.contains_key(&node_id) {
                self.trigger(node_id, duration);
            }
        }
    }
}

impl Default for RefractoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
// Combined Attenuation
// ─────────────────────────────────────────────

/// Compute the combined attenuation factor for a bid from source to target.
///
/// ```text
/// attenuation = angular_focus × hub_penalty × refractory_factor
/// ```
///
/// Returns a value in (0, ~1.5]. Values > 1.0 are possible for isolated
/// nodes (hub_penalty > 1 for degree 0), which is intentional: it slightly
/// boosts attention to peripheral/novel nodes.
pub fn compute_attenuation(
    theta_source: f32,
    theta_target: f32,
    target_degree: usize,
    target_id: &Uuid,
    config: &HubAttenuationConfig,
    refractory: &RefractoryTracker,
) -> f32 {
    if !config.enabled {
        return 1.0;
    }

    let mut factor = 1.0f32;

    // 1. Angular focus
    if config.angular_enabled {
        factor *= angular_focus(theta_source, theta_target, config.angular_sigma);
    }

    // 2. Hub penalty
    if config.hub_penalty_enabled {
        factor *= hub_penalty(target_degree);
    }

    // 3. Refractory period
    if config.refractory_enabled {
        factor *= refractory.factor(target_id, config.refractory_leak);
    }

    factor
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Angular Focus ──

    #[test]
    fn angular_distance_same_angle() {
        let d = angular_distance(1.0, 1.0);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn angular_distance_opposite() {
        let d = angular_distance(0.0, std::f32::consts::PI);
        assert!((d - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn angular_distance_wraps_around() {
        // 0.1 and TAU - 0.1 should be 0.2 apart
        let d = angular_distance(0.1, std::f32::consts::TAU - 0.1);
        assert!((d - 0.2).abs() < 1e-5);
    }

    #[test]
    fn angular_focus_same_direction_is_one() {
        let f = angular_focus(1.0, 1.0, 1.0);
        assert!((f - 1.0).abs() < 1e-6);
    }

    #[test]
    fn angular_focus_orthogonal_is_small() {
        // π/2 apart with σ=1.0: exp(-(π/2)² / 1) ≈ exp(-2.47) ≈ 0.085
        let f = angular_focus(0.0, std::f32::consts::FRAC_PI_2, 1.0);
        assert!(f < 0.15, "orthogonal focus should be small, got {f}");
        assert!(f > 0.0, "focus should be positive");
    }

    #[test]
    fn angular_focus_wider_sigma_more_permissive() {
        let f_tight = angular_focus(0.0, 1.0, 0.5);
        let f_wide = angular_focus(0.0, 1.0, 2.0);
        assert!(f_wide > f_tight, "wider sigma should allow more cross-domain");
    }

    // ── Hub Penalty ──

    #[test]
    fn hub_penalty_decreases_with_degree() {
        let p_low = hub_penalty(2);
        let p_high = hub_penalty(100);
        assert!(p_high < p_low, "higher degree should get lower penalty");
    }

    #[test]
    fn hub_penalty_isolated_node_boosted() {
        let p = hub_penalty(0);
        // 1/ln(2) ≈ 1.44
        assert!(p > 1.0, "isolated nodes should be slightly boosted, got {p}");
    }

    #[test]
    fn hub_penalty_cancels_semantic_mass_amplification() {
        // semantic_mass uses ln(degree + 1)
        // hub_penalty uses 1/ln(degree + 2)
        // For degree=50: ln(51) ≈ 3.93, 1/ln(52) ≈ 0.25
        // Product ≈ 0.98 — roughly cancels!
        let degree = 50;
        let mass_factor = ((degree as f32) + 1.0).ln();
        let penalty = hub_penalty(degree);
        let product = mass_factor * penalty;
        // Should be roughly 1.0 (within the ln(d+1)/ln(d+2) ratio)
        assert!(product > 0.8 && product < 1.2,
            "mass×penalty should roughly cancel, got {product}");
    }

    // ── Refractory Tracker ──

    #[test]
    fn refractory_new_node_not_refractory() {
        let tracker = RefractoryTracker::new();
        let id = Uuid::new_v4();
        assert!(!tracker.is_refractory(&id));
        assert!((tracker.factor(&id, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn refractory_trigger_and_decay() {
        let mut tracker = RefractoryTracker::new();
        let id = Uuid::new_v4();

        tracker.trigger(id, 3);
        assert!(tracker.is_refractory(&id));
        assert!((tracker.factor(&id, 0.1) - 0.1).abs() < 1e-6);

        tracker.tick(); // 2 remaining
        assert!(tracker.is_refractory(&id));

        tracker.tick(); // 1 remaining
        assert!(tracker.is_refractory(&id));

        tracker.tick(); // 0 → removed
        assert!(!tracker.is_refractory(&id));
        assert!((tracker.factor(&id, 0.1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn refractory_mark_high_receivers() {
        let mut tracker = RefractoryTracker::new();
        let hub = Uuid::new_v4();
        let normal = Uuid::new_v4();

        let mut received = HashMap::new();
        received.insert(hub, 0.8);     // above threshold
        received.insert(normal, 0.2);  // below threshold

        tracker.mark_high_receivers(&received, 0.5, 5);

        assert!(tracker.is_refractory(&hub));
        assert!(!tracker.is_refractory(&normal));
        assert_eq!(tracker.active_count(), 1);
        assert_eq!(tracker.total_refractory_events, 1);
    }

    #[test]
    fn refractory_does_not_re_trigger_during_cooldown() {
        let mut tracker = RefractoryTracker::new();
        let id = Uuid::new_v4();

        tracker.trigger(id, 5);

        let mut received = HashMap::new();
        received.insert(id, 1.0);

        // Should not re-trigger (already in cooldown)
        let events_before = tracker.total_refractory_events;
        tracker.mark_high_receivers(&received, 0.5, 5);
        assert_eq!(tracker.total_refractory_events, events_before);
    }

    // ── Combined Attenuation ──

    #[test]
    fn attenuation_disabled_returns_one() {
        let config = HubAttenuationConfig { enabled: false, ..Default::default() };
        let refractory = RefractoryTracker::new();
        let id = Uuid::new_v4();

        let a = compute_attenuation(0.0, 1.0, 100, &id, &config, &refractory);
        assert!((a - 1.0).abs() < 1e-6);
    }

    #[test]
    fn attenuation_same_domain_low_degree_near_one() {
        let config = HubAttenuationConfig {
            enabled: true,
            angular_sigma: 1.0,
            ..Default::default()
        };
        let refractory = RefractoryTracker::new();
        let id = Uuid::new_v4();

        // Same angle (θ=0), low degree → attenuation should be high (near 1.0+)
        let a = compute_attenuation(0.0, 0.0, 2, &id, &config, &refractory);
        // angular_focus = 1.0, hub_penalty = 1/ln(4) ≈ 0.72, refractory = 1.0
        assert!(a > 0.6, "same domain + low degree should have high attenuation, got {a}");
    }

    #[test]
    fn attenuation_cross_domain_high_degree_refractory_very_low() {
        let mut config = HubAttenuationConfig {
            enabled: true,
            angular_sigma: 0.5,
            refractory_leak: 0.1,
            ..Default::default()
        };

        let mut refractory = RefractoryTracker::new();
        let hub_id = Uuid::new_v4();
        refractory.trigger(hub_id, 5);

        // Cross-domain (θ=π/2), high degree, refractory
        let a = compute_attenuation(
            0.0, std::f32::consts::FRAC_PI_2,
            500, &hub_id, &config, &refractory,
        );

        // angular ≈ exp(-(π/2)²/0.25) ≈ exp(-9.87) ≈ 0.000052
        // hub_penalty = 1/ln(502) ≈ 0.16
        // refractory = 0.1
        // total ≈ 0.0000008
        assert!(a < 0.01, "cross-domain + hub + refractory should be near zero, got {a}");
    }

    #[test]
    fn snapshot_serialization() {
        let mut tracker = RefractoryTracker::new();
        tracker.trigger(Uuid::new_v4(), 5);
        tracker.trigger(Uuid::new_v4(), 3);

        let snap = tracker.snapshot();
        assert_eq!(snap.active_refractory_nodes, 2);
        assert_eq!(snap.total_refractory_events, 2);

        // Should be serializable
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("active_refractory_nodes"));
    }
}
