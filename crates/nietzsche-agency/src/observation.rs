// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XV.1 — ObservationBridge: live metrics for visualization.
//!
//! Aggregates thermodynamic, attention, and gravity data into a format
//! optimized for real-time visualization (perspektive.js / d3.js).
//!
//! ## Output Format
//!
//! The bridge produces `ObservationFrame` — a snapshot of the cognitive
//! state with per-node visual properties:
//!
//! - **color**: RGB derived from cognitive temperature (blue=cold → red=hot)
//! - **brightness**: proportional to arousal / energy
//! - **size**: proportional to semantic mass
//! - **pulse**: oscillation frequency from attention flow
//!
//! Plus system-wide gauges for dashboard panels.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cognitive_dashboard::CognitiveDashboard;

// ─────────────────────────────────────────────
// Observation Frame
// ─────────────────────────────────────────────

/// A complete observation frame for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationFrame {
    /// Timestamp (Unix millis).
    pub timestamp_ms: u64,
    /// Collection name.
    pub collection: String,

    // ── System-wide gauges ──────────────────
    pub gauges: SystemGauges,

    // ── Per-node visual properties ──────────
    /// Node visual states (sampled, not all nodes).
    pub nodes: Vec<NodeVisual>,

    // ── Gravity field (top attractions) ─────
    pub gravity_lines: Vec<GravityLine>,
}

/// System-wide gauges for dashboard panels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemGauges {
    /// Cognitive temperature [0, ∞). Optimal: 0.15–0.85.
    pub temperature: f64,
    /// Shannon entropy of energy distribution.
    pub entropy: f64,
    /// Helmholtz free energy (lower = more organized).
    pub free_energy: f64,
    /// Phase state: "Solid", "Liquid", "Gas", "Critical".
    pub phase: String,
    /// Entropy production rate (dS/dt).
    pub entropy_rate: f64,
    /// ECAN inflation price (1.0 = no inflation).
    pub attention_price: f32,
    /// Total attention flow this tick.
    pub attention_flow: f32,
    /// Hebbian active traces count.
    pub hebbian_traces: usize,
    /// Exploration modifier from temperature.
    pub explore_modifier: f64,
    /// Gravity wells count.
    pub gravity_wells: usize,
    /// Mean gravitational force.
    pub mean_gravity: f64,
    /// DirtySet adaptive ratio (dirty / total).
    pub dirty_ratio: f64,
    /// Total nodes.
    pub total_nodes: usize,
    /// Mean energy.
    pub mean_energy: f64,
}

/// Per-node visual properties for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVisual {
    /// Node ID.
    pub id: String,
    /// RGB color: temperature-mapped (blue cold → red hot).
    pub color: [u8; 3],
    /// Brightness: energy level [0.0, 1.0].
    pub brightness: f32,
    /// Size: semantic mass (log scale).
    pub size: f32,
    /// Pulse frequency: attention received this tick.
    pub pulse: f32,
}

/// A gravity attraction line for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityLine {
    pub source: String,
    pub target: String,
    pub force: f64,
    /// Line thickness proportional to force.
    pub thickness: f32,
}

// ─────────────────────────────────────────────
// Color mapping
// ─────────────────────────────────────────────

/// Map cognitive temperature to an RGB color.
///
/// - T < 0.15 (Solid): deep blue (cold, crystallized)
/// - T ≈ 0.50 (Liquid): green (optimal, self-organizing)
/// - T > 0.85 (Gas): red (chaotic, fragmented)
///
/// Uses a smooth gradient via HSV interpolation.
pub fn temperature_to_rgb(temperature: f64) -> [u8; 3] {
    // Clamp to [0, 2] for extreme cases
    let t = temperature.clamp(0.0, 2.0);

    // Map temperature to hue: 240° (blue) → 120° (green) → 0° (red)
    let hue = if t <= 0.5 {
        // Blue (240°) → Green (120°)
        240.0 - (t / 0.5) * 120.0
    } else {
        // Green (120°) → Red (0°)
        120.0 - ((t - 0.5) / 1.5) * 120.0
    };

    // Saturation: high for extreme temperatures, moderate for optimal
    let saturation = 0.7 + 0.3 * (2.0 * (t - 0.5).abs()).min(1.0);

    // Value: always bright
    let value = 0.85 + 0.15 * (1.0 - (t - 0.5).abs().min(1.0));

    hsv_to_rgb(hue, saturation, value)
}

/// Convert HSV to RGB.
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [u8; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    [
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    ]
}

// ─────────────────────────────────────────────
// Frame builder
// ─────────────────────────────────────────────

impl ObservationFrame {
    /// Build an observation frame from a CognitiveDashboard.
    ///
    /// Node visuals require additional per-node data (energy, mass, attention).
    /// If `node_data` is empty, only system gauges and gravity lines are populated.
    pub fn from_dashboard(
        dashboard: &CognitiveDashboard,
        node_data: &[NodeObservation],
    ) -> Self {
        let temperature = dashboard.thermodynamics.as_ref()
            .map(|t| t.temperature)
            .unwrap_or(0.5);

        let gauges = SystemGauges {
            temperature,
            entropy: dashboard.thermodynamics.as_ref().map(|t| t.entropy).unwrap_or(0.0),
            free_energy: dashboard.thermodynamics.as_ref().map(|t| t.free_energy).unwrap_or(0.0),
            phase: dashboard.thermodynamics.as_ref()
                .map(|t| t.phase.clone())
                .unwrap_or_else(|| "Unknown".into()),
            entropy_rate: dashboard.thermodynamics.as_ref().map(|t| t.entropy_rate).unwrap_or(0.0),
            attention_price: dashboard.attention.as_ref().map(|a| a.price).unwrap_or(1.0),
            attention_flow: dashboard.attention.as_ref().map(|a| a.total_flow).unwrap_or(0.0),
            hebbian_traces: dashboard.hebbian.as_ref().map(|h| h.active_traces).unwrap_or(0),
            explore_modifier: dashboard.thermodynamics.as_ref().map(|t| t.explore_modifier).unwrap_or(1.0),
            gravity_wells: dashboard.gravity.as_ref().map(|g| g.gravity_wells).unwrap_or(0),
            mean_gravity: dashboard.gravity.as_ref().map(|g| g.mean_force).unwrap_or(0.0),
            dirty_ratio: 0.0, // filled by caller if available
            total_nodes: dashboard.health.as_ref().map(|h| h.total_nodes).unwrap_or(0),
            mean_energy: dashboard.thermodynamics.as_ref()
                .map(|t| t.mean_energy)
                .unwrap_or(dashboard.health.as_ref().map(|h| h.mean_energy as f64).unwrap_or(0.0)),
        };

        // Build per-node visuals
        let system_color = temperature_to_rgb(temperature);
        let nodes: Vec<NodeVisual> = node_data.iter().map(|n| {
            // Per-node temperature = energy deviation from mean
            let node_temp = if gauges.mean_energy > 1e-6 {
                (n.energy as f64 - gauges.mean_energy).abs() / gauges.mean_energy
            } else {
                0.0
            };
            let color = temperature_to_rgb(node_temp);

            NodeVisual {
                id: n.id.to_string(),
                color,
                brightness: n.energy,
                size: n.mass.max(0.1),
                pulse: n.attention_received,
            }
        }).collect();

        // Gravity lines from dashboard
        let gravity_lines = dashboard.gravity.as_ref()
            .map(|g| g.top_attractions.iter().map(|a| {
                let max_force = g.top_attractions.first().map(|f| f.force).unwrap_or(1.0);
                GravityLine {
                    source: a.source.clone(),
                    target: a.target.clone(),
                    force: a.force,
                    thickness: (a.force / max_force.max(1e-6) * 5.0) as f32,
                }
            }).collect())
            .unwrap_or_default();

        let _ = system_color; // used for fallback

        ObservationFrame {
            timestamp_ms: dashboard.timestamp_ms,
            collection: dashboard.collection.clone(),
            gauges,
            nodes,
            gravity_lines,
        }
    }

    /// CF_META key for observation frame persistence.
    pub fn meta_key() -> &'static str {
        "agency:observation_frame"
    }
}

/// Input data for per-node visual computation.
#[derive(Debug, Clone)]
pub struct NodeObservation {
    pub id: Uuid,
    pub energy: f32,
    pub mass: f32,
    pub attention_received: f32,
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_dashboard::*;

    #[test]
    fn temperature_cold_is_blue() {
        let [r, g, b] = temperature_to_rgb(0.0);
        assert!(b > r, "cold should be blue-ish: r={r} g={g} b={b}");
    }

    #[test]
    fn temperature_optimal_is_green() {
        let [r, g, b] = temperature_to_rgb(0.5);
        assert!(g >= r && g >= b, "optimal should be green-ish: r={r} g={g} b={b}");
    }

    #[test]
    fn temperature_hot_is_red() {
        let [r, g, b] = temperature_to_rgb(1.5);
        assert!(r > b, "hot should be red-ish: r={r} g={g} b={b}");
    }

    #[test]
    fn temperature_extreme_clamped() {
        // Should not panic
        let _ = temperature_to_rgb(-1.0);
        let _ = temperature_to_rgb(100.0);
    }

    #[test]
    fn frame_from_empty_dashboard() {
        let dash = CognitiveDashboard {
            timestamp_ms: 1000,
            collection: "test".into(),
            tick_duration_ms: 10,
            health: None,
            attention: None,
            hebbian: None,
            thermodynamics: None,
            maturity: MaturitySnapshot { promotions: 0, demotions: 0 },
            gravity: None,
            shatter: None,
            healing: None,
            learning: None,
            compression: None,
            sharding: None,
            world_model: None,
            flywheel: None,
        };

        let frame = ObservationFrame::from_dashboard(&dash, &[]);
        assert_eq!(frame.collection, "test");
        assert!(frame.nodes.is_empty());
        assert!(frame.gravity_lines.is_empty());
        assert_eq!(frame.gauges.phase, "Unknown");
    }

    #[test]
    fn frame_with_nodes() {
        let dash = CognitiveDashboard {
            timestamp_ms: 1000,
            collection: "test".into(),
            tick_duration_ms: 10,
            health: Some(HealthSnapshot {
                global_hausdorff: 1.0,
                mean_energy: 0.5,
                coherence_score: 0.8,
                gap_count: 0,
                entropy_spike_count: 0,
                total_nodes: 100,
                total_edges: 200,
            }),
            attention: Some(AttentionSnapshot {
                participants: 50,
                total_bids: 100,
                winning_bids: 25,
                price: 1.1,
                total_flow: 10.0,
                top_receivers: vec![],
            }),
            hebbian: None,
            thermodynamics: Some(ThermoSnapshot {
                temperature: 0.4,
                entropy: 3.0,
                free_energy: -1.0,
                entropy_rate: 0.01,
                phase: "Liquid".into(),
                explore_modifier: 1.0,
                nodes_analysed: 100,
                mean_energy: 0.5,
                energy_std: 0.2,
                heat_flows_count: 5,
            }),
            maturity: MaturitySnapshot { promotions: 3, demotions: 1 },
            gravity: None,
            shatter: None,
            healing: None,
            learning: None,
            compression: None,
            sharding: None,
            world_model: None,
            flywheel: None,
        };

        let nodes = vec![
            NodeObservation { id: Uuid::from_u128(1), energy: 0.8, mass: 1.5, attention_received: 0.3 },
            NodeObservation { id: Uuid::from_u128(2), energy: 0.1, mass: 0.2, attention_received: 0.0 },
        ];

        let frame = ObservationFrame::from_dashboard(&dash, &nodes);
        assert_eq!(frame.nodes.len(), 2);
        assert_eq!(frame.gauges.temperature, 0.4);
        assert_eq!(frame.gauges.phase, "Liquid");
        assert!(frame.nodes[0].brightness > frame.nodes[1].brightness);
        assert!(frame.nodes[0].size > frame.nodes[1].size);
    }

    #[test]
    fn frame_serializes() {
        let dash = CognitiveDashboard {
            timestamp_ms: 1000,
            collection: "ser".into(),
            tick_duration_ms: 5,
            health: None,
            attention: None,
            hebbian: None,
            thermodynamics: Some(ThermoSnapshot {
                temperature: 0.3,
                entropy: 2.0,
                free_energy: -0.5,
                entropy_rate: 0.0,
                phase: "Liquid".into(),
                explore_modifier: 1.2,
                nodes_analysed: 50,
                mean_energy: 0.4,
                energy_std: 0.1,
                heat_flows_count: 0,
            }),
            maturity: MaturitySnapshot { promotions: 0, demotions: 0 },
            gravity: None,
            shatter: None,
            healing: None,
            learning: None,
            compression: None,
            sharding: None,
            world_model: None,
            flywheel: None,
        };

        let frame = ObservationFrame::from_dashboard(&dash, &[]);
        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.contains("\"temperature\":0.3"));
        assert!(json.contains("\"phase\":\"Liquid\""));
    }

    #[test]
    fn gravity_lines_from_dashboard() {
        let dash = CognitiveDashboard {
            timestamp_ms: 1000,
            collection: "grav".into(),
            tick_duration_ms: 5,
            health: None,
            attention: None,
            hebbian: None,
            thermodynamics: None,
            maturity: MaturitySnapshot { promotions: 0, demotions: 0 },
            gravity: Some(GravitySnapshot {
                forces_computed: 10,
                top_attractions: vec![
                    GravityAttraction {
                        source: "a".into(),
                        target: "b".into(),
                        force: 2.0,
                        source_mass: 1.0,
                        target_mass: 0.5,
                        distance: 0.3,
                    },
                    GravityAttraction {
                        source: "c".into(),
                        target: "d".into(),
                        force: 0.5,
                        source_mass: 0.3,
                        target_mass: 0.2,
                        distance: 0.8,
                    },
                ],
                mean_force: 1.25,
                gravity_wells: 2,
            }),
            shatter: None,
            healing: None,
            learning: None,
            compression: None,
            sharding: None,
            world_model: None,
            flywheel: None,
        };

        let frame = ObservationFrame::from_dashboard(&dash, &[]);
        assert_eq!(frame.gravity_lines.len(), 2);
        // First line should be thicker (higher force)
        assert!(frame.gravity_lines[0].thickness > frame.gravity_lines[1].thickness);
    }
}
