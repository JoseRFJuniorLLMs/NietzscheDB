//! Cognitive Dashboard — unified snapshot of all agency subsystems.
//!
//! Aggregates the latest reports from ECAN, Hebbian LTP, Thermodynamics,
//! Health Observer, Evolution, and Forgetting into a single serializable
//! snapshot. Persisted to CF_META after each agency tick and served via
//! `GET /api/agency/dashboard?collection=`.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────
// Dashboard snapshot
// ─────────────────────────────────────────────

/// Unified cognitive dashboard — single JSON snapshot of the entire agency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDashboard {
    /// Timestamp (Unix millis) when this snapshot was taken.
    pub timestamp_ms: u64,
    /// Collection name.
    pub collection: String,
    /// Duration of the last agency tick (ms).
    pub tick_duration_ms: u64,

    // ── Health ──────────────────────────────────
    pub health: Option<HealthSnapshot>,

    // ── ECAN (Attention Economy) ────────────────
    pub attention: Option<AttentionSnapshot>,

    // ── Hebbian LTP ─────────────────────────────
    pub hebbian: Option<HebbianSnapshot>,

    // ── Cognitive Thermodynamics ────────────────
    pub thermodynamics: Option<ThermoSnapshot>,

    // ── Maturity ────────────────────────────────
    pub maturity: MaturitySnapshot,

    // ── Semantic Gravity (Phase XIV) ────────────
    pub gravity: Option<GravitySnapshot>,

    // ── Shatter Protocol (Phase XVI) ──────────────
    pub shatter: Option<ShatterSnapshot>,

    // ── Self-Healing (Phase XIX) ──────────────────
    #[serde(default)]
    pub healing: Option<HealingSnapshot>,

    // ── Learning Engine (Phase XX) ──────────────────
    #[serde(default)]
    pub learning: Option<LearningSnapshot>,

    // ── Knowledge Compression (Phase XXI) ──────────
    #[serde(default)]
    pub compression: Option<CompressionSnapshot>,

    // ── Hyperbolic Sharding (Phase XXII) ───────────
    #[serde(default)]
    pub sharding: Option<ShardingSnapshot>,

    // ── World Model (Phase XXIII) ──────────────────
    #[serde(default)]
    pub world_model: Option<WorldModelSnapshot>,

    // ── Cognitive Flywheel (Phase XXIV) ────────────
    #[serde(default)]
    pub flywheel: Option<FlywheelSnapshot>,
}

/// Health subsystem snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub global_hausdorff: f32,
    pub mean_energy: f32,
    pub coherence_score: f64,
    pub gap_count: usize,
    pub entropy_spike_count: usize,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// ECAN attention economy snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSnapshot {
    pub participants: usize,
    pub total_bids: usize,
    pub winning_bids: usize,
    pub price: f32,
    pub total_flow: f32,
    /// Top 10 attention receivers (node_id, amount).
    pub top_receivers: Vec<(String, f32)>,
}

/// Hebbian LTP snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HebbianSnapshot {
    pub active_traces: usize,
    pub edges_potentiated: usize,
    pub total_delta: f32,
}

/// Cognitive thermodynamics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermoSnapshot {
    pub temperature: f64,
    pub entropy: f64,
    pub free_energy: f64,
    pub entropy_rate: f64,
    pub phase: String,
    pub explore_modifier: f64,
    pub nodes_analysed: usize,
    pub mean_energy: f64,
    pub energy_std: f64,
    pub heat_flows_count: usize,
}

/// Maturity (promotion/demotion) snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaturitySnapshot {
    pub promotions: usize,
    pub demotions: usize,
}

/// Semantic gravity field snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitySnapshot {
    /// Number of gravitational forces computed.
    pub forces_computed: usize,
    /// Top-K strongest gravitational attractions.
    pub top_attractions: Vec<GravityAttraction>,
    /// Mean gravitational force magnitude.
    pub mean_force: f64,
    /// Number of "gravity wells" (nodes with mass > threshold).
    pub gravity_wells: usize,
}

/// A single gravitational attraction between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityAttraction {
    pub source: String,
    pub target: String,
    pub force: f64,
    pub source_mass: f64,
    pub target_mass: f64,
    pub distance: f64,
}

/// Shatter protocol snapshot (Phase XVI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShatterSnapshot {
    /// Number of super-nodes detected above degree threshold.
    pub super_nodes_detected: usize,
    /// Shatter plans emitted as intents.
    pub plans_emitted: usize,
    /// Total nodes scanned for degree.
    pub nodes_scanned: usize,
    /// Top super-node IDs with their degree.
    pub top_super_nodes: Vec<(String, usize)>,
    /// Count of phantom (ghost) nodes — previously shattered originals.
    pub ghost_nodes: usize,
    /// Count of avatar nodes created by shatter events.
    pub avatar_nodes: usize,
    /// Largest degree (in + out) in the graph.
    pub largest_degree: usize,
    /// Average degree across all non-phantom nodes.
    pub avg_degree: f64,
}

/// Self-Healing subsystem snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealingSnapshot {
    pub nodes_scanned: usize,
    pub boundary_drift_count: usize,
    pub orphan_count: usize,
    pub dead_edge_count: usize,
    pub exhausted_count: usize,
    pub ghost_count: usize,
    pub live_count: usize,
    pub phantom_ratio: f64,
}

/// Learning Engine snapshot (Phase XX).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSnapshot {
    pub total_observations: u64,
    pub access_hotspots: usize,
    pub mutation_hotspots: usize,
    pub booming_sectors: usize,
    pub tracked_entries: usize,
}

/// Knowledge Compression snapshot (Phase XXI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSnapshot {
    pub nodes_scanned: usize,
    pub duplicate_pairs: usize,
    pub stale_clusters: usize,
    pub redundant_paths: usize,
    pub estimated_reduction: usize,
}

/// Hyperbolic Sharding snapshot (Phase XXII).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingSnapshot {
    pub nodes_analyzed: usize,
    pub total_shards: usize,
    pub occupied_shards: usize,
    pub empty_shards: usize,
    pub imbalance_ratio: f64,
    pub rebalance_recommended: bool,
}

/// World Model snapshot (Phase XXIII).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelSnapshot {
    pub observations: u64,
    pub query_rate_mean: f64,
    pub mutation_rate_mean: f64,
    pub anomaly_count: usize,
    pub is_quiet_period: bool,
    pub is_busy_period: bool,
}

/// Cognitive Flywheel snapshot (Phase XXIV).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlywheelSnapshot {
    pub momentum: f64,
    pub cycles: u64,
    pub healthy_streak: u64,
    pub overall_status: String,
    pub is_spinning: bool,
    pub efficiency: f64,
    pub recommendations_count: usize,
}

// ─────────────────────────────────────────────
// Builder
// ─────────────────────────────────────────────

impl CognitiveDashboard {
    /// Build a dashboard from an `AgencyTickReport`.
    pub fn from_tick_report(
        report: &crate::engine::AgencyTickReport,
        collection: &str,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let health = report.health_report.as_ref().map(|h| HealthSnapshot {
            global_hausdorff: h.global_hausdorff,
            mean_energy: h.mean_energy,
            coherence_score: h.coherence_score,
            gap_count: h.gap_count,
            entropy_spike_count: h.entropy_spike_count,
            total_nodes: h.total_nodes,
            total_edges: h.total_edges,
        });

        let attention = report.attention_report.as_ref().map(|a| AttentionSnapshot {
            participants: a.participants,
            total_bids: a.total_bids,
            winning_bids: a.winning_bids,
            price: a.price,
            total_flow: a.total_flow,
            top_receivers: a.top_receivers.iter().take(10)
                .map(|(id, v)| (id.to_string(), *v))
                .collect(),
        });

        let hebbian = report.hebbian_report.as_ref().map(|h| HebbianSnapshot {
            active_traces: h.active_traces,
            edges_potentiated: h.potentiated,
            total_delta: h.total_delta,
        });

        let thermodynamics = report.thermodynamic_report.as_ref().map(|t| ThermoSnapshot {
            temperature: t.temperature,
            entropy: t.entropy,
            free_energy: t.free_energy,
            entropy_rate: t.entropy_rate,
            phase: format!("{}", t.phase),
            explore_modifier: t.explore_modifier,
            nodes_analysed: t.nodes_analysed,
            mean_energy: t.mean_energy,
            energy_std: t.energy_std,
            heat_flows_count: t.heat_flows.len(),
        });

        Self {
            timestamp_ms: now,
            collection: collection.to_string(),
            tick_duration_ms: report.duration_ms,
            health,
            attention,
            hebbian,
            thermodynamics,
            maturity: MaturitySnapshot {
                promotions: report.promotions,
                demotions: report.demotions,
            },
            gravity: report.gravity_report.as_ref().map(|g| GravitySnapshot {
                forces_computed: g.pairs_evaluated,
                top_attractions: g.top_attractions.iter().take(10).map(|f| GravityAttraction {
                    source: f.source.to_string(),
                    target: f.target.to_string(),
                    force: f.force,
                    source_mass: f.source_mass,
                    target_mass: f.target_mass,
                    distance: f.distance,
                }).collect(),
                mean_force: g.mean_force,
                gravity_wells: g.wells.len(),
            }),
            shatter: report.shatter_report.as_ref().map(|s| ShatterSnapshot {
                super_nodes_detected: s.super_nodes_detected,
                plans_emitted: s.plans_emitted,
                nodes_scanned: s.nodes_scanned,
                top_super_nodes: s.candidates.iter().take(10)
                    .map(|c| (c.node_id.to_string(), c.degree))
                    .collect(),
                ghost_nodes: s.ghost_nodes,
                avatar_nodes: s.avatar_nodes,
                largest_degree: s.largest_degree,
                avg_degree: s.avg_degree,
            }),
            healing: report.healing_report.as_ref().map(|h| HealingSnapshot {
                nodes_scanned: h.nodes_scanned,
                boundary_drift_count: h.boundary_drift_count,
                orphan_count: h.orphan_count,
                dead_edge_count: h.dead_edge_count,
                exhausted_count: h.exhausted_count,
                ghost_count: h.ghost_count,
                live_count: h.live_count,
                phantom_ratio: h.phantom_ratio,
            }),
            learning: report.learning_report.as_ref().map(|l| LearningSnapshot {
                total_observations: l.total_observations,
                access_hotspots: l.access_hotspots.len(),
                mutation_hotspots: l.mutation_hotspots.len(),
                booming_sectors: l.booming_sectors.len(),
                tracked_entries: l.tracked_access + l.tracked_mutations,
            }),
            compression: report.compression_report.as_ref().map(|c| CompressionSnapshot {
                nodes_scanned: c.nodes_scanned,
                duplicate_pairs: c.duplicate_pairs,
                stale_clusters: c.stale_clusters,
                redundant_paths: c.redundant_paths,
                estimated_reduction: c.estimated_reduction,
            }),
            sharding: report.sharding_report.as_ref().map(|s| ShardingSnapshot {
                nodes_analyzed: s.nodes_analyzed,
                total_shards: s.total_shards,
                occupied_shards: s.occupied_shards,
                empty_shards: s.empty_shards,
                imbalance_ratio: s.imbalance_ratio,
                rebalance_recommended: s.rebalance_recommended,
            }),
            world_model: report.world_snapshot.as_ref().map(|w| WorldModelSnapshot {
                observations: w.observations,
                query_rate_mean: w.query_rate_mean,
                mutation_rate_mean: w.mutation_rate_mean,
                anomaly_count: w.anomalies.len(),
                is_quiet_period: w.is_quiet_period,
                is_busy_period: w.is_busy_period,
            }),
            flywheel: report.flywheel_report.as_ref().map(|f| FlywheelSnapshot {
                momentum: f.momentum,
                cycles: f.cycles,
                healthy_streak: f.healthy_streak,
                overall_status: f.overall_status.clone(),
                is_spinning: f.is_spinning,
                efficiency: f.efficiency,
                recommendations_count: f.recommendations.len(),
            }),
        }
    }

    /// CF_META key for dashboard persistence.
    pub fn meta_key() -> &'static str {
        "agency:cognitive_dashboard"
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_serializes_to_json() {
        let dash = CognitiveDashboard {
            timestamp_ms: 1709000000000,
            collection: "test".to_string(),
            tick_duration_ms: 42,
            health: Some(HealthSnapshot {
                global_hausdorff: 1.2,
                mean_energy: 0.5f32,
                coherence_score: 0.8,
                gap_count: 3,
                entropy_spike_count: 1,
                total_nodes: 1000,
                total_edges: 5000,
            }),
            attention: Some(AttentionSnapshot {
                participants: 100,
                total_bids: 200,
                winning_bids: 50,
                price: 1.1,
                total_flow: 25.0,
                top_receivers: vec![("abc".into(), 5.0)],
            }),
            hebbian: Some(HebbianSnapshot {
                active_traces: 30,
                edges_potentiated: 10,
                total_delta: 0.15,
            }),
            thermodynamics: Some(ThermoSnapshot {
                temperature: 0.42,
                entropy: 3.5,
                free_energy: -1.2,
                entropy_rate: 0.01,
                phase: "Liquid".to_string(),
                explore_modifier: 1.0,
                nodes_analysed: 1000,
                mean_energy: 0.5,
                energy_std: 0.21,
                heat_flows_count: 15,
            }),
            maturity: MaturitySnapshot { promotions: 5, demotions: 2 },
            gravity: Some(GravitySnapshot {
                forces_computed: 100,
                top_attractions: vec![GravityAttraction {
                    source: "a".into(),
                    target: "b".into(),
                    force: 2.5,
                    source_mass: 1.0,
                    target_mass: 0.8,
                    distance: 0.5,
                }],
                mean_force: 0.3,
                gravity_wells: 12,
            }),
            shatter: None,
            healing: None,
            learning: None,
            compression: None,
            sharding: None,
            world_model: None,
            flywheel: None,
        };

        let json = serde_json::to_string(&dash).unwrap();
        assert!(json.contains("\"temperature\":0.42"));
        assert!(json.contains("\"phase\":\"Liquid\""));
        assert!(json.contains("\"gravity_wells\":12"));
    }

    #[test]
    fn dashboard_roundtrip() {
        let dash = CognitiveDashboard {
            timestamp_ms: 123,
            collection: "rt".into(),
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

        let bytes = serde_json::to_vec(&dash).unwrap();
        let back: CognitiveDashboard = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.collection, "rt");
        assert_eq!(back.timestamp_ms, 123);
    }
}
