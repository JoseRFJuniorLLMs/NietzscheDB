//! Nezhmetdinov Daemon — The Forgetting Engine.
//!
//! Named after Rashid Nezhmetdinov, the chess grandmaster famous for
//! brilliant sacrifices. This daemon sacrifices data persistence for
//! topological clarity — proving that intelligent forgetting creates
//! a more lucid mind.
//!
//! ## Pipeline (per tick)
//!
//! 1. Scan all non-phantom nodes (up to max_scan)
//! 2. Compute vitality V(n) for each node
//! 3. Apply Triple Condition: V(n) < θ AND e(n) < θ_e AND κ(n) = 0
//! 4. Ricci curvature veto: simulate removal, abort if ΔRicci < -ε
//! 5. Emit `ForgettingJudgment` events for condemned nodes
//! 6. Build MikhailThallReport with cycle statistics
//!
//! This daemon is **read-only** — it only scans and emits events.
//! The actual hard delete is performed by the server via AgencyIntent.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage, CausalType};

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};
use crate::forgetting::{
    VitalityWeights, VitalityInput, nezhmetdinov_vitality,
    Verdict, ForgetteringJudgment, MikhailThallReport,
    NezhmetdinovConfig, RicciShield,
};

use super::{AgencyDaemon, DaemonReport};

/// The Nezhmetdinov Forgetting Engine Daemon.
///
/// Scans the graph, evaluates each node's vitality, and emits
/// forgetting judgments for the reactor to execute.
pub struct NezhmetdinovDaemon {
    weights: VitalityWeights,
    nezhmetdinov_config: NezhmetdinovConfig,
}

impl NezhmetdinovDaemon {
    pub fn new() -> Self {
        Self {
            weights: VitalityWeights::default(),
            nezhmetdinov_config: NezhmetdinovConfig::from_env(),
        }
    }

    pub fn with_config(config: NezhmetdinovConfig) -> Self {
        Self {
            weights: VitalityWeights::default(),
            nezhmetdinov_config: config,
        }
    }

    /// Count Minkowski timelike/lightlike edges for a node (causal centrality κ).
    fn causal_centrality(
        &self,
        node_id: &Uuid,
        adjacency: &AdjacencyIndex,
        storage: &GraphStorage,
    ) -> usize {
        let mut causal_count = 0usize;

        // Check outgoing edges
        for entry in adjacency.entries_out(node_id) {
            if let Ok(Some(edge)) = storage.get_edge(&entry.edge_id) {
                match edge.causal_type {
                    CausalType::Timelike | CausalType::Lightlike => {
                        causal_count += 1;
                    }
                    _ => {}
                }
            }
        }

        // Check incoming edges
        for entry in adjacency.entries_in(node_id) {
            if let Ok(Some(edge)) = storage.get_edge(&entry.edge_id) {
                match edge.causal_type {
                    CausalType::Timelike | CausalType::Lightlike => {
                        causal_count += 1;
                    }
                    _ => {}
                }
            }
        }

        causal_count
    }

    /// Compute emotional toxicity from valence and arousal.
    /// τ = max(0, -valence) × arousal
    fn compute_toxicity(valence: f32, arousal: f32) -> f32 {
        let negative_valence = (-valence).max(0.0);
        (negative_valence * arousal).clamp(0.0, 1.0)
    }

    /// Compute elite proximity: distance from node to nearest elite centroid.
    /// For MVP, we use a simple heuristic based on energy and depth.
    fn compute_elite_distance(energy: f32, depth: f32) -> f32 {
        // Elites tend to have high energy and moderate depth.
        // Distance from "ideal elite" profile:
        let elite_energy = 0.8;
        let elite_depth = 0.3;
        let de = (energy - elite_energy).abs();
        let dd = (depth - elite_depth).abs();
        (de * de + dd * dd).sqrt().min(1.0)
    }
}

impl Default for NezhmetdinovDaemon {
    fn default() -> Self {
        Self::new()
    }
}

impl AgencyDaemon for NezhmetdinovDaemon {
    fn name(&self) -> &str { "nezhmetdinov" }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        _config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();
        let nezh = &self.nezhmetdinov_config;

        if !nezh.enabled {
            return Ok(DaemonReport {
                daemon_name: "nezhmetdinov".into(),
                events_emitted: 0,
                nodes_scanned: 0,
                duration_us: t0.elapsed().as_micros() as u64,
                details: vec!["disabled".into()],
            });
        }

        // Build adjacency maps for Ricci calculation
        let mut adj_out: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        let mut adj_in: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        let mut active_nodes: Vec<Uuid> = Vec::new();

        // ── Phase 1: Scan nodes and collect data ──────────────────────
        let mut nodes_scanned = 0usize;
        let mut judgments: Vec<ForgetteringJudgment> = Vec::new();

        struct NodeData {
            id: Uuid,
            energy: f32,
            hausdorff: f32,
            depth: f32,
            valence: f32,
            arousal: f32,
            is_phantom: bool,
        }

        let mut node_data: Vec<NodeData> = Vec::new();

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };
            nodes_scanned += 1;

            if nodes_scanned > nezh.max_scan_per_tick {
                break;
            }

            if meta.is_phantom {
                continue;
            }

            active_nodes.push(meta.id);

            // Build adjacency for this node using entries (edge_id + neighbor_id)
            let out_entries = adjacency.entries_out(&meta.id);
            let in_entries = adjacency.entries_in(&meta.id);

            let out_neighbors: Vec<Uuid> = out_entries.iter().map(|e| e.neighbor_id).collect();
            let in_neighbors: Vec<Uuid> = in_entries.iter().map(|e| e.neighbor_id).collect();

            adj_out.insert(meta.id, out_neighbors);
            adj_in.insert(meta.id, in_neighbors);

            node_data.push(NodeData {
                id: meta.id,
                energy: meta.energy,
                hausdorff: meta.hausdorff_local,
                depth: meta.depth,
                valence: meta.valence,
                arousal: meta.arousal,
                is_phantom: meta.is_phantom,
            });
        }

        // ── Phase 2: Evaluate each node ──────────────────────────────
        let ricci_shield = RicciShield::new(nezh.ricci_epsilon);

        for nd in &node_data {
            let causal = self.causal_centrality(&nd.id, adjacency, storage);
            let toxicity = Self::compute_toxicity(nd.valence, nd.arousal);
            let elite_distance = Self::compute_elite_distance(nd.energy, nd.depth);

            // Compute entropy delta (simplified: use hausdorff as proxy)
            let entropy_delta = 1.0 - nd.hausdorff.clamp(0.0, 2.0) / 2.0;

            let input = VitalityInput::new(
                nd.energy,
                nd.hausdorff,
                entropy_delta,
                elite_distance,
                causal,
                toxicity,
            );

            let vitality = nezhmetdinov_vitality(&self.weights, &input);

            // ── Triple Condition ──────────────────────────────────
            let cond1_low_vitality = vitality < nezh.vitality_threshold;
            let cond2_low_energy = nd.energy < nezh.energy_threshold;
            let cond3_no_causal = causal == 0;

            let verdict = if cond1_low_vitality && cond2_low_energy && cond3_no_causal {
                // Triple condition met — check Ricci shield
                if ricci_shield.quick_veto(&nd.id, &adj_out, &adj_in) {
                    Verdict::RicciShielded
                } else {
                    Verdict::Condemned
                }
            } else if toxicity > nezh.toxicity_threshold && causal == 0 {
                Verdict::Toxic
            } else if vitality > nezh.sacred_vitality_threshold
                || causal >= nezh.bounds.sacred_causal_threshold
            {
                Verdict::Sacred
            } else {
                Verdict::Dormant
            };

            let judgment = ForgetteringJudgment {
                node_id: nd.id,
                vitality_score: vitality,
                energy: nd.energy,
                causal_centrality: causal,
                ricci_delta: 0.0, // Quick veto doesn't compute full delta
                toxicity,
                verdict,
                cycle: 0, // Set by caller
                reason: match verdict {
                    Verdict::Condemned => format!(
                        "Triple: V={:.3}<{:.3}, e={:.3}<{:.3}, κ=0",
                        vitality, nezh.vitality_threshold,
                        nd.energy, nezh.energy_threshold,
                    ),
                    Verdict::RicciShielded => "Triple met but Ricci veto saved".into(),
                    Verdict::Toxic => format!("τ={:.3} > {:.3}, κ=0", toxicity, nezh.toxicity_threshold),
                    Verdict::Sacred => format!("V={:.3} or κ={}", vitality, causal),
                    Verdict::Dormant => "awaiting".into(),
                },
            };

            // Emit events for condemned and toxic nodes
            if verdict.is_death_sentence() {
                bus.publish(AgencyEvent::ForgettingCondemned {
                    node_id: nd.id,
                    vitality: vitality,
                    energy: nd.energy,
                    reason: judgment.reason.clone(),
                });
            }

            judgments.push(judgment);
        }

        // ── Phase 3: Build report ────────────────────────────────────
        let duration_us = t0.elapsed().as_micros() as u64;
        let report = MikhailThallReport::from_judgments(0, &judgments, duration_us);

        let details = vec![
            format!("evaluated={}", report.nodes_evaluated),
            format!("condemned={}", report.condemned_count),
            format!("toxic={}", report.toxic_count),
            format!("ricci_shielded={}", report.ricci_shielded_count),
            format!("sacred={}", report.sacred_count),
            format!("mean_V={:.3}", report.mean_vitality),
        ];

        Ok(DaemonReport {
            daemon_name: "nezhmetdinov".into(),
            events_emitted: report.condemned_count + report.toxic_count,
            nodes_scanned,
            duration_us,
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, PoincareVector};
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.meta.energy = energy;
        node.meta.is_phantom = false;
        node
    }

    #[test]
    fn daemon_runs_on_empty_graph() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();

        let daemon = NezhmetdinovDaemon::new();
        let report = daemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.events_emitted, 0);
    }

    #[test]
    fn high_energy_nodes_are_sacred() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();

        // Create high-energy nodes
        for i in 0..5 {
            let node = make_node(0.1 * i as f32, 0.05, 0.9);
            storage.put_node(&node).unwrap();
        }

        let daemon = NezhmetdinovDaemon::new();
        let report = daemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 5);
        // High energy nodes should NOT be condemned
        assert_eq!(report.events_emitted, 0, "high energy nodes should be sacred");
    }

    #[test]
    fn low_energy_isolated_nodes_condemned() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let config = AgencyConfig::default();

        // Create low-energy isolated nodes (no edges, no causal links)
        for i in 0..5 {
            let mut node = make_node(0.1 * i as f32, 0.05, 0.02);
            node.meta.hausdorff_local = 0.1;
            storage.put_node(&node).unwrap();
        }

        // Use higher vitality threshold so that low-energy isolated nodes
        // fall below it (default weights make V ≈ 0.32 for these nodes)
        let mut nezh_config = NezhmetdinovConfig::default();
        nezh_config.vitality_threshold = 0.40;
        let daemon = NezhmetdinovDaemon::with_config(nezh_config);
        let report = daemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert!(report.events_emitted > 0, "low energy isolated nodes should be condemned");
    }
}
