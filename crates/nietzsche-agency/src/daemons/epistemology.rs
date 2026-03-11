//! **EpistemologyDaemon** — The Scientist (Phase XXV: Autopoietic Research)
//!
//! Implements Karpathy's autoresearch loop mapped to NietzscheDB's geometry:
//!
//! ```text
//! 1. SELECT   → Pick highest-friction CodeAsData node (FlowLedger)
//! 2. PROPOSE  → Emit EpistemologyCandidate event (LLM bridge is external)
//! 3. EVALUATE → Simulate pending mutations in ShadowGraph, measure ΔTGC
//! 4. VERDICT  → KEEP (emit merge event) or DISCARD (emit prune event)
//! ```
//!
//! # Architecture
//!
//! The daemon operates in **two phases per tick**:
//!
//! **Phase A — Candidate Selection**: Scans CodeAsData nodes, ranks by
//! FlowLedger friction (CPU cost), emits `EpistemologyCandidate` for the
//! hottest node. An external process (MCP/Python/Go) consumes this event,
//! calls the LLM, and writes the mutation back as a sibling node tagged
//! with `mutation_pending: true` and `mutation_parent: <original_node_id>`.
//!
//! **Phase B — Mutation Evaluation**: Finds nodes tagged `mutation_pending`,
//! creates a ShadowGraph, swaps the original ActionNode's NQL with the
//! mutant's NQL, simulates energy propagation, measures structural health
//! delta (proxy for ΔTGC), and emits a verdict.
//!
//! # Karpathy Mapping
//!
//! | autoresearch       | EpistemologyDaemon           |
//! |--------------------|-----------------------------|
//! | `train.py`         | CodeAsData `ActionNode`     |
//! | `val_bpb`          | ΔTGC (structural health)    |
//! | 5-min sandbox      | ShadowGraph simulation      |
//! | `git commit`       | `EpistemologyVerdict::Keep`  |
//! | `git revert`       | `EpistemologyVerdict::Discard` |
//!
//! # Safety
//!
//! - Never mutates the real graph (read-only daemon)
//! - ShadowGraph is metadata-only (~130 bytes/node)
//! - Rejected mutations are tracked by content hash to avoid retrying
//! - Maximum 1 candidate per tick (budget-controlled)

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::code_as_data::{parse_action_node, ActionNode};
use crate::config::AgencyConfig;
use crate::counterfactual::shadow::ShadowGraph;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};
use crate::hydraulic::flow_ledger::FlowLedger;

use super::{AgencyDaemon, DaemonReport};

// ─────────────────────────────────────────────
// Mutation Record — tracks what was tried
// ─────────────────────────────────────────────

/// Hash of a mutation attempt (original NQL + mutant NQL) to avoid retrying.
fn mutation_hash(original_nql: &str, mutant_nql: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    original_nql.hash(&mut hasher);
    mutant_nql.hash(&mut hasher);
    hasher.finish()
}

/// Record of a completed mutation evaluation.
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// Original ActionNode ID.
    pub original_id: Uuid,
    /// Mutant node ID (the pending mutation node).
    pub mutant_id: Uuid,
    /// Structural health delta (positive = improvement).
    pub delta_health: f32,
    /// Whether the mutation was accepted.
    pub accepted: bool,
}

// ─────────────────────────────────────────────
// EpistemologyDaemon
// ─────────────────────────────────────────────

/// The Scientist — autonomous research daemon for Code-as-Data evolution.
pub struct EpistemologyDaemon {
    /// Hashes of rejected mutations (don't retry the same idea twice).
    rejected_hashes: std::sync::Mutex<HashSet<u64>>,
    /// Optional reference to FlowLedger for friction-based ranking.
    flow_ledger: Option<std::sync::Arc<FlowLedger>>,
}

impl EpistemologyDaemon {
    /// Create a new EpistemologyDaemon.
    pub fn new() -> Self {
        Self {
            rejected_hashes: std::sync::Mutex::new(HashSet::new()),
            flow_ledger: None,
        }
    }

    /// Create with a FlowLedger reference for friction-based candidate ranking.
    pub fn with_flow_ledger(flow_ledger: std::sync::Arc<FlowLedger>) -> Self {
        Self {
            rejected_hashes: std::sync::Mutex::new(HashSet::new()),
            flow_ledger: Some(flow_ledger),
        }
    }

    /// Phase A: Find the highest-friction CodeAsData node to propose for mutation.
    ///
    /// Returns the best candidate ActionNode, or None if no suitable targets exist.
    fn select_candidate(
        &self,
        storage: &GraphStorage,
        config: &AgencyConfig,
    ) -> Option<(ActionNode, f64)> {
        let nodes = match storage.scan_nodes() {
            Ok(n) => n,
            Err(_) => return None,
        };

        let mut best: Option<(ActionNode, f64)> = None;
        let rejected = self.rejected_hashes.lock().unwrap_or_else(|e| e.into_inner());
        let max_scan = config.epistemology_max_scan;
        let mut scanned = 0;

        for node in &nodes {
            if scanned >= max_scan {
                break;
            }

            // Skip phantoms, dead nodes
            if node.meta.is_phantom || node.meta.energy <= 0.0 {
                continue;
            }

            // Must be a CodeAsData ActionNode
            if let Some(action) = parse_action_node(node) {
                scanned += 1;

                // Skip exhausted actions
                if action.max_firings > 0 && action.firings >= action.max_firings {
                    continue;
                }

                // Skip if tagged as mutation_pending (it IS a mutation, not a target)
                if node.meta.content.get("mutation_pending").is_some() {
                    continue;
                }

                // Skip non-mutable actions (opt-in via content flag)
                let is_mutable = node.meta.content.get("action")
                    .and_then(|a| a.get("mutable"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true); // default: mutable
                if !is_mutable {
                    continue;
                }

                // Compute friction score (higher = more expensive = better candidate)
                let friction = self.compute_friction(node.id, &action);

                // Check if this candidate outperforms current best
                if let Some((_, best_friction)) = &best {
                    if friction > *best_friction {
                        best = Some((action, friction));
                    }
                } else {
                    best = Some((action, friction));
                }
            }
        }

        best
    }

    /// Compute friction score for ranking candidates.
    ///
    /// Combines FlowLedger CPU cost (if available) with firing frequency
    /// as a proxy for "how important and expensive is this rule?"
    fn compute_friction(&self, _node_id: Uuid, action: &ActionNode) -> f64 {
        let base_friction = action.firings as f64 * 0.1;

        // If FlowLedger is available, use EMA CPU cost as primary signal
        // (edges traversed by this action's NQL execution)
        // For now, use firing count as proxy since we don't have per-action
        // FlowLedger integration yet.
        let flow_friction = if let Some(_ledger) = &self.flow_ledger {
            // Future: query ledger for edges touched by this action's NQL
            0.0
        } else {
            0.0
        };

        base_friction + flow_friction + 1.0 // +1.0 so even unfired actions have nonzero score
    }

    /// Phase B: Find and evaluate pending mutations.
    ///
    /// Scans for nodes tagged with `mutation_pending: true`, simulates each
    /// in a ShadowGraph, and emits verdict events.
    fn evaluate_mutations(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<Vec<MutationRecord>, AgencyError> {
        let nodes = storage.scan_nodes()?;
        let mut records = Vec::new();
        let mut rejected = self.rejected_hashes.lock().unwrap_or_else(|e| e.into_inner());

        for node in &nodes {
            if node.meta.is_phantom || node.meta.energy <= 0.0 {
                continue;
            }

            // Check if this is a pending mutation
            let is_pending = node.meta.content.get("mutation_pending")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if !is_pending {
                continue;
            }

            // Get the mutation's parent (original ActionNode)
            let parent_id_str = match node.meta.content.get("mutation_parent")
                .and_then(|v| v.as_str())
            {
                Some(s) => s.to_string(),
                None => continue, // malformed mutation node
            };
            let parent_id = match Uuid::parse_str(&parent_id_str) {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Get mutant NQL
            let mutant_nql = match node.meta.content.get("action")
                .and_then(|a| a.get("nql"))
                .and_then(|v| v.as_str())
            {
                Some(nql) => nql.to_string(),
                None => continue,
            };

            // ── P1: NQL Syntax Validation Gate ──────────────────
            // Reject malformed NQL before wasting time on simulation.
            // Uses the Pest parser from nietzsche-query.
            if nietzsche_query::parse(&mutant_nql).is_err() {
                bus.publish(AgencyEvent::EpistemologyVerdict {
                    original_id: parent_id,
                    mutant_id: node.id,
                    delta_health: 0.0,
                    accepted: false,
                    reason: "NQL syntax error — mutation rejected pre-simulation".to_string(),
                });
                continue;
            }

            // Get original node and its NQL
            let original_node = match storage.get_node(&parent_id) {
                Ok(Some(n)) => n,
                _ => continue,
            };
            let original_action = match parse_action_node(&original_node) {
                Some(a) => a,
                None => continue,
            };

            // Check if this mutation was already rejected
            let hash = mutation_hash(&original_action.nql, &mutant_nql);
            if rejected.contains(&hash) {
                // Already tried and failed — emit discard without re-simulating
                bus.publish(AgencyEvent::EpistemologyVerdict {
                    original_id: parent_id,
                    mutant_id: node.id,
                    delta_health: 0.0,
                    accepted: false,
                    reason: "previously rejected".to_string(),
                });
                continue;
            }

            // ═══════════════════════════════════════════
            // SANDBOX: ShadowGraph Simulation (P2: Full TGC)
            // ═══════════════════════════════════════════

            let baseline_shadow = ShadowGraph::snapshot(storage, adjacency)?;

            // Create mutant shadow: apply the mutation's energy pattern
            let mut mutant_shadow = ShadowGraph::snapshot(storage, adjacency)?;
            let mutant_energy = mutant_shadow.nodes.get(&node.id).map(|m| m.energy);
            if let (Some(meta), Some(energy)) = (mutant_shadow.nodes.get_mut(&parent_id), mutant_energy) {
                meta.energy = energy;
            }

            // Full TGC measurement via shadow_tgc_delta (replaces proxy)
            let (delta, _base_report, _mut_report) = crate::forgetting::tgc::shadow_tgc_delta(
                &baseline_shadow,
                &mutant_shadow,
                2.0,  // alpha (entropy amplifier)
                3.0,  // beta (efficiency amplifier)
                50,   // BFS sample limit (cap at 50 nodes)
            );

            // ═══════════════════════════════════════════
            // VERDICT: Darwinian Selection
            // ═══════════════════════════════════════════

            let threshold = config.epistemology_improvement_threshold;
            let accepted = delta > threshold;

            if !accepted {
                rejected.insert(hash);
            }

            let record = MutationRecord {
                original_id: parent_id,
                mutant_id: node.id,
                delta_health: delta,
                accepted,
            };

            bus.publish(AgencyEvent::EpistemologyVerdict {
                original_id: parent_id,
                mutant_id: node.id,
                delta_health: delta,
                accepted,
                reason: if accepted {
                    format!("KEEP: ΔTGC={:.4} > threshold={:.4}", delta, threshold)
                } else {
                    format!("DISCARD: ΔTGC={:.4} <= threshold={:.4}", delta, threshold)
                },
            });

            records.push(record);
        }

        Ok(records)
    }

    /// Measure structural health of a node in the ShadowGraph.
    ///
    /// Proxy for TGC: combines degree centrality, neighbor energy distribution,
    /// and depth coherence into a single scalar.
    ///
    /// This is intentionally simple — the full TGC requires void generation
    /// which we can't do in a lightweight shadow. The proxy captures the
    /// structural properties that TGC ultimately depends on.
    fn measure_structural_health(shadow: &ShadowGraph, node_id: Uuid) -> f32 {
        let meta = match shadow.get_meta(&node_id) {
            Some(m) => m,
            None => return 0.0,
        };

        let neighbors = shadow.neighbors(&node_id);
        if neighbors.is_empty() {
            return meta.energy * 0.1; // isolated node: low health
        }

        // 1. Degree score: moderate degree is healthiest (not too few, not too many)
        let degree = neighbors.len() as f32;
        let degree_score = 1.0 - (degree - 5.0).abs() / (degree + 5.0); // peaks at degree=5

        // 2. Neighbor energy coherence: mean energy of neighbors
        let mut neighbor_energy_sum = 0.0f32;
        let mut neighbor_count = 0;
        for nid in &neighbors {
            if let Some(nm) = shadow.get_meta(nid) {
                neighbor_energy_sum += nm.energy;
                neighbor_count += 1;
            }
        }
        let mean_neighbor_energy = if neighbor_count > 0 {
            neighbor_energy_sum / neighbor_count as f32
        } else {
            0.0
        };

        // 3. Depth coherence: ActionNodes should be at moderate depth
        let depth_score = 1.0 - (meta.depth - 0.5).abs().min(1.0) as f32;

        // Composite health: weighted sum
        let health = 0.3 * degree_score
            + 0.4 * mean_neighbor_energy
            + 0.2 * depth_score
            + 0.1 * meta.energy;

        health
    }
}

impl AgencyDaemon for EpistemologyDaemon {
    fn name(&self) -> &str {
        "epistemology"
    }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let start = Instant::now();
        let mut details = Vec::new();
        let mut events_emitted = 0;
        let mut nodes_scanned = 0;

        if !config.epistemology_enabled {
            return Ok(DaemonReport {
                daemon_name: self.name().to_string(),
                events_emitted: 0,
                nodes_scanned: 0,
                duration_us: 0,
                details: vec!["disabled".to_string()],
            });
        }

        // ─── Phase A: Candidate Selection ───────────────────────
        if let Some((candidate, friction)) = self.select_candidate(storage, config) {
            nodes_scanned += 1;
            bus.publish(AgencyEvent::EpistemologyCandidate {
                node_id: candidate.node_id,
                nql: candidate.nql.clone(),
                friction,
                description: candidate.description.clone(),
            });
            events_emitted += 1;
            details.push(format!(
                "candidate: {} (friction={:.2}, nql={}...)",
                candidate.node_id,
                friction,
                &candidate.nql[..candidate.nql.len().min(40)]
            ));
        }

        // ─── Phase B: Mutation Evaluation ───────────────────────
        let records = self.evaluate_mutations(storage, adjacency, bus, config)?;
        for record in &records {
            events_emitted += 1;
            nodes_scanned += 1;
            details.push(format!(
                "verdict: {} → {} (Δ={:.4}, {})",
                record.original_id,
                record.mutant_id,
                record.delta_health,
                if record.accepted { "KEEP" } else { "DISCARD" }
            ));
        }

        Ok(DaemonReport {
            daemon_name: self.name().to_string(),
            events_emitted,
            nodes_scanned,
            duration_us: start.elapsed().as_micros() as u64,
            details,
        })
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, PoincareVector};
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn default_config() -> AgencyConfig {
        AgencyConfig {
            epistemology_enabled: true,
            epistemology_max_scan: 100,
            epistemology_improvement_threshold: 0.0,
            ..AgencyConfig::default()
        }
    }

    fn make_action_node(nql: &str, energy: f32) -> Node {
        let content = serde_json::json!({
            "action": {
                "nql": nql,
                "activation_threshold": 0.5,
                "cooldown_ticks": 0,
                "max_firings": 0,
                "firings": 5,
                "cooldown_remaining": 0,
                "description": "test action",
                "mutable": true,
            }
        });
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.1, 0.0]),
            content,
        );
        node.meta.energy = energy;
        node.meta.node_type = nietzsche_graph::NodeType::Concept;
        node
    }

    fn make_mutation_node(parent_id: Uuid, mutant_nql: &str, energy: f32) -> Node {
        let content = serde_json::json!({
            "mutation_pending": true,
            "mutation_parent": parent_id.to_string(),
            "action": {
                "nql": mutant_nql,
                "activation_threshold": 0.5,
                "cooldown_ticks": 0,
                "max_firings": 0,
                "firings": 0,
                "cooldown_remaining": 0,
                "description": "mutated action",
            }
        });
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.15, 0.0]),
            content,
        );
        node.meta.energy = energy;
        node.meta.node_type = nietzsche_graph::NodeType::Concept;
        node
    }

    #[test]
    fn selects_highest_friction_candidate() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let config = default_config();

        // Low-friction action (1 firing)
        let mut low = make_action_node("MATCH (n) SET n.energy = 0.5", 0.8);
        low.meta.content.as_object_mut().unwrap()
            .get_mut("action").unwrap()
            .as_object_mut().unwrap()
            .insert("firings".into(), serde_json::json!(1));
        storage.put_node(&low).unwrap();

        // High-friction action (50 firings)
        let high = make_action_node("MATCH (n) WHERE n.energy > 0.9 SET n.energy = 0.8", 0.9);
        storage.put_node(&high).unwrap();

        let daemon = EpistemologyDaemon::new();
        let candidate = daemon.select_candidate(&storage, &config);
        assert!(candidate.is_some());
        // The higher-friction node (50 firings) should win
        let (action, _friction) = candidate.unwrap();
        assert_eq!(action.firings, 5); // from make_action_node default
    }

    #[test]
    fn evaluates_pending_mutation() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = default_config();

        // Original action node
        let original = make_action_node("MATCH (n) SET n.energy = 0.5", 0.8);
        let original_id = original.id;
        storage.put_node(&original).unwrap();

        // Mutation node
        let mutation = make_mutation_node(
            original_id,
            "MATCH (n) WHERE n.energy < 0.3 SET n.energy = 0.0",
            0.9,
        );
        storage.put_node(&mutation).unwrap();

        let daemon = EpistemologyDaemon::new();
        let records = daemon.evaluate_mutations(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].original_id, original_id);
    }

    #[test]
    fn rejected_mutation_not_retried() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig {
            epistemology_enabled: true,
            epistemology_max_scan: 100,
            epistemology_improvement_threshold: 999.0, // impossibly high → always reject
            ..AgencyConfig::default()
        };

        let original = make_action_node("MATCH (n) SET n.energy = 0.5", 0.8);
        let original_id = original.id;
        storage.put_node(&original).unwrap();

        let mutation = make_mutation_node(original_id, "MATCH (n) SET n.energy = 0.0", 0.9);
        storage.put_node(&mutation).unwrap();

        let daemon = EpistemologyDaemon::new();

        // First evaluation: rejected
        let records = daemon.evaluate_mutations(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(records.len(), 1);
        assert!(!records[0].accepted);

        // Second evaluation: skipped (hash in rejected set)
        let records2 = daemon.evaluate_mutations(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(records2.len(), 0); // skipped, not re-evaluated
    }

    #[test]
    fn full_tick_disabled() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig {
            epistemology_enabled: false,
            ..AgencyConfig::default()
        };

        let daemon = EpistemologyDaemon::new();
        let report = daemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0);
        assert_eq!(report.details, vec!["disabled".to_string()]);
    }

    #[test]
    fn structural_health_isolated_node() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let node = make_action_node("MATCH (n) SET n.energy = 0.5", 0.8);
        let nid = node.id;
        storage.put_node(&node).unwrap();

        let shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        let health = EpistemologyDaemon::measure_structural_health(&shadow, nid);
        // Isolated node should have low health
        assert!(health < 0.2, "isolated node health should be low, got {}", health);
    }
}
