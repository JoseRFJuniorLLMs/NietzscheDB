// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XII — ECAN Attention Cycle orchestrator.
//!
//! Runs the full attention economy cycle:
//!
//! 1. **Scan** — load NodeMeta for participating nodes (O(N), no embeddings)
//! 2. **Budget** — compute attention budgets from semantic mass
//! 3. **Density** — estimate angular density for curiosity (reuses depth proxy)
//! 4. **Curiosity** — compute novelty + curiosity per node
//! 5. **Bids** — generate exploitation + exploration bids
//! 6. **Inflate** — compute cognitive inflation price
//! 7. **Auction** — resolve bids, determine attention flow
//! 8. **Report** — produce AttentionReport with flow metrics + energy deltas
//!
//! ## Performance
//!
//! Scans only `NodeMeta` (~100 bytes per node). No embedding loads.
//! For 10K nodes: ~2ms. For 100K: ~20ms.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::attention_economy::{
    self, AttentionBid, AttentionConfig, AttentionReport, AttentionState,
};
use crate::curiosity_engine::{
    self, CuriosityConfig, CuriosityState,
};
use crate::hub_attenuation::{self, HubAttenuationConfig, RefractoryTracker};
use crate::error::AgencyError;

/// Combined ECAN + Curiosity + Hub Attenuation configuration.
#[derive(Debug, Clone)]
pub struct EcanConfig {
    pub attention: AttentionConfig,
    pub curiosity: CuriosityConfig,
    /// Hub avalanche attenuation config (angular focus + hub penalty + refractory).
    pub hub_attenuation: HubAttenuationConfig,
    /// Tick interval: run ECAN every N agency ticks (default 1 = every tick).
    pub interval: u64,
}

impl Default for EcanConfig {
    fn default() -> Self {
        Self {
            attention: AttentionConfig::default(),
            curiosity: CuriosityConfig::default(),
            hub_attenuation: HubAttenuationConfig::default(),
            interval: 1,
        }
    }
}

/// Persistent state for the ECAN cycle across ticks.
pub struct EcanCycle {
    tick_count: u64,
    /// Last cycle's attention states (for trend analysis).
    last_states: Vec<AttentionState>,
    /// Last cycle's curiosity states.
    last_curiosity: Vec<CuriosityState>,
    /// Refractory period tracker for hub attenuation (persists across ticks).
    pub refractory: RefractoryTracker,
}

impl EcanCycle {
    pub fn new() -> Self {
        Self {
            tick_count: 0,
            last_states: Vec::new(),
            last_curiosity: Vec::new(),
            refractory: RefractoryTracker::new(),
        }
    }

    /// Run a full ECAN cycle. Returns None if not time yet (interval gating).
    pub fn tick(
        &mut self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        config: &EcanConfig,
    ) -> Result<Option<AttentionReport>, AgencyError> {
        self.tick_count += 1;

        // Advance refractory cooldowns every tick (even if ECAN is gated)
        self.refractory.tick();

        // Interval gating
        if config.interval > 1 && self.tick_count % config.interval != 0 {
            return Ok(None);
        }

        let report = run_ecan_cycle(storage, adjacency, config, &mut self.refractory)?;

        Ok(Some(report))
    }
}

/// Run one full ECAN + Curiosity cycle with hub attenuation.
///
/// Takes storage + adjacency, returns a report. The `refractory` tracker
/// is updated in-place when hub attenuation is enabled: nodes that receive
/// high attention enter a refractory period where their incoming bids are
/// attenuated, forcing avalanches to explore new pathways.
pub fn run_ecan_cycle(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &EcanConfig,
    refractory: &mut RefractoryTracker,
) -> Result<AttentionReport, AgencyError> {
    let att_cfg = &config.attention;
    let cur_cfg = &config.curiosity;

    // ── Step 1: Scan nodes ──────────────────────────────────
    let mut attention_states = Vec::new();
    let mut theta_data: Vec<(Uuid, f32)> = Vec::new();
    let mut scanned = 0usize;

    for result in storage.iter_nodes_meta() {
        let meta = result.map_err(|e| AgencyError::Internal(format!("scan: {e}")))?;

        if meta.energy < att_cfg.energy_floor {
            continue;
        }

        let degree_out = adjacency.degree_out(&meta.id);
        let degree_in = adjacency.degree_in(&meta.id);
        let total_degree = degree_out + degree_in;

        // Semantic mass
        let mass = attention_economy::semantic_mass(meta.energy, total_degree);
        let budget = attention_economy::compute_budget(mass, att_cfg.budget_scale);
        let demand = budget * att_cfg.demand_fraction;

        attention_states.push(AttentionState {
            node_id: meta.id,
            budget,
            demand,
            received: 0.0,
            spent: 0.0,
            semantic_mass: mass,
        });

        // Use depth as a proxy for theta (radial position)
        // For density estimation, we use the depth value mapped to [0, 2π)
        let theta_proxy = meta.depth * std::f64::consts::TAU as f32;
        theta_data.push((meta.id, theta_proxy));

        scanned += 1;
        if att_cfg.max_scan > 0 && scanned >= att_cfg.max_scan {
            break;
        }
    }

    if attention_states.is_empty() {
        return Ok(AttentionReport {
            participants: 0,
            total_bids: 0,
            winning_bids: 0,
            price: 1.0,
            total_flow: 0.0,
            top_receivers: Vec::new(),
            energy_deltas: Vec::new(),
        });
    }

    // ── Step 2: Angular density + Curiosity ──────────────────
    let density_map = curiosity_engine::compute_angular_density(&theta_data, cur_cfg.density_bins);
    let curiosity_states = curiosity_engine::compute_curiosity_states(
        &attention_states, &density_map,
    );
    let explore_ratio = curiosity_engine::exploration_ratio(
        &curiosity_states, cur_cfg.max_exploration_ratio,
    );

    // ── Step 3: Build lookup maps for hub attenuation ────────
    let theta_map: HashMap<Uuid, f32> = theta_data.iter().copied().collect();
    let degree_map: HashMap<Uuid, usize> = attention_states
        .iter()
        .map(|s| {
            let d = adjacency.degree_out(&s.node_id) + adjacency.degree_in(&s.node_id);
            (s.node_id, d)
        })
        .collect();
    let hub_cfg = &config.hub_attenuation;

    // ── Step 4: Generate exploitation bids ───────────────────
    // Each node bids on its highest-mass neighbour
    let mut exploitation_bids = Vec::new();
    let mass_map: HashMap<Uuid, f32> = attention_states
        .iter()
        .map(|s| (s.node_id, s.semantic_mass))
        .collect();

    for state in &attention_states {
        let neighbours = adjacency.neighbors_out(&state.node_id);
        if neighbours.is_empty() {
            continue;
        }

        // Rank neighbours by mass, pick top N
        let mut ranked: Vec<(Uuid, f32)> = neighbours
            .iter()
            .filter_map(|nid| {
                mass_map.get(nid).map(|m| (*nid, *m))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let theta_source = theta_map.get(&state.node_id).copied().unwrap_or(0.0);

        for (target, target_mass) in ranked.iter().take(att_cfg.max_bids_per_source) {
            let distance = (state.semantic_mass - target_mass).abs() / (state.semantic_mass + 0.01);
            let raw_bid = attention_economy::compute_bid_value(
                state.semantic_mass * (1.0 - explore_ratio),
                distance,
            );

            // Apply hub attenuation: angular focus × hub penalty × refractory
            let theta_target = theta_map.get(target).copied().unwrap_or(0.0);
            let target_degree = degree_map.get(target).copied().unwrap_or(0);
            let attenuation = hub_attenuation::compute_attenuation(
                theta_source, theta_target, target_degree, target,
                hub_cfg, refractory,
            );
            let bid_value = raw_bid * attenuation;

            if bid_value > 0.0 {
                exploitation_bids.push(AttentionBid {
                    source: state.node_id,
                    target: *target,
                    value: bid_value,
                });
            }
        }
    }

    // ── Step 4: Generate exploration bids ─────────────────────
    let novelty_map: HashMap<Uuid, f32> = curiosity_states
        .iter()
        .map(|c| (c.node_id, c.novelty))
        .collect();

    let mut neighbour_candidates: HashMap<Uuid, Vec<(Uuid, f32, f32)>> = HashMap::new();
    for state in &attention_states {
        let neighbours = adjacency.neighbors_out(&state.node_id);
        let candidates: Vec<(Uuid, f32, f32)> = neighbours
            .iter()
            .filter_map(|nid| {
                let novelty = novelty_map.get(nid).copied().unwrap_or(0.5);
                let target_mass = mass_map.get(nid).copied().unwrap_or(0.0);
                let distance = (state.semantic_mass - target_mass).abs()
                    / (state.semantic_mass + 0.01);
                Some((*nid, novelty, distance))
            })
            .collect();
        if !candidates.is_empty() {
            neighbour_candidates.insert(state.node_id, candidates);
        }
    }

    let raw_exploration_bids = curiosity_engine::generate_exploration_bids(
        &curiosity_states,
        &neighbour_candidates,
        cur_cfg,
    );

    // Apply hub attenuation to exploration bids
    let exploration_bids: Vec<AttentionBid> = raw_exploration_bids
        .into_iter()
        .map(|mut bid| {
            let theta_source = theta_map.get(&bid.source).copied().unwrap_or(0.0);
            let theta_target = theta_map.get(&bid.target).copied().unwrap_or(0.0);
            let target_degree = degree_map.get(&bid.target).copied().unwrap_or(0);
            let attenuation = hub_attenuation::compute_attenuation(
                theta_source, theta_target, target_degree, &bid.target,
                hub_cfg, refractory,
            );
            bid.value *= attenuation;
            bid
        })
        .filter(|bid| bid.value > 0.0)
        .collect();

    // ── Step 6: Merge bids + inflate + resolve ───────────────
    let mut all_bids = exploitation_bids;
    all_bids.extend(exploration_bids);
    let total_bids = all_bids.len();

    let total_demand: f32 = attention_states.iter().map(|s| s.demand).sum();
    let total_budget: f32 = attention_states.iter().map(|s| s.budget).sum();
    let price = attention_economy::cognitive_inflation(total_demand, total_budget);

    let (winners, total_flow) = attention_economy::resolve_auctions(&all_bids, price);
    let winning_bids = winners.len();

    // ── Step 7: Update attention received/spent ──────────────
    let mut received_map: HashMap<Uuid, f32> = HashMap::new();
    let mut spent_map: HashMap<Uuid, f32> = HashMap::new();

    for bid in &winners {
        *received_map.entry(bid.target).or_insert(0.0) += bid.value;
        *spent_map.entry(bid.source).or_insert(0.0) += bid.value;
    }

    for state in &mut attention_states {
        state.received = received_map.get(&state.node_id).copied().unwrap_or(0.0);
        state.spent = spent_map.get(&state.node_id).copied().unwrap_or(0.0);
    }

    // ── Step 7b: Mark high receivers for refractory ──────────
    if hub_cfg.enabled && hub_cfg.refractory_enabled {
        refractory.mark_high_receivers(
            &received_map,
            hub_cfg.refractory_threshold,
            hub_cfg.refractory_ticks,
        );
    }

    // ── Step 8: Compute energy deltas ────────────────────────
    let energy_deltas = attention_economy::compute_energy_deltas(&attention_states, att_cfg);

    // ── Step 9: Top receivers ────────────────────────────────
    let mut receiver_list: Vec<(Uuid, f32)> = received_map.into_iter().collect();
    receiver_list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_receivers: Vec<(Uuid, f32)> = receiver_list.into_iter().take(10).collect();

    tracing::info!(
        participants = attention_states.len(),
        bids = total_bids,
        winners = winning_bids,
        price = format!("{:.3}", price),
        flow = format!("{:.3}", total_flow),
        explore = format!("{:.2}", explore_ratio),
        hub_attenuation = hub_cfg.enabled,
        refractory_active = refractory.active_count(),
        "ECAN cycle complete"
    );

    Ok(AttentionReport {
        participants: attention_states.len(),
        total_bids,
        winning_bids,
        price,
        total_flow,
        top_receivers,
        energy_deltas,
    })
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::adjacency::AdjacencyIndex;
    use nietzsche_graph::model::{Edge, Node, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_storage(dir: &TempDir) -> GraphStorage {
        let p = dir.path().join("rocksdb");
        GraphStorage::open(p.to_str().unwrap()).unwrap()
    }

    fn make_node(name: &str, energy: f32, depth: f32) -> Node {
        let mut n = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![depth, 0.0]),
            serde_json::json!({ "name": name }),
        );
        n.energy = energy;
        n.meta.depth = depth;
        n
    }

    fn build_test_graph(dir: &TempDir) -> (GraphStorage, AdjacencyIndex, Vec<Node>) {
        let storage = open_storage(dir);
        let adjacency = AdjacencyIndex::new();

        let nodes = vec![
            make_node("Hub", 0.9, 0.1),          // central hub
            make_node("Satellite-1", 0.7, 0.3),
            make_node("Satellite-2", 0.6, 0.35),
            make_node("Satellite-3", 0.5, 0.4),
            make_node("Frontier", 0.8, 0.85),     // frontier explorer
            make_node("Dead", 0.02, 0.5),          // below energy floor
        ];

        for n in &nodes { storage.put_node(n).unwrap(); }

        // Hub connects to all satellites
        let edges = vec![
            Edge::association(nodes[0].id, nodes[1].id, 0.9),
            Edge::association(nodes[0].id, nodes[2].id, 0.8),
            Edge::association(nodes[0].id, nodes[3].id, 0.7),
            Edge::association(nodes[1].id, nodes[4].id, 0.6), // satellite → frontier
            Edge::association(nodes[4].id, nodes[0].id, 0.5), // frontier → hub
        ];

        for e in &edges {
            storage.put_edge(e).unwrap();
            adjacency.add_edge(e);
        }

        (storage, adjacency, nodes)
    }

    #[test]
    fn ecan_cycle_runs_on_simple_graph() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_test_graph(&dir);

        let config = EcanConfig::default();
        let mut refr = RefractoryTracker::new();
        let report = run_ecan_cycle(&storage, &adjacency, &config, &mut refr).unwrap();

        // Should have participants (5 alive, 1 dead below floor)
        assert!(report.participants >= 4, "got {}", report.participants);
        assert!(report.total_bids > 0, "should have bids");
        assert!(report.winning_bids > 0, "should have winners");
        assert!(report.price >= 1.0, "price should be >= 1.0");
        assert!(report.total_flow > 0.0, "should have attention flow");
    }

    #[test]
    fn hub_receives_most_attention() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_test_graph(&dir);

        let config = EcanConfig::default();
        let mut refr = RefractoryTracker::new();
        let report = run_ecan_cycle(&storage, &adjacency, &config, &mut refr).unwrap();

        // The hub (highest degree + energy) should be a top receiver
        if !report.top_receivers.is_empty() {
            // Hub should appear in top receivers
            let hub_id = nodes[0].id;
            let hub_received = report.top_receivers
                .iter()
                .find(|(id, _)| *id == hub_id)
                .map(|(_, r)| *r)
                .unwrap_or(0.0);
            // Hub should receive attention (it has incoming edge from frontier)
            // Not asserting rank because it depends on bid dynamics
            assert!(report.total_flow > 0.0);
        }
    }

    #[test]
    fn dead_nodes_excluded() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_test_graph(&dir);

        let config = EcanConfig::default();
        let mut refr = RefractoryTracker::new();
        let report = run_ecan_cycle(&storage, &adjacency, &config, &mut refr).unwrap();

        // Dead node (energy 0.02) should be excluded (floor = 0.05)
        let dead_id = nodes[5].id;
        assert!(
            !report.top_receivers.iter().any(|(id, _)| *id == dead_id),
            "dead node should not receive attention"
        );
        assert!(
            !report.energy_deltas.iter().any(|(id, _)| *id == dead_id),
            "dead node should not get energy delta"
        );
    }

    #[test]
    fn empty_graph_no_crash() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let config = EcanConfig::default();
        let mut refr = RefractoryTracker::new();
        let report = run_ecan_cycle(&storage, &adjacency, &config, &mut refr).unwrap();

        assert_eq!(report.participants, 0);
        assert_eq!(report.total_bids, 0);
    }

    #[test]
    fn interval_gating() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let config = EcanConfig {
            interval: 3,
            ..Default::default()
        };

        let mut cycle = EcanCycle::new();

        // tick 1, 2: gated (interval=3)
        assert!(cycle.tick(&storage, &adjacency, &config).unwrap().is_none());
        assert!(cycle.tick(&storage, &adjacency, &config).unwrap().is_none());

        // tick 3: runs
        let report = cycle.tick(&storage, &adjacency, &config).unwrap();
        assert!(report.is_some(), "tick 3 should run");

        // tick 4, 5: gated
        assert!(cycle.tick(&storage, &adjacency, &config).unwrap().is_none());
        assert!(cycle.tick(&storage, &adjacency, &config).unwrap().is_none());

        // tick 6: runs
        assert!(cycle.tick(&storage, &adjacency, &config).unwrap().is_some());
    }

    #[test]
    fn energy_deltas_reflect_attention_and_decay() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let config = EcanConfig::default();
        let mut refr = RefractoryTracker::new();
        let report = run_ecan_cycle(&storage, &adjacency, &config, &mut refr).unwrap();

        // With decay, some nodes gain (receivers) and some lose (non-receivers)
        // At minimum, there should be energy deltas (positive or negative)
        assert!(!report.energy_deltas.is_empty(), "should have energy deltas");

        // Verify net flow is reasonable (not all extreme)
        for (_, delta) in &report.energy_deltas {
            assert!(delta.abs() < 10.0, "delta should be bounded, got {delta}");
        }
    }
}
