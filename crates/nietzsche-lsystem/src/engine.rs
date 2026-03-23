// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! [`LSystemEngine`] — applies production rules to the hyperbolic graph
//! and manages Hausdorff-based auto-pruning.
//!
//! ## Tick protocol
//!
//! 1. **Scan** — collect all live nodes (owned copy; no borrow held).
//! 2. **Hausdorff** — recompute local dimension for every node; persist via
//!    `NietzscheDB::update_hausdorff`.
//! 3. **Sensory degrade** *(Phase 11, optional)* — for each node with sensory
//!    data, degrade the latent vector based on current energy level
//!    (f32 → f16 → int8 → PQ → gone). Only runs when [`tick_with_sensory`]
//!    is used with a [`SensoryDegrader`] implementation.
//! 4. **Match** — for each node, find the first matching production rule and
//!    record a `PendingAction` (no mutations yet — avoids borrow conflicts).
//!    Nodes outside the Hausdorff target `[hausdorff_lo, hausdorff_hi]` are
//!    auto-queued for pruning.
//! 5. **Apply** — consume the pending list: insert child/sibling nodes +
//!    L-System edges, prune candidates, update energies.
//! 6. **Report** — return a [`LSystemReport`] with counts and global D.

use std::collections::HashSet;

use rand::Rng;
use uuid::Uuid;

use nietzsche_graph::{
    Edge, EdgeType, GraphError, Node, PoincareVector, VectorStore,
    db::NietzscheDB,
};

use crate::hausdorff::{batch_local_hausdorff, global_hausdorff, LOCAL_K, DEFAULT_HAUSDORFF_SAMPLE};
use crate::mobius::{spawn_child_diversified, spawn_sibling};
use crate::rules::{check_condition, ProductionRule, RuleAction};

// ─────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────

/// Default lower bound for the Hausdorff auto-prune gate.
pub const DEFAULT_HAUSDORFF_LO: f32 = 0.5;
/// Default upper bound for the Hausdorff auto-prune gate.
pub const DEFAULT_HAUSDORFF_HI: f32 = 1.9;
/// Default sigma multiplier for the energy circuit breaker.
///
/// Spawning is blocked for any node whose energy exceeds `μ + σ_threshold × σ`
/// (where μ and σ are the global mean and std of node energies). This prevents
/// "semantic tumors" — runaway feedback loops where high-energy regions spawn
/// children faster than the Hausdorff gate can contain them.
///
/// Set to `0.0` to disable the circuit breaker.
pub const DEFAULT_CIRCUIT_BREAKER_SIGMA: f32 = 2.0;

/// Trait for optional Phase 11 sensory degradation during L-System ticks.
///
/// Implement this to plug sensory degradation into the tick loop.
/// The engine calls `degrade_node_sensory` for each node after Hausdorff
/// update but before rule matching.
pub trait SensoryDegrader {
    /// Degrade the sensory latent for a node based on its current energy.
    ///
    /// Returns `true` if the node's sensory data was actually degraded
    /// (i.e., the quantization level changed). Implementations should:
    /// - Look up the node's sensory data in the `sensory` CF
    /// - Call `LatentVector::degrade(energy)` if the target level is lower
    /// - Persist the updated sensory data
    /// - Mark sensory as irrecoverable if energy < 0.1
    fn degrade_node_sensory(&self, node_id: Uuid, energy: f32) -> Result<bool, LSystemError>;
}

/// L-System fractal growth engine.
///
/// Holds a list of [`ProductionRule`]s applied each tick. The engine is
/// stateless (no internal counter) — callers track the generation number.
#[derive(Debug, Default)]
pub struct LSystemEngine {
    /// Rules evaluated in order; first match wins per node per tick.
    pub rules:              Vec<ProductionRule>,
    /// Auto-prune nodes with local Hausdorff D < `hausdorff_lo`.
    pub hausdorff_lo:       f32,
    /// Auto-prune nodes with local Hausdorff D > `hausdorff_hi`.
    pub hausdorff_hi:       f32,
    /// Energy circuit breaker: block spawning for nodes with energy > μ + k × σ.
    ///
    /// When set to a positive value, the engine computes the global mean (μ) and
    /// standard deviation (σ) of all node energies each tick. Any node whose
    /// energy exceeds `μ + circuit_breaker_sigma × σ` is prevented from spawning
    /// children or siblings (prune and energy-update rules still apply).
    ///
    /// Set to `0.0` to disable. Default: [`DEFAULT_CIRCUIT_BREAKER_SIGMA`] (2.0).
    pub circuit_breaker_sigma: f32,
    /// Angular jitter (radians) for diversified child spawning.
    ///
    /// Controls the cone half-angle around the radial direction when placing
    /// children via Möbius addition. Prevents geodesic convergence (semantic
    /// black holes) where children of nearby parents collapse into the same
    /// angular region.
    ///
    /// - `0.0` = pure radial (legacy behaviour)
    /// - `0.3` = recommended for production (~17° spread)
    ///
    /// Default: `0.3`.
    pub angular_jitter: f64,
    /// Maximum number of nodes to sample for batch Hausdorff computation.
    ///
    /// When a collection has more nodes than this limit, a stratified random
    /// sample is selected. Only sampled nodes get their `hausdorff_local`
    /// recomputed; the rest retain their previous values. This reduces the
    /// GPU distance matrix from O(N²) to O(S²) where S = sample size.
    ///
    /// - `0` = no sampling (compute all — original behaviour)
    /// - `8000` = recommended for production (distance matrix ≈ 256 MB)
    ///
    /// Default: [`DEFAULT_HAUSDORFF_SAMPLE`] (8000).
    pub hausdorff_sample_size: usize,
    /// Number of neighbours for local Hausdorff estimation (k in kNN).
    ///
    /// Lower values speed up both GPU kernel and CPU box-counting.
    /// Must be ≥ 3 for meaningful dimension estimation.
    ///
    /// Default: [`LOCAL_K`] (12).
    pub hausdorff_k: usize,
    /// Maximum number of nodes to spawn per tick (children + siblings).
    ///
    /// Once this limit is reached, remaining spawn actions are silently
    /// dropped (prune and energy-update actions still apply). This prevents
    /// runaway growth when the Hausdorff gate and circuit breaker are
    /// insufficient to contain the L-System.
    ///
    /// - `0` = unlimited (legacy behaviour)
    ///
    /// Default: `0` (unlimited). Set `LSYSTEM_MAX_SPAWNS_PER_TICK` to override.
    pub max_spawns_per_tick: usize,
}

impl LSystemEngine {
    /// Construct an engine with the global Hausdorff target `[0.5, 1.9]`
    /// and the default energy circuit breaker (`2σ`).
    pub fn new(rules: Vec<ProductionRule>) -> Self {
        Self {
            rules,
            hausdorff_lo: DEFAULT_HAUSDORFF_LO,
            hausdorff_hi: DEFAULT_HAUSDORFF_HI,
            circuit_breaker_sigma: DEFAULT_CIRCUIT_BREAKER_SIGMA,
            angular_jitter: 0.3,
            hausdorff_sample_size: DEFAULT_HAUSDORFF_SAMPLE,
            hausdorff_k: LOCAL_K,
            max_spawns_per_tick: 0,
        }
    }

    /// Run one generation tick against `db`.
    ///
    /// Returns a [`LSystemReport`] describing the mutations performed.
    pub fn tick<V: VectorStore>(&self, db: &mut NietzscheDB<V>) -> Result<LSystemReport, LSystemError> {
        self.tick_inner(db, None::<&NoopDegrader>)
    }

    /// Run one generation tick with Phase 11 sensory degradation.
    ///
    /// Same as [`tick`], but between Hausdorff update and rule matching,
    /// degrades sensory latents for every node based on current energy.
    ///
    /// ## Extended tick protocol
    ///
    /// 1. Scan all nodes
    /// 2. Recompute local Hausdorff
    /// 3. **Phase 11: degrade sensory latents** (energy → quantization)
    /// 4. Match rules → build pending list
    /// 5. Apply mutations
    /// 6. Report
    pub fn tick_with_sensory<V: VectorStore, S: SensoryDegrader>(
        &self,
        db: &mut NietzscheDB<V>,
        degrader: &S,
    ) -> Result<LSystemReport, LSystemError> {
        self.tick_inner(db, Some(degrader))
    }

    fn tick_inner<V: VectorStore, S: SensoryDegrader>(
        &self,
        db: &mut NietzscheDB<V>,
        degrader: Option<&S>,
    ) -> Result<LSystemReport, LSystemError> {
        let mut rng = rand::thread_rng();

        // ── Step 1: collect all nodes (single scan — reused throughout) ──
        let mut all_nodes: Vec<Node> = db.storage().scan_nodes()?;
        let total_n = all_nodes.len();
        let k = self.hausdorff_k;

        // ── Step 2: recompute local Hausdorff (with stochastic sampling) ─
        //
        // For large collections, computing all-pairs kNN is O(N²) in both
        // VRAM and CPU post-processing. Stochastic sampling reduces this to
        // O(S²) where S = hausdorff_sample_size, while non-sampled nodes
        // retain their previous hausdorff_local values from the last tick.
        let use_sampling = self.hausdorff_sample_size > 0
            && total_n > self.hausdorff_sample_size;

        if use_sampling {
            // Stratified random sample: shuffle indices, take first S.
            use rand::seq::SliceRandom;
            let mut indices: Vec<usize> = (0..total_n).collect();
            indices.shuffle(&mut rng);
            indices.truncate(self.hausdorff_sample_size);
            indices.sort_unstable(); // sort for cache-friendly access

            let sampled_nodes: Vec<Node> = indices.iter().map(|&i| all_nodes[i].clone()).collect();

            eprintln!(
                "[L-System] Hausdorff sampling: {} of {} nodes (k={})",
                sampled_nodes.len(), total_n, k,
            );

            let hausdorff_values = batch_local_hausdorff(&sampled_nodes, k);

            // Persist + update in-memory for sampled nodes only
            for (idx, h) in indices.iter().zip(hausdorff_values.iter()) {
                db.update_hausdorff(all_nodes[*idx].id, *h)?;
                all_nodes[*idx].hausdorff_local = *h;
            }
        } else {
            // Small collection: compute for all nodes (original path)
            let hausdorff_values = batch_local_hausdorff(&all_nodes, k);
            for (node, h) in all_nodes.iter_mut().zip(hausdorff_values.iter()) {
                db.update_hausdorff(node.id, *h)?;
                node.hausdorff_local = *h;
            }
        }

        // ── Step 3 (Phase 11): degrade sensory latents ──────────────────
        let mut sensory_degraded = 0usize;
        if let Some(degrader) = degrader {
            for node in &all_nodes {
                match degrader.degrade_node_sensory(node.id, node.energy) {
                    Ok(true) => sensory_degraded += 1,
                    Ok(false) => {}
                    Err(e) => {
                        // Non-fatal: log and continue. Sensory degradation
                        // must not block graph evolution.
                        #[cfg(feature = "tracing")]
                        tracing::warn!("sensory degrade failed for {}: {e}", node.id);
                        let _ = e;
                    }
                }
            }
        }

        // ── Step 4: energy circuit breaker threshold ─────────────────────
        // (No re-scan needed — hausdorff_local updated in-memory above)
        //
        // Compute μ + k·σ across all node energies. Nodes above this
        // threshold are "overheated" and will be blocked from spawning
        // children/siblings, preventing semantic tumors (runaway feedback
        // loops where high-energy regions grow uncontrollably).
        let cb_threshold = if all_nodes.len() >= 2 && self.circuit_breaker_sigma > 0.0 {
            let n = all_nodes.len() as f32;
            let mean = all_nodes.iter().map(|nd| nd.energy).sum::<f32>() / n;
            let variance = all_nodes.iter()
                .map(|nd| (nd.energy - mean).powi(2))
                .sum::<f32>() / n;
            let std = variance.sqrt();
            Some(mean + self.circuit_breaker_sigma * std)
        } else {
            None // disabled or < 2 nodes
        };
        let mut nodes_halted = 0usize;

        // ── Step 5: match rules → build pending list ─────────────────────
        let mut pending: Vec<PendingAction> = Vec::new();

        for node in &all_nodes {
            // Hausdorff auto-prune (takes priority over user rules)
            if node.hausdorff_local < self.hausdorff_lo
                || node.hausdorff_local > self.hausdorff_hi
            {
                pending.push(PendingAction::Prune { node_id: node.id });
                continue;
            }

            // First matching user rule
            for rule in &self.rules {
                if check_condition(
                    node.energy,
                    node.depth,
                    node.hausdorff_local,
                    node.lsystem_generation,
                    &rule.condition,
                ) {
                    // Circuit breaker: block spawn actions for overheated nodes.
                    // Prune and UpdateEnergy actions pass through — we only
                    // freeze *growth*, not maintenance.
                    if let Some(threshold) = cb_threshold {
                        if node.energy > threshold {
                            match &rule.action {
                                RuleAction::SpawnChild { .. }
                                | RuleAction::SpawnSibling { .. } => {
                                    nodes_halted += 1;
                                    break;
                                }
                                _ => {} // allow prune / energy updates
                            }
                        }
                    }

                    if let Some(action) = make_pending(node, rule, self.angular_jitter, &mut rng) {
                        pending.push(action);
                    }
                    break;
                }
            }
        }

        // ── Step 6: apply mutations ──────────────────────────────────────
        let (nodes_spawned, nodes_pruned, edges_created) = apply_pending(db, pending, self.max_spawns_per_tick)?;

        // ── Step 7: global Hausdorff (sampled, no full re-scan) ──────────
        //
        // Use a stride-based sample of the current in-memory nodes to
        // estimate global Hausdorff without a third full scan_nodes().
        // After mutations, node count ≈ total_n ± mutations, but
        // all_nodes still represents the pre-mutation snapshot which is
        // a valid approximation for global D.
        let global_sample_cap = 1000usize;
        let global_d = if all_nodes.len() <= global_sample_cap {
            global_hausdorff(&all_nodes)
        } else {
            let step = all_nodes.len() / global_sample_cap;
            let sampled: Vec<Node> = all_nodes.iter()
                .step_by(step.max(1))
                .take(global_sample_cap)
                .cloned()
                .collect();
            global_hausdorff(&sampled)
        };

        let final_total = total_n + nodes_spawned - nodes_pruned;

        Ok(LSystemReport {
            nodes_spawned,
            nodes_pruned,
            edges_created,
            global_hausdorff: global_d,
            total_nodes: final_total,
            sensory_degraded,
            nodes_halted,
        })
    }
}

// ─────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────

/// Statistics returned by [`LSystemEngine::tick`].
#[derive(Debug, Clone)]
pub struct LSystemReport {
    pub nodes_spawned:      usize,
    pub nodes_pruned:       usize,
    pub edges_created:      usize,
    pub global_hausdorff:   f32,
    pub total_nodes:        usize,
    /// Phase 11: number of nodes whose sensory latents were degraded this tick.
    pub sensory_degraded:   usize,
    /// Number of nodes whose spawning was blocked by the energy circuit breaker.
    ///
    /// When > 0, the graph contains "overheated" regions with energy above
    /// `μ + k·σ`. Growth is frozen in those regions until energy dissipates.
    pub nodes_halted:       usize,
}

impl LSystemReport {
    /// `true` if the graph is within the target fractal regime `(1.2, 1.8)`.
    pub fn is_fractal(&self) -> bool {
        self.global_hausdorff > 1.2 && self.global_hausdorff < 1.8
    }
}

// ─────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum LSystemError {
    #[error("graph error: {0}")]
    Graph(#[from] GraphError),
}

/// No-op degrader used by the plain `tick()` method.
struct NoopDegrader;

impl SensoryDegrader for NoopDegrader {
    fn degrade_node_sensory(&self, _node_id: Uuid, _energy: f32) -> Result<bool, LSystemError> {
        Ok(false)
    }
}

// ─────────────────────────────────────────────
// Pending action (pre-computed, no borrow of db)
// ─────────────────────────────────────────────

enum PendingAction {
    SpawnChild {
        parent_id:         Uuid,
        parent_generation: u32,
        child_coords:      Vec<f64>,
        weight:            f32,
        content:           serde_json::Value,
        rule_name:         String,
    },
    SpawnSibling {
        parent_id:          Uuid,
        parent_generation:  u32,
        sibling_coords:     Vec<f64>,
        weight:             f32,
        content:            serde_json::Value,
        rule_name:          String,
    },
    Prune { node_id: Uuid },
    UpdateEnergy { node_id: Uuid, new_energy: f32 },
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Convert a rule action into a `PendingAction`, pre-computing new positions.
fn make_pending(
    node: &Node,
    rule: &ProductionRule,
    angular_jitter: f64,
    rng:  &mut impl Rng,
) -> Option<PendingAction> {
    match &rule.action {
        RuleAction::SpawnChild { depth_offset, weight, content } => {
            let parent_f64 = node.embedding.coords_f64();
            let coords = spawn_child_diversified(&parent_f64, *depth_offset, angular_jitter, rng);
            Some(PendingAction::SpawnChild {
                parent_id:         node.id,
                parent_generation: node.lsystem_generation,
                child_coords:      coords,
                weight:            *weight,
                content:           content.clone(),
                rule_name:         rule.name.clone(),
            })
        }
        RuleAction::SpawnSibling { angle, distance, weight, content } => {
            let parent_f64 = node.embedding.coords_f64();
            let coords = spawn_sibling(&parent_f64, *angle, *distance, rng);
            Some(PendingAction::SpawnSibling {
                parent_id:         node.id,
                parent_generation: node.lsystem_generation,
                sibling_coords:    coords,
                weight:            *weight,
                content:           content.clone(),
                rule_name:         rule.name.clone(),
            })
        }
        RuleAction::Prune => Some(PendingAction::Prune { node_id: node.id }),
        RuleAction::UpdateEnergy { delta } => {
            let new_energy = (node.energy + delta).clamp(0.0, 1.0);
            Some(PendingAction::UpdateEnergy { node_id: node.id, new_energy })
        }
    }
}

/// Execute all pending actions against `db`.
///
/// Returns `(nodes_spawned, nodes_pruned, edges_created)`.
fn apply_pending<V: VectorStore>(
    db:      &mut NietzscheDB<V>,
    pending: Vec<PendingAction>,
    max_spawns: usize,
) -> Result<(usize, usize, usize), LSystemError> {
    let mut nodes_spawned = 0usize;
    let mut nodes_pruned  = 0usize;
    let mut edges_created = 0usize;
    let mut pruned: HashSet<Uuid> = HashSet::new();

    for action in pending {
        match action {
            PendingAction::SpawnChild {
                parent_id, parent_generation, child_coords, weight, content, rule_name,
            } => {
                if max_spawns > 0 && nodes_spawned >= max_spawns {
                    continue; // spawn cap reached
                }
                if pruned.contains(&parent_id) {
                    continue; // parent was pruned earlier in this tick
                }
                let emb = PoincareVector::from_f64(child_coords);
                if !emb.is_valid() {
                    continue; // numeric drift — skip
                }
                let mut child = Node::new(Uuid::new_v4(), emb, content);
                child.lsystem_generation = parent_generation + 1;
                let child_id = child.id;

                db.insert_node(child)?;
                nodes_spawned += 1;

                let mut edge = Edge::new(parent_id, child_id, EdgeType::LSystemGenerated, weight);
                edge.lsystem_rule = Some(rule_name);
                db.insert_edge(edge)?;
                edges_created += 1;
            }

            PendingAction::SpawnSibling {
                parent_id, parent_generation, sibling_coords, weight, content, rule_name,
            } => {
                if max_spawns > 0 && nodes_spawned >= max_spawns {
                    continue; // spawn cap reached
                }
                if pruned.contains(&parent_id) {
                    continue;
                }
                let emb = PoincareVector::from_f64(sibling_coords);
                if !emb.is_valid() {
                    continue;
                }
                let mut sibling = Node::new(Uuid::new_v4(), emb, content);
                sibling.lsystem_generation = parent_generation + 1;
                let sibling_id = sibling.id;

                db.insert_node(sibling)?;
                nodes_spawned += 1;

                let mut edge = Edge::new(parent_id, sibling_id, EdgeType::LSystemGenerated, weight);
                edge.lsystem_rule = Some(rule_name);
                db.insert_edge(edge)?;
                edges_created += 1;
            }

            PendingAction::Prune { node_id } => {
                if pruned.insert(node_id) {
                    db.prune_node(node_id)?;
                    nodes_pruned += 1;
                }
            }

            PendingAction::UpdateEnergy { node_id, new_energy } => {
                if !pruned.contains(&node_id) {
                    db.update_energy(node_id, new_energy)?;
                }
            }
        }
    }

    Ok((nodes_spawned, nodes_pruned, edges_created))
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{MockVectorStore, Node, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_db(dir: &TempDir) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir.path(), MockVectorStore::default(), 2).unwrap()
    }

    fn seed_node(x: f64, energy: f32) -> Node {
        let mut n = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, 0.0]),
            serde_json::json!({}),
        );
        n.energy = energy;
        n
    }

    #[test]
    fn tick_spawns_children_for_high_energy_nodes() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let seed = seed_node(0.2, 0.9);
        db.insert_node(seed.clone()).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 3),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert!(report.nodes_spawned >= 1, "expected at least one child spawned");
        assert!(report.edges_created >= 1);
    }

    #[test]
    fn tick_prunes_low_energy_nodes() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let fading = seed_node(0.3, 0.05); // energy below prune threshold
        db.insert_node(fading.clone()).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::prune_fading("prune", 0.1),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert!(report.nodes_pruned >= 1);

        // Node still exists but energy = 0
        let node = db.get_node(fading.id).unwrap().unwrap();
        assert_eq!(node.energy, 0.0);
    }

    #[test]
    fn generation_limit_prevents_unbounded_growth() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let seed = seed_node(0.1, 0.9);
        db.insert_node(seed).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 1), // max gen = 1
        ]);

        // Tick 1: seed (gen=0) spawns a child (gen=1)
        let r1 = engine.tick(&mut db).unwrap();
        assert_eq!(r1.nodes_spawned, 1);

        // Tick 2: seed (gen=0) still matches; child (gen=1) does NOT
        let r2 = engine.tick(&mut db).unwrap();
        // Only the original seed node can still fire (if energy stays high)
        // Child nodes (gen=1) are blocked by GenerationBelow(1)
        assert!(r2.nodes_spawned <= 1, "child should not grow: spawned {}", r2.nodes_spawned);
    }

    #[test]
    fn spawned_child_is_deeper_than_parent() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let seed = seed_node(0.3, 0.9);
        let parent_depth = seed.depth;
        db.insert_node(seed).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 3),
        ]);

        engine.tick(&mut db).unwrap();

        // Scan for nodes with generation == 1
        let all = db.storage().scan_nodes().unwrap();
        let children: Vec<_> = all.iter().filter(|n| n.lsystem_generation == 1).collect();
        assert!(!children.is_empty());
        for child in children {
            assert!(
                child.depth > parent_depth,
                "child.depth {} should exceed parent.depth {}", child.depth, parent_depth
            );
        }
    }

    #[test]
    fn sibling_spawning_rule() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let seed = seed_node(0.4, 0.8);
        db.insert_node(seed).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::lateral_association("lateral", 3),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert!(report.nodes_spawned >= 1);
        assert!(report.edges_created >= 1);

        // Sibling edges should be LSystemGenerated
        let all_edges = db.storage().scan_edges().unwrap();
        let lsys_edges: Vec<_> = all_edges.iter()
            .filter(|e| e.edge_type == EdgeType::LSystemGenerated)
            .collect();
        assert!(!lsys_edges.is_empty());
    }

    #[test]
    fn report_global_hausdorff_is_in_range() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Seed a few nodes
        for i in 0..5 {
            db.insert_node(seed_node(0.1 * (i as f64 + 1.0), 0.9)).unwrap();
        }

        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 2),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert!(
            report.global_hausdorff >= 0.0 && report.global_hausdorff <= 3.0,
            "Hausdorff out of range: {}",
            report.global_hausdorff
        );
    }

    #[test]
    fn no_rules_no_mutations() {
        let dir = tmp();
        let mut db = open_db(&dir);
        db.insert_node(seed_node(0.2, 1.0)).unwrap();

        let engine = LSystemEngine::new(vec![]); // empty rule set
        let report = engine.tick(&mut db).unwrap();

        // No rules = no spawns; only Hausdorff auto-prune could fire
        // (default node has hausdorff_local = 1.0 which is within [0.5, 1.9])
        assert_eq!(report.nodes_spawned, 0);
        assert_eq!(report.edges_created, 0);
    }

    #[test]
    fn circuit_breaker_halts_overheated_node() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // 5 "normal" nodes at energy 0.3
        for i in 0..5 {
            db.insert_node(seed_node(0.05 * (i as f64 + 1.0), 0.3)).unwrap();
        }
        // 1 "overheated" outlier at energy 0.99
        let hot = seed_node(0.4, 0.99);
        let hot_id = hot.id;
        db.insert_node(hot).unwrap();

        // Mean ≈ 0.415, std ≈ 0.254, threshold ≈ 0.923
        // The hot node (0.99) exceeds threshold → should be halted.
        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 5),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert!(
            report.nodes_halted >= 1,
            "circuit breaker should halt the overheated node, halted={}",
            report.nodes_halted,
        );

        // Verify the overheated node did NOT spawn a child
        let all_edges = db.storage().scan_edges().unwrap();
        let hot_children: Vec<_> = all_edges.iter()
            .filter(|e| e.from == hot_id && e.edge_type == EdgeType::LSystemGenerated)
            .collect();
        assert!(
            hot_children.is_empty(),
            "overheated node should have no L-System children"
        );
    }

    #[test]
    fn circuit_breaker_allows_uniform_energy() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Two nodes at the same energy → std = 0 → threshold = mean.
        // Neither exceeds mean → no halting.
        // (Using 2 nodes ensures local_hausdorff returns 1.0, avoiding
        // spurious auto-prune from box-counting on tiny collinear sets.)
        db.insert_node(seed_node(0.2, 0.8)).unwrap();
        db.insert_node(seed_node(0.4, 0.8)).unwrap();

        let engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 3),
        ]);

        let report = engine.tick(&mut db).unwrap();
        assert_eq!(
            report.nodes_halted, 0,
            "uniform energy should not trigger circuit breaker"
        );
        assert!(
            report.nodes_spawned > 0,
            "nodes should still spawn normally"
        );
    }

    #[test]
    fn circuit_breaker_disabled_when_sigma_zero() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Same outlier setup as the halt test
        for i in 0..5 {
            db.insert_node(seed_node(0.05 * (i as f64 + 1.0), 0.3)).unwrap();
        }
        db.insert_node(seed_node(0.4, 0.99)).unwrap();

        let mut engine = LSystemEngine::new(vec![
            ProductionRule::growth_child("grow", 5),
        ]);
        engine.circuit_breaker_sigma = 0.0; // disabled

        let report = engine.tick(&mut db).unwrap();
        assert_eq!(
            report.nodes_halted, 0,
            "circuit breaker should be disabled when sigma = 0"
        );
    }
}
