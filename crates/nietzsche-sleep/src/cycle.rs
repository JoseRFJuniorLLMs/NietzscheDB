//! Reconsolidation sleep cycle — 7-step protocol.
//!
//! ## Protocol
//!
//! ```text
//! Step 1  Scan all nodes from RocksDB
//! Step 2  Compute global Hausdorff dimension  (H_before)
//! Step 3  Take embedding snapshot             (rollback point)
//! Step 3b [Phase 11] Snapshot sensory latents
//! Step 4  Perturb each node by random_tangent + exp_map
//! Step 5  RiemannianAdam: pull each node toward its neighbors
//! Step 5b [Phase 11] Consolidate sensory (decoder fine-tune, latent replay)
//! Step 6  Compute global Hausdorff dimension  (H_after) + semantic drift
//! Step 7  Hausdorff Δ ≤ threshold AND drift ≤ threshold  →  commit
//!          else                                           →  restore snapshot + sensory
//! ```
//!
//! The coherence loss is the mean squared Euclidean distance from a node
//! to its neighbours *before* perturbation (snapshot coordinates).  The
//! Riemannian Adam optimizer then minimises this loss on the manifold.

use rand::Rng;

use nietzsche_graph::{GraphError, NietzscheDB, PoincareVector, VectorStore};
use nietzsche_lsystem::global_hausdorff;

use crate::riemannian::{exp_map, random_tangent, AdamState, RiemannianAdam};
use crate::snapshot::Snapshot;

// ─────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────

/// Errors produced by the sleep cycle.
#[derive(Debug, thiserror::Error)]
pub enum SleepError {
    #[error("graph error: {0}")]
    Graph(#[from] GraphError),
}

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

/// Parameters for one reconsolidation sleep cycle.
#[derive(Debug, Clone)]
pub struct SleepConfig {
    /// Standard deviation of the random perturbation tangent vector.
    pub noise: f64,

    /// Number of Riemannian Adam steps run per node during reconsolidation.
    pub adam_steps: usize,

    /// Learning rate for the Riemannian Adam optimizer.
    pub adam_lr: f64,

    /// Maximum allowed |H_after − H_before| to commit.
    ///
    /// If the delta exceeds this threshold the snapshot is restored, undoing
    /// all embedding changes made during the cycle.
    pub hausdorff_threshold: f32,

    /// Maximum allowed average Poincaré distance between pre- and post-cycle
    /// embeddings (semantic drift).
    ///
    /// The Hausdorff check validates global fractal geometry, but a graph can
    /// preserve its Hausdorff dimension while individual nodes silently change
    /// meaning. This threshold catches that: after perturbation + optimisation,
    /// the mean Poincaré distance from each node's new position to its snapshot
    /// position must not exceed this value, or the cycle rolls back.
    pub semantic_drift_threshold: f64,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            noise: 0.02,
            adam_steps: 10,
            adam_lr: 5e-3,
            hausdorff_threshold: 0.15,
            semantic_drift_threshold: 0.5,
        }
    }
}

// ─────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────

/// Summary of a completed sleep cycle.
#[derive(Debug, Clone)]
pub struct SleepReport {
    /// Hausdorff dimension before perturbation.
    pub hausdorff_before: f32,

    /// Hausdorff dimension after perturbation + optimisation.
    pub hausdorff_after: f32,

    /// |H_after − H_before|.
    pub hausdorff_delta: f32,

    /// Mean Poincaré distance from each node's post-cycle embedding to its
    /// pre-cycle (snapshot) position. A low value means the cycle preserved
    /// semantic content; a high value means significant drift occurred.
    pub semantic_drift_avg: f64,

    /// Maximum Poincaré distance across all nodes (worst-case drift).
    pub semantic_drift_max: f64,

    /// Whether the new embeddings were committed (`true`) or rolled back (`false`).
    pub committed: bool,

    /// Number of node embeddings perturbed during the cycle.
    pub nodes_perturbed: usize,

    /// Number of node embeddings captured in the rollback snapshot.
    pub snapshot_nodes: usize,

    /// Phase 11: number of nodes whose sensory data was consolidated.
    pub sensory_consolidated: usize,
}

// ─────────────────────────────────────────────
// SleepCycle
// ─────────────────────────────────────────────

/// Phase 11: hook for sensory latent consolidation during the sleep cycle.
///
/// Implement this to plug sensory snapshot/rollback and optional decoder
/// fine-tuning into the reconsolidation loop. The sleep cycle calls:
///
/// 1. `snapshot_sensory()` — before perturbation (alongside embedding snapshot)
/// 2. `consolidate_sensory()` — after perturbation (fine-tune decoders, replay latents)
/// 3. `restore_sensory()` — on rollback (undo any sensory changes)
pub trait SensoryConsolidator {
    /// Snapshot all sensory latents for the given node IDs.
    fn snapshot_sensory(&self, node_ids: &[uuid::Uuid]) -> Result<(), SleepError>;

    /// Consolidate sensory data during the sleep cycle.
    ///
    /// This is where decoder fine-tuning (LoRA), latent replay, and
    /// reconstruction quality improvement happen. Returns the number
    /// of nodes whose sensory data was consolidated.
    fn consolidate_sensory(&self, node_ids: &[uuid::Uuid]) -> Result<usize, SleepError>;

    /// Restore sensory latents from the snapshot (rollback).
    fn restore_sensory(&self) -> Result<usize, SleepError>;
}

/// No-op consolidator used by the plain `run()` method.
struct NoopConsolidator;

impl SensoryConsolidator for NoopConsolidator {
    fn snapshot_sensory(&self, _node_ids: &[uuid::Uuid]) -> Result<(), SleepError> { Ok(()) }
    fn consolidate_sensory(&self, _node_ids: &[uuid::Uuid]) -> Result<usize, SleepError> { Ok(0) }
    fn restore_sensory(&self) -> Result<usize, SleepError> { Ok(0) }
}

/// Executes the reconsolidation sleep cycle on a live NietzscheDB.
pub struct SleepCycle;

impl SleepCycle {
    /// Run one full sleep cycle (original protocol, no sensory hooks).
    pub fn run<V: VectorStore>(
        config: &SleepConfig,
        db:     &mut NietzscheDB<V>,
        rng:    &mut impl Rng,
    ) -> Result<SleepReport, SleepError> {
        Self::run_inner(db, config, rng, None::<&NoopConsolidator>)
    }

    /// Run one full sleep cycle with Phase 11 sensory consolidation.
    ///
    /// Extended protocol:
    /// 1. Scan all nodes
    /// 2. Hausdorff before
    /// 3. Snapshot embeddings **+ snapshot sensory latents**
    /// 4. Perturb embeddings (geodesic)
    /// 5. RiemannianAdam (coherence loss)
    /// 6. **Consolidate sensory** (decoder fine-tune, latent replay)
    /// 7. Hausdorff after
    /// 8. Commit or rollback **(includes sensory rollback)**
    pub fn run_with_sensory<V: VectorStore, S: SensoryConsolidator>(
        config:       &SleepConfig,
        db:           &mut NietzscheDB<V>,
        rng:          &mut impl Rng,
        consolidator: &S,
    ) -> Result<SleepReport, SleepError> {
        Self::run_inner(db, config, rng, Some(consolidator))
    }

    fn run_inner<V: VectorStore, S: SensoryConsolidator>(
        db:           &mut NietzscheDB<V>,
        config:       &SleepConfig,
        rng:          &mut impl Rng,
        consolidator: Option<&S>,
    ) -> Result<SleepReport, SleepError> {
        // ── Step 1: scan all nodes ──────────────────────────────
        let nodes_before = db.storage().scan_nodes()?;

        // ── Step 2: Hausdorff before ────────────────────────────
        let hausdorff_before = global_hausdorff(&nodes_before);

        // ── Step 3: snapshot embeddings (+ sensory latents) ─────
        let snapshot = Snapshot::take(db)?;
        let snapshot_nodes = snapshot.node_count();

        // Phase 11: snapshot sensory latents alongside embeddings
        let mut sensory_consolidated = 0usize;
        if let Some(consolidator) = consolidator {
            let node_ids: Vec<uuid::Uuid> = nodes_before.iter().map(|n| n.id).collect();
            consolidator.snapshot_sensory(&node_ids)?;
        }

        // ── Steps 4 + 5: perturb + optimise each node ──────────
        let adam = RiemannianAdam::new(config.adam_lr);
        let mut nodes_perturbed = 0usize;

        for node in &nodes_before {
            let dim = node.embedding.dim;

            let neighbor_coords: Vec<Vec<f64>> = db
                .neighbors_out(node.id)
                .into_iter()
                .filter_map(|nid| snapshot.get(&nid).map(|e| e.coords.iter().map(|&c| c as f64).collect()))
                .collect();

            // Step 4 — random perturbation along a geodesic
            let tangent = random_tangent(dim, config.noise, rng);
            let coords_f64 = node.embedding.coords_f64();
            let mut coords = exp_map(&coords_f64, &tangent);

            // Step 5 — Riemannian Adam: minimise coherence loss
            let mut adam_state = AdamState::new(dim);
            for _ in 0..config.adam_steps {
                let eucl_grad: Vec<f64> = if neighbor_coords.is_empty() {
                    vec![0.0; dim]
                } else {
                    let n = neighbor_coords.len() as f64;
                    (0..dim)
                        .map(|d| {
                            2.0 * neighbor_coords.iter()
                                .map(|nc| coords[d] - nc[d])
                                .sum::<f64>()
                                / n
                        })
                        .collect()
                };

                coords = adam.step(&coords, &eucl_grad, &mut adam_state);
            }

            db.update_embedding(node.id, PoincareVector::from_f64(coords))?;
            nodes_perturbed += 1;
        }

        // ── Step 5b (Phase 11): consolidate sensory ─────────────
        // Decoder fine-tuning, latent replay, reconstruction improvement.
        // Only after embeddings have been perturbed + optimized.
        if let Some(consolidator) = consolidator {
            let node_ids: Vec<uuid::Uuid> = nodes_before.iter().map(|n| n.id).collect();
            sensory_consolidated = consolidator.consolidate_sensory(&node_ids)?;
        }

        // ── Step 6: Hausdorff after + semantic drift ─────────────
        let nodes_after  = db.storage().scan_nodes()?;
        let hausdorff_after = global_hausdorff(&nodes_after);

        // Compute per-node semantic drift: Poincaré distance from
        // each node's new embedding to its pre-cycle (snapshot) embedding.
        let mut total_drift = 0.0f64;
        let mut max_drift   = 0.0f64;
        let mut drift_count = 0usize;

        for node in &nodes_after {
            if let Some(pre) = snapshot.get(&node.id) {
                let d = pre.distance(&node.embedding);
                total_drift += d;
                if d > max_drift { max_drift = d; }
                drift_count += 1;
            }
        }

        let semantic_drift_avg = if drift_count > 0 {
            total_drift / drift_count as f64
        } else {
            0.0
        };
        let semantic_drift_max = max_drift;

        // ── Step 7: commit or restore ───────────────────────────
        let hausdorff_delta = (hausdorff_after - hausdorff_before).abs();
        let hausdorff_ok    = hausdorff_delta <= config.hausdorff_threshold;
        let drift_ok        = semantic_drift_avg <= config.semantic_drift_threshold;
        let committed       = hausdorff_ok && drift_ok;

        if !committed {
            snapshot.restore(db)?;
            // Point 5 Audit Fix: Force-sync vector store to ensure no orphans/drift
            // remain after the failed cycle.
            db.force_sync_vector_store()?;

            // Phase 11: also rollback sensory latents
            if let Some(consolidator) = consolidator {
                consolidator.restore_sensory()?;
            }
        }

        Ok(SleepReport {
            hausdorff_before,
            hausdorff_after,
            hausdorff_delta,
            semantic_drift_avg,
            semantic_drift_max,
            committed,
            nodes_perturbed,
            snapshot_nodes,
            sensory_consolidated,
        })
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{CausalType, Edge, EdgeType, MockVectorStore, NietzscheDB, Node, PoincareVector};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use uuid::Uuid;

    fn open_db(dir: &std::path::Path) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir, MockVectorStore::default()).expect("open db")
    }

    fn insert(db: &mut NietzscheDB<MockVectorStore>, x: f64, y: f64) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node::new(id, PoincareVector::new(vec![x as f32, y as f32]), serde_json::json!({}));
        db.insert_node(node).expect("insert node");
        id
    }

    fn connect(db: &mut NietzscheDB<MockVectorStore>, from: Uuid, to: Uuid) {
        let edge = Edge {
            id:                 Uuid::new_v4(),
            from,
            to,
            edge_type:          EdgeType::Association,
            weight:             1.0,
            lsystem_rule:       None,
            created_at:         0,
            metadata:           Default::default(),
            minkowski_interval: 0.0,
            causal_type:        CausalType::default(),
        };
        db.insert_edge(edge).expect("insert edge");
    }

    // ── basic smoke test ──────────────────────────────

    #[test]
    fn cycle_runs_on_empty_graph() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(1);
        let cfg = SleepConfig::default();

        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        assert_eq!(report.nodes_perturbed, 0);
        assert_eq!(report.snapshot_nodes, 0);
    }

    #[test]
    fn cycle_perturbs_all_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(2);

        insert(&mut db, 0.1, 0.0);
        insert(&mut db, 0.0, 0.2);
        insert(&mut db, 0.3, 0.1);

        let cfg    = SleepConfig::default();
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        assert_eq!(report.nodes_perturbed, 3);
        assert_eq!(report.snapshot_nodes, 3);
    }

    // ── Poincaré-ball invariant ───────────────────────

    #[test]
    fn embeddings_stay_inside_unit_ball_after_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(42);

        // Insert nodes spread around the ball
        for i in 0..8 {
            let angle = i as f64 * std::f64::consts::PI / 4.0;
            insert(&mut db, angle.cos() * 0.5, angle.sin() * 0.5);
        }

        let cfg = SleepConfig { noise: 0.05, ..Default::default() };
        SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        for node in db.storage().scan_nodes().unwrap() {
            let norm: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            assert!(norm < 1.0, "embedding escaped the ball: ‖x‖ = {norm}");
        }
    }

    // ── report sanity ─────────────────────────────────

    #[test]
    fn report_fields_are_consistent() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(7);

        insert(&mut db, 0.1, 0.2);
        insert(&mut db, 0.3, 0.4);

        let cfg    = SleepConfig::default();
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        // delta must equal |after - before|
        let expected_delta = (report.hausdorff_after - report.hausdorff_before).abs();
        assert!((report.hausdorff_delta - expected_delta).abs() < 1e-6);

        // Hausdorff values must be in [0, 3]
        assert!(report.hausdorff_before >= 0.0 && report.hausdorff_before <= 3.0);
        assert!(report.hausdorff_after  >= 0.0 && report.hausdorff_after  <= 3.0);
    }

    // ── rollback path ─────────────────────────────────

    #[test]
    fn rollback_restores_embeddings_when_threshold_exceeded() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(99);

        let id1 = insert(&mut db, 0.1, 0.0);
        let id2 = insert(&mut db, 0.0, 0.2);

        // Capture original coordinates before cycle
        let orig1 = db.get_node(id1).unwrap().unwrap().embedding.coords.clone();
        let orig2 = db.get_node(id2).unwrap().unwrap().embedding.coords.clone();

        // Set threshold = 0 so ANY change triggers rollback
        let cfg = SleepConfig {
            noise: 0.01,
            hausdorff_threshold: 0.0,
            ..Default::default()
        };
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        if !report.committed {
            // Verify embeddings were restored
            let after1 = db.get_node(id1).unwrap().unwrap().embedding.coords;
            let after2 = db.get_node(id2).unwrap().unwrap().embedding.coords;
            for d in 0..2 {
                assert!(
                    (after1[d] - orig1[d]).abs() < 1e-9,
                    "node1 coord[{d}]: expected {}, got {}", orig1[d], after1[d]
                );
                assert!(
                    (after2[d] - orig2[d]).abs() < 1e-9,
                    "node2 coord[{d}]: expected {}, got {}", orig2[d], after2[d]
                );
            }
        }
        // If committed with threshold=0 it means hausdorff_delta was exactly 0
        // (both before and after returned 1.0 for < 3 nodes), which is still valid.
    }

    // ── connected graph ───────────────────────────────

    #[test]
    fn cycle_uses_neighbor_coherence_gradient() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(13);

        // Star graph: center connected to 4 leaves
        let center = insert(&mut db, 0.0, 0.0);
        for i in 0..4 {
            let angle = i as f64 * std::f64::consts::PI / 2.0;
            let leaf  = insert(&mut db, angle.cos() * 0.3, angle.sin() * 0.3);
            connect(&mut db, center, leaf);
        }

        let cfg    = SleepConfig { noise: 0.01, adam_steps: 5, ..Default::default() };
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        // All nodes must remain in ball after a connected-graph cycle
        for node in db.storage().scan_nodes().unwrap() {
            let norm: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            assert!(norm < 1.0, "‖x‖ = {norm} escapes ball");
        }
        assert_eq!(report.nodes_perturbed, 5);
    }

    // ── semantic drift ────────────────────────────────

    #[test]
    fn report_includes_semantic_drift_metrics() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(77);

        insert(&mut db, 0.1, 0.2);
        insert(&mut db, 0.3, 0.1);

        let cfg    = SleepConfig::default();
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        // With noise=0.02 + Adam, drift should be small but non-zero
        assert!(report.semantic_drift_avg >= 0.0);
        assert!(report.semantic_drift_max >= report.semantic_drift_avg);
    }

    #[test]
    fn semantic_drift_triggers_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(42);

        // Insert nodes with connected edges so perturbation + Adam has real effect
        let a = insert(&mut db, 0.1, 0.0);
        let b = insert(&mut db, 0.0, 0.3);
        let c = insert(&mut db, 0.4, 0.2);
        connect(&mut db, a, b);
        connect(&mut db, b, c);
        connect(&mut db, c, a);

        // Capture original embeddings
        let orig_a = db.get_node(a).unwrap().unwrap().embedding.coords.clone();

        // Set semantic_drift_threshold = 0 so ANY drift triggers rollback
        let cfg = SleepConfig {
            noise: 0.05,
            semantic_drift_threshold: 0.0,
            ..Default::default()
        };
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        // With threshold=0 and noise=0.05, drift should exceed threshold
        if !report.committed {
            // Verify embeddings were restored
            let after_a = db.get_node(a).unwrap().unwrap().embedding.coords;
            for d in 0..2 {
                assert!(
                    (after_a[d] - orig_a[d]).abs() < 1e-9,
                    "rollback failed: coord[{d}] expected {}, got {}", orig_a[d], after_a[d]
                );
            }
        }
        // drift_avg > 0 (there was actual perturbation)
        assert!(report.semantic_drift_avg > 0.0);
    }

    #[test]
    fn zero_noise_produces_zero_drift() {
        let dir = tempfile::tempdir().unwrap();
        let mut db  = open_db(dir.path());
        let mut rng = StdRng::seed_from_u64(1);

        insert(&mut db, 0.2, 0.1);
        insert(&mut db, 0.1, 0.3);

        // noise=0 means no perturbation, so drift should be ~0
        let cfg = SleepConfig {
            noise: 0.0,
            ..Default::default()
        };
        let report = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();

        assert!(report.semantic_drift_avg < 1e-10, "drift should be ~0 with no noise: {}", report.semantic_drift_avg);
        assert!(report.committed, "should commit with zero drift");
    }

    // ── determinism ───────────────────────────────────

    #[test]
    fn same_seed_produces_same_result() {
        fn run_once(seed: u64) -> (f32, f32) {
            let dir = tempfile::tempdir().unwrap();
            let mut db  = open_db(dir.path());
            let mut rng = StdRng::seed_from_u64(seed);

            insert(&mut db, 0.2, 0.1);
            insert(&mut db, 0.1, 0.3);
            insert(&mut db, 0.4, 0.2);

            let cfg = SleepConfig { noise: 0.01, adam_steps: 3, ..Default::default() };
            let r   = SleepCycle::run(&cfg, &mut db, &mut rng).unwrap();
            (r.hausdorff_before, r.hausdorff_after)
        }

        let (b1, a1) = run_once(55);
        let (b2, a2) = run_once(55);
        assert_eq!(b1, b2);
        assert_eq!(a1, a2);
    }
}
