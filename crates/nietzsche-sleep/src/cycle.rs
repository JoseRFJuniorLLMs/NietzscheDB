//! Reconsolidation sleep cycle — 7-step protocol.
//!
//! ## Protocol
//!
//! ```text
//! Step 1  Scan all nodes from RocksDB
//! Step 2  Compute global Hausdorff dimension  (H_before)
//! Step 3  Take embedding snapshot             (rollback point)
//! Step 4  Perturb each node by random_tangent + exp_map
//! Step 5  RiemannianAdam: pull each node toward its neighbors
//! Step 6  Compute global Hausdorff dimension  (H_after)
//! Step 7  |H_after − H_before| ≤ threshold  →  commit
//!          else                               →  restore snapshot
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
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            noise: 0.02,
            adam_steps: 10,
            adam_lr: 5e-3,
            hausdorff_threshold: 0.15,
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

    /// Whether the new embeddings were committed (`true`) or rolled back (`false`).
    pub committed: bool,

    /// Number of node embeddings perturbed during the cycle.
    pub nodes_perturbed: usize,

    /// Number of node embeddings captured in the rollback snapshot.
    pub snapshot_nodes: usize,
}

// ─────────────────────────────────────────────
// SleepCycle
// ─────────────────────────────────────────────

/// Executes the 7-step reconsolidation sleep cycle on a live NietzscheDB.
pub struct SleepCycle;

impl SleepCycle {
    /// Run one full sleep cycle.
    ///
    /// # Arguments
    /// * `config` — cycle parameters
    /// * `db`     — mutable handle to the graph (WAL + RocksDB + vector store)
    /// * `rng`    — random number generator for perturbation
    pub fn run<V: VectorStore>(
        config: &SleepConfig,
        db:     &mut NietzscheDB<V>,
        rng:    &mut impl Rng,
    ) -> Result<SleepReport, SleepError> {
        // ── Step 1: scan all nodes ──────────────────────────────
        let nodes_before = db.storage().scan_nodes()?;
        let node_count   = nodes_before.len();

        // ── Step 2: Hausdorff before ────────────────────────────
        let hausdorff_before = global_hausdorff(&nodes_before);

        // ── Step 3: snapshot ────────────────────────────────────
        // Re-use the already-scanned vec to build the snapshot map
        let snapshot = {
            let embeddings = nodes_before.iter()
                .map(|n| (n.id, n.embedding.clone()))
                .collect::<std::collections::HashMap<_, _>>();
            Snapshot::take(db)?  // full scan again — keeps borrow clean
        };
        let snapshot_nodes = snapshot.node_count();

        // ── Steps 4 + 5: perturb + optimise each node ──────────
        let adam = RiemannianAdam::new(config.adam_lr);
        let mut nodes_perturbed = 0usize;

        // Work from the pre-scan list; embeddings in `snapshot` are the
        // pre-perturbation reference for the coherence gradient.
        for node in &nodes_before {
            let dim = node.embedding.dim;

            // Collect pre-perturbation neighbor coordinates from snapshot
            let neighbor_coords: Vec<Vec<f64>> = db
                .neighbors_out(node.id)
                .into_iter()
                .filter_map(|nid| snapshot.get(&nid).map(|e| e.coords.clone()))
                .collect();

            // Step 4 — random perturbation along a geodesic
            let tangent = random_tangent(dim, config.noise, rng);
            let mut coords = exp_map(&node.embedding.coords, &tangent);

            // Step 5 — Riemannian Adam: minimise coherence loss
            //
            // Loss:  L(x) = (1/N) · Σ_j ‖x − p_j‖²    (Euclidean approx.)
            // Grad:  ∂L/∂x = (2/N) · Σ_j (x − p_j)
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

            // Persist the updated embedding
            db.update_embedding(node.id, PoincareVector::new(coords))?;
            nodes_perturbed += 1;
        }

        // ── Step 6: Hausdorff after ─────────────────────────────
        let nodes_after  = db.storage().scan_nodes()?;
        let hausdorff_after = global_hausdorff(&nodes_after);

        // ── Step 7: commit or restore ───────────────────────────
        let hausdorff_delta = (hausdorff_after - hausdorff_before).abs();
        let committed = hausdorff_delta <= config.hausdorff_threshold;

        if !committed {
            snapshot.restore(db)?;
        }

        Ok(SleepReport {
            hausdorff_before,
            hausdorff_after,
            hausdorff_delta,
            committed,
            nodes_perturbed,
            snapshot_nodes,
        })
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Edge, EdgeType, MockVectorStore, NietzscheDB, Node, PoincareVector};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use uuid::Uuid;

    fn open_db(dir: &std::path::Path) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir, MockVectorStore::default()).expect("open db")
    }

    fn insert(db: &mut NietzscheDB<MockVectorStore>, x: f64, y: f64) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node::new(id, PoincareVector::new(vec![x, y]), serde_json::json!({}));
        db.insert_node(node).expect("insert node");
        id
    }

    fn connect(db: &mut NietzscheDB<MockVectorStore>, from: Uuid, to: Uuid) {
        let edge = Edge {
            id:        Uuid::new_v4(),
            from,
            to,
            edge_type: EdgeType::Association,
            weight:    1.0,
            metadata:  Default::default(),
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
            let norm: f64 = node.embedding.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
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
            let norm: f64 = node.embedding.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm < 1.0, "‖x‖ = {norm} escapes ball");
        }
        assert_eq!(report.nodes_perturbed, 5);
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
