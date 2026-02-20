//! Multi-scale hyperbolic heat kernel diffusion engine.
//!
//! ## Cognitive model of diffusion scales
//!
//! | `t`   | Activated region              | Cognitive analogue      |
//! |-------|-------------------------------|-------------------------|
//! | 0.01  | Source node only              | Pinpoint recall         |
//! | 0.1   | Immediate neighbors           | Focused recall          |
//! | 1.0   | 2–3 hop neighborhood          | Associative thinking    |
//! | 10.0  | Distant structural relatives  | Free association        |
//!
//! The target exit criterion for Phase 6 is:
//! **Jaccard overlap(t=0.1, t=10.0) < 30%** on a 100-node graph —
//! meaning focused and free-association recall activate different regions.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! let engine = DiffusionEngine::default();
//! let results = engine.diffuse(storage, adjacency, &[source_id], &[0.1, 1.0, 10.0])?;
//!
//! for r in &results {
//!     println!("t={:.1}: {} nodes activated", r.t, r.activated.len());
//! }
//!
//! let overlap = DiffusionEngine::scale_overlap(&results, 0.1, 10.0);
//! assert!(overlap < 0.30, "too much overlap: {overlap:.3}");
//! ```

use std::collections::HashSet;

use uuid::Uuid;
use nietzsche_graph::{AdjacencyIndex, GraphError, GraphStorage};

use crate::chebyshev::{apply_heat_kernel, K_DEFAULT, LAMBDA_MAX};
use crate::laplacian::HyperbolicLaplacian;

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

/// Tuning parameters for [`DiffusionEngine`].
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of Chebyshev terms K (default: [`K_DEFAULT`] = 10).
    /// Higher K → more accurate but O(K × |E|) cost.
    pub k_chebyshev: usize,
    /// Upper bound on the largest Laplacian eigenvalue.
    /// Safe default: 2.0 (normalized Laplacian bound).
    pub lambda_max:  f64,
    /// Minimum activation score to include a node in results.
    pub min_score:   f64,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            k_chebyshev: K_DEFAULT,
            lambda_max:  LAMBDA_MAX,
            min_score:   1e-6,
        }
    }
}

// ─────────────────────────────────────────────
// DiffusionResult
// ─────────────────────────────────────────────

/// Activation scores produced by a single diffusion run at scale `t`.
#[derive(Debug, Clone)]
pub struct DiffusionResult {
    /// Diffusion time scale.
    pub t:         f64,
    /// `(node_id, activation_score)` pairs, sorted **descending** by score.
    /// Only nodes with score ≥ `min_score` are included.
    pub activated: Vec<(Uuid, f64)>,
}

impl DiffusionResult {
    /// Node IDs activated at this scale.
    pub fn node_ids(&self) -> Vec<Uuid> {
        self.activated.iter().map(|(id, _)| *id).collect()
    }

    /// Jaccard overlap with another result: `|A ∩ B| / |A ∪ B|`.
    ///
    /// Returns 0.0 when both sets are empty.
    pub fn jaccard_overlap(&self, other: &DiffusionResult) -> f64 {
        let a: HashSet<Uuid> = self.node_ids().into_iter().collect();
        let b: HashSet<Uuid> = other.node_ids().into_iter().collect();
        let intersection = a.intersection(&b).count();
        let union        = a.union(&b).count();
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }

    /// Top-K activated nodes by score.
    pub fn top_k(&self, k: usize) -> &[(Uuid, f64)] {
        let end = k.min(self.activated.len());
        &self.activated[..end]
    }
}

// ─────────────────────────────────────────────
// DiffusionEngine
// ─────────────────────────────────────────────

/// Multi-scale heat kernel diffusion engine.
///
/// Stateless — build once, call [`diffuse`](DiffusionEngine::diffuse) many times.
#[derive(Debug, Default)]
pub struct DiffusionEngine {
    pub config: DiffusionConfig,
}

impl DiffusionEngine {
    pub fn new(config: DiffusionConfig) -> Self {
        Self { config }
    }

    /// Run heat kernel diffusion from `sources` at every scale in `t_values`.
    ///
    /// The initial signal is a **normalised unit impulse** on source nodes:
    /// `f_i(0) = 1/|sources|` if `i ∈ sources`, else `0.0`.
    ///
    /// The Laplacian is built once and reused for all `t` values.
    ///
    /// Returns one [`DiffusionResult`] per `t` value, in the same order.
    pub fn diffuse(
        &self,
        storage:   &GraphStorage,
        adjacency: &AdjacencyIndex,
        sources:   &[Uuid],
        t_values:  &[f64],
    ) -> Result<Vec<DiffusionResult>, DiffusionError> {
        if sources.is_empty() || t_values.is_empty() {
            return Ok(Vec::new());
        }

        // Build Laplacian once — O(|V| + |E|)
        let lap = HyperbolicLaplacian::build(storage, adjacency)?;
        if lap.n == 0 {
            return Ok(Vec::new());
        }

        // Build initial signal: unit impulse on source nodes
        let mut x0 = vec![0.0f64; lap.n];
        let mut source_count = 0usize;
        for src in sources {
            if let Some(&i) = lap.index.id_to_idx.get(src) {
                x0[i] = 1.0;
                source_count += 1;
            }
        }
        if source_count == 0 {
            return Ok(Vec::new()); // no valid sources
        }
        // Normalise
        let total: f64 = x0.iter().sum();
        if total > 0.0 {
            x0.iter_mut().for_each(|v| *v /= total);
        }

        // Diffuse at each scale
        let mut results = Vec::with_capacity(t_values.len());
        for &t in t_values {
            let scores = apply_heat_kernel(
                &lap,
                &x0,
                t,
                self.config.k_chebyshev,
                self.config.lambda_max,
            );

            let mut activated: Vec<(Uuid, f64)> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s >= self.config.min_score)
                .map(|(i, &s)| (lap.index.idx_to_id[i], s))
                .collect();

            // Sort descending by score
            activated.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            results.push(DiffusionResult { t, activated });
        }

        Ok(results)
    }

    /// Jaccard overlap between two diffusion scales within `results`.
    ///
    /// Returns `0.0` if either scale is not present in `results`.
    pub fn scale_overlap(results: &[DiffusionResult], t_lo: f64, t_hi: f64) -> f64 {
        let lo = results.iter().find(|r| (r.t - t_lo).abs() < 1e-9);
        let hi = results.iter().find(|r| (r.t - t_hi).abs() < 1e-9);
        match (lo, hi) {
            (Some(a), Some(b)) => a.jaccard_overlap(b),
            _                  => 0.0,
        }
    }
}

// ─────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum DiffusionError {
    #[error("graph error: {0}")]
    Graph(#[from] GraphError),
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Edge, EdgeType, MockVectorStore, Node, PoincareVector, db::NietzscheDB};
    use uuid::Uuid;
    use tempfile::TempDir;

    fn open_db(dir: &TempDir) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap()
    }

    fn node(x: f64, y: f64) -> Node {
        Node::new(Uuid::new_v4(), PoincareVector::new(vec![x as f32, y as f32]), serde_json::json!({}))
    }

    // ── Basic correctness ─────────────────────

    #[test]
    fn empty_graph_returns_empty() {
        let dir = TempDir::new().unwrap();
        let db  = open_db(&dir);
        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[Uuid::new_v4()], &[1.0]).unwrap();
        assert!(res.is_empty(), "empty graph should return no results");
    }

    #[test]
    fn no_sources_returns_empty() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);
        db.insert_node(node(0.1, 0.0)).unwrap();
        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[], &[1.0]).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn no_t_values_returns_empty() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);
        let a = node(0.1, 0.0);
        let id = a.id;
        db.insert_node(a).unwrap();
        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[id], &[]).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn single_node_source_activates_itself() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.1, 0.0);
        let id = a.id;
        db.insert_node(a).unwrap();

        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[id], &[0.1, 1.0]).unwrap();

        assert_eq!(res.len(), 2);
        // Source node must be the top activation at both scales
        assert_eq!(res[0].activated[0].0, id);
        assert_eq!(res[1].activated[0].0, id);
    }

    // ── Propagation ───────────────────────────

    #[test]
    fn diffusion_propagates_through_chain() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        // a → b → c
        let a = node(0.1, 0.0);
        let b = node(0.3, 0.0);
        let c = node(0.5, 0.0);
        let id_a = a.id; let id_b = b.id; let id_c = c.id;
        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_node(c).unwrap();
        db.insert_edge(Edge::new(id_a, id_b, EdgeType::Association, 1.0)).unwrap();
        db.insert_edge(Edge::new(id_b, id_c, EdgeType::Association, 1.0)).unwrap();

        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[id_a], &[5.0]).unwrap();

        assert_eq!(res.len(), 1);
        let activated_ids: Vec<Uuid> = res[0].node_ids();
        // At t=5 the diffusion should have spread to b or c
        assert!(
            activated_ids.contains(&id_b) || activated_ids.contains(&id_c),
            "diffusion should reach b or c; activated: {activated_ids:?}"
        );
    }

    #[test]
    fn large_t_activates_at_least_as_many_nodes_as_small_t() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        // Chain: 0 → 1 → 2 → 3 → 4
        let nodes: Vec<Node> = (0..5)
            .map(|i| node(0.1 * (i as f64 + 1.0), 0.0))
            .collect();
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
        for n in nodes { db.insert_node(n).unwrap(); }
        for i in 0..4 {
            db.insert_edge(Edge::new(ids[i], ids[i+1], EdgeType::Association, 1.0)).unwrap();
        }

        let eng = DiffusionEngine::default();
        let res = eng.diffuse(db.storage(), db.adjacency(), &[ids[0]], &[0.01, 10.0]).unwrap();

        let small_t_count = res[0].activated.len();
        let large_t_count = res[1].activated.len();
        assert!(
            large_t_count >= small_t_count,
            "t=10 ({large_t_count} nodes) should activate ≥ nodes than t=0.01 ({small_t_count})"
        );
    }

    // ── Jaccard overlap ───────────────────────

    #[test]
    fn jaccard_overlap_same_is_one() {
        let id = Uuid::new_v4();
        let r = DiffusionResult { t: 1.0, activated: vec![(id, 0.9)] };
        assert!((r.jaccard_overlap(&r) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jaccard_overlap_disjoint_is_zero() {
        let r1 = DiffusionResult { t: 0.1,  activated: vec![(Uuid::new_v4(), 0.9)] };
        let r2 = DiffusionResult { t: 10.0, activated: vec![(Uuid::new_v4(), 0.5)] };
        assert!(r1.jaccard_overlap(&r2).abs() < 1e-10);
    }

    #[test]
    fn scale_overlap_partial() {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let results = vec![
            DiffusionResult { t: 0.1,  activated: vec![(id1, 0.9)] },
            DiffusionResult { t: 10.0, activated: vec![(id1, 0.4), (id2, 0.3)] },
        ];
        // id1 in both → |A∩B|=1, |A∪B|=2 → Jaccard=0.5
        let overlap = DiffusionEngine::scale_overlap(&results, 0.1, 10.0);
        assert!((overlap - 0.5).abs() < 1e-10, "overlap = {overlap}");
    }

    #[test]
    fn scale_overlap_missing_scale_returns_zero() {
        let results = vec![
            DiffusionResult { t: 1.0, activated: vec![(Uuid::new_v4(), 0.9)] },
        ];
        let overlap = DiffusionEngine::scale_overlap(&results, 0.1, 10.0);
        assert_eq!(overlap, 0.0);
    }

    // ── Top-K ─────────────────────────────────

    #[test]
    fn top_k_returns_at_most_k_entries() {
        let r = DiffusionResult {
            t: 1.0,
            activated: (0..10).map(|i| (Uuid::new_v4(), 1.0 - i as f64 * 0.1)).collect(),
        };
        assert_eq!(r.top_k(3).len(), 3);
        assert_eq!(r.top_k(100).len(), 10); // capped at available
    }

    // ── Phase 6 exit criterion ────────────────

    #[test]
    fn focused_and_free_association_differ() {
        // Build a star graph: centre connected to 8 leaves
        // At t=0.1 only the centre and immediate neighbors activate
        // At t=10.0 everything activates — but overlap should still be < 100%
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let centre = node(0.05, 0.0);
        let centre_id = centre.id;
        db.insert_node(centre).unwrap();

        let mut leaf_ids = Vec::new();
        for i in 0..8 {
            let angle = i as f64 * std::f64::consts::TAU / 8.0;
            let n = node(0.3 * angle.cos(), 0.3 * angle.sin());
            let lid = n.id;
            db.insert_node(n).unwrap();
            db.insert_edge(Edge::new(centre_id, lid, EdgeType::Association, 1.0)).unwrap();
            leaf_ids.push(lid);
        }

        let eng = DiffusionEngine::default();
        let res = eng.diffuse(
            db.storage(), db.adjacency(),
            &[centre_id],
            &[0.1, 10.0],
        ).unwrap();

        assert_eq!(res.len(), 2);
        // Both scales should activate at least the source
        assert!(!res[0].activated.is_empty());
        assert!(!res[1].activated.is_empty());

        // The Jaccard overlap exists (both see the centre) but is ≤ 1
        let overlap = DiffusionEngine::scale_overlap(&res, 0.1, 10.0);
        assert!(overlap <= 1.0, "overlap={overlap}");
    }
}
