//! Hyperbolic graph Laplacian construction.
//!
//! Builds a sparse, weighted Laplacian from the graph's Poincaré-distance
//! edge weights, enabling spectral operations used by the Chebyshev diffusion.
//!
//! ## Edge weight convention
//!
//! ```text
//! w(u, v) = 1 / (1 + d_H(u, v))
//! ```
//!
//! Short hyperbolic distances → high weight (strong coupling).
//! Long distances → low weight (weak coupling).
//!
//! ## Laplacian variants
//!
//! | Variant              | Formula               | Eigenvalue range |
//! |----------------------|-----------------------|------------------|
//! | Combinatorial L      | D − A                 | [0, 2 d_max]     |
//! | Normalized L (sym.)  | I − D^{−½} A D^{−½}  | [0, 2]           |
//!
//! The **normalized** form is used; its eigenvalues are bounded by 2, which
//! is required for stable Chebyshev recurrence (λ_max ≤ 2 always holds).

use std::collections::HashMap;

use uuid::Uuid;
use nietzsche_graph::{AdjacencyIndex, GraphError, GraphStorage};

// ─────────────────────────────────────────────
// NodeIndex — Uuid ↔ usize bijection
// ─────────────────────────────────────────────

/// Bidirectional mapping between node UUIDs and dense integer indices.
///
/// Required because the Laplacian works on contiguous arrays, but the
/// graph stores nodes by UUID.
pub struct NodeIndex {
    pub id_to_idx: HashMap<Uuid, usize>,
    pub idx_to_id: Vec<Uuid>,
}

impl NodeIndex {
    pub fn build(storage: &GraphStorage) -> Result<Self, GraphError> {
        let nodes = storage.scan_nodes()?;
        let mut id_to_idx = HashMap::with_capacity(nodes.len());
        let mut idx_to_id  = Vec::with_capacity(nodes.len());

        for node in nodes {
            let idx = idx_to_id.len();
            id_to_idx.insert(node.id, idx);
            idx_to_id.push(node.id);
        }

        Ok(Self { id_to_idx, idx_to_id })
    }

    #[inline] pub fn len(&self) -> usize { self.idx_to_id.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.idx_to_id.is_empty() }
}

// ─────────────────────────────────────────────
// HyperbolicLaplacian
// ─────────────────────────────────────────────

/// Sparse normalized Laplacian `L̃ = I − D^{−½} A D^{−½}` built from
/// Poincaré-distance edge weights.
///
/// Stored as a symmetric adjacency list: `adj[i]` = `[(j, w_ij)]` for
/// every edge incident to node `i` (both directions).
pub struct HyperbolicLaplacian {
    /// Number of nodes.
    pub n:      usize,
    /// UUID ↔ index bijection.
    pub index:  NodeIndex,
    /// `adj[i]` = `[(j, w_ij)]`  (undirected — both directions included)
    pub adj:    Vec<Vec<(usize, f64)>>,
    /// Weighted degree `d_i = Σ_j w_ij`.
    pub degree: Vec<f64>,
}

impl HyperbolicLaplacian {
    /// Build the Laplacian from the current graph state.
    ///
    /// Edge weight: `w(u, v) = 1 / (1 + d_H(u, v))`
    ///
    /// Each directed edge `u → v` contributes weight to both `adj[u]`
    /// and `adj[v]` so the resulting matrix is symmetric.
    pub fn build(
        storage:   &GraphStorage,
        adjacency: &AdjacencyIndex,
    ) -> Result<Self, GraphError> {
        let index = NodeIndex::build(storage)?;
        let n = index.len();

        let mut adj:    Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut degree: Vec<f64>               = vec![0.0;         n];

        let nodes = storage.scan_nodes()?;
        for node in &nodes {
            let Some(&i) = index.id_to_idx.get(&node.id) else { continue };

            for neighbor_id in adjacency.neighbors_out(&node.id) {
                let Some(&j) = index.id_to_idx.get(&neighbor_id) else { continue };

                let Some(neighbour) = storage.get_node(&neighbor_id)? else { continue };
                let d = node.embedding.distance(&neighbour.embedding);
                // Valence modulation: edges between nodes of matching emotional
                // polarity propagate heat faster (emotional clustering).
                let valence_mod = nietzsche_graph::valence_edge_modifier(
                    node.meta.valence,
                    neighbour.meta.valence,
                );
                let w = valence_mod / (1.0 + d);

                // Undirected: add both directions
                adj[i].push((j, w));
                adj[j].push((i, w));

                degree[i] += w;
                degree[j] += w;
            }
        }

        Ok(Self { n, index, adj, degree })
    }

    /// Compute `(L̃ · x)_i = x_i − Σ_j  w_ij / √(d_i · d_j) · x_j`.
    pub fn apply_normalized(&self, x: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.n);
        let mut result = x.to_vec(); // identity part

        for i in 0..self.n {
            let di = self.degree[i].max(1e-12);
            for &(j, w) in &self.adj[i] {
                let dj = self.degree[j].max(1e-12);
                result[i] -= w / (di * dj).sqrt() * x[j];
            }
        }
        result
    }

    /// Compute the **scaled Laplacian** product `(2/λ_max · L̃ − I) · x`.
    ///
    /// This maps eigenvalues from `[0, λ_max]` to `[−1, 1]`, as required
    /// for the Chebyshev recurrence to be numerically stable.
    pub fn apply_scaled(&self, x: &[f64], lambda_max: f64) -> Vec<f64> {
        let lx = self.apply_normalized(x);
        lx.iter().zip(x.iter())
            .map(|(lxi, xi)| (2.0 / lambda_max) * lxi - xi)
            .collect()
    }
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

    #[test]
    fn builds_from_empty_graph() {
        let dir = TempDir::new().unwrap();
        let db = open_db(&dir);
        let lap = HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        assert_eq!(lap.n, 0);
    }

    #[test]
    fn correct_size_after_insert() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.1, 0.0);
        let b = node(0.3, 0.0);
        let id_a = a.id;
        let id_b = b.id;
        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(Edge::new(id_a, id_b, EdgeType::Association, 1.0)).unwrap();

        let lap = HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        assert_eq!(lap.n, 2);
    }

    #[test]
    fn isolated_node_laplacian_is_identity() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);
        db.insert_node(node(0.1, 0.0)).unwrap();

        let lap = HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        let x = vec![3.14];
        let lx = lap.apply_normalized(&x);
        // degree = 0, no neighbors → L̃·x = x
        assert!((lx[0] - 3.14).abs() < 1e-10);
    }

    #[test]
    fn apply_scaled_is_finite() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.2, 0.0);
        let b = node(0.4, 0.0);
        let c = node(0.0, 0.2);
        let ids = [a.id, b.id, c.id];
        for n in [a, b, c] { db.insert_node(n).unwrap(); }
        db.insert_edge(Edge::new(ids[0], ids[1], EdgeType::Association, 1.0)).unwrap();
        db.insert_edge(Edge::new(ids[1], ids[2], EdgeType::Association, 1.0)).unwrap();

        let lap = HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        let x   = vec![1.0, 1.0, 1.0];
        let lx  = lap.apply_scaled(&x, 2.0);
        for v in &lx {
            assert!(v.is_finite(), "apply_scaled produced non-finite value: {v}");
        }
    }

    #[test]
    fn degree_is_positive_for_connected_node() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.1, 0.0);
        let b = node(0.3, 0.0);
        let id_a = a.id;
        let id_b = b.id;
        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(Edge::new(id_a, id_b, EdgeType::Association, 1.0)).unwrap();

        let lap = HyperbolicLaplacian::build(db.storage(), db.adjacency()).unwrap();
        assert!(lap.degree.iter().all(|&d| d >= 0.0));
        // Both nodes should have positive degree (edge is treated as undirected)
        assert!(lap.degree.iter().any(|&d| d > 0.0));
    }
}
