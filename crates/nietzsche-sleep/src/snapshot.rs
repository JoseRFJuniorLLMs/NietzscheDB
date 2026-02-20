//! Embedding snapshot — checkpoint and restore for the reconsolidation cycle.
//!
//! The snapshot captures all node embeddings at a point in time so that
//! the sleep cycle can roll back if the Hausdorff dimension diverges after
//! perturbation + optimisation.
//!
//! ## Lifecycle
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │  Snapshot::take(db)                                              │
//! │       │                                                          │
//! │       ▼                                                          │
//! │  perturb + optimise embeddings …                                 │
//! │       │                                                          │
//! │       ├── Hausdorff OK ──► keep new embeddings (no restore)     │
//! │       └── Hausdorff bad ► Snapshot::restore(db)                 │
//! └──────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;

use uuid::Uuid;

use nietzsche_graph::{GraphError, NietzscheDB, PoincareVector, VectorStore};

// ─────────────────────────────────────────────
// Snapshot
// ─────────────────────────────────────────────

/// Immutable checkpoint of every node's Poincaré-ball embedding.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// `node_id → embedding` at the time [`Snapshot::take`] was called.
    embeddings: HashMap<Uuid, PoincareVector>,
}

impl Snapshot {
    // ── Construction ───────────────────────────────────

    /// Capture the current embeddings of **all** nodes in the graph.
    ///
    /// Performs a full table scan over RocksDB — call once per sleep cycle,
    /// not in a hot path.
    pub fn take<V: VectorStore>(db: &NietzscheDB<V>) -> Result<Self, GraphError> {
        let nodes = db.storage().scan_nodes()?;
        let embeddings = nodes.into_iter()
            .map(|n| (n.id, n.embedding))
            .collect();
        Ok(Self { embeddings })
    }

    // ── Restore ────────────────────────────────────────

    /// Write all captured embeddings back to the graph (WAL + RocksDB + vector store).
    ///
    /// Nodes that were deleted between [`take`] and [`restore`] are silently skipped.
    ///
    /// # Returns
    /// The number of embeddings actually restored.
    pub fn restore<V: VectorStore>(&self, db: &mut NietzscheDB<V>) -> Result<usize, GraphError> {
        let mut count = 0;

        for (&id, embedding) in &self.embeddings {
            // Check existence via the public db API (avoids holding a borrow into
            // db.storage() across the subsequent &mut call).
            let exists = db.get_node(id)?.is_some();
            if exists {
                db.update_embedding(id, embedding.clone())?;
                count += 1;
            }
        }

        Ok(count)
    }

    // ── Inspection ────────────────────────────────────

    /// Number of node embeddings captured.
    pub fn node_count(&self) -> usize {
        self.embeddings.len()
    }

    /// Retrieve the captured embedding for `id`, if present.
    pub fn get(&self, id: &Uuid) -> Option<&PoincareVector> {
        self.embeddings.get(id)
    }

    /// Iterate over all captured `(id, embedding)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Uuid, &PoincareVector)> {
        self.embeddings.iter()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{MockVectorStore, NietzscheDB, Node, PoincareVector};
    use uuid::Uuid;

    fn open_db(dir: &std::path::Path) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir, MockVectorStore::default()).expect("open db")
    }

    fn insert(db: &mut NietzscheDB<MockVectorStore>, coords: Vec<f64>) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node::new(id, PoincareVector::from_f64(coords), serde_json::json!({}));
        db.insert_node(node).expect("insert node");
        id
    }

    #[test]
    fn snapshot_captures_all_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let _a = insert(&mut db, vec![0.1, 0.0]);
        let _b = insert(&mut db, vec![0.0, 0.2]);
        let _c = insert(&mut db, vec![0.3, 0.3]);

        let snap = Snapshot::take(&db).unwrap();
        assert_eq!(snap.node_count(), 3);
    }

    #[test]
    fn snapshot_is_empty_when_no_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let db = open_db(dir.path());
        let snap = Snapshot::take(&db).unwrap();
        assert_eq!(snap.node_count(), 0);
    }

    #[test]
    fn get_returns_captured_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id = insert(&mut db, vec![0.1, 0.2]);
        let snap = Snapshot::take(&db).unwrap();

        let emb = snap.get(&id).unwrap();
        assert!((emb.coords[0] - 0.1).abs() < 1e-10);
        assert!((emb.coords[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn restore_reverts_modified_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id = insert(&mut db, vec![0.1, 0.0]);

        // Take snapshot at the original position
        let snap = Snapshot::take(&db).unwrap();

        // Move the embedding
        db.update_embedding(id, PoincareVector::new(vec![0.5, 0.5])).unwrap();

        // Verify it moved
        let moved = db.get_node(id).unwrap().unwrap();
        assert!((moved.embedding.coords[0] - 0.5).abs() < 1e-10);

        // Restore
        let restored_count = snap.restore(&mut db).unwrap();
        assert_eq!(restored_count, 1);

        // Verify it's back
        let after = db.get_node(id).unwrap().unwrap();
        assert!((after.embedding.coords[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn restore_skips_deleted_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id = insert(&mut db, vec![0.1, 0.0]);
        let snap = Snapshot::take(&db).unwrap();
        assert_eq!(snap.node_count(), 1);

        // Delete the node
        db.delete_node(id).unwrap();

        // Restore should silently skip
        let count = snap.restore(&mut db).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn iter_yields_all_captured_pairs() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id1 = insert(&mut db, vec![0.1, 0.0]);
        let id2 = insert(&mut db, vec![0.0, 0.1]);

        let snap = Snapshot::take(&db).unwrap();
        let ids: std::collections::HashSet<Uuid> = snap.iter().map(|(&id, _)| id).collect();

        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }
}
