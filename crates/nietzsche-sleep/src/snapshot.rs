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
use std::time::{SystemTime, UNIX_EPOCH};

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
// NamedSnapshot — time-travel / versioning
// ─────────────────────────────────────────────

/// A named, timestamped snapshot for time-travel queries.
///
/// Unlike [`Snapshot`], which is an anonymous, ephemeral checkpoint used
/// during the reconsolidation sleep cycle, a `NamedSnapshot` carries a
/// human-readable label and a creation timestamp so that users can
/// bookmark specific graph states and restore them later.
#[derive(Debug, Clone)]
pub struct NamedSnapshot {
    /// Human-readable label (unique within a [`SnapshotRegistry`]).
    pub name: String,
    /// Unix timestamp (seconds since epoch) when this snapshot was created.
    pub created_at: i64,
    /// Number of node embeddings captured (cached for fast listing).
    pub node_count: usize,
    /// `node_id → embedding` at creation time.
    pub embeddings: HashMap<Uuid, PoincareVector>,
}

// ─────────────────────────────────────────────
// SnapshotRegistry
// ─────────────────────────────────────────────

/// Registry of named snapshots for time-travel.
///
/// Holds an in-memory map of [`NamedSnapshot`]s keyed by name.
/// Typical lifecycle:
///
/// ```text
/// registry.create("before-migration", &db)?;
/// // … dangerous operation …
/// registry.restore("before-migration", &mut db)?;
/// ```
pub struct SnapshotRegistry {
    snapshots: HashMap<String, NamedSnapshot>,
}

impl SnapshotRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
        }
    }

    /// Take a snapshot of **all** current node embeddings and store it under `name`.
    ///
    /// Returns a reference to the newly created [`NamedSnapshot`].
    ///
    /// # Errors
    ///
    /// - [`GraphError::Storage`] if a name collision occurs (snapshot already exists).
    /// - Any error propagated from `db.storage().scan_nodes()`.
    pub fn create<V: VectorStore>(
        &mut self,
        name: &str,
        db: &NietzscheDB<V>,
    ) -> Result<&NamedSnapshot, GraphError> {
        if self.snapshots.contains_key(name) {
            return Err(GraphError::Storage(format!(
                "snapshot '{}' already exists",
                name
            )));
        }

        let nodes = db.storage().scan_nodes()?;
        let embeddings: HashMap<Uuid, PoincareVector> =
            nodes.into_iter().map(|n| (n.id, n.embedding)).collect();
        let node_count = embeddings.len();

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let snap = NamedSnapshot {
            name: name.to_owned(),
            created_at,
            node_count,
            embeddings,
        };

        self.snapshots.insert(name.to_owned(), snap);
        Ok(self.snapshots.get(name).unwrap())
    }

    /// Retrieve a named snapshot by name.
    pub fn get(&self, name: &str) -> Option<&NamedSnapshot> {
        self.snapshots.get(name)
    }

    /// List all snapshots as `(name, created_at, node_count)` tuples,
    /// sorted by creation time (oldest first).
    pub fn list(&self) -> Vec<(&str, i64, usize)> {
        let mut entries: Vec<(&str, i64, usize)> = self
            .snapshots
            .values()
            .map(|s| (s.name.as_str(), s.created_at, s.node_count))
            .collect();
        entries.sort_by_key(|&(_, ts, _)| ts);
        entries
    }

    /// Remove a named snapshot. Returns `true` if it existed.
    pub fn delete(&mut self, name: &str) -> bool {
        self.snapshots.remove(name).is_some()
    }

    /// Restore embeddings from a named snapshot back into the graph.
    ///
    /// Nodes that were deleted between snapshot creation and restore are
    /// silently skipped (same semantics as [`Snapshot::restore`]).
    ///
    /// # Returns
    ///
    /// The number of embeddings actually restored.
    ///
    /// # Errors
    ///
    /// - [`GraphError::Storage`] if no snapshot with `name` exists.
    /// - Any error propagated from the underlying `db.update_embedding()`.
    pub fn restore<V: VectorStore>(
        &self,
        name: &str,
        db: &mut NietzscheDB<V>,
    ) -> Result<usize, GraphError> {
        let snap = self.snapshots.get(name).ok_or_else(|| {
            GraphError::Storage(format!("snapshot '{}' not found", name))
        })?;

        let mut count = 0;
        for (&id, embedding) in &snap.embeddings {
            let exists = db.get_node(id)?.is_some();
            if exists {
                db.update_embedding(id, embedding.clone())?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Number of named snapshots currently held.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns `true` if the registry contains no snapshots.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

impl Default for SnapshotRegistry {
    fn default() -> Self {
        Self::new()
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

    // ── NamedSnapshot / SnapshotRegistry tests ──

    #[test]
    fn named_snapshot_create_and_retrieve() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id_a = insert(&mut db, vec![0.1, 0.0]);
        let id_b = insert(&mut db, vec![0.0, 0.2]);

        let mut registry = SnapshotRegistry::new();
        let snap = registry.create("v1", &db).unwrap();

        assert_eq!(snap.name, "v1");
        assert_eq!(snap.node_count, 2);
        assert!(snap.created_at > 0);
        assert!(snap.embeddings.contains_key(&id_a));
        assert!(snap.embeddings.contains_key(&id_b));

        // Retrieve by name
        let retrieved = registry.get("v1").unwrap();
        assert_eq!(retrieved.name, "v1");
        assert_eq!(retrieved.node_count, 2);

        let emb_a = retrieved.embeddings.get(&id_a).unwrap();
        assert!((emb_a.coords[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn named_snapshot_duplicate_name_errors() {
        let dir = tempfile::tempdir().unwrap();
        let db = open_db(dir.path());

        let mut registry = SnapshotRegistry::new();
        registry.create("dup", &db).unwrap();

        let err = registry.create("dup", &db).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("already exists"), "unexpected error: {msg}");
    }

    #[test]
    fn named_snapshot_delete() {
        let dir = tempfile::tempdir().unwrap();
        let db = open_db(dir.path());

        let mut registry = SnapshotRegistry::new();
        registry.create("ephemeral", &db).unwrap();
        assert_eq!(registry.len(), 1);

        let removed = registry.delete("ephemeral");
        assert!(removed);
        assert_eq!(registry.len(), 0);
        assert!(registry.get("ephemeral").is_none());

        // Deleting non-existent returns false
        assert!(!registry.delete("ghost"));
    }

    #[test]
    fn named_snapshot_list() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let _a = insert(&mut db, vec![0.1, 0.0]);

        let mut registry = SnapshotRegistry::new();
        registry.create("alpha", &db).unwrap();

        // Insert another node, then create a second snapshot
        let _b = insert(&mut db, vec![0.0, 0.2]);
        registry.create("beta", &db).unwrap();

        let listing = registry.list();
        assert_eq!(listing.len(), 2);

        // Sorted by created_at (both created within the same second, so
        // order is stable but both should be present)
        let names: Vec<&str> = listing.iter().map(|&(n, _, _)| n).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));

        // alpha has 1 node, beta has 2
        let alpha_entry = listing.iter().find(|&&(n, _, _)| n == "alpha").unwrap();
        assert_eq!(alpha_entry.2, 1);
        let beta_entry = listing.iter().find(|&&(n, _, _)| n == "beta").unwrap();
        assert_eq!(beta_entry.2, 2);
    }

    #[test]
    fn named_snapshot_restore() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id = insert(&mut db, vec![0.1, 0.0]);

        let mut registry = SnapshotRegistry::new();
        registry.create("checkpoint", &db).unwrap();

        // Mutate the embedding
        db.update_embedding(id, PoincareVector::new(vec![0.9, 0.0])).unwrap();
        let moved = db.get_node(id).unwrap().unwrap();
        assert!((moved.embedding.coords[0] - 0.9).abs() < 1e-10);

        // Restore from named snapshot
        let count = registry.restore("checkpoint", &mut db).unwrap();
        assert_eq!(count, 1);

        let after = db.get_node(id).unwrap().unwrap();
        assert!((after.embedding.coords[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn named_snapshot_restore_nonexistent_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let registry = SnapshotRegistry::new();
        let err = registry.restore("no-such", &mut db).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn named_snapshot_restore_skips_deleted_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = open_db(dir.path());

        let id = insert(&mut db, vec![0.1, 0.0]);

        let mut registry = SnapshotRegistry::new();
        registry.create("before-delete", &db).unwrap();

        // Delete the node after snapshot
        db.delete_node(id).unwrap();

        let count = registry.restore("before-delete", &mut db).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn registry_is_empty_and_len() {
        let registry = SnapshotRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn registry_default_is_empty() {
        let registry = SnapshotRegistry::default();
        assert!(registry.is_empty());
    }
}
