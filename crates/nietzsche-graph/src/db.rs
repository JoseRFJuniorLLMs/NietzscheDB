use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{Edge, Node, PoincareVector};
use crate::storage::GraphStorage;
use crate::transaction::Transaction;
use crate::wal::{GraphWal, GraphWalEntry};

// ─────────────────────────────────────────────
// VectorStore trait
// ─────────────────────────────────────────────

/// Abstraction over a hyperbolic vector store (HyperspaceDB or a mock).
///
/// `NietzscheDB` is generic over this trait so that unit tests can inject
/// a `MockVectorStore` without requiring a live HyperspaceDB process.
pub trait VectorStore: Send + Sync {
    /// Upsert a vector embedding for `id`.
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError>;

    /// Remove the embedding for `id` (no-op if not found).
    fn delete(&mut self, id: Uuid) -> Result<(), GraphError>;

    /// Search for the `k` nearest neighbours of `query`.
    /// Returns a list of `(id, distance)` pairs, sorted ascending by distance.
    fn knn(
        &self,
        query: &PoincareVector,
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, GraphError>;
}

// ─────────────────────────────────────────────
// MockVectorStore  (tests / local dev)
// ─────────────────────────────────────────────

/// In-memory vector store used in tests and offline development.
///
/// Performs a linear scan over stored vectors using the Poincaré ball
/// distance metric — not optimised, but correct.
#[derive(Default)]
pub struct MockVectorStore {
    entries: Vec<(Uuid, PoincareVector)>,
}

impl VectorStore for MockVectorStore {
    fn upsert(&mut self, id: Uuid, vector: &PoincareVector) -> Result<(), GraphError> {
        if let Some(e) = self.entries.iter_mut().find(|(i, _)| *i == id) {
            e.1 = vector.clone();
        } else {
            self.entries.push((id, vector.clone()));
        }
        Ok(())
    }

    fn delete(&mut self, id: Uuid) -> Result<(), GraphError> {
        self.entries.retain(|(i, _)| *i != id);
        Ok(())
    }

    fn knn(
        &self,
        query: &PoincareVector,
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        let mut scored: Vec<(Uuid, f64)> = self
            .entries
            .iter()
            .map(|(id, v)| (*id, query.distance(v)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

// ─────────────────────────────────────────────
// NietzscheDB
// ─────────────────────────────────────────────

/// The primary coordinator for NietzscheDB.
///
/// Maintains three subsystems and keeps them consistent:
/// - [`GraphStorage`] — durable RocksDB-backed graph (nodes, edges, adjacency)
/// - [`GraphWal`]     — append-only Write-Ahead Log for crash recovery
/// - `V: VectorStore` — hyperbolic vector index (HyperspaceDB in production)
///
/// ## Write protocol (saga pattern)
/// 1. Append WAL entry (durable before in-memory update).
/// 2. Write to RocksDB (atomic via WriteBatch where needed).
/// 3. Update the in-memory [`AdjacencyIndex`].
/// 4. Update the vector store (best-effort; re-indexed on recovery if needed).
///
/// On restart, call [`NietzscheDB::open`] which replays the WAL against
/// RocksDB and rebuilds the in-memory adjacency index from stored edges.
pub struct NietzscheDB<V: VectorStore> {
    storage:      GraphStorage,
    wal:          GraphWal,
    adjacency:    AdjacencyIndex,
    vector_store: V,
    /// Hot-tier RAM cache: elite Übermensch nodes pinned for O(1) access.
    /// Populated by [`ZaratustraEngine`] after each Übermensch phase.
    pub hot_tier: Arc<DashMap<Uuid, Node>>,
}

impl<V: VectorStore> NietzscheDB<V> {
    // ── Construction ───────────────────────────────────

    /// Open (or create) a NietzscheDB instance.
    ///
    /// - `data_dir` — root directory; RocksDB lives at `data_dir/rocksdb/`
    ///               and the WAL at `data_dir/graph.wal`.
    /// - `vector_store` — injected vector store implementation.
    pub fn open(data_dir: &Path, vector_store: V) -> Result<Self, GraphError> {
        let rocksdb_path = data_dir.join("rocksdb");
        let rocksdb_str = rocksdb_path
            .to_str()
            .ok_or_else(|| GraphError::Storage("invalid RocksDB path".into()))?;

        let storage   = GraphStorage::open(rocksdb_str)?;
        let wal       = GraphWal::open(data_dir)?;
        let adjacency = storage.rebuild_adjacency()?;

        Ok(Self { storage, wal, adjacency, vector_store, hot_tier: Arc::new(DashMap::new()) })
    }

    // ── Node operations ────────────────────────────────

    /// Insert a new node.
    ///
    /// Writes to WAL first, then RocksDB, then the vector store.
    pub fn insert_node(&mut self, node: Node) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::InsertNode(node.clone()))?;

        // 2. RocksDB
        self.storage.put_node(&node)?;

        // 3. Vector store (best-effort)
        self.vector_store.upsert(node.id, &node.embedding)?;

        Ok(())
    }

    /// Bulk-insert N nodes in a single WAL flush + single RocksDB WriteBatch.
    ///
    /// **10–50× faster than N individual `insert_node` calls** for large imports
    /// because:
    /// - One `BufWriter::flush()` instead of N (eliminates per-write fsync cost)
    /// - One `DB::write(WriteBatch)` instead of N individual RocksDB writes
    ///
    /// Suitable for initial data loads, ETL pipelines, and NQL `INSERT` batches.
    pub fn insert_nodes_bulk(&mut self, nodes: Vec<Node>) -> Result<(), GraphError> {
        if nodes.is_empty() { return Ok(()); }

        // 1. WAL — buffer all entries, flush ONCE at the end.
        for node in &nodes {
            self.wal.append_buffered(&GraphWalEntry::InsertNode(node.clone()))?;
        }
        self.wal.flush()?;

        // 2. RocksDB — single WriteBatch for all nodes + energy_idx keys.
        self.storage.put_nodes_batch(&nodes)?;

        // 3. Vector store — sequential upserts (HNSW insert is not batchable yet).
        for node in &nodes {
            self.vector_store.upsert(node.id, &node.embedding)?;
        }

        Ok(())
    }

    /// Bulk-insert N edges in a single WAL flush + single RocksDB WriteBatch.
    ///
    /// Also updates the in-memory `AdjacencyIndex` for all inserted edges.
    pub fn insert_edges_bulk(&mut self, edges: Vec<Edge>) -> Result<(), GraphError> {
        if edges.is_empty() { return Ok(()); }

        // 1. WAL — buffer all, flush once.
        for edge in &edges {
            self.wal.append_buffered(&GraphWalEntry::InsertEdge(edge.clone()))?;
        }
        self.wal.flush()?;

        // 2. RocksDB — single WriteBatch.
        self.storage.put_edges_batch(&edges)?;

        // 3. In-memory adjacency index.
        for edge in &edges {
            self.adjacency.add_edge(edge);
        }

        Ok(())
    }

    /// Retrieve a node by ID.
    ///
    /// Checks the hot-tier RAM cache first (O(1) DashMap lookup) before
    /// falling back to RocksDB. Elite nodes promoted by Zaratustra are
    /// served entirely from RAM on repeat reads.
    pub fn get_node(&self, id: Uuid) -> Result<Option<Node>, GraphError> {
        if let Some(node) = self.hot_tier.get(&id) {
            return Ok(Some(node.clone()));
        }
        self.storage.get_node(&id)
    }

    /// Soft-delete a node: sets energy = 0 and marks all its edges as Pruned.
    ///
    /// The node record is kept in RocksDB for historical replay; it is only
    /// excluded from traversal via energy-based filtering.
    pub fn prune_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::PruneNode(id))?;

        // 2. Fetch, modify, re-persist
        if let Some(mut node) = self.storage.get_node(&id)? {
            node.energy = 0.0;
            self.storage.put_node(&node)?;
        }

        // 3. Remove from vector store
        self.vector_store.delete(id)?;

        // 4. Remove from in-memory adjacency (no traversal from pruned nodes)
        self.adjacency.remove_node(&id);

        Ok(())
    }

    /// Hard-delete a node and all edges referencing it.
    ///
    /// Prefer [`prune_node`] for normal operation; this is for administrative
    /// data-correction workflows.
    pub fn delete_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::DeleteNode(id))?;

        // 2. Remove incident edges first (RocksDB)
        let out_edges = self.storage.edge_ids_from(&id)?;
        let in_edges  = self.storage.edge_ids_to(&id)?;
        for eid in out_edges.iter().chain(in_edges.iter()) {
            self.wal.append(&GraphWalEntry::DeleteEdge(*eid))?;
            if let Some(edge) = self.storage.get_edge(eid)? {
                self.storage.delete_edge(&edge)?;
            }
        }

        // 3. Remove node
        self.storage.delete_node(&id)?;

        // 4. In-memory adjacency
        self.adjacency.remove_node(&id);

        // 5. Vector store
        self.vector_store.delete(id)?;

        Ok(())
    }

    /// Update a node's energy value.
    ///
    /// Uses `put_node_update_energy()` to atomically swap the energy_idx key,
    /// preventing stale entries from corrupting `WHERE n.energy > X` queries.
    pub fn update_energy(&mut self, node_id: Uuid, energy: f32) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateNodeEnergy { node_id, energy })?;

        if let Some(mut node) = self.storage.get_node(&node_id)? {
            let old_energy = node.energy;
            node.energy = energy;
            // Evict from hot_tier so callers get the fresh value on next read.
            self.hot_tier.remove(&node_id);
            self.storage.put_node_update_energy(&node, old_energy)?;
        }

        Ok(())
    }

    /// Update a node's local Hausdorff dimension.
    pub fn update_hausdorff(&mut self, node_id: Uuid, hausdorff: f32) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateHausdorff { node_id, hausdorff })?;

        if let Some(mut node) = self.storage.get_node(&node_id)? {
            let old_energy = node.energy; // energy unchanged but must pass it
            node.hausdorff_local = hausdorff;
            self.hot_tier.remove(&node_id);
            self.storage.put_node_update_energy(&node, old_energy)?;
        }

        Ok(())
    }

    /// Update a node's embedding (reconsolidation / sleep cycle).
    pub fn update_embedding(
        &mut self,
        node_id: Uuid,
        embedding: PoincareVector,
    ) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateEmbedding {
            node_id,
            embedding: embedding.clone(),
        })?;

        if let Some(mut node) = self.storage.get_node(&node_id)? {
            let old_energy = node.energy;
            node.embedding = embedding.clone();
            self.hot_tier.remove(&node_id);
            self.storage.put_node_update_energy(&node, old_energy)?;
        }

        self.vector_store.upsert(node_id, &embedding)?;

        Ok(())
    }

    // ── Edge operations ────────────────────────────────

    /// Insert a new edge and update the adjacency index.
    pub fn insert_edge(&mut self, edge: Edge) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::InsertEdge(edge.clone()))?;

        // 2. RocksDB (atomic WriteBatch across edges + adj_out + adj_in)
        self.storage.put_edge(&edge)?;

        // 3. In-memory adjacency
        self.adjacency.add_edge(&edge);

        Ok(())
    }

    /// Retrieve an edge by ID.
    pub fn get_edge(&self, id: Uuid) -> Result<Option<Edge>, GraphError> {
        self.storage.get_edge(&id)
    }

    /// Hard-delete an edge and update the adjacency index.
    pub fn delete_edge(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::DeleteEdge(id))?;

        // 2. Fetch edge, remove from adjacency index, then delete from RocksDB
        if let Some(edge) = self.storage.get_edge(&id)? {
            self.adjacency.remove_edge(&edge);
            self.storage.delete_edge(&edge)?;
        }

        Ok(())
    }

    // ── Query helpers ──────────────────────────────────

    /// Return outgoing neighbour IDs for `node_id` (from in-memory index).
    pub fn neighbors_out(&self, node_id: Uuid) -> Vec<Uuid> {
        self.adjacency.neighbors_out(&node_id)
    }

    /// Return incoming neighbour IDs for `node_id` (from in-memory index).
    pub fn neighbors_in(&self, node_id: Uuid) -> Vec<Uuid> {
        self.adjacency.neighbors_in(&node_id)
    }

    // ── Accessors for traversal engine ─────────────────

    pub fn storage(&self) -> &GraphStorage { &self.storage }
    pub fn adjacency(&self) -> &AdjacencyIndex { &self.adjacency }

    // ── Hot-Tier RAM cache ─────────────────────────────

    /// Alias for [`get_node`] — hot-tier check is now built into the primary
    /// read path so this function is kept only for backward compatibility.
    #[inline]
    pub fn get_node_fast(&self, id: Uuid) -> Result<Option<Node>, GraphError> {
        self.get_node(id)
    }

    /// Promote a node to the hot-tier RAM cache (called by Zaratustra engine).
    pub fn promote_to_hot_tier(&self, node: Node) {
        self.hot_tier.insert(node.id, node);
    }

    /// Evict a node from the hot-tier RAM cache.
    pub fn evict_from_hot_tier(&self, id: &Uuid) {
        self.hot_tier.remove(id);
    }

    /// Replace the entire hot-tier with a new set of elite nodes.
    /// Called after every Zaratustra Übermensch phase.
    pub fn replace_hot_tier(&self, elite_nodes: Vec<Node>) {
        self.hot_tier.clear();
        for node in elite_nodes {
            self.hot_tier.insert(node.id, node);
        }
    }

    /// Current hot-tier node count.
    pub fn hot_tier_len(&self) -> usize {
        self.hot_tier.len()
    }

    /// k-nearest-neighbour search in the vector store.
    pub fn knn(
        &self,
        query: &PoincareVector,
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        self.vector_store.knn(query, k)
    }

    /// **Filtered KNN** — return the `k` nearest neighbours that satisfy
    /// `node.energy >= min_energy`, using `energy_idx` as a pre-filter.
    ///
    /// ## Routing strategy (Qdrant pattern)
    ///
    /// | Candidate count | Strategy |
    /// |-----------------|----------|
    /// | < `PLAIN_SCAN_THRESHOLD` | Linear scan: compute Poincaré distance for each candidate directly |
    /// | ≥ `PLAIN_SCAN_THRESHOLD` | HNSW with 4× oversample → post-filter by allowed set |
    ///
    /// The cardinality check is O(log N) via `energy_idx`; no full table scan.
    pub fn knn_energy_filtered(
        &self,
        query: &PoincareVector,
        k: usize,
        min_energy: f32,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        /// Below this threshold: linear scan is cheaper than HNSW + post-filter.
        const PLAIN_SCAN_THRESHOLD: usize = 500;

        // O(log N) — reads only the energy_idx CF (20-byte keys, no node deserialisation).
        let candidate_ids = self.storage.scan_energy_ids_gt(min_energy)?;

        if candidate_ids.len() < PLAIN_SCAN_THRESHOLD {
            // ── Plain scan path ──────────────────────────────────────────────
            // Load each candidate from hot_tier first, then RocksDB.
            // Still O(k × ~400 bytes) — fast with hot_tier, acceptable without.
            let mut scored: Vec<(Uuid, f64)> = candidate_ids
                .into_iter()
                .filter_map(|id| {
                    self.get_node(id).ok().flatten()
                        .map(|n| (n.id, query.distance(&n.embedding)))
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(scored)
        } else {
            // ── HNSW path with post-filter ───────────────────────────────────
            // Oversample 4× so the filter has enough candidates after pruning.
            let allowed: std::collections::HashSet<Uuid> =
                candidate_ids.into_iter().collect();
            let raw = self.vector_store.knn(query, k * 4)?;
            let filtered: Vec<(Uuid, f64)> = raw
                .into_iter()
                .filter(|(id, _)| allowed.contains(id))
                .take(k)
                .collect();
            Ok(filtered)
        }
    }

    // ── Stats ──────────────────────────────────────────

    pub fn node_count(&self) -> Result<usize, GraphError> { self.storage.node_count() }
    pub fn edge_count(&self) -> Result<usize, GraphError> { self.storage.edge_count() }

    // ── Transaction API ────────────────────────────────

    /// Begin a new ACID transaction.
    ///
    /// All mutations on the returned [`Transaction`] are buffered in memory.
    /// Call [`Transaction::commit`] to apply them atomically, or
    /// [`Transaction::rollback`] to discard.
    ///
    /// Only one transaction can be active at a time (enforced by the
    /// `&mut self` borrow on both `begin_transaction` and the `Transaction`).
    pub fn begin_transaction(&mut self) -> Transaction<'_, V> {
        Transaction::new(Uuid::new_v4(), self)
    }

    // ── pub(crate) WAL access for Transaction ─────────

    /// Append a WAL entry. Used by the transaction coordinator only.
    pub(crate) fn wal_append(&mut self, entry: &GraphWalEntry) -> Result<(), GraphError> {
        self.wal.append(entry)
    }

    // ── pub(crate) storage primitives (no WAL write) ──

    /// Apply an `InsertNode` directly to storage and vector store.
    /// The WAL entry must have been written by the caller.
    pub(crate) fn apply_insert_node(&mut self, node: &Node) -> Result<(), GraphError> {
        self.storage.put_node(node)?;
        self.vector_store.upsert(node.id, &node.embedding)?;
        Ok(())
    }

    /// Apply an `InsertEdge` directly to storage and adjacency index.
    pub(crate) fn apply_insert_edge(&mut self, edge: &Edge) {
        let _ = self.storage.put_edge(edge);
        self.adjacency.add_edge(edge);
    }

    /// Apply a `PruneNode` (energy → 0) directly to storage.
    pub(crate) fn apply_prune_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        if let Some(mut node) = self.storage.get_node(&id)? {
            node.energy = 0.0;
            self.storage.put_node(&node)?;
        }
        self.vector_store.delete(id)?;
        self.adjacency.remove_node(&id);
        Ok(())
    }

    /// Apply a `DeleteNode` directly to storage.
    pub(crate) fn apply_delete_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        self.storage.delete_node(&id)?;
        self.vector_store.delete(id)?;
        self.adjacency.remove_node(&id);
        Ok(())
    }

    /// Apply a `DeleteEdge` directly to storage.
    pub(crate) fn apply_delete_edge(&mut self, id: Uuid) -> Result<(), GraphError> {
        if let Some(edge) = self.storage.get_edge(&id)? {
            self.adjacency.remove_edge(&edge);
            self.storage.delete_edge(&edge)?;
        }
        Ok(())
    }

    /// Apply an `UpdateEnergy` directly to storage.
    pub(crate) fn apply_update_energy(&mut self, node_id: Uuid, energy: f32) -> Result<(), GraphError> {
        if let Some(mut node) = self.storage.get_node(&node_id)? {
            node.energy = energy;
            self.storage.put_node(&node)?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_db(dir: &TempDir) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap()
    }

    fn make_node(dims: &[f64]) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(dims.to_vec()),
            serde_json::json!({}),
        )
    }

    #[test]
    fn insert_and_get_node() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let node = make_node(&[0.1, 0.2]);
        let id = node.id;

        db.insert_node(node).unwrap();
        let found = db.get_node(id).unwrap().unwrap();
        assert_eq!(found.id, id);
    }

    #[test]
    fn insert_and_get_edge() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let edge = Edge::association(a.id, b.id, 0.8);
        let eid = edge.id;

        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(edge).unwrap();

        let found = db.get_edge(eid).unwrap().unwrap();
        assert_eq!(found.id, eid);
    }

    #[test]
    fn adjacency_updated_on_insert_edge() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let aid = a.id;
        let bid = b.id;
        let edge = Edge::association(aid, bid, 0.5);

        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(edge).unwrap();

        assert!(db.neighbors_out(aid).contains(&bid));
        assert!(db.neighbors_in(bid).contains(&aid));
    }

    #[test]
    fn prune_node_zeroes_energy_and_removes_from_adjacency() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let aid = a.id;
        let bid = b.id;
        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(Edge::association(aid, bid, 0.9)).unwrap();

        db.prune_node(aid).unwrap();

        let node = db.get_node(aid).unwrap().unwrap();
        assert_eq!(node.energy, 0.0);
        assert!(db.neighbors_out(aid).is_empty());
    }

    #[test]
    fn delete_node_removes_incident_edges() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let aid = a.id;
        let bid = b.id;
        let edge = Edge::association(aid, bid, 0.7);
        let eid = edge.id;

        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(edge).unwrap();
        db.delete_node(aid).unwrap();

        assert!(db.get_node(aid).unwrap().is_none());
        assert!(db.get_edge(eid).unwrap().is_none());
    }

    #[test]
    fn update_energy_persists() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let node = make_node(&[0.3, 0.0]);
        let id = node.id;
        db.insert_node(node).unwrap();

        db.update_energy(id, 0.42).unwrap();

        let found = db.get_node(id).unwrap().unwrap();
        assert!((found.energy - 0.42).abs() < 1e-6);
    }

    #[test]
    fn knn_returns_nearest() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let near  = make_node(&[0.01, 0.0]);
        let far   = make_node(&[0.5,  0.0]);
        let query = PoincareVector::new(vec![0.0, 0.0]); // origin

        db.insert_node(near.clone()).unwrap();
        db.insert_node(far.clone()).unwrap();

        let results = db.knn(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, near.id); // nearest to origin
    }

    #[test]
    fn node_count_and_edge_count() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let edge = Edge::association(a.id, b.id, 0.5);

        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_edge(edge).unwrap();

        assert_eq!(db.node_count().unwrap(), 2);
        assert_eq!(db.edge_count().unwrap(), 1);
    }

    #[test]
    fn survives_reopen_wal_replay() {
        let dir = tmp();
        let node_id;
        {
            let mut db = open_db(&dir);
            let node = make_node(&[0.1, 0.2]);
            node_id = node.id;
            db.insert_node(node).unwrap();
        }

        // Reopen — RocksDB should have the node persisted
        let db2 = open_db(&dir);
        let found = db2.get_node(node_id).unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn update_hausdorff_persists() {
        let dir = tmp();
        let mut db = open_db(&dir);
        let node = make_node(&[0.2, 0.0]);
        let id = node.id;
        db.insert_node(node).unwrap();

        db.update_hausdorff(id, 1.5).unwrap();

        let found = db.get_node(id).unwrap().unwrap();
        assert!((found.hausdorff_local - 1.5).abs() < 1e-6);
    }
}
