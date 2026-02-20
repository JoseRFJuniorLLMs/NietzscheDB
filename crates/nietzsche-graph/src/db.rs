use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use dashmap::DashMap;
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{Edge, Node, NodeMeta, PoincareVector};
use crate::schema::SchemaValidator;
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
    /// Hot-tier RAM cache: elite Übermensch node metadata pinned for O(1) access.
    ///
    /// ## BUG A fix (Committee 2026-02-19)
    /// Changed from `DashMap<Uuid, Node>` (~24 KB/entry) to `DashMap<Uuid, NodeMeta>`
    /// (~100 bytes/entry) — **240× less RAM per cached node**.
    /// Populated by [`ZaratustraEngine`] after each Übermensch phase.
    pub hot_tier: Arc<DashMap<Uuid, NodeMeta>>,
    /// Metadata fields to maintain secondary indexes on (set via `NIETZSCHE_INDEXED_FIELDS`).
    /// When a node is inserted/deleted, index entries are written/removed for these fields.
    indexed_fields: HashSet<String>,
    /// Optional schema validator for per-NodeType constraints.
    schema_validator: Option<SchemaValidator>,
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

        // Load persisted schema constraints if any exist
        let schema_validator = SchemaValidator::load_all(&storage).ok();

        Ok(Self {
            storage, wal, adjacency, vector_store,
            hot_tier: Arc::new(DashMap::new()),
            indexed_fields: HashSet::new(),
            schema_validator,
        })
    }

    /// Replace the vector store backend at runtime.
    ///
    /// Used at server startup to inject an alternative backend (e.g. GPU)
    /// before any requests are served. Not safe to call while requests are
    /// in-flight — callers must hold the write lock on the collection.
    pub fn set_vector_store(&mut self, vs: V) {
        self.vector_store = vs;
    }

    /// Set the metadata fields to maintain secondary indexes on.
    /// Call this before inserting nodes to enable automatic index maintenance.
    pub fn set_indexed_fields(&mut self, fields: HashSet<String>) {
        self.indexed_fields = fields;
    }

    /// Get a reference to the schema validator (if any).
    pub fn schema_validator(&self) -> Option<&SchemaValidator> {
        self.schema_validator.as_ref()
    }

    /// Register a schema constraint and persist it to RocksDB.
    pub fn set_schema_constraint(
        &mut self,
        constraint: crate::schema::SchemaConstraint,
    ) -> Result<(), GraphError> {
        let validator = self.schema_validator.get_or_insert_with(SchemaValidator::new);
        validator.save_constraint(&self.storage, &constraint)?;
        validator.set_constraint(constraint);
        Ok(())
    }

    // ── NodeMeta-only accessors (BUG A fast path) ────

    /// Retrieve node metadata by ID (~100 bytes, no embedding).
    ///
    /// Checks the hot-tier RAM cache first (O(1) DashMap lookup) before
    /// falling back to RocksDB `CF_NODES`.
    pub fn get_node_meta(&self, id: Uuid) -> Result<Option<NodeMeta>, GraphError> {
        if let Some(meta) = self.hot_tier.get(&id) {
            return Ok(Some(meta.clone()));
        }
        self.storage.get_node_meta(&id)
    }

    // ── Node operations ────────────────────────────────

    /// Insert a new node.
    ///
    /// Writes to WAL first, then RocksDB, then the vector store.
    pub fn insert_node(&mut self, node: Node) -> Result<(), GraphError> {
        // 0. Schema validation (if enabled)
        if let Some(ref validator) = self.schema_validator {
            if let Err(violations) = validator.validate_node(&node.meta) {
                return Err(GraphError::Storage(format!(
                    "schema validation failed: {}",
                    violations.join("; "),
                )));
            }
        }

        // 1. WAL
        self.wal.append(&GraphWalEntry::InsertNode(node.clone()))?;

        // 2. RocksDB
        self.storage.put_node(&node)?;

        // 3. Metadata secondary index
        if !self.indexed_fields.is_empty() {
            self.storage.put_meta_index(&node.id, &node.meta.metadata, &self.indexed_fields)?;
        }

        // 4. Vector store (best-effort)
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

        // 3. Metadata secondary index
        if !self.indexed_fields.is_empty() {
            for node in &nodes {
                self.storage.put_meta_index(&node.id, &node.meta.metadata, &self.indexed_fields)?;
            }
        }

        // 4. Vector store — sequential upserts (HNSW insert is not batchable yet).
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

    /// Retrieve a full node (metadata + embedding) by ID.
    ///
    /// If the hot-tier has the metadata cached, uses that + a single
    /// `CF_EMBEDDINGS` read. Otherwise falls back to the full
    /// `GraphStorage::get_node()` join.
    pub fn get_node(&self, id: Uuid) -> Result<Option<Node>, GraphError> {
        if let Some(meta) = self.hot_tier.get(&id) {
            // Hot-tier hit: we have meta, just need the embedding
            return match self.storage.get_embedding(&id)? {
                Some(emb) => Ok(Some(Node::from((meta.clone(), emb)))),
                None => Ok(None),
            };
        }
        self.storage.get_node(&id)
    }

    /// Soft-delete a node: sets energy = 0 and marks all its edges as Pruned.
    ///
    /// The node record is kept in RocksDB for historical replay; it is only
    /// excluded from traversal via energy-based filtering.
    ///
    /// **BUG A optimized:** reads only `NodeMeta` (~100 bytes), never touches
    /// the ~24 KB embedding in `CF_EMBEDDINGS`.
    pub fn prune_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::PruneNode(id))?;

        // 2. Fetch meta only, set energy = 0, re-persist meta only
        if let Some(mut meta) = self.storage.get_node_meta(&id)? {
            let old_energy = meta.energy;
            meta.energy = 0.0;
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
        }

        // 3. Remove from vector store
        self.vector_store.delete(id)?;

        // 4. Remove from hot-tier + in-memory adjacency
        self.hot_tier.remove(&id);
        self.adjacency.remove_node(&id);

        Ok(())
    }

    /// Hard-delete a node and all edges referencing it.
    ///
    /// Prefer [`prune_node`] for normal operation; this is for administrative
    /// data-correction workflows.
    pub fn delete_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 0. Read metadata for index cleanup before deletion
        let meta_for_idx = if !self.indexed_fields.is_empty() {
            self.storage.get_node_meta(&id)?
        } else {
            None
        };

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

        // 4. Clean up metadata secondary index
        if let Some(meta) = meta_for_idx {
            self.storage.delete_meta_index(&id, &meta.metadata, &self.indexed_fields)?;
        }

        // 5. In-memory adjacency
        self.adjacency.remove_node(&id);

        // 6. Vector store
        self.vector_store.delete(id)?;

        Ok(())
    }

    /// Reap all expired nodes (TTL enforcement).
    ///
    /// Scans `CF_NODES` for nodes where `expires_at <= now`, then hard-deletes
    /// each one (including incident edges, vector store entry, adjacency).
    /// Returns the number of reaped nodes.
    ///
    /// Intended to be called periodically by a background task.
    pub fn reap_expired(&mut self) -> Result<usize, GraphError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let expired_ids = self.storage.scan_expired_node_ids(now)?;
        let count = expired_ids.len();
        for id in expired_ids {
            self.delete_node(id)?;
        }
        Ok(count)
    }

    /// Update a node's energy value.
    ///
    /// **BUG A optimized:** reads only `NodeMeta` (~100 bytes) and writes only
    /// to `CF_NODES` + `CF_ENERGY_IDX` — never touches the ~24 KB embedding.
    pub fn update_energy(&mut self, node_id: Uuid, energy: f32) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateNodeEnergy { node_id, energy })?;

        if let Some(mut meta) = self.storage.get_node_meta(&node_id)? {
            let old_energy = meta.energy;
            meta.energy = energy;
            self.hot_tier.remove(&node_id);
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
        }

        Ok(())
    }

    /// Update a node's local Hausdorff dimension.
    ///
    /// **BUG A optimized:** reads/writes only `NodeMeta` (~100 bytes).
    pub fn update_hausdorff(&mut self, node_id: Uuid, hausdorff: f32) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateHausdorff { node_id, hausdorff })?;

        if let Some(mut meta) = self.storage.get_node_meta(&node_id)? {
            let old_energy = meta.energy; // energy unchanged but must pass it
            meta.hausdorff_local = hausdorff;
            self.hot_tier.remove(&node_id);
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
        }

        Ok(())
    }

    /// Update a node's embedding (reconsolidation / sleep cycle).
    ///
    /// Writes the new embedding to `CF_EMBEDDINGS` and updates the vector store.
    /// Also updates `depth` in `NodeMeta` since it derives from `‖embedding‖`.
    pub fn update_embedding(
        &mut self,
        node_id: Uuid,
        embedding: PoincareVector,
    ) -> Result<(), GraphError> {
        self.wal.append(&GraphWalEntry::UpdateEmbedding {
            node_id,
            embedding: embedding.clone(),
        })?;

        // Update embedding in CF_EMBEDDINGS
        self.storage.put_embedding(&node_id, &embedding)?;

        // Update depth in NodeMeta (derives from ‖embedding‖)
        if let Some(mut meta) = self.storage.get_node_meta(&node_id)? {
            meta.depth = embedding.depth() as f32;
            self.storage.put_node_meta(&meta)?;
            self.hot_tier.remove(&node_id);
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

    // ── MERGE helpers (FASE D — Neo4j MERGE replacement) ─────────

    /// Find a node by `node_type` and content key match.
    ///
    /// Scans all nodes in CF_NODES, filtering by `node_type` and checking
    /// that the node's JSON content contains ALL keys in `match_keys` with
    /// matching values.
    ///
    /// Returns the first matching node, or `None` if no match.
    pub fn find_node_by_content(
        &self,
        node_type: &str,
        match_keys: &serde_json::Value,
    ) -> Result<Option<Node>, GraphError> {
        let match_obj = match match_keys.as_object() {
            Some(obj) => obj,
            None => return Ok(None),
        };

        for result in self.storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Filter by node_type (Debug format matches proto strings)
            let type_str = format!("{:?}", meta.node_type);
            if type_str != node_type {
                continue;
            }

            // Check content match
            if let Some(content_obj) = meta.content.as_object() {
                let all_match = match_obj.iter().all(|(k, v)| {
                    content_obj.get(k).map_or(false, |cv| cv == v)
                });
                if all_match {
                    // Found — fetch full node with embedding
                    return self.get_node(meta.id);
                }
            }
        }

        Ok(None)
    }

    /// Update a node's content JSON by merging new keys.
    ///
    /// Existing keys are overwritten, new keys are added.
    /// The node's embedding is NOT changed.
    pub fn update_node_content(
        &mut self,
        id: Uuid,
        updates: &serde_json::Value,
    ) -> Result<(), GraphError> {
        if let Some(mut meta) = self.storage.get_node_meta(&id)? {
            if let (Some(content_obj), Some(update_obj)) = (
                meta.content.as_object_mut(),
                updates.as_object(),
            ) {
                for (k, v) in update_obj {
                    content_obj.insert(k.clone(), v.clone());
                }
            }

            // Re-persist meta (energy unchanged)
            let old_energy = meta.energy;
            self.hot_tier.remove(&id);
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
        }

        Ok(())
    }

    /// Find an edge between two nodes with a specific type.
    ///
    /// Scans outgoing edges of `from_id` looking for one that points to
    /// `to_id` with the given `edge_type`.
    pub fn find_edge(
        &self,
        from_id: Uuid,
        to_id: Uuid,
        edge_type: &str,
    ) -> Result<Option<Edge>, GraphError> {
        let edge_ids = self.storage.edge_ids_from(&from_id)?;
        for eid in edge_ids {
            if let Some(edge) = self.storage.get_edge(&eid)? {
                if edge.to == to_id {
                    let type_str = format!("{:?}", edge.edge_type);
                    if type_str == edge_type {
                        return Ok(Some(edge));
                    }
                }
            }
        }
        Ok(None)
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

    /// Promote a node's metadata to the hot-tier RAM cache (called by Zaratustra engine).
    ///
    /// Only caches `NodeMeta` (~100 bytes) — the embedding stays in `CF_EMBEDDINGS`
    /// and is read on demand via `get_node()`.
    pub fn promote_to_hot_tier(&self, node: &Node) {
        self.hot_tier.insert(node.id, node.meta.clone());
    }

    /// Promote `NodeMeta` directly (avoids needing the full Node).
    pub fn promote_meta_to_hot_tier(&self, meta: NodeMeta) {
        self.hot_tier.insert(meta.id, meta);
    }

    /// Evict a node from the hot-tier RAM cache.
    pub fn evict_from_hot_tier(&self, id: &Uuid) {
        self.hot_tier.remove(id);
    }

    /// Replace the entire hot-tier with metadata from a set of elite nodes.
    /// Called after every Zaratustra Übermensch phase.
    pub fn replace_hot_tier(&self, elite_nodes: &[Node]) {
        self.hot_tier.clear();
        for node in elite_nodes {
            self.hot_tier.insert(node.id, node.meta.clone());
        }
    }

    /// Replace the entire hot-tier with pre-extracted metadata.
    pub fn replace_hot_tier_meta(&self, elite_metas: Vec<NodeMeta>) {
        self.hot_tier.clear();
        for meta in elite_metas {
            self.hot_tier.insert(meta.id, meta);
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

    // ── Hybrid BM25 + ANN search ────────────────────────

    /// Hybrid search combining BM25 full-text scoring with KNN vector
    /// similarity, fused via Reciprocal Rank Fusion (RRF).
    ///
    /// ## Algorithm
    ///
    /// 1. Run BM25 search → ranked list R_text
    /// 2. Run KNN search  → ranked list R_vec
    /// 3. Fuse: `score(d) = text_weight / (60 + rank_text(d)) + vector_weight / (60 + rank_vec(d))`
    /// 4. Sort by fused score descending, return top-k
    ///
    /// The constant 60 is the standard RRF parameter from the original paper
    /// (Cormack et al., 2009).
    pub fn hybrid_search(
        &self,
        text_query: &str,
        vector_query: &PoincareVector,
        k: usize,
        text_weight: f64,
        vector_weight: f64,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        const RRF_K: f64 = 60.0;

        // Over-fetch from both sources to improve fusion quality
        let fetch_k = k * 3;

        // BM25 ranked list
        let fts = crate::fulltext::FullTextIndex::new(&self.storage);
        let bm25_results = fts.search(text_query, fetch_k)?;

        // KNN ranked list
        let knn_results = self.vector_store.knn(vector_query, fetch_k)?;

        // Build RRF fusion scores
        let mut fused_scores: HashMap<Uuid, f64> = HashMap::new();

        for (rank, result) in bm25_results.iter().enumerate() {
            *fused_scores.entry(result.node_id).or_default() +=
                text_weight / (RRF_K + rank as f64 + 1.0);
        }

        for (rank, (id, _dist)) in knn_results.iter().enumerate() {
            *fused_scores.entry(*id).or_default() +=
                vector_weight / (RRF_K + rank as f64 + 1.0);
        }

        // Sort by fused score descending
        let mut results: Vec<(Uuid, f64)> = fused_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results)
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
        if let Some(mut meta) = self.storage.get_node_meta(&id)? {
            let old_energy = meta.energy;
            meta.energy = 0.0;
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
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

    /// Apply an `UpdateEnergy` directly to storage (NodeMeta only — no embedding I/O).
    pub(crate) fn apply_update_energy(&mut self, node_id: Uuid, energy: f32) -> Result<(), GraphError> {
        if let Some(mut meta) = self.storage.get_node_meta(&node_id)? {
            let old_energy = meta.energy;
            meta.energy = energy;
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
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
            PoincareVector::from_f64(dims.to_vec()),
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
    fn reap_expired_removes_only_expired_nodes() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Expired node
        let mut expired = make_node(&[0.1, 0.0]);
        expired.meta.expires_at = Some(now - 60);
        let expired_id = expired.id;
        db.insert_node(expired).unwrap();

        // Fresh node (TTL in the future)
        let mut fresh = make_node(&[0.2, 0.0]);
        fresh.meta.expires_at = Some(now + 3600);
        let fresh_id = fresh.id;
        db.insert_node(fresh).unwrap();

        // Eternal node (no TTL)
        let eternal = make_node(&[0.3, 0.0]);
        let eternal_id = eternal.id;
        db.insert_node(eternal).unwrap();

        let reaped = db.reap_expired().unwrap();
        assert_eq!(reaped, 1);

        assert!(db.get_node(expired_id).unwrap().is_none());
        assert!(db.get_node(fresh_id).unwrap().is_some());
        assert!(db.get_node(eternal_id).unwrap().is_some());
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
