use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use tracing;
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{Edge, Node, NodeMeta, PoincareVector};
use crate::schema::SchemaValidator;
use crate::storage::GraphStorage;
use crate::transaction::Transaction;
use crate::wal::{GraphWal, GraphWalEntry};

// ─────────────────────────────────────────────
// Backpressure signal
// ─────────────────────────────────────────────

/// Signal returned by [`NietzscheDB::check_backpressure`] indicating whether
/// the client should slow down its write rate.
///
/// Included in `InsertNode` and `BatchInsertNodes` gRPC responses so that
/// EVA-Mind (or any SDK) can respect the database's consolidation needs.
#[derive(Debug, Clone)]
pub struct BackpressureSignal {
    /// Whether the write was accepted. Currently always `true` — the database
    /// never rejects writes, only suggests delays.
    pub accept: bool,
    /// Reason for backpressure. Empty string means no pressure.
    /// Possible values: `"energy_inflated"`, `"capacity_high"`, `"capacity_warning"`.
    pub reason: String,
    /// Suggested delay in milliseconds before the next write. `0` = no delay.
    pub suggested_delay_ms: u32,
}

impl BackpressureSignal {
    /// No backpressure — everything is healthy.
    pub fn ok() -> Self {
        Self { accept: true, reason: String::new(), suggested_delay_ms: 0 }
    }

    /// `true` if the signal recommends slowing down.
    pub fn is_pressured(&self) -> bool {
        self.suggested_delay_ms > 0
    }
}

// ─────────────────────────────────────────────
// MetadataFilter (pushed-down KNN filter)
// ─────────────────────────────────────────────

/// Filter for KNN metadata pre-filtering (pushed down to the HNSW index).
///
/// Converts to `hyperspace_core::FilterExpr` at the HNSW boundary.
/// Supports the filter patterns used by EVA-Mind's Qdrant calls
/// (e.g. `SearchWithScore(ctx, col, emb, k, 0.7, userID)`).
#[derive(Debug, Clone)]
pub enum MetadataFilter {
    /// Exact string match: `field = value`.
    Eq { field: String, value: String },
    /// Field value is one of the allowed values.
    In { field: String, values: Vec<String> },
    /// Numeric range: `gte <= field <= lte` (either bound may be absent).
    Range { field: String, gte: Option<f64>, lte: Option<f64> },
    /// All sub-filters must match (intersection).
    And(Vec<MetadataFilter>),
    /// At least one sub-filter must match (union).
    Or(Vec<MetadataFilter>),
    /// Invert the match of the sub-filter.
    Not(Box<MetadataFilter>),
    /// Field contains the given value (JSONB containment match).
    Contains { field: String, value: serde_json::Value },
    /// Field exists in the metadata.
    Exists { field: String },
    /// No filtering — return all K results.
    None,
}

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

    /// Upsert a vector embedding with additional metadata fields to index.
    ///
    /// The `meta` map is stored in the HNSW metadata index so that
    /// [`knn_filtered`](VectorStore::knn_filtered) can use it for
    /// pushed-down pre-filtering during search.
    fn upsert_with_meta(
        &mut self,
        id: Uuid,
        vector: &PoincareVector,
        _meta: HashMap<String, String>,
    ) -> Result<(), GraphError> {
        self.upsert(id, vector)
    }

    /// Remove the embedding for `id` (no-op if not found).
    fn delete(&mut self, id: Uuid) -> Result<(), GraphError>;

    /// Search for the `k` nearest neighbours of `query`.
    /// Returns a list of `(id, distance)` pairs, sorted ascending by distance.
    fn knn(
        &self,
        query: &PoincareVector,
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, GraphError>;

    /// Filtered KNN: search for the `k` nearest neighbours that match `filter`.
    ///
    /// The filter is pushed down to the HNSW index (RoaringBitmap pre-filter)
    /// for sub-linear candidate pruning. Falls back to [`knn`](VectorStore::knn)
    /// when `filter` is [`MetadataFilter::None`].
    fn knn_filtered(
        &self,
        query: &PoincareVector,
        k: usize,
        _filter: &MetadataFilter,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        self.knn(query, k)
    }
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
    storage:      Arc<GraphStorage>,
    wal:          GraphWal,
    adjacency:    AdjacencyIndex,
    vector_store: V,
    /// Hot-tier RAM cache: elite Übermensch node metadata pinned for O(1) access.
    ///
    /// ## Point 8 Audit Fix: LRU Eviction
    /// Changed from `DashMap` to `Mutex<LruCache>` to support fixed-size pinning
    /// with Least-Recently-Used (LRU) eviction strategy.
    pub hot_tier: Arc<Mutex<LruCache<Uuid, NodeMeta>>>,
    /// Metadata fields to maintain secondary indexes on (set via `NIETZSCHE_INDEXED_FIELDS`).
    /// When a node is inserted/deleted, index entries are written/removed for these fields.
    indexed_fields: HashSet<String>,
    /// Optional schema validator for per-NodeType constraints.
    schema_validator: Option<SchemaValidator>,

    /// Quantum purity cache: maps node_id → purity ∈ [0, 1].
    ///
    /// Computed from the Arousal→Purity mapping (poincare_to_bloch):
    /// - High energy/arousal → pure quantum state (purity ≈ 1.0, laser-like)
    /// - Low energy/arousal → mixed state (purity ≈ 0.0, center of Bloch sphere)
    ///
    /// Stored separately from NodeMeta to avoid bincode serialization breakage.
    /// Updated on insert_node and update_energy.
    pub quantum_purity_cache: Arc<Mutex<HashMap<Uuid, f64>>>,

    /// Configured vector dimension for this collection.
    /// Populated from the vector store at open time. Used by `vector_dim()`.
    dim: usize,
}

impl<V: VectorStore> NietzscheDB<V> {
    // ── Construction ───────────────────────────────────

    /// Open (or create) a NietzscheDB instance.
    ///
    /// - `data_dir`     — root directory; RocksDB lives at `data_dir/rocksdb/`
    ///                    and the WAL at `data_dir/graph.wal`.
    /// - `vector_store` — injected vector store implementation.
    /// - `dim`          — configured vector dimension for this collection.
    ///                    Stored on the struct so that `vector_dim()` is O(1).
    pub fn open(data_dir: &Path, vector_store: V, dim: usize) -> Result<Self, GraphError> {
        let rocksdb_path = data_dir.join("rocksdb");
        let rocksdb_str = rocksdb_path
            .to_str()
            .ok_or_else(|| GraphError::Storage("invalid RocksDB path".into()))?;

        let storage   = Arc::new(GraphStorage::open(rocksdb_str)?);

        // ── WAL crash recovery ──────────────────────────────────────────
        // Replay committed transactions that may not have been fully applied.
        // The WAL is opened with tail truncation, then replayed before normal
        // operation. This ensures crash consistency.
        let mut vector_store = vector_store;
        let replay_entries = GraphWal::replay(data_dir)?;
        if !replay_entries.is_empty() {
            let replayed = Self::replay_wal_entries(&storage, &mut vector_store, &replay_entries);
            if replayed > 0 {
                tracing::info!(
                    total_entries = replay_entries.len(),
                    replayed = replayed,
                    "WAL crash recovery: replayed committed operations"
                );
            }
        }

        let wal       = GraphWal::open(data_dir)?;

        // ── One-time edge migration (pre-Minkowski → current format) ──
        match storage.repair_legacy_edges() {
            Ok(0) => {} // nothing to repair
            Ok(n) => tracing::info!(repaired = n, "migrated legacy edges to current format"),
            Err(e) => tracing::warn!(error = %e, "edge migration failed (non-fatal, edges will be repaired on access)"),
        }

        let adjacency = storage.rebuild_adjacency()?;

        // Load persisted schema constraints if any exist
        let schema_validator = SchemaValidator::load_all(&storage).ok();

        let mut db = Self {
            storage, wal, adjacency, vector_store,
            hot_tier: Arc::new(Mutex::new(LruCache::new(std::num::NonZeroUsize::new(10000).unwrap()))),
            indexed_fields: HashSet::new(),
            schema_validator,
            quantum_purity_cache: Arc::new(Mutex::new(HashMap::new())),
            dim,
        };

        // Load persisted secondary indexes from CF_META
        let _ = db.load_persisted_indexes();

        Ok(db)
    }

    /// Replay WAL entries to RocksDB storage for crash recovery.
    ///
    /// Scans the WAL for `TxBegin`/`TxCommitted` pairs and re-applies all
    /// operations from committed transactions. Non-transactional entries
    /// (outside a tx) are also re-applied since they were already durably
    /// flushed to the WAL.
    ///
    /// Returns the number of operations replayed.
    fn replay_wal_entries(
        storage: &Arc<GraphStorage>,
        vector_store: &mut V,
        entries: &[GraphWalEntry],
    ) -> usize {
        let mut replayed = 0;
        let mut in_tx: Option<Uuid> = None;
        let mut tx_ops: Vec<&GraphWalEntry> = Vec::new();
        let mut committed_txs: HashSet<Uuid> = HashSet::new();

        // First pass: identify committed transactions
        for entry in entries {
            if let GraphWalEntry::TxCommitted(tx_id) = entry {
                committed_txs.insert(*tx_id);
            }
        }

        // Second pass: replay operations
        for entry in entries {
            match entry {
                GraphWalEntry::TxBegin(tx_id) => {
                    in_tx = Some(*tx_id);
                    tx_ops.clear();
                }
                GraphWalEntry::TxCommitted(tx_id) => {
                    if in_tx == Some(*tx_id) {
                        // Replay all buffered ops for this committed transaction
                        for op in &tx_ops {
                            if Self::replay_single_entry(storage, vector_store, op) {
                                replayed += 1;
                            }
                        }
                    }
                    in_tx = None;
                    tx_ops.clear();
                }
                GraphWalEntry::TxRolledBack(tx_id) => {
                    if in_tx == Some(*tx_id) {
                        // Discard buffered ops — transaction was rolled back
                        in_tx = None;
                        tx_ops.clear();
                    }
                }
                _ => {
                    if in_tx.is_some() {
                        // Buffer ops inside a transaction
                        tx_ops.push(entry);
                    } else {
                        // Non-transactional entry — replay immediately
                        // (these were flushed with fsync before the WAL returned)
                        if Self::replay_single_entry(storage, vector_store, entry) {
                            replayed += 1;
                        }
                    }
                }
            }
        }

        replayed
    }

    /// Replay a single WAL entry to RocksDB storage.
    /// Returns true if the entry was applied, false on error (logged and skipped).
    fn replay_single_entry(
        storage: &Arc<GraphStorage>,
        vector_store: &mut V,
        entry: &GraphWalEntry,
    ) -> bool {
        let result = match entry {
            GraphWalEntry::InsertNode(node) => {
                // Point 9: InsertNode in WAL replay MUST also populate vector store
                // if it's a synchronous recovery path.
                let _ = vector_store.upsert(node.id, &node.embedding);
                storage.put_node(node)
            }
            GraphWalEntry::InsertEdge(edge) => storage.put_edge(edge),
            GraphWalEntry::DeleteNode(id) => {
                // Point 9 Audit Fix: Propagate deletion to vector store during recovery
                let _ = vector_store.delete(*id);
                storage.delete_node(id)
            }
            GraphWalEntry::DeleteEdge(id) => {
                // Point 9: Implement direct deletion without existence check during recovery
                // We use a dummy edge with given ID to trigger deletion in RocksDB
                let mut dummy = Edge::association(*id, *id, 0.0);
                dummy.id = *id;
                storage.delete_edge(&dummy)
            }
            GraphWalEntry::PruneNode(id) => {
                match storage.get_node_meta(id) {
                    Ok(Some(mut meta)) => {
                        let old_energy = meta.energy;
                        meta.energy = 0.0;
                        storage.put_node_meta_update_energy(&meta, old_energy)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            GraphWalEntry::UpdateNodeEnergy { node_id, energy } => {
                match storage.get_node_meta(node_id) {
                    Ok(Some(mut meta)) => {
                        let old_energy = meta.energy;
                        meta.energy = *energy;
                        storage.put_node_meta_update_energy(&meta, old_energy)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            GraphWalEntry::UpdateHausdorff { node_id, hausdorff } => {
                match storage.get_node_meta(node_id) {
                    Ok(Some(mut meta)) => {
                        meta.hausdorff_local = *hausdorff;
                        storage.put_node_meta(&meta)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            GraphWalEntry::UpdateEmbedding { node_id: _, embedding: _ } => {
                // Embedding updates are applied to the vector store, not RocksDB.
                // The vector store rebuilds from its own WAL/snapshot on startup,
                // so we skip this entry during graph WAL replay.
                Ok(())
            }
            GraphWalEntry::PhantomizeNode(id) => {
                match storage.get_node_meta(id) {
                    Ok(Some(mut meta)) => {
                        meta.is_phantom = true;
                        storage.put_node_meta(&meta)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            GraphWalEntry::ReanimateNode { node_id, energy } => {
                match storage.get_node_meta(node_id) {
                    Ok(Some(mut meta)) => {
                        meta.is_phantom = false;
                        let old_energy = meta.energy;
                        meta.energy = *energy;
                        storage.put_node_meta_update_energy(&meta, old_energy)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            GraphWalEntry::UpdateEdgeMeta(edge) => {
                storage.update_edge_only(edge)
            }
            GraphWalEntry::UpdateNodeContent { node_id, content } => {
                match storage.get_node_meta(node_id) {
                    Ok(Some(mut meta)) => {
                        meta.content = content.clone();
                        storage.put_node_meta(&meta)
                    }
                    Ok(None) => Ok(()),
                    Err(e) => Err(e),
                }
            }
            // Transaction markers are handled by the caller
            GraphWalEntry::TxBegin(_) | GraphWalEntry::TxCommitted(_) | GraphWalEntry::TxRolledBack(_) => {
                return false;
            }
        };

        match result {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(entry = %entry.tag(), error = %e, "WAL replay: failed to apply entry, skipping");
                false
            }
        }
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

    // ── Secondary index management (Phase E) ──────────

    /// Create a persistent secondary index on a metadata field.
    ///
    /// The field is added to `indexed_fields` and persisted to RocksDB
    /// (key `"index:{field}"` in CF_META). If nodes already exist, their
    /// metadata is backfilled into `CF_META_IDX`.
    pub fn create_index(&mut self, field: &str) -> Result<(), GraphError> {
        if self.indexed_fields.contains(field) {
            return Ok(()); // already indexed
        }

        // 1. Persist to registry
        self.storage.put_meta(&format!("index:{field}"), &[])?;

        // 2. Add to in-memory set
        self.indexed_fields.insert(field.to_string());

        // 3. Backfill: scan all nodes and write index entries for this field
        let field_set: HashSet<String> = [field.to_string()].into_iter().collect();
        for result in self.storage.iter_nodes_meta() {
            if let Ok(meta) = result {
                if meta.metadata.contains_key(field) {
                    self.storage.put_meta_index(&meta.id, &meta.metadata, &field_set)?;
                }
            }
        }

        Ok(())
    }

    /// Drop a secondary index on a metadata field.
    ///
    /// Removes the field from `indexed_fields`, deletes the registry entry,
    /// and purges all index entries for this field from `CF_META_IDX`.
    pub fn drop_index(&mut self, field: &str) -> Result<(), GraphError> {
        if !self.indexed_fields.remove(field) {
            return Ok(()); // wasn't indexed
        }

        // 1. Remove from registry
        self.storage.delete_meta(&format!("index:{field}"))?;

        // 2. Purge index entries: scan all nodes and delete their index entries
        let field_set: HashSet<String> = [field.to_string()].into_iter().collect();
        for result in self.storage.iter_nodes_meta() {
            if let Ok(meta) = result {
                if meta.metadata.contains_key(field) {
                    self.storage.delete_meta_index(&meta.id, &meta.metadata, &field_set)?;
                }
            }
        }

        Ok(())
    }

    /// List all active secondary indexes.
    pub fn list_indexes(&self) -> Vec<String> {
        self.indexed_fields.iter().cloned().collect()
    }

    /// Check if a field has a secondary index.
    pub fn has_index(&self, field: &str) -> bool {
        self.indexed_fields.contains(field)
    }

    /// Load persisted indexes from CF_META on startup.
    ///
    /// Scans for keys with prefix `"index:"` and populates `indexed_fields`.
    fn load_persisted_indexes(&mut self) -> Result<(), GraphError> {
        let entries = self.storage.scan_meta_prefix(b"index:")?;
        for (key_bytes, _) in entries {
            if let Ok(key_str) = String::from_utf8(key_bytes) {
                if let Some(field) = key_str.strip_prefix("index:") {
                    self.indexed_fields.insert(field.to_string());
                }
            }
        }
        Ok(())
    }

    /// Scan the metadata index for a field with exact match.
    ///
    /// Returns node IDs where `metadata[field] == value`.
    /// O(log N + k) via `CF_META_IDX`.
    pub fn index_scan_eq(
        &self,
        field: &str,
        value: &serde_json::Value,
    ) -> Result<Vec<Uuid>, GraphError> {
        self.storage.scan_meta_index_eq(field, value)
    }

    /// Scan the metadata index for a field with range.
    ///
    /// Returns node IDs where `min <= metadata[field] <= max`.
    /// O(log N + k) via `CF_META_IDX`.
    pub fn index_scan_range(
        &self,
        field: &str,
        min_val: &serde_json::Value,
        max_val: &serde_json::Value,
    ) -> Result<Vec<Uuid>, GraphError> {
        self.storage.scan_meta_index_range(field, min_val, max_val)
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
        if let Some(meta) = self.hot_tier.lock().get(&id) {
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

        // 2 & 3. RocksDB + Metadata Index (Atomic)
        self.storage.put_node_atomic(&node, &self.indexed_fields)?;

        // 4. Vector store — pass indexed metadata for pushed-down KNN filtering
        if self.indexed_fields.is_empty() {
            self.vector_store.upsert(node.id, &node.embedding)?;
        } else {
            let hnsw_meta = Self::extract_hnsw_meta(&node, &self.indexed_fields);
            self.vector_store.upsert_with_meta(node.id, &node.embedding, hnsw_meta)?;
        }

        // 5. Immediate hot-tier promotion (Industrial Consistency - Point 3)
        if node.energy >= 0.85 {
            self.promote_to_hot_tier(&node);
        }

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

        // 2 & 3. RocksDB + Metadata Index (Atomic - Point 2)
        self.storage.put_nodes_batch_atomic(&nodes, &self.indexed_fields)?;

        // 4. Vector store — pass indexed metadata for pushed-down KNN filtering
        let has_indexed = !self.indexed_fields.is_empty();
        for node in &nodes {
            if has_indexed {
                let hnsw_meta = Self::extract_hnsw_meta(node, &self.indexed_fields);
                self.vector_store.upsert_with_meta(node.id, &node.embedding, hnsw_meta)?;
            } else {
                self.vector_store.upsert(node.id, &node.embedding)?;
            }
        }

        // 5. Immediate promotion for large batches
        for node in &nodes {
            if node.energy >= 0.85 {
                self.promote_to_hot_tier(node);
            }
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
        if let Some(meta) = self.hot_tier.lock().get(&id) {
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
        self.hot_tier.lock().pop(&id);
        self.adjacency.remove_node(&id);

        Ok(())
    }

    /// Phantomize a node: structural scar that preserves topology.
    ///
    /// Unlike [`prune_node`], this method **keeps adjacency connections intact**
    /// so the hyperbolic geometry doesn't collapse. The node is removed from
    /// the vector store (no KNN results) and from the hot-tier cache, but its
    /// edges and position in the Poincaré ball are preserved.
    ///
    /// Phantom nodes can be reanimated via [`reanimate_node`] if fresh data
    /// arrives at a similar position, mimicking biological memory re-learning.
    pub fn phantomize_node(&mut self, id: Uuid) -> Result<(), GraphError> {
        // 1. WAL
        self.wal.append(&GraphWalEntry::PhantomizeNode(id))?;

        // 2. Set energy = 0, is_phantom = true
        if let Some(mut meta) = self.storage.get_node_meta(&id)? {
            let old_energy = meta.energy;
            meta.energy = 0.0;
            meta.is_phantom = true;
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
        }

        // 3. Remove from vector store (no KNN results)
        self.vector_store.delete(id)?;

        // 4. Remove from hot-tier cache
        self.hot_tier.lock().pop(&id);

        // 5. KEEP adjacency intact — topology is preserved!
        Ok(())
    }

    /// Reanimate a phantom node, restoring it to active state.
    ///
    /// Re-inserts the embedding into the vector store and sets the new energy.
    /// Returns `true` if the node was phantom and was reanimated, `false` if
    /// the node doesn't exist or wasn't phantom.
    pub fn reanimate_node(&mut self, id: Uuid, energy: f32) -> Result<bool, GraphError> {
        let meta = match self.storage.get_node_meta(&id)? {
            Some(m) if m.is_phantom => m,
            _ => return Ok(false),
        };

        // 1. WAL
        self.wal.append(&GraphWalEntry::ReanimateNode { node_id: id, energy })?;

        // 2. Update meta: clear phantom flag, restore energy
        let old_energy = meta.energy;
        let mut meta = meta;
        meta.energy = energy;
        meta.is_phantom = false;
        self.storage.put_node_meta_update_energy(&meta, old_energy)?;

        // 3. Re-insert embedding into vector store
        if let Some(emb) = self.storage.get_embedding(&id)? {
            self.vector_store.upsert(id, &emb)?;
        }

        Ok(true)
    }

    /// Hard-delete a node and all edges referencing it.
    ///
    /// Prefer [`phantomize_node`] for normal operation; this is for
    /// administrative data-correction workflows.
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

        // 7. Quantum purity cache
        self.quantum_purity_cache.lock().remove(&id);

        Ok(())
    }

    /// Reap all expired nodes (TTL enforcement).
    ///
    /// Scans `CF_NODES` for nodes where `expires_at <= now`, then
    /// **phantomizes** each one — preserving topological connections so the
    /// hyperbolic geometry doesn't collapse (anti-esquecimento).
    ///
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
            self.phantomize_node(id)?;
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
            self.hot_tier.lock().pop(&node_id);
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
            self.hot_tier.lock().pop(&node_id);
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
            self.hot_tier.lock().pop(&node_id);
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

    // ── Edge metadata updates (FASE D) ──────────────────────────

    /// Update an edge's metadata fields (merge new keys into existing).
    ///
    /// Used by MERGE ON MATCH SET for edges. Adjacency lists are unchanged.
    pub fn update_edge_metadata(
        &mut self,
        edge_id: Uuid,
        updates: &serde_json::Value,
    ) -> Result<(), GraphError> {
        let mut edge = self.storage.get_edge(&edge_id)?
            .ok_or_else(|| GraphError::EdgeNotFound(edge_id))?;

        if let Some(obj) = updates.as_object() {
            for (k, v) in obj {
                edge.metadata.insert(k.clone(), v.clone());
            }
        }

        self.wal.append(&GraphWalEntry::UpdateEdgeMeta(edge.clone()))?;
        self.storage.update_edge_only(&edge)?;
        Ok(())
    }

    /// Atomically increment a numeric field in an edge's metadata.
    ///
    /// If the field doesn't exist, it's created with `delta` as its value.
    /// Returns the new value after increment.
    ///
    /// Use case: `MERGE (p)-[r:MENTIONED]->(t) ON MATCH SET r.count = r.count + 1`
    pub fn increment_edge_metadata(
        &mut self,
        edge_id: Uuid,
        field: &str,
        delta: f64,
    ) -> Result<f64, GraphError> {
        let mut edge = self.storage.get_edge(&edge_id)?
            .ok_or_else(|| GraphError::EdgeNotFound(edge_id))?;

        let current = edge.metadata
            .get(field)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let new_val = current + delta;
        edge.metadata.insert(field.to_string(), serde_json::json!(new_val));

        self.wal.append(&GraphWalEntry::UpdateEdgeMeta(edge.clone()))?;
        self.storage.update_edge_only(&edge)?;
        Ok(new_val)
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

            // WAL
            self.wal.append(&GraphWalEntry::UpdateNodeContent {
                node_id: id,
                content: meta.content.clone(),
            })?;

            // Re-persist meta (energy unchanged)
            let old_energy = meta.energy;
            self.hot_tier.lock().pop(&id);
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
    /// Get a shared Arc reference to the GraphStorage for Swartz SQL engine.
    pub fn storage_arc(&self) -> Arc<GraphStorage> { Arc::clone(&self.storage) }
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
        self.hot_tier.lock().put(node.id, node.meta.clone());
    }

    /// Promote `NodeMeta` directly (avoids needing the full Node).
    pub fn promote_meta_to_hot_tier(&self, meta: NodeMeta) {
        let id = meta.id;
        self.hot_tier.lock().put(id, meta);
    }

    /// Evict a node from the hot-tier RAM cache.
    pub fn evict_from_hot_tier(&self, id: &Uuid) {
        self.hot_tier.lock().pop(id);
    }

    /// Replace the entire hot-tier with metadata from a set of elite nodes.
    /// Called after every Zaratustra Übermensch phase.
    pub fn replace_hot_tier(&self, elite_nodes: &[Node]) {
        let mut guard = self.hot_tier.lock();
        guard.clear();
        for node in elite_nodes {
            guard.put(node.id, node.meta.clone());
        }
    }

    /// Replace the entire hot-tier with pre-extracted metadata.
    pub fn replace_hot_tier_meta(&self, elite_metas: Vec<NodeMeta>) {
        let mut guard = self.hot_tier.lock();
        guard.clear();
        for meta in elite_metas {
            guard.put(meta.id, meta);
        }
    }

    /// Current hot-tier node count.
    pub fn hot_tier_len(&self) -> usize {
        self.hot_tier.lock().len()
    }

    // ── Quantum Purity Cache ──────────────────────────────────────────────

    /// Store quantum purity for a node (Arousal→Purity mapping).
    ///
    /// Called after `insert_node` or `update_energy` by the API layer,
    /// which computes purity via `poincare_to_bloch(embedding, energy).purity`.
    pub fn set_quantum_purity(&self, node_id: Uuid, purity: f64) {
        self.quantum_purity_cache.lock().insert(node_id, purity);
    }

    /// Retrieve the cached quantum purity for a node.
    ///
    /// Returns `None` if the node has not had its purity computed yet
    /// (e.g. legacy nodes inserted before this feature).
    pub fn get_quantum_purity(&self, node_id: &Uuid) -> Option<f64> {
        self.quantum_purity_cache.lock().get(node_id).copied()
    }

    /// Remove quantum purity entry when a node is deleted.
    pub fn evict_quantum_purity(&self, node_id: &Uuid) {
        self.quantum_purity_cache.lock().remove(node_id);
    }

    /// Batch-set quantum purity for multiple nodes.
    pub fn set_quantum_purity_batch(&self, entries: &[(Uuid, f64)]) {
        let mut guard = self.quantum_purity_cache.lock();
        for &(id, purity) in entries {
            guard.insert(id, purity);
        }
    }

    /// Current quantum purity cache entry count.
    pub fn quantum_purity_cache_len(&self) -> usize {
        self.quantum_purity_cache.lock().len()
    }

    /// Configured vector dimension for this collection.
    pub fn vector_dim(&self) -> usize {
        self.dim
    }

    /// Set the vector dimension (called after opening).
    pub fn set_vector_dim(&mut self, dim: usize) {
        self.dim = dim;
    }

    /// k-nearest-neighbour search in the vector store.
    pub fn knn(
        &self,
        query: &PoincareVector,
        k: usize,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        self.vector_store.knn(query, k)
    }

    /// Filtered KNN: search for the `k` nearest neighbours that match `filter`.
    ///
    /// The filter is pushed down to the HNSW RoaringBitmap index for sub-linear
    /// candidate pruning. Equivalent to Qdrant's `SearchWithScore(..., filter)`.
    pub fn knn_filtered(
        &self,
        query: &PoincareVector,
        k: usize,
        filter: &MetadataFilter,
    ) -> Result<Vec<(Uuid, f64)>, GraphError> {
        self.vector_store.knn_filtered(query, k, filter)
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

    // ── HNSW metadata helpers ──────────────────────────

    /// Extract a flat `HashMap<String, String>` from a node's metadata for
    /// indexing in the HNSW forward/inverted index. Only extracts fields
    /// listed in `indexed_fields`. Also adds `node_type` and `energy`
    /// (as string) for common filter patterns.
    fn extract_hnsw_meta(node: &Node, indexed_fields: &HashSet<String>) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        // Always include node_type and energy — universally useful for filtering.
        meta.insert("node_type".to_string(), format!("{:?}", node.meta.node_type));
        meta.insert("energy".to_string(), format!("{}", node.meta.energy));

        // Extract declared indexed fields from the node metadata HashMap.
        for field in indexed_fields {
            if let Some(val) = node.meta.metadata.get(field) {
                let s = match val {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                meta.insert(field.clone(), s);
            }
        }
        meta
    }

    // ── Stats ──────────────────────────────────────────

    pub fn node_count(&self) -> Result<usize, GraphError> { self.storage.node_count() }
    pub fn edge_count(&self) -> Result<usize, GraphError> { self.storage.edge_count() }

    // ── Backpressure ────────────────────────────────────

    /// Check whether the database is under ingestion pressure.
    ///
    /// Returns a [`BackpressureSignal`] that write-path RPCs (InsertNode,
    /// BatchInsertNodes) should include in their response. The signal tells
    /// the client how long to wait before the next write.
    ///
    /// ## Checks performed (all O(1) or O(limit)):
    ///
    /// 1. **Capacity** — node count via `rocksdb.estimate-num-keys` (O(1)).
    /// 2. **Energy inflation** — count of nodes with energy > 0.85 via the
    ///    energy secondary index, capped at 100 iterations (O(100)).
    pub fn check_backpressure(&self) -> BackpressureSignal {
        let total = self.node_count().unwrap_or(0);

        // ── capacity check (O(1)) ─────────────────────────
        if total > 100_000 {
            return BackpressureSignal {
                accept: true,
                reason: "capacity_high".into(),
                suggested_delay_ms: 1000,
            };
        }
        if total > 50_000 {
            return BackpressureSignal {
                accept: true,
                reason: "capacity_warning".into(),
                suggested_delay_ms: 200,
            };
        }

        // ── energy inflation check (O(sample_limit)) ─────
        // If more than half the (sampled) nodes have energy > 0.85, the
        // graph is "overheated" and needs time to consolidate via
        // Zaratustra or Sleep before ingesting more data.
        //
        // NOTE: we use the energy index to count nodes rather than
        // `node_count()` (RocksDB estimate-num-keys), because the
        // estimate can be 0 for small/fresh databases whose memtable
        // hasn't flushed to SST files yet.
        let sample_limit = 100usize;
        let total_sampled = self.storage
            .count_energy_above(0.0, sample_limit)
            .unwrap_or(0);
        if total_sampled >= 10 {
            let inflated = self.storage
                .count_energy_above(0.85, sample_limit)
                .unwrap_or(0);
            if inflated * 2 > total_sampled {
                return BackpressureSignal {
                    accept: true,
                    reason: "energy_inflated".into(),
                    suggested_delay_ms: 500,
                };
            }
        }

        BackpressureSignal::ok()
    }

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
    pub(crate) fn apply_insert_edge(&mut self, edge: &Edge) -> Result<(), GraphError> {
        self.storage.put_edge(edge)?;
        self.adjacency.add_edge(edge);
        Ok(())
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

    /// Update a node's energy directly in storage and hot-tier.
    pub(crate) fn apply_update_energy(&mut self, id: Uuid, energy: f32) -> Result<(), GraphError> {
        if let Some(mut meta) = self.storage.get_node_meta(&id)? {
            let old_energy = meta.energy;
            meta.energy = energy;
            self.storage.put_node_meta_update_energy(&meta, old_energy)?;
            
            // Sync hot-tier RAM cache
            let mut guard = self.hot_tier.lock();
            if let Some(entry) = guard.get_mut(&id) {
                entry.energy = energy;
            } else if energy >= 0.85 {
                // Point 3: Auto-promote to hot-tier if energy becomes high
                guard.put(id, meta);
            }
        }
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

    /// Force-sync the vector store against the RocksDB node table ("DeepRecovery").
    ///
    /// Point 5 Audit Fix: Scans all nodes in `CF_NODES` and ensures the
    /// vector store has exactly those IDs with their current embeddings.
    /// This resolves drift after failed SleepCycle rollbacks.
    pub fn force_sync_vector_store(&mut self) -> Result<usize, GraphError> {
        tracing::info!("Starting DeepRecovery: force-syncing vector store against RocksDB...");
        let mut count = 0;
        
        let nodes = self.storage.scan_nodes()?;
        for node in nodes {
            self.vector_store.upsert(node.id, &node.embedding)?;
            count += 1;
        }
        
        tracing::info!(count, "DeepRecovery complete: vector store synced.");
        Ok(count)
    }

    /// Prune nodes from the hot-tier cache whose energy is below `threshold`.
    ///
    /// Point 8 Audit Fix: Energy-based LRU eviction to prevent memory growth.
    pub fn prune_low_energy_nodes(&self, threshold: f32) {
        let mut guard = self.hot_tier.lock();
        let mut to_remove = Vec::new();
        for (id, meta) in guard.iter() {
            if meta.energy < threshold {
                to_remove.push(*id);
            }
        }
        for id in to_remove {
            guard.pop(&id);
        }
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

        // Expired node is phantomized (not hard-deleted) — it still exists
        // but with energy=0 and is_phantom=true.
        let expired_meta = db.storage.get_node_meta(&expired_id).unwrap().unwrap();
        assert!(expired_meta.is_phantom, "reaped node should be phantom");
        assert_eq!(expired_meta.energy, 0.0);

        // Fresh and eternal nodes are unaffected
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

    // ── Backpressure tests ─────────────────────────────

    #[test]
    fn backpressure_ok_on_small_graph() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Insert a few healthy nodes
        for i in 0..5 {
            let mut node = make_node(&[0.1 * (i as f64 + 1.0), 0.0]);
            node.energy = 0.5;
            db.insert_node(node).unwrap();
        }

        let bp = db.check_backpressure();
        assert!(bp.accept);
        assert!(!bp.is_pressured(), "small healthy graph should not trigger backpressure");
    }

    #[test]
    fn backpressure_detects_energy_inflation() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Insert 20 nodes, all with very high energy (> 0.85)
        for i in 0..20 {
            let mut node = make_node(&[0.04 * (i as f64 + 1.0), 0.0]);
            node.energy = 0.95;
            db.insert_node(node).unwrap();
        }

        let bp = db.check_backpressure();
        assert!(bp.accept, "writes should still be accepted");
        assert!(bp.is_pressured(), "inflated energy should trigger backpressure");
        assert_eq!(bp.reason, "energy_inflated");
        assert!(bp.suggested_delay_ms > 0);
    }

    // ── Phantom Nodes ────────────────────────────────────────────────

    #[test]
    fn phantomize_preserves_topology() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Build A → B → C chain
        let a = make_node(&[0.1, 0.0]);
        let b = make_node(&[0.2, 0.0]);
        let c = make_node(&[0.3, 0.0]);
        let (aid, bid, cid) = (a.id, b.id, c.id);
        db.insert_node(a).unwrap();
        db.insert_node(b).unwrap();
        db.insert_node(c).unwrap();
        db.insert_edge(Edge::association(aid, bid, 1.0)).unwrap();
        db.insert_edge(Edge::association(bid, cid, 1.0)).unwrap();

        // Phantomize B
        db.phantomize_node(bid).unwrap();

        // B's meta should be phantom with energy 0
        let meta = db.storage.get_node_meta(&bid).unwrap().unwrap();
        assert!(meta.is_phantom, "node should be phantom");
        assert_eq!(meta.energy, 0.0);

        // Topology preserved: A and C still connected through B
        let out_a = db.adjacency.neighbors_out(&aid);
        assert!(out_a.contains(&bid),
            "A→B edge should survive phantomization");
        let out_b = db.adjacency.neighbors_out(&bid);
        assert!(out_b.contains(&cid),
            "B→C edge should survive phantomization");

        // Embedding still in RocksDB (geometry preserved)
        assert!(db.storage.get_embedding(&bid).unwrap().is_some(),
            "embedding should survive in RocksDB");
    }

    #[test]
    fn reanimate_restores_phantom_node() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let node = make_node(&[0.1, 0.0]);
        let id = node.id;
        db.insert_node(node).unwrap();

        // Phantomize
        db.phantomize_node(id).unwrap();
        let meta = db.storage.get_node_meta(&id).unwrap().unwrap();
        assert!(meta.is_phantom);

        // Reanimate with energy 0.8
        let ok = db.reanimate_node(id, 0.8).unwrap();
        assert!(ok, "should return true for phantom node");

        // Verify restored state
        let meta = db.storage.get_node_meta(&id).unwrap().unwrap();
        assert!(!meta.is_phantom, "phantom flag should be cleared");
        assert!((meta.energy - 0.8).abs() < 1e-6);
    }

    #[test]
    fn reanimate_returns_false_for_non_phantom() {
        let dir = tmp();
        let mut db = open_db(&dir);

        let node = make_node(&[0.1, 0.0]);
        let id = node.id;
        db.insert_node(node).unwrap();

        // Try to reanimate a non-phantom node
        let ok = db.reanimate_node(id, 0.5).unwrap();
        assert!(!ok, "should return false for active node");
    }

    #[test]
    fn reap_expired_phantomizes_instead_of_deleting() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Insert node with TTL already expired
        let mut node = make_node(&[0.1, 0.0]);
        let id = node.id;
        node.meta.expires_at = Some(1); // Unix epoch + 1 second = long expired
        db.insert_node(node).unwrap();

        // Build an edge so we can verify topology
        let other = make_node(&[0.2, 0.0]);
        let oid = other.id;
        db.insert_node(other).unwrap();
        db.insert_edge(Edge::association(id, oid, 1.0)).unwrap();

        // Reap expired
        let count = db.reap_expired().unwrap();
        assert_eq!(count, 1);

        // Node should be phantom, NOT hard-deleted
        let meta = db.storage.get_node_meta(&id).unwrap();
        assert!(meta.is_some(), "node should still exist as phantom");
        let meta = meta.unwrap();
        assert!(meta.is_phantom);
        assert_eq!(meta.energy, 0.0);

        // Topology preserved
        let out = db.adjacency.neighbors_out(&id);
        assert!(!out.is_empty(), "edges should survive reap");
    }
}
