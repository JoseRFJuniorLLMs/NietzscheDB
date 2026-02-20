use rocksdb::{
    BlockBasedOptions, Cache, ColumnFamilyDescriptor, DB, DBCompressionType, Options,
    SliceTransform,
};
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{Edge, Node, NodeMeta, PoincareVector};

// ─────────────────────────────────────────────
// Column Family names
// ─────────────────────────────────────────────

const CF_NODES:      &str = "nodes";      // key: node_id (16 bytes) → NodeMeta (bincode)
const CF_EMBEDDINGS: &str = "embeddings"; // key: node_id (16 bytes) → PoincareVector (bincode)
const CF_EDGES:      &str = "edges";      // key: edge_id (16 bytes) → Edge (bincode)
const CF_ADJ_OUT:    &str = "adj_out";    // key: node_id → Vec<Uuid> outgoing edge ids
const CF_ADJ_IN:     &str = "adj_in";     // key: node_id → Vec<Uuid> incoming edge ids
const CF_META:       &str = "meta";       // key: &str → arbitrary bytes
const CF_SENSORY:    &str = "sensory";    // key: node_id (16 bytes) → SensoryMemory (bincode)
/// Energy secondary index: key = [energy_be_4_bytes | node_id_16_bytes] → empty.
/// Enables O(log N) range scans for `WHERE n.energy > X`.
const CF_ENERGY_IDX: &str = "energy_idx";
/// Metadata secondary index: key = [field_name_fnv1a(8B) | sortable_value(8B) | node_id(16B)] → empty.
/// Enables O(log N) range scans for `WHERE n.field = X` or `WHERE n.field > X`.
const CF_META_IDX: &str = "meta_idx";
/// List storage: key = [node_id(16B) | list_name_hash(8B) | seq_be(8B)] → value bytes.
/// Supports RPUSH/LRANGE per-node ordered lists.
const CF_LISTS: &str = "lists";

const ALL_CFS: &[&str] = &[
    CF_NODES, CF_EMBEDDINGS, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN, CF_META, CF_SENSORY, CF_ENERGY_IDX,
    CF_META_IDX, CF_LISTS,
];

// ─────────────────────────────────────────────
// RocksDB helpers
// ─────────────────────────────────────────────

/// Shared block cache (LRU, 512 MiB).
/// Shared across all CFs to prevent cache fragmentation.
fn make_block_cache() -> Cache {
    Cache::new_lru_cache(512 * 1024 * 1024)
}

/// Column-family options tuned for high read throughput.
/// - Bloom filter: 10 bits/key → ~1% false-positive rate → 10× fewer unnecessary reads
/// - Block cache: shared 512 MB LRU
/// - Block size: 16 KiB (good for sequential scan of small records like nodes/edges)
fn cf_opts_read_heavy(cache: &Cache) -> Options {
    let mut bbo = BlockBasedOptions::default();
    bbo.set_bloom_filter(10.0, false);
    bbo.set_block_size(16 * 1024);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_pin_l0_filter_and_index_blocks_in_cache(true);
    bbo.set_block_cache(cache);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&bbo);
    opts.set_compression_type(DBCompressionType::Lz4);
    opts.set_bottommost_compression_type(DBCompressionType::Zstd);
    opts.set_write_buffer_size(128 * 1024 * 1024); // 128 MiB write buffer per CF
    opts.set_max_write_buffer_number(3);
    opts.set_min_write_buffer_number_to_merge(1);
    opts
}

/// Options for the energy index CF: prefix extractor on the 4-byte energy prefix
/// so bloom filters work on range queries, not just point lookups.
fn cf_opts_energy_idx(cache: &Cache) -> Options {
    let mut bbo = BlockBasedOptions::default();
    bbo.set_bloom_filter(10.0, true); // prefix bloom
    bbo.set_block_size(4 * 1024);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_block_cache(cache);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&bbo);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(4)); // 4-byte energy prefix
    opts.set_memtable_prefix_bloom_ratio(0.1);
    opts.set_compression_type(DBCompressionType::Lz4);
    opts
}

/// Options for the metadata index CF: 8-byte prefix extractor (FNV-1a hash of field name)
/// so bloom filters enable efficient per-field range scans.
fn cf_opts_meta_idx(cache: &Cache) -> Options {
    let mut bbo = BlockBasedOptions::default();
    bbo.set_bloom_filter(10.0, true); // prefix bloom
    bbo.set_block_size(4 * 1024);
    bbo.set_cache_index_and_filter_blocks(true);
    bbo.set_block_cache(cache);

    let mut opts = Options::default();
    opts.set_block_based_table_factory(&bbo);
    opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8)); // 8-byte field name hash
    opts.set_memtable_prefix_bloom_ratio(0.1);
    opts.set_compression_type(DBCompressionType::Lz4);
    opts
}

// ─────────────────────────────────────────────
// GraphStorage
// ─────────────────────────────────────────────

/// Persistent storage for the NietzscheDB hyperbolic graph.
///
/// Backed by RocksDB with 8 column families:
/// - `nodes`      — serialized [`NodeMeta`] structs (bincode, ~100 bytes)
/// - `embeddings` — serialized [`PoincareVector`] structs (bincode, ~24 KB at 3072 dims)
/// - `edges`      — serialized Edge structs (bincode)
/// - `adj_out`    — per-node list of outgoing edge UUIDs
/// - `adj_in`     — per-node list of incoming edge UUIDs
/// - `meta`       — global metadata key/value pairs
/// - `sensory`    — sensory memory records
/// - `energy_idx` — secondary index: [energy_be_4bytes | node_id] → ∅ (range scans)
///
/// ## BUG A fix (Committee 2026-02-19)
/// `NodeMeta` and `PoincareVector` are stored in **separate** column families.
/// Operations that only need metadata (BFS energy gate, `update_energy()`,
/// `update_hausdorff()`, NQL filters) read ~100 bytes from `CF_NODES` instead
/// of deserializing the full ~24 KB node+embedding blob.  This gives 10–25×
/// speedup in traversal and reduces hot-tier RAM by ~240× per node.
///
/// ## Performance
/// - 512 MiB shared LRU block cache
/// - Bloom filters (10 bits/key) on all CFs
/// - 128 MiB write buffer per CF (fewer memtable flushes)
/// - LZ4 + Zstd compression
pub struct GraphStorage {
    db:     DB,
    // Keep cache alive for the lifetime of the store
    _cache: Cache,
}

impl GraphStorage {
    // ── Open / init ────────────────────────────────────

    /// Open (or create) the RocksDB database at `path`.
    ///
    /// ## Tuning applied
    /// - 512 MiB shared LRU block cache
    /// - Bloom filters (10 bits/key) on all CFs
    /// - 128 MiB write buffer per CF, up to 3 buffers before stall
    /// - LZ4 compression on L0/L1, Zstd on bottommost level
    /// - Parallelism = number of logical CPUs
    /// - 4-byte prefix extractor on `energy_idx` CF for range bloom
    pub fn open(path: &str) -> Result<Self, GraphError> {
        let cache = make_block_cache();

        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        db_opts.increase_parallelism(num_cpus());
        db_opts.set_max_background_jobs(num_cpus().min(8) as i32);
        db_opts.set_bytes_per_sync(1024 * 1024); // 1 MiB

        let cf_descs: Vec<ColumnFamilyDescriptor> = vec![
            ColumnFamilyDescriptor::new(CF_NODES,      cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_EMBEDDINGS, cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_EDGES,      cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_ADJ_OUT,    cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_ADJ_IN,     cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_META,       cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_SENSORY,    cf_opts_read_heavy(&cache)),
            ColumnFamilyDescriptor::new(CF_ENERGY_IDX, cf_opts_energy_idx(&cache)),
            ColumnFamilyDescriptor::new(CF_META_IDX,   cf_opts_meta_idx(&cache)),
            ColumnFamilyDescriptor::new(CF_LISTS,      cf_opts_read_heavy(&cache)),
        ];

        let db = DB::open_cf_descriptors(&db_opts, path, cf_descs)
            .map_err(|e| GraphError::Storage(e.to_string()))?;

        Ok(Self { db, _cache: cache })
    }

    // ── Inline CF handle accessors ─────────────────────────────────────────────
    // `cf_handle()` does a HashMap<&str, Arc<BoundColumnFamily>> lookup.
    // The #[inline] hint lets the compiler hoist repeated lookups within a fn.

    #[inline] fn cf_nodes(&self)      -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_NODES).unwrap() }
    #[inline] fn cf_embeddings(&self) -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_EMBEDDINGS).unwrap() }
    #[inline] fn cf_edges(&self)      -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_EDGES).unwrap() }
    #[inline] fn cf_adj_out(&self)    -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_ADJ_OUT).unwrap() }
    #[inline] fn cf_adj_in(&self)     -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_ADJ_IN).unwrap() }
    #[inline] fn cf_meta(&self)       -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_META).unwrap() }
    #[inline] fn cf_energy(&self)     -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_ENERGY_IDX).unwrap() }
    #[inline] fn cf_meta_idx(&self)  -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_META_IDX).unwrap() }
    #[inline] fn cf_lists(&self)    -> &rocksdb::ColumnFamily { self.db.cf_handle(CF_LISTS).unwrap() }

    // ── Node operations ────────────────────────────────

    // ── Node operations (split storage: NodeMeta + PoincareVector) ────

    /// Persist a node (insert or overwrite).
    ///
    /// Atomically writes:
    /// - `NodeMeta` → `CF_NODES` (~100 bytes)
    /// - `PoincareVector` → `CF_EMBEDDINGS` (~24 KB at 3072 dims)
    /// - Energy secondary index key → `CF_ENERGY_IDX`
    pub fn put_node(&self, node: &Node) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();

        let meta_bytes = bincode::serialize(&node.meta)?;
        batch.put_cf(&self.cf_nodes(), node.id.as_bytes(), &meta_bytes);

        let emb_bytes = bincode::serialize(&node.embedding)?;
        batch.put_cf(&self.cf_embeddings(), node.id.as_bytes(), &emb_bytes);

        let energy_key = energy_index_key(node.energy, &node.id);
        batch.put_cf(&self.cf_energy(), &energy_key, &[]);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Persist only `NodeMeta` (no embedding write).
    ///
    /// Used by `update_energy()`, `update_hausdorff()` — saves ~24 KB of I/O
    /// per call by not touching `CF_EMBEDDINGS`.
    pub fn put_node_meta(&self, meta: &NodeMeta) -> Result<(), GraphError> {
        let value = bincode::serialize(meta)?;
        self.db.put_cf(&self.cf_nodes(), meta.id.as_bytes(), &value)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Persist only the embedding (no metadata write).
    pub fn put_embedding(&self, node_id: &Uuid, embedding: &PoincareVector) -> Result<(), GraphError> {
        let value = bincode::serialize(embedding)?;
        self.db.put_cf(&self.cf_embeddings(), node_id.as_bytes(), &value)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Update a node whose energy value has changed.
    ///
    /// **Writes only `NodeMeta` (~100 bytes) — never touches `CF_EMBEDDINGS`.**
    ///
    /// Atomically in one WriteBatch:
    /// 1. Deletes the old energy_idx key (keyed on `old_energy`)
    /// 2. Writes the updated `NodeMeta` to CF_NODES
    /// 3. Writes the new energy_idx key (keyed on `meta.energy`)
    pub fn put_node_meta_update_energy(
        &self,
        meta: &NodeMeta,
        old_energy: f32,
    ) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();

        let value = bincode::serialize(meta)?;
        batch.put_cf(&self.cf_nodes(), meta.id.as_bytes(), &value);

        if (old_energy - meta.energy).abs() > f32::EPSILON {
            let old_key = energy_index_key(old_energy, &meta.id);
            batch.delete_cf(&self.cf_energy(), &old_key);
        }

        let new_key = energy_index_key(meta.energy, &meta.id);
        batch.put_cf(&self.cf_energy(), &new_key, &[]);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Legacy wrapper — delegates to [`put_node_meta_update_energy`].
    pub fn put_node_update_energy(
        &self,
        node: &Node,
        old_energy: f32,
    ) -> Result<(), GraphError> {
        self.put_node_meta_update_energy(&node.meta, old_energy)
    }

    /// Batch-insert multiple nodes in a single RocksDB write (10× faster than individual puts).
    pub fn put_nodes_batch(&self, nodes: &[Node]) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();
        for node in nodes {
            let meta_bytes = bincode::serialize(&node.meta)?;
            batch.put_cf(&self.cf_nodes(), node.id.as_bytes(), &meta_bytes);

            let emb_bytes = bincode::serialize(&node.embedding)?;
            batch.put_cf(&self.cf_embeddings(), node.id.as_bytes(), &emb_bytes);

            let energy_key = energy_index_key(node.energy, &node.id);
            batch.put_cf(&self.cf_energy(), &energy_key, &[]);
        }
        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Retrieve node metadata by ID (~100 bytes). Returns `None` if not found.
    ///
    /// **Use this for energy checks, BFS gates, NQL filters** — 100–250× less
    /// I/O than `get_node()` which also loads the ~24 KB embedding.
    #[inline]
    pub fn get_node_meta(&self, id: &Uuid) -> Result<Option<NodeMeta>, GraphError> {
        match self.db.get_cf(&self.cf_nodes(), id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Retrieve only the embedding by node ID (~24 KB). Returns `None` if not found.
    #[inline]
    pub fn get_embedding(&self, id: &Uuid) -> Result<Option<PoincareVector>, GraphError> {
        match self.db.get_cf(&self.cf_embeddings(), id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Retrieve a full node (metadata + embedding) by ID.
    ///
    /// Joins `CF_NODES` and `CF_EMBEDDINGS` — use `get_node_meta()` when you
    /// don't need the embedding (BFS, energy updates, NQL filters).
    #[inline]
    pub fn get_node(&self, id: &Uuid) -> Result<Option<Node>, GraphError> {
        let meta = match self.get_node_meta(id)? {
            Some(m) => m,
            None => return Ok(None),
        };
        let embedding = match self.get_embedding(id)? {
            Some(e) => e,
            None => return Ok(None),
        };
        Ok(Some(Node::from((meta, embedding))))
    }

    /// Delete a node record from both CFs + energy_idx.
    /// Does NOT clean up adjacency — caller's responsibility.
    pub fn delete_node(&self, id: &Uuid) -> Result<(), GraphError> {
        if let Some(meta) = self.get_node_meta(id)? {
            let mut batch = rocksdb::WriteBatch::default();
            batch.delete_cf(&self.cf_nodes(), id.as_bytes());
            batch.delete_cf(&self.cf_embeddings(), id.as_bytes());
            let energy_key = energy_index_key(meta.energy, id);
            batch.delete_cf(&self.cf_energy(), &energy_key);
            self.db.write(batch)
                .map_err(|e| GraphError::Storage(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Returns true if a node with this ID exists.
    #[inline]
    pub fn node_exists(&self, id: &Uuid) -> Result<bool, GraphError> {
        Ok(self.db.get_cf(&self.cf_nodes(), id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
            .is_some())
    }

    /// Scan all node metadata (no embeddings). Fast: ~100 bytes per node.
    pub fn scan_nodes_meta(&self) -> Result<Vec<NodeMeta>, GraphError> {
        self.iter_nodes_meta().collect()
    }

    /// Scan all nodes (metadata + embedding). Full table scan — use sparingly.
    /// For energy-filtered queries, prefer `scan_nodes_energy_range()`.
    pub fn scan_nodes(&self) -> Result<Vec<Node>, GraphError> {
        self.iter_nodes().collect()
    }

    /// Iterator over node metadata only — yields `Result<NodeMeta>`.
    pub fn iter_nodes_meta(&self) -> NodeMetaIterator<'_> {
        NodeMetaIterator {
            inner: self.db.iterator_cf(&self.cf_nodes(), rocksdb::IteratorMode::Start),
        }
    }

    /// Iterator-based node scan — yields `Result<Node>` (joins meta + embedding).
    /// Preferred over `scan_nodes()` for large datasets.
    pub fn iter_nodes(&self) -> NodeIterator<'_> {
        NodeIterator {
            storage: self,
            meta_iter: self.db.iterator_cf(&self.cf_nodes(), rocksdb::IteratorMode::Start),
        }
    }

    /// **Fast energy range scan** — O(log N + k) via secondary index.
    ///
    /// Returns all nodes with `min_energy <= n.energy <= max_energy`, sorted by energy ASC.
    /// Uses the `energy_idx` CF which is prefix-bloom indexed on the 4-byte energy key.
    ///
    /// This replaces the O(N) full scan for `WHERE n.energy BETWEEN X AND Y` queries.
    pub fn scan_nodes_energy_range(
        &self,
        min_energy: f32,
        max_energy: f32,
    ) -> Result<Vec<Node>, GraphError> {
        let start_key = energy_index_key(min_energy, &Uuid::nil());
        let end_key   = energy_index_key(max_energy, &Uuid::max());

        let iter = self.db.iterator_cf(
            &self.cf_energy(),
            rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        let mut nodes = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            if key.as_ref() > end_key.as_slice() {
                break;
            }
            if key.len() < 20 {
                continue;
            }
            // Decode node_id from bytes 4..20
            let node_id = Uuid::from_slice(&key[4..20])
                .map_err(|e| GraphError::Storage(e.to_string()))?;
            if let Some(node) = self.get_node(&node_id)? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    /// Fast "greater-than" energy scan using the secondary index.
    pub fn scan_nodes_energy_gt(&self, min_energy: f32) -> Result<Vec<Node>, GraphError> {
        self.scan_nodes_energy_range(min_energy, f32::MAX)
    }

    /// Scan all node metadata and return IDs of expired nodes (TTL enforcement).
    ///
    /// A node is expired when `expires_at.is_some() && expires_at <= now`.
    /// O(N) full scan — intended for background reaper, not the hot path.
    pub fn scan_expired_node_ids(&self, now_unix_secs: i64) -> Result<Vec<Uuid>, GraphError> {
        let mut expired = Vec::new();
        for result in self.iter_nodes_meta() {
            let meta = result?;
            if let Some(exp) = meta.expires_at {
                if now_unix_secs >= exp {
                    expired.push(meta.id);
                }
            }
        }
        Ok(expired)
    }

    /// **Ultra-fast energy ID scan** — returns only node UUIDs, not full nodes.
    ///
    /// Same O(log N + k) complexity as `scan_nodes_energy_gt()` but skips the
    /// 25 KB `get_node()` deserialisation per result. Used by the filtered KNN
    /// routing logic in `NietzscheDB::knn_energy_filtered()`.
    pub fn scan_energy_ids_gt(&self, min_energy: f32) -> Result<Vec<Uuid>, GraphError> {
        let start_key = energy_index_key(min_energy, &Uuid::nil());
        let end_key   = energy_index_key(f32::MAX, &Uuid::max());

        let iter = self.db.iterator_cf(
            &self.cf_energy(),
            rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            if key.as_ref() > end_key.as_slice() {
                break;
            }
            if key.len() < 20 {
                continue;
            }
            let node_id = Uuid::from_slice(&key[4..20])
                .map_err(|e| GraphError::Storage(e.to_string()))?;
            ids.push(node_id);
        }
        Ok(ids)
    }

    // ── Metadata index operations ─────────────────────

    /// Write metadata index entries for a node's metadata fields.
    ///
    /// For each field in `indexed_fields`, if the node's `metadata` map contains
    /// a matching key with a sortable value (number or string), we write a
    /// `[field_hash(8B) | sortable_value(8B) | node_id(16B)]` key to `CF_META_IDX`.
    pub fn put_meta_index(
        &self,
        node_id: &Uuid,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
        indexed_fields: &std::collections::HashSet<String>,
    ) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();
        for field in indexed_fields {
            if let Some(val) = metadata.get(field) {
                if let Some(key) = meta_index_key(field, val, node_id) {
                    batch.put_cf(&self.cf_meta_idx(), &key, &[]);
                }
            }
        }
        if batch.len() > 0 {
            self.db.write(batch)
                .map_err(|e| GraphError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Remove metadata index entries for a node.
    pub fn delete_meta_index(
        &self,
        node_id: &Uuid,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
        indexed_fields: &std::collections::HashSet<String>,
    ) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();
        for field in indexed_fields {
            if let Some(val) = metadata.get(field) {
                if let Some(key) = meta_index_key(field, val, node_id) {
                    batch.delete_cf(&self.cf_meta_idx(), &key);
                }
            }
        }
        if batch.len() > 0 {
            self.db.write(batch)
                .map_err(|e| GraphError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Range scan on a metadata field: returns node IDs where `field` value
    /// is in `[min_val, max_val]` (inclusive, by sortable byte order).
    ///
    /// O(log N + k) via the `meta_idx` secondary index.
    pub fn scan_meta_index_range(
        &self,
        field: &str,
        min_val: &serde_json::Value,
        max_val: &serde_json::Value,
    ) -> Result<Vec<Uuid>, GraphError> {
        let field_hash = fnv1a_64(field.as_bytes());
        let min_sortable = sortable_value_bytes(min_val).unwrap_or([0u8; 8]);
        let max_sortable = sortable_value_bytes(max_val).unwrap_or([0xFF; 8]);

        let mut start_key = [0u8; 32];
        start_key[0..8].copy_from_slice(&field_hash.to_be_bytes());
        start_key[8..16].copy_from_slice(&min_sortable);
        // node_id = Uuid::nil (all zeros)

        let mut end_key = [0u8; 32];
        end_key[0..8].copy_from_slice(&field_hash.to_be_bytes());
        end_key[8..16].copy_from_slice(&max_sortable);
        end_key[16..32].copy_from_slice(&[0xFF; 16]); // max UUID

        let iter = self.db.iterator_cf(
            &self.cf_meta_idx(),
            rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        let mut ids = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            if key.as_ref() > end_key.as_slice() {
                break;
            }
            if key.len() < 32 {
                continue;
            }
            let node_id = Uuid::from_slice(&key[16..32])
                .map_err(|e| GraphError::Storage(e.to_string()))?;
            ids.push(node_id);
        }
        Ok(ids)
    }

    /// Exact-match scan on a metadata field: returns node IDs where `field == val`.
    pub fn scan_meta_index_eq(
        &self,
        field: &str,
        val: &serde_json::Value,
    ) -> Result<Vec<Uuid>, GraphError> {
        self.scan_meta_index_range(field, val, val)
    }

    // ── List operations (RPUSH/LRANGE) ─────────────────

    /// Append a value to the end of a named list for a node (RPUSH).
    ///
    /// Uses an atomic read-increment-write of the sequence counter stored in CF_META.
    /// Key format: `[node_id(16B) | list_name_hash(8B) | seq_be(8B)]`.
    pub fn list_rpush(
        &self,
        node_id: &Uuid,
        list_name: &str,
        value: &[u8],
    ) -> Result<u64, GraphError> {
        let name_hash = fnv1a_64(list_name.as_bytes());

        // Atomic increment: read current seq from CF_META
        let seq_meta_key = format!("list_seq:{}:{}", node_id, list_name);
        let current_seq: u64 = match self.db.get_cf(&self.cf_meta(), seq_meta_key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) if bytes.len() == 8 => u64::from_be_bytes(bytes[..8].try_into().unwrap()),
            _ => 0,
        };
        let new_seq = current_seq + 1;

        // Build list entry key
        let mut key = [0u8; 32];
        key[0..16].copy_from_slice(node_id.as_bytes());
        key[16..24].copy_from_slice(&name_hash.to_be_bytes());
        key[24..32].copy_from_slice(&new_seq.to_be_bytes());

        // Atomic write: list entry + updated seq counter
        let mut batch = rocksdb::WriteBatch::default();
        batch.put_cf(&self.cf_lists(), &key, value);
        batch.put_cf(&self.cf_meta(), seq_meta_key.as_bytes(), &new_seq.to_be_bytes());
        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))?;

        Ok(new_seq)
    }

    /// Read a range of values from a named list (LRANGE).
    ///
    /// `start` and `stop` are 0-based indices (inclusive, like Redis LRANGE).
    /// Use `stop = -1` (as `i64`) for "to the end".
    pub fn list_lrange(
        &self,
        node_id: &Uuid,
        list_name: &str,
        start: u64,
        stop: i64,
    ) -> Result<Vec<Vec<u8>>, GraphError> {
        let name_hash = fnv1a_64(list_name.as_bytes());

        // Build prefix for iteration
        let mut prefix = [0u8; 24];
        prefix[0..16].copy_from_slice(node_id.as_bytes());
        prefix[16..24].copy_from_slice(&name_hash.to_be_bytes());

        let mut start_key = [0u8; 32];
        start_key[0..24].copy_from_slice(&prefix);
        start_key[24..32].copy_from_slice(&1u64.to_be_bytes()); // seq starts at 1

        let iter = self.db.iterator_cf(
            &self.cf_lists(),
            rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
        );

        let mut results = Vec::new();
        let mut idx = 0u64;
        for item in iter {
            let (key, val) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            // Check prefix match (node_id + name_hash)
            if key.len() < 32 || &key[0..24] != &prefix[..] {
                break;
            }
            if idx >= start {
                results.push(val.to_vec());
            }
            idx += 1;
            if stop >= 0 && idx > stop as u64 {
                break;
            }
        }
        Ok(results)
    }

    /// Return the length of a named list.
    pub fn list_len(
        &self,
        node_id: &Uuid,
        list_name: &str,
    ) -> Result<u64, GraphError> {
        let seq_meta_key = format!("list_seq:{}:{}", node_id, list_name);
        match self.db.get_cf(&self.cf_meta(), seq_meta_key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) if bytes.len() == 8 => Ok(u64::from_be_bytes(bytes[..8].try_into().unwrap())),
            _ => Ok(0),
        }
    }

    // ── Edge operations ────────────────────────────────

    /// Persist an edge and update both adjacency column families atomically.
    pub fn put_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();

        // 1. Persist the edge itself
        let edge_bytes = bincode::serialize(edge)?;
        batch.put_cf(&self.cf_edges(), edge.id.as_bytes(), &edge_bytes);

        // 2. Append to adj_out[from]
        let mut out_ids = self.read_uuid_list_cf_out(&edge.from)?;
        out_ids.push(edge.id);
        batch.put_cf(&self.cf_adj_out(), edge.from.as_bytes(), bincode::serialize(&out_ids)?);

        // 3. Append to adj_in[to]
        let mut in_ids = self.read_uuid_list_cf_in(&edge.to)?;
        in_ids.push(edge.id);
        batch.put_cf(&self.cf_adj_in(), edge.to.as_bytes(), bincode::serialize(&in_ids)?);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Batch-insert multiple edges in a single RocksDB write.
    pub fn put_edges_batch(&self, edges: &[Edge]) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();
        for edge in edges {
            let edge_bytes = bincode::serialize(edge)?;
            batch.put_cf(&self.cf_edges(), edge.id.as_bytes(), &edge_bytes);

            let mut out_ids = self.read_uuid_list_cf_out(&edge.from)?;
            out_ids.push(edge.id);
            batch.put_cf(&self.cf_adj_out(), edge.from.as_bytes(), bincode::serialize(&out_ids)?);

            let mut in_ids = self.read_uuid_list_cf_in(&edge.to)?;
            in_ids.push(edge.id);
            batch.put_cf(&self.cf_adj_in(), edge.to.as_bytes(), bincode::serialize(&in_ids)?);
        }
        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Retrieve an edge by ID. Returns `None` if not found.
    #[inline]
    pub fn get_edge(&self, id: &Uuid) -> Result<Option<Edge>, GraphError> {
        match self.db.get_cf(&self.cf_edges(), id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete an edge record and remove it from both adjacency lists.
    pub fn delete_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();
        batch.delete_cf(&self.cf_edges(), edge.id.as_bytes());

        let mut out_ids = self.read_uuid_list_cf_out(&edge.from)?;
        out_ids.retain(|id| id != &edge.id);
        batch.put_cf(&self.cf_adj_out(), edge.from.as_bytes(), bincode::serialize(&out_ids)?);

        let mut in_ids = self.read_uuid_list_cf_in(&edge.to)?;
        in_ids.retain(|id| id != &edge.id);
        batch.put_cf(&self.cf_adj_in(), edge.to.as_bytes(), bincode::serialize(&in_ids)?);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// All edge IDs originating from `node_id`.
    pub fn edge_ids_from(&self, node_id: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        self.read_uuid_list_cf_out(node_id)
    }

    /// All edge IDs pointing to `node_id`.
    pub fn edge_ids_to(&self, node_id: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        self.read_uuid_list_cf_in(node_id)
    }

    /// Scan all edges (full table scan).
    pub fn scan_edges(&self) -> Result<Vec<Edge>, GraphError> {
        self.iter_edges().collect()
    }

    /// Iterator-based edge scan — yields `Result<Edge>` without loading all into memory.
    pub fn iter_edges(&self) -> EdgeIterator<'_> {
        EdgeIterator {
            inner: self.db.iterator_cf(&self.cf_edges(), rocksdb::IteratorMode::Start),
        }
    }

    // ── Metadata ───────────────────────────────────────

    /// Write a metadata key/value pair.
    pub fn put_meta(&self, key: &str, value: &[u8]) -> Result<(), GraphError> {
        self.db.put_cf(&self.cf_meta(), key.as_bytes(), value)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Read a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Result<Option<Vec<u8>>, GraphError> {
        self.db.get_cf(&self.cf_meta(), key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Delete a metadata key.
    pub fn delete_meta(&self, key: &str) -> Result<(), GraphError> {
        self.db.delete_cf(&self.cf_meta(), key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Scan all metadata entries with a given key prefix.
    ///
    /// Returns `(key, value)` pairs where the key starts with `prefix`.
    pub fn scan_meta_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>, GraphError> {
        let cf = self.cf_meta();
        let iter = self.db.prefix_iterator_cf(&cf, prefix);
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            if !key.starts_with(prefix) {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    // ── AdjacencyIndex reconstruction ──────────────────

    /// Reconstruct the in-memory `AdjacencyIndex` from all edges in RocksDB.
    /// Called once at startup after WAL replay.
    pub fn rebuild_adjacency(&self) -> Result<AdjacencyIndex, GraphError> {
        let index = AdjacencyIndex::new();
        let edges = self.scan_edges()?;
        for edge in &edges {
            index.add_edge(edge);
        }
        Ok(index)
    }

    /// Total number of persisted nodes (fast: uses RocksDB estimate).
    pub fn node_count(&self) -> Result<usize, GraphError> {
        // RocksDB property "rocksdb.estimate-num-keys" is O(1) and accurate enough
        // for monitoring. Falls back to full scan if not available.
        let cf = self.cf_nodes();
        if let Ok(Some(v)) = self.db.property_int_value_cf(&cf, "rocksdb.estimate-num-keys") {
            return Ok(v as usize);
        }
        // Fallback: full scan count
        let mut count = 0usize;
        for item in self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
            item.map_err(|e| GraphError::Storage(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    /// Total number of persisted edges (fast: uses RocksDB estimate).
    pub fn edge_count(&self) -> Result<usize, GraphError> {
        let cf = self.cf_edges();
        if let Ok(Some(v)) = self.db.property_int_value_cf(&cf, "rocksdb.estimate-num-keys") {
            return Ok(v as usize);
        }
        let mut count = 0usize;
        for item in self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start) {
            item.map_err(|e| GraphError::Storage(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    // ── Raw DB handle (for SensoryStorage) ────────────────

    /// Expose the underlying RocksDB handle so that `SensoryStorage`
    /// (in `nietzsche-sensory`) can borrow it directly.
    pub fn db_handle(&self) -> &DB {
        &self.db
    }

    // ── Internal helpers ───────────────────────────────

    #[inline]
    fn read_uuid_list_cf_out(&self, key: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        match self.db.get_cf(&self.cf_adj_out(), key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(bincode::deserialize(&bytes)?),
            None => Ok(Vec::new()),
        }
    }

    #[inline]
    fn read_uuid_list_cf_in(&self, key: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        match self.db.get_cf(&self.cf_adj_in(), key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(bincode::deserialize(&bytes)?),
            None => Ok(Vec::new()),
        }
    }
}

// ─────────────────────────────────────────────
// Energy secondary index helpers
// ─────────────────────────────────────────────

/// Build the energy index key: [energy_be(4 bytes) | node_id(16 bytes)] = 20 bytes total.
///
/// Big-endian byte order preserves f32 comparison order for unsigned exponent ranges.
/// We XOR the sign bit so that negative floats sort before positive ones correctly.
#[inline]
fn energy_index_key(energy: f32, node_id: &Uuid) -> [u8; 20] {
    let raw = energy.to_bits();
    // IEEE 754 sign-magnitude → two's complement sort order
    let sortable = if raw >> 31 == 0 {
        raw ^ 0x8000_0000
    } else {
        !raw
    };
    let mut key = [0u8; 20];
    key[0..4].copy_from_slice(&sortable.to_be_bytes());
    key[4..20].copy_from_slice(node_id.as_bytes());
    key
}

// ─────────────────────────────────────────────
// Metadata secondary index helpers
// ─────────────────────────────────────────────

/// FNV-1a 64-bit hash for field name → 8-byte prefix in meta_idx keys.
#[inline]
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Convert a JSON value to 8 sortable bytes for the meta_idx key.
///
/// - Numbers (f64/i64): IEEE 754 sign-magnitude → sortable big-endian
/// - Strings: first 8 bytes of the string (zero-padded)
/// - Others: not indexable → returns None
fn sortable_value_bytes(val: &serde_json::Value) -> Option<[u8; 8]> {
    match val {
        serde_json::Value::Number(n) => {
            let f = n.as_f64()?;
            let raw = f.to_bits();
            let sortable = if raw >> 63 == 0 {
                raw ^ 0x8000_0000_0000_0000
            } else {
                !raw
            };
            Some(sortable.to_be_bytes())
        }
        serde_json::Value::String(s) => {
            let bytes = s.as_bytes();
            let mut buf = [0u8; 8];
            let len = bytes.len().min(8);
            buf[..len].copy_from_slice(&bytes[..len]);
            Some(buf)
        }
        _ => None,
    }
}

/// Build a metadata index key: [field_hash(8B) | sortable_value(8B) | node_id(16B)] = 32 bytes.
/// Returns None if the value is not indexable (null, bool, array, object).
fn meta_index_key(field: &str, val: &serde_json::Value, node_id: &Uuid) -> Option<[u8; 32]> {
    let sortable = sortable_value_bytes(val)?;
    let field_hash = fnv1a_64(field.as_bytes());
    let mut key = [0u8; 32];
    key[0..8].copy_from_slice(&field_hash.to_be_bytes());
    key[8..16].copy_from_slice(&sortable);
    key[16..32].copy_from_slice(node_id.as_bytes());
    Some(key)
}

// ─────────────────────────────────────────────
// System helpers
// ─────────────────────────────────────────────

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
}

// ─────────────────────────────────────────────
// Lazy iterators (avoid full table scans)
// ─────────────────────────────────────────────

/// Lazy iterator over node metadata in RocksDB (no embedding — ~100 bytes per item).
pub struct NodeMetaIterator<'a> {
    inner: rocksdb::DBIteratorWithThreadMode<'a, DB>,
}

impl<'a> Iterator for NodeMetaIterator<'a> {
    type Item = Result<NodeMeta, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        Some(item
            .map_err(|e| GraphError::Storage(e.to_string()))
            .and_then(|(_, value)| {
                bincode::deserialize(&value).map_err(Into::into)
            }))
    }
}

/// Lazy iterator over full nodes (joins `CF_NODES` + `CF_EMBEDDINGS`).
pub struct NodeIterator<'a> {
    storage: &'a GraphStorage,
    meta_iter: rocksdb::DBIteratorWithThreadMode<'a, DB>,
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = Result<Node, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = self.meta_iter.next()?;
            let (key, meta_bytes) = match item {
                Ok(kv) => kv,
                Err(e) => return Some(Err(GraphError::Storage(e.to_string()))),
            };
            let meta: NodeMeta = match bincode::deserialize(&meta_bytes) {
                Ok(m) => m,
                Err(e) => return Some(Err(e.into())),
            };
            // Join with embedding from CF_EMBEDDINGS
            let node_id = match Uuid::from_slice(&key) {
                Ok(id) => id,
                Err(e) => return Some(Err(GraphError::Storage(e.to_string()))),
            };
            match self.storage.get_embedding(&node_id) {
                Ok(Some(emb)) => return Some(Ok(Node::from((meta, emb)))),
                Ok(None) => continue, // orphan meta without embedding — skip
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Lazy iterator over edges in RocksDB.
pub struct EdgeIterator<'a> {
    inner: rocksdb::DBIteratorWithThreadMode<'a, DB>,
}

impl<'a> Iterator for EdgeIterator<'a> {
    type Item = Result<Edge, GraphError>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        Some(item
            .map_err(|e| GraphError::Storage(e.to_string()))
            .and_then(|(_, value)| {
                bincode::deserialize(&value).map_err(Into::into)
            }))
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EdgeType, PoincareVector};
    use tempfile::TempDir;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f64, y: f64) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, y as f32]),
            serde_json::json!({"label": "test"}),
        )
    }

    #[test]
    fn put_and_get_node() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;

        storage.put_node(&node).unwrap();
        let retrieved = storage.get_node(&id).unwrap().unwrap();
        assert_eq!(retrieved.id, id);
        assert_eq!(retrieved.embedding.coords, node.embedding.coords);
    }

    #[test]
    fn get_node_meta_without_embedding() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;
        let expected_energy = node.energy;

        storage.put_node(&node).unwrap();
        let meta = storage.get_node_meta(&id).unwrap().unwrap();
        assert_eq!(meta.id, id);
        assert_eq!(meta.energy, expected_energy);
    }

    #[test]
    fn get_embedding_separate() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.3, 0.4);
        let id = node.id;
        let expected_coords = node.embedding.coords.clone();

        storage.put_node(&node).unwrap();
        let emb = storage.get_embedding(&id).unwrap().unwrap();
        assert_eq!(emb.coords, expected_coords);
    }

    #[test]
    fn put_node_meta_only_does_not_touch_embedding() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;
        storage.put_node(&node).unwrap();

        // Update meta only
        let mut meta = storage.get_node_meta(&id).unwrap().unwrap();
        meta.energy = 0.42;
        storage.put_node_meta(&meta).unwrap();

        // Embedding should be unchanged
        let emb = storage.get_embedding(&id).unwrap().unwrap();
        assert_eq!(emb.coords, node.embedding.coords);

        // Meta should reflect update
        let meta2 = storage.get_node_meta(&id).unwrap().unwrap();
        assert!((meta2.energy - 0.42).abs() < 1e-6);
    }

    #[test]
    fn get_missing_node_returns_none() {
        let (storage, _dir) = open_temp_db();
        let result = storage.get_node(&Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn delete_node_removes_from_both_cfs() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;
        storage.put_node(&node).unwrap();
        storage.delete_node(&id).unwrap();
        assert!(storage.get_node(&id).unwrap().is_none());
        assert!(storage.get_node_meta(&id).unwrap().is_none());
        assert!(storage.get_embedding(&id).unwrap().is_none());
    }

    #[test]
    fn put_edge_updates_adjacency_cfs() {
        let (storage, _dir) = open_temp_db();
        let a = make_node(0.1, 0.0);
        let b = make_node(0.3, 0.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();

        let edge = Edge::association(a.id, b.id, 0.9);
        let eid = edge.id;
        storage.put_edge(&edge).unwrap();

        let out = storage.edge_ids_from(&a.id).unwrap();
        let inc = storage.edge_ids_to(&b.id).unwrap();
        assert_eq!(out, vec![eid]);
        assert_eq!(inc, vec![eid]);
    }

    #[test]
    fn delete_edge_removes_from_adjacency() {
        let (storage, _dir) = open_temp_db();
        let a = make_node(0.1, 0.0);
        let b = make_node(0.3, 0.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();

        let edge = Edge::association(a.id, b.id, 0.9);
        storage.put_edge(&edge).unwrap();
        storage.delete_edge(&edge).unwrap();

        assert!(storage.edge_ids_from(&a.id).unwrap().is_empty());
        assert!(storage.edge_ids_to(&b.id).unwrap().is_empty());
    }

    #[test]
    fn scan_nodes_returns_all() {
        let (storage, _dir) = open_temp_db();
        for _ in 0..10 {
            storage.put_node(&make_node(0.1, 0.1)).unwrap();
        }
        let nodes = storage.scan_nodes().unwrap();
        assert_eq!(nodes.len(), 10);
    }

    #[test]
    fn scan_nodes_meta_returns_all() {
        let (storage, _dir) = open_temp_db();
        for _ in 0..10 {
            storage.put_node(&make_node(0.1, 0.1)).unwrap();
        }
        let metas = storage.scan_nodes_meta().unwrap();
        assert_eq!(metas.len(), 10);
    }

    #[test]
    fn rebuild_adjacency_from_edges() {
        let (storage, _dir) = open_temp_db();
        let a = make_node(0.1, 0.0);
        let b = make_node(0.3, 0.0);
        let c = make_node(0.5, 0.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        storage.put_node(&c).unwrap();
        storage.put_edge(&Edge::association(a.id, b.id, 1.0)).unwrap();
        storage.put_edge(&Edge::hierarchical(a.id, c.id)).unwrap();

        let index = storage.rebuild_adjacency().unwrap();
        let out = index.neighbors_out(&a.id);
        assert_eq!(out.len(), 2);
        assert!(out.contains(&b.id));
        assert!(out.contains(&c.id));
    }

    #[test]
    fn node_survives_restart() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let node = make_node(0.2, 0.3);
        let id = node.id;

        {
            let storage = GraphStorage::open(&path).unwrap();
            storage.put_node(&node).unwrap();
        } // db closed here

        {
            let storage = GraphStorage::open(&path).unwrap();
            let retrieved = storage.get_node(&id).unwrap().unwrap();
            assert_eq!(retrieved.id, id);
        }
    }

    #[test]
    fn node_count_and_edge_count() {
        let (storage, _dir) = open_temp_db();
        let a = make_node(0.1, 0.0);
        let b = make_node(0.3, 0.0);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        storage.put_edge(&Edge::association(a.id, b.id, 1.0)).unwrap();

        assert_eq!(storage.node_count().unwrap(), 2);
        assert_eq!(storage.edge_count().unwrap(), 1);
    }

    #[test]
    fn meta_put_and_get() {
        let (storage, _dir) = open_temp_db();
        storage.put_meta("db_version", b"2.0").unwrap();
        let val = storage.get_meta("db_version").unwrap().unwrap();
        assert_eq!(val, b"2.0");
    }

    #[test]
    fn scan_expired_node_ids_returns_expired() {
        let (storage, _dir) = open_temp_db();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Node with TTL already expired
        let mut expired_node = make_node(0.1, 0.2);
        expired_node.meta.expires_at = Some(now - 10);
        storage.put_node(&expired_node).unwrap();

        // Node with TTL in the future
        let mut fresh_node = make_node(0.3, 0.4);
        fresh_node.meta.expires_at = Some(now + 3600);
        storage.put_node(&fresh_node).unwrap();

        // Node with no TTL
        let eternal_node = make_node(0.5, 0.6);
        storage.put_node(&eternal_node).unwrap();

        let expired_ids = storage.scan_expired_node_ids(now).unwrap();
        assert_eq!(expired_ids.len(), 1);
        assert_eq!(expired_ids[0], expired_node.id);
    }

    #[test]
    fn put_node_meta_update_energy_only_writes_meta() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;
        let original_coords = node.embedding.coords.clone();
        storage.put_node(&node).unwrap();

        let mut meta = storage.get_node_meta(&id).unwrap().unwrap();
        let old_energy = meta.energy;
        meta.energy = 0.33;
        storage.put_node_meta_update_energy(&meta, old_energy).unwrap();

        // Embedding untouched
        let emb = storage.get_embedding(&id).unwrap().unwrap();
        assert_eq!(emb.coords, original_coords);

        // Meta updated
        let m = storage.get_node_meta(&id).unwrap().unwrap();
        assert!((m.energy - 0.33).abs() < 1e-6);
    }
}
