use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{Edge, Node};

// ─────────────────────────────────────────────
// Column Family names
// ─────────────────────────────────────────────

const CF_NODES:   &str = "nodes";    // key: node_id (16 bytes) → Node (bincode)
const CF_EDGES:   &str = "edges";    // key: edge_id (16 bytes) → Edge (bincode)
const CF_ADJ_OUT: &str = "adj_out";  // key: node_id → Vec<Uuid> outgoing edge ids
const CF_ADJ_IN:  &str = "adj_in";   // key: node_id → Vec<Uuid> incoming edge ids
const CF_META:    &str = "meta";     // key: &str → arbitrary bytes

const ALL_CFS: &[&str] = &[CF_NODES, CF_EDGES, CF_ADJ_OUT, CF_ADJ_IN, CF_META];

// ─────────────────────────────────────────────
// GraphStorage
// ─────────────────────────────────────────────

/// Persistent storage for the NietzscheDB hyperbolic graph.
///
/// Backed by RocksDB with five column families:
/// - `nodes`   — serialized Node structs (bincode)
/// - `edges`   — serialized Edge structs (bincode)
/// - `adj_out` — per-node list of outgoing edge UUIDs
/// - `adj_in`  — per-node list of incoming edge UUIDs
/// - `meta`    — global metadata key/value pairs
pub struct GraphStorage {
    db: DB,
}

impl GraphStorage {
    // ── Open / init ────────────────────────────────────

    /// Open (or create) the RocksDB database at `path`.
    pub fn open(path: &str) -> Result<Self, GraphError> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descs: Vec<ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, Options::default()))
            .collect();

        let db = DB::open_cf_descriptors(&opts, path, cf_descs)
            .map_err(|e| GraphError::Storage(e.to_string()))?;

        Ok(Self { db })
    }

    // ── Node operations ────────────────────────────────

    /// Persist a node (insert or overwrite).
    pub fn put_node(&self, node: &Node) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        let value = bincode::serialize(node)?;
        self.db.put_cf(&cf, node.id.as_bytes(), value)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Retrieve a node by ID. Returns `None` if not found.
    pub fn get_node(&self, id: &Uuid) -> Result<Option<Node>, GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        match self.db.get_cf(&cf, id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete a node record (does NOT clean up adjacency — caller's responsibility).
    pub fn delete_node(&self, id: &Uuid) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        self.db.delete_cf(&cf, id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Returns true if a node with this ID exists.
    pub fn node_exists(&self, id: &Uuid) -> Result<bool, GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        Ok(self.db.get_cf(&cf, id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
            .is_some())
    }

    /// Scan all nodes. Returns a Vec (full table scan — use sparingly in production).
    pub fn scan_nodes(&self) -> Result<Vec<Node>, GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        let mut nodes = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            let node: Node = bincode::deserialize(&value)?;
            nodes.push(node);
        }
        Ok(nodes)
    }

    // ── Edge operations ────────────────────────────────

    /// Persist an edge and update both adjacency column families atomically.
    pub fn put_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        // Use a write batch for atomicity across 3 CFs
        let mut batch = rocksdb::WriteBatch::default();

        // 1. Persist the edge itself
        let cf_edges = self.db.cf_handle(CF_EDGES).unwrap();
        let edge_bytes = bincode::serialize(edge)?;
        batch.put_cf(&cf_edges, edge.id.as_bytes(), &edge_bytes);

        // 2. Append to adj_out[from]
        let cf_out = self.db.cf_handle(CF_ADJ_OUT).unwrap();
        let mut out_ids = self.read_uuid_list(&cf_out, &edge.from)?;
        out_ids.push(edge.id);
        batch.put_cf(&cf_out, edge.from.as_bytes(), bincode::serialize(&out_ids)?);

        // 3. Append to adj_in[to]
        let cf_in = self.db.cf_handle(CF_ADJ_IN).unwrap();
        let mut in_ids = self.read_uuid_list(&cf_in, &edge.to)?;
        in_ids.push(edge.id);
        batch.put_cf(&cf_in, edge.to.as_bytes(), bincode::serialize(&in_ids)?);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Retrieve an edge by ID. Returns `None` if not found.
    pub fn get_edge(&self, id: &Uuid) -> Result<Option<Edge>, GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        match self.db.get_cf(&cf, id.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(Some(bincode::deserialize(&bytes)?)),
            None => Ok(None),
        }
    }

    /// Delete an edge record and remove it from both adjacency lists.
    pub fn delete_edge(&self, edge: &Edge) -> Result<(), GraphError> {
        let mut batch = rocksdb::WriteBatch::default();

        // Remove edge record
        let cf_edges = self.db.cf_handle(CF_EDGES).unwrap();
        batch.delete_cf(&cf_edges, edge.id.as_bytes());

        // Remove from adj_out[from]
        let cf_out = self.db.cf_handle(CF_ADJ_OUT).unwrap();
        let mut out_ids = self.read_uuid_list(&cf_out, &edge.from)?;
        out_ids.retain(|id| id != &edge.id);
        batch.put_cf(&cf_out, edge.from.as_bytes(), bincode::serialize(&out_ids)?);

        // Remove from adj_in[to]
        let cf_in = self.db.cf_handle(CF_ADJ_IN).unwrap();
        let mut in_ids = self.read_uuid_list(&cf_in, &edge.to)?;
        in_ids.retain(|id| id != &edge.id);
        batch.put_cf(&cf_in, edge.to.as_bytes(), bincode::serialize(&in_ids)?);

        self.db.write(batch)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// All edge IDs originating from `node_id`.
    pub fn edge_ids_from(&self, node_id: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        let cf = self.db.cf_handle(CF_ADJ_OUT).unwrap();
        self.read_uuid_list(&cf, node_id)
    }

    /// All edge IDs pointing to `node_id`.
    pub fn edge_ids_to(&self, node_id: &Uuid) -> Result<Vec<Uuid>, GraphError> {
        let cf = self.db.cf_handle(CF_ADJ_IN).unwrap();
        self.read_uuid_list(&cf, node_id)
    }

    /// Scan all edges (full table scan).
    pub fn scan_edges(&self) -> Result<Vec<Edge>, GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let mut edges = Vec::new();
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (_, value) = item.map_err(|e| GraphError::Storage(e.to_string()))?;
            let edge: Edge = bincode::deserialize(&value)?;
            edges.push(edge);
        }
        Ok(edges)
    }

    // ── Metadata ───────────────────────────────────────

    /// Write a metadata key/value pair.
    pub fn put_meta(&self, key: &str, value: &[u8]) -> Result<(), GraphError> {
        let cf = self.db.cf_handle(CF_META).unwrap();
        self.db.put_cf(&cf, key.as_bytes(), value)
            .map_err(|e| GraphError::Storage(e.to_string()))
    }

    /// Read a metadata value by key.
    pub fn get_meta(&self, key: &str) -> Result<Option<Vec<u8>>, GraphError> {
        let cf = self.db.cf_handle(CF_META).unwrap();
        self.db.get_cf(&cf, key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))
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

    /// Total number of persisted nodes.
    pub fn node_count(&self) -> Result<usize, GraphError> {
        let cf = self.db.cf_handle(CF_NODES).unwrap();
        let mut count = 0usize;
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            item.map_err(|e| GraphError::Storage(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    /// Total number of persisted edges.
    pub fn edge_count(&self) -> Result<usize, GraphError> {
        let cf = self.db.cf_handle(CF_EDGES).unwrap();
        let mut count = 0usize;
        let iter = self.db.iterator_cf(&cf, rocksdb::IteratorMode::Start);
        for item in iter {
            item.map_err(|e| GraphError::Storage(e.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    // ── Internal helpers ───────────────────────────────

    fn read_uuid_list(
        &self,
        cf: &rocksdb::BoundColumnFamily,
        key: &Uuid,
    ) -> Result<Vec<Uuid>, GraphError> {
        match self.db.get_cf(cf, key.as_bytes())
            .map_err(|e| GraphError::Storage(e.to_string()))?
        {
            Some(bytes) => Ok(bincode::deserialize(&bytes)?),
            None => Ok(Vec::new()),
        }
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
            PoincareVector::new(vec![x, y]),
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
    fn get_missing_node_returns_none() {
        let (storage, _dir) = open_temp_db();
        let result = storage.get_node(&Uuid::new_v4()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn delete_node_removes_it() {
        let (storage, _dir) = open_temp_db();
        let node = make_node(0.1, 0.2);
        let id = node.id;
        storage.put_node(&node).unwrap();
        storage.delete_node(&id).unwrap();
        assert!(storage.get_node(&id).unwrap().is_none());
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
}
