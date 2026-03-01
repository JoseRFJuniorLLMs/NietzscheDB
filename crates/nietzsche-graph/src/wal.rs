use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::GraphError;
use crate::model::{Edge, Node, PoincareVector};

// ─────────────────────────────────────────────
// WAL Entry types
// ─────────────────────────────────────────────

/// All mutations that go through the graph WAL.
///
/// This WAL covers *graph* operations only — the NietzscheDB
/// vector store has its own separate WAL for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphWalEntry {
    /// A new node was inserted.
    InsertNode(Node),
    /// A new edge was inserted.
    InsertEdge(Edge),
    /// A node was hard-deleted (rare — prefer PruneNode).
    DeleteNode(Uuid),
    /// A node was soft-deleted (energy set to 0, edge_type → Pruned).
    PruneNode(Uuid),
    /// An edge was hard-deleted.
    DeleteEdge(Uuid),
    /// A node's energy value was updated.
    UpdateNodeEnergy { node_id: Uuid, energy: f32 },
    /// A node's local Hausdorff dimension was recomputed.
    UpdateHausdorff { node_id: Uuid, hausdorff: f32 },
    /// A node's embedding was updated (during sleep/reconsolidation).
    UpdateEmbedding { node_id: Uuid, embedding: PoincareVector },
    /// A node was phantomized (structural scar, topology preserved).
    PhantomizeNode(Uuid),
    /// A phantom node was reanimated (brought back to active state).
    ReanimateNode { node_id: Uuid, energy: f32 },
    /// A transaction was started (saga pattern begin).
    TxBegin(Uuid),
    /// A transaction was committed (saga pattern checkpoint).
    TxCommitted(Uuid),
    /// A transaction was rolled back.
    TxRolledBack(Uuid),
    /// An edge's metadata was updated (MERGE ON MATCH / increment).
    UpdateEdgeMeta(Edge),
    /// A node's content was updated (MERGE ON MATCH SET).
    UpdateNodeContent { node_id: Uuid, content: serde_json::Value },
}

impl GraphWalEntry {
    /// Human-readable tag for logging.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::InsertNode(_)       => "INSERT_NODE",
            Self::InsertEdge(_)       => "INSERT_EDGE",
            Self::DeleteNode(_)       => "DELETE_NODE",
            Self::PruneNode(_)        => "PRUNE_NODE",
            Self::DeleteEdge(_)       => "DELETE_EDGE",
            Self::UpdateNodeEnergy {..}=> "UPDATE_ENERGY",
            Self::UpdateHausdorff {..} => "UPDATE_HAUSDORFF",
            Self::UpdateEmbedding {..} => "UPDATE_EMBEDDING",
            Self::PhantomizeNode(_)   => "PHANTOMIZE_NODE",
            Self::ReanimateNode {..}  => "REANIMATE_NODE",
            Self::TxBegin(_)          => "TX_BEGIN",
            Self::TxCommitted(_)      => "TX_COMMITTED",
            Self::TxRolledBack(_)     => "TX_ROLLED_BACK",
            Self::UpdateEdgeMeta(_)  => "UPDATE_EDGE_META",
            Self::UpdateNodeContent {..} => "UPDATE_NODE_CONTENT",
        }
    }
}

// ─────────────────────────────────────────────
// Wire format
// ─────────────────────────────────────────────
//
// Each record:  [magic: u32][len: u32][crc32: u32][payload: [u8; len]]
//
// On replay, a record with a bad CRC or truncated payload is treated
// as a corrupted tail — everything after it is discarded.

const MAGIC: u32 = 0x4E5A4757; // "NZGW" (NietzscheGraphWAL)

/// Maximum WAL record payload size (64 MiB). Prevents OOM on corrupted files.
const MAX_WAL_RECORD_SIZE: u32 = 64 * 1024 * 1024;

// ─────────────────────────────────────────────
// GraphWal
// ─────────────────────────────────────────────

/// Append-only binary Write-Ahead Log for graph mutations.
///
/// Stored at `<data_dir>/graph.wal`.
/// Opened with `O_APPEND` so writes are atomic at the OS level
/// (no partial record corruption during normal operation).
pub struct GraphWal {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl GraphWal {
    /// Open (or create) the WAL file at `dir/graph.wal`.
    /// Verifies the tail on open — truncates any incomplete record.
    pub fn open(dir: &Path) -> Result<Self, GraphError> {
        let path = dir.join("graph.wal");

        // Verify existing file integrity and truncate corrupted tail
        if path.exists() {
            Self::truncate_corrupted_tail(&path)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| GraphError::Storage(format!("WAL open: {e}")))?;

        Ok(Self {
            path,
            writer: BufWriter::new(file),
        })
    }

    /// Append one entry to the WAL and **immediately flush + fsync** to stable storage.
    ///
    /// Guarantees durability for single-entry writes. For bulk operations,
    /// use [`append_buffered`] followed by a single [`flush`] call.
    pub fn append(&mut self, entry: &GraphWalEntry) -> Result<(), GraphError> {
        self.write_record(entry)?;
        self.writer.flush()
            .map_err(|e| GraphError::Storage(format!("WAL flush: {e}")))?;
        // fsync to guarantee data reaches stable storage (not just OS page cache)
        self.writer.get_ref().sync_data()
            .map_err(|e| GraphError::Storage(format!("WAL fsync: {e}")))?;
        Ok(())
    }

    /// Append one entry to the WAL **without flushing**.
    ///
    /// The entry is buffered in the `BufWriter`. Call [`flush`] once after
    /// all entries in a batch to commit them all durably in one OS write.
    ///
    /// ## Safety
    /// If the process crashes before `flush()`, buffered entries are lost.
    /// Use only inside `insert_nodes_bulk` / `insert_edges_bulk` patterns
    /// where RocksDB durability is guaranteed by the subsequent `write_batch`.
    pub fn append_buffered(&mut self, entry: &GraphWalEntry) -> Result<(), GraphError> {
        self.write_record(entry)
    }

    /// Flush all buffered WAL entries to stable storage (flush + fsync).
    ///
    /// Call once after a series of [`append_buffered`] calls.
    pub fn flush(&mut self) -> Result<(), GraphError> {
        self.writer.flush()
            .map_err(|e| GraphError::Storage(format!("WAL flush: {e}")))?;
        self.writer.get_ref().sync_data()
            .map_err(|e| GraphError::Storage(format!("WAL fsync: {e}")))?;
        Ok(())
    }

    /// Write a single record to the BufWriter (no flush).
    #[inline]
    fn write_record(&mut self, entry: &GraphWalEntry) -> Result<(), GraphError> {
        let payload = bincode::serialize(entry)?;
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);

        self.writer.write_all(&MAGIC.to_le_bytes())
            .map_err(|e| GraphError::Storage(format!("WAL write magic: {e}")))?;
        self.writer.write_all(&len.to_le_bytes())
            .map_err(|e| GraphError::Storage(format!("WAL write len: {e}")))?;
        self.writer.write_all(&crc.to_le_bytes())
            .map_err(|e| GraphError::Storage(format!("WAL write crc: {e}")))?;
        self.writer.write_all(&payload)
            .map_err(|e| GraphError::Storage(format!("WAL write payload: {e}")))?;
        Ok(())
    }

    /// Replay all valid entries from the WAL file.
    /// Stops at the first corrupted record (truncated tail).
    pub fn replay(dir: &Path) -> Result<Vec<GraphWalEntry>, GraphError> {
        let path = dir.join("graph.wal");
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path)
            .map_err(|e| GraphError::Storage(format!("WAL replay open: {e}")))?;
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Read magic
            let mut magic_buf = [0u8; 4];
            match reader.read_exact(&mut magic_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(GraphError::Storage(format!("WAL read magic: {e}"))),
            }
            if u32::from_le_bytes(magic_buf) != MAGIC {
                break; // corrupted
            }

            // Read len
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf);
            if len > MAX_WAL_RECORD_SIZE {
                break; // corrupted — unreasonable payload size
            }
            let len = len as usize;

            // Read CRC
            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() { break; }
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Read payload
            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() { break; }

            // Verify CRC
            if crc32fast::hash(&payload) != stored_crc {
                break; // corrupted tail
            }

            match bincode::deserialize::<GraphWalEntry>(&payload) {
                Ok(entry) => entries.push(entry),
                Err(_) => break, // unknown entry type — corrupted tail
            }
        }

        Ok(entries)
    }

    /// Number of valid records currently in the WAL (replay-count).
    pub fn record_count(dir: &Path) -> Result<usize, GraphError> {
        Ok(Self::replay(dir)?.len())
    }

    // ── Internal ───────────────────────────────────────

    /// Read all records, find last valid offset, truncate file there.
    fn truncate_corrupted_tail(path: &Path) -> Result<(), GraphError> {
        let file = File::open(path)
            .map_err(|e| GraphError::Storage(format!("WAL integrity check open: {e}")))?;
        let mut reader = BufReader::new(file);
        let mut last_good_offset: u64 = 0;
        let mut offset: u64 = 0;

        loop {
            let mut magic_buf = [0u8; 4];
            match reader.read_exact(&mut magic_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(_) => break,
            }
            if u32::from_le_bytes(magic_buf) != MAGIC { break; }

            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf);
            if len > MAX_WAL_RECORD_SIZE { break; }
            let len = len as usize;

            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() { break; }
            let stored_crc = u32::from_le_bytes(crc_buf);

            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() { break; }

            if crc32fast::hash(&payload) != stored_crc { break; }
            if bincode::deserialize::<GraphWalEntry>(&payload).is_err() { break; }

            offset += 4 + 4 + 4 + len as u64;
            last_good_offset = offset;
        }

        // Truncate to the last good record boundary
        let file = OpenOptions::new().write(true).open(path)
            .map_err(|e| GraphError::Storage(format!("WAL truncate open: {e}")))?;
        file.set_len(last_good_offset)
            .map_err(|e| GraphError::Storage(format!("WAL truncate: {e}")))?;

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

    fn make_node() -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.1, 0.2]),
            serde_json::json!({}),
        )
    }

    #[test]
    fn append_and_replay_insert_node() {
        let dir = tmp();
        let node = make_node();
        let id = node.id;

        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            wal.append(&GraphWalEntry::InsertNode(node)).unwrap();
        }

        let entries = GraphWal::replay(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        if let GraphWalEntry::InsertNode(n) = &entries[0] {
            assert_eq!(n.id, id);
        } else {
            panic!("expected InsertNode");
        }
    }

    #[test]
    fn replay_multiple_entry_types() {
        let dir = tmp();
        let node = make_node();
        let edge = Edge::association(Uuid::new_v4(), Uuid::new_v4(), 0.5);
        let tx_id = Uuid::new_v4();

        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            wal.append(&GraphWalEntry::InsertNode(node.clone())).unwrap();
            wal.append(&GraphWalEntry::InsertEdge(edge.clone())).unwrap();
            wal.append(&GraphWalEntry::UpdateNodeEnergy { node_id: node.id, energy: 0.5 }).unwrap();
            wal.append(&GraphWalEntry::TxCommitted(tx_id)).unwrap();
        }

        let entries = GraphWal::replay(dir.path()).unwrap();
        assert_eq!(entries.len(), 4);
        assert!(matches!(entries[0], GraphWalEntry::InsertNode(_)));
        assert!(matches!(entries[1], GraphWalEntry::InsertEdge(_)));
        assert!(matches!(entries[2], GraphWalEntry::UpdateNodeEnergy { .. }));
        assert!(matches!(entries[3], GraphWalEntry::TxCommitted(_)));
    }

    #[test]
    fn empty_wal_replay_returns_empty() {
        let dir = tmp();
        let entries = GraphWal::replay(dir.path()).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn corrupted_tail_is_truncated() {
        let dir = tmp();

        // Write two valid entries then corrupt the file
        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            wal.append(&GraphWalEntry::InsertNode(make_node())).unwrap();
            wal.append(&GraphWalEntry::InsertNode(make_node())).unwrap();
        }

        // Append garbage bytes
        let wal_path = dir.path().join("graph.wal");
        let mut f = OpenOptions::new().append(true).open(&wal_path).unwrap();
        f.write_all(b"\xDE\xAD\xBE\xEF\x00\x01\x02\x03").unwrap();
        drop(f);

        // Open triggers tail truncation
        let _ = GraphWal::open(dir.path()).unwrap();

        // After opening, replay should still return only the 2 valid entries
        let entries = GraphWal::replay(dir.path()).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn survives_reopen_appends_more() {
        let dir = tmp();
        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            wal.append(&GraphWalEntry::InsertNode(make_node())).unwrap();
        }
        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            wal.append(&GraphWalEntry::InsertNode(make_node())).unwrap();
        }
        let entries = GraphWal::replay(dir.path()).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn record_count_matches_appends() {
        let dir = tmp();
        {
            let mut wal = GraphWal::open(dir.path()).unwrap();
            for _ in 0..5 {
                wal.append(&GraphWalEntry::PruneNode(Uuid::new_v4())).unwrap();
            }
        }
        assert_eq!(GraphWal::record_count(dir.path()).unwrap(), 5);
    }
}
