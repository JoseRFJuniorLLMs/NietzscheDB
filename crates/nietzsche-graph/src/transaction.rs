//! ACID transaction coordinator for NietzscheDB.
//!
//! Implements a **Saga pattern** spanning RocksDB (graph) and the vector
//! store, with WAL-based atomicity and crash recovery.
//!
//! ## Commit protocol
//!
//! ```text
//! 1.  TxBegin(tx_id)          → WAL           (durable marker)
//! 2.  op₁ … opₙ               → WAL           (buffered ops logged)
//! 3.  TxCommitted(tx_id)      → WAL           (point of no return)
//! 4.  apply op₁ … opₙ        → RocksDB + adjacency + vector store
//! ```
//!
//! If the process crashes **before step 3**: the WAL has `TxBegin` but no
//! `TxCommitted` — no ops were applied, so no rollback is needed.
//!
//! If the process crashes **after step 3 but before step 4 completes**:
//! on next `NietzscheDB::open`, `recover_from_wal` detects the committed
//! transaction and re-applies its ops (all storage writes are idempotent).
//!
//! ## Rollback protocol
//!
//! ```text
//! 1.  TxRolledBack(tx_id)     → WAL
//! ```
//!
//! Because ops are buffered and never applied before commit, rollback is
//! a single WAL append — no undo logic required.

use uuid::Uuid;

use crate::db::{NietzscheDB, VectorStore};
use crate::error::GraphError;
use crate::model::{Edge, Node};
use crate::wal::GraphWalEntry;

// ─────────────────────────────────────────────
// TxOp — buffered operation
// ─────────────────────────────────────────────

/// A single operation buffered inside a [`Transaction`].
///
/// Ops are applied to storage **only** when [`Transaction::commit`] is called.
#[derive(Debug, Clone)]
pub enum TxOp {
    InsertNode(Node),
    InsertEdge(Edge),
    PruneNode(Uuid),
    DeleteNode(Uuid),
    DeleteEdge(Uuid),
    UpdateEnergy { node_id: Uuid, energy: f32 },
}

impl TxOp {
    /// Convert to the corresponding [`GraphWalEntry`] for WAL logging.
    pub(crate) fn to_wal_entry(&self) -> GraphWalEntry {
        match self {
            Self::InsertNode(n) => GraphWalEntry::InsertNode(n.clone()),
            Self::InsertEdge(e) => GraphWalEntry::InsertEdge(e.clone()),
            Self::PruneNode(id) => GraphWalEntry::PruneNode(*id),
            Self::DeleteNode(id) => GraphWalEntry::DeleteNode(*id),
            Self::DeleteEdge(id) => GraphWalEntry::DeleteEdge(*id),
            Self::UpdateEnergy { node_id, energy } => {
                GraphWalEntry::UpdateNodeEnergy { node_id: *node_id, energy: *energy }
            }
        }
    }
}

// ─────────────────────────────────────────────
// TxReport
// ─────────────────────────────────────────────

/// Statistics returned by a successful [`Transaction::commit`].
#[derive(Debug, Clone)]
pub struct TxReport {
    pub tx_id:          Uuid,
    pub nodes_inserted: usize,
    pub edges_inserted: usize,
    pub nodes_pruned:   usize,
    pub nodes_deleted:  usize,
    pub edges_deleted:  usize,
    /// Total ops applied (includes `UpdateEnergy` and others).
    pub ops_applied:    usize,
}

impl TxReport {
    pub fn new(tx_id: Uuid) -> Self {
        Self {
            tx_id,
            nodes_inserted: 0,
            edges_inserted: 0,
            nodes_pruned:   0,
            nodes_deleted:  0,
            edges_deleted:  0,
            ops_applied:    0,
        }
    }
}

// ─────────────────────────────────────────────
// TxError
// ─────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum TxError {
    #[error("graph error during transaction: {0}")]
    Graph(#[from] GraphError),

    #[error("transaction {0} was already completed")]
    AlreadyCompleted(Uuid),
}

// ─────────────────────────────────────────────
// Transaction
// ─────────────────────────────────────────────

/// An ACID transaction over [`NietzscheDB`].
///
/// Obtained via [`NietzscheDB::begin_transaction`].
///
/// All mutations are **buffered in memory** until [`commit`](Transaction::commit)
/// or [`rollback`](Transaction::rollback) is called.
///
/// Dropping a `Transaction` without committing does not write to storage,
/// but also does not write `TxRolledBack` to the WAL — call
/// [`rollback`](Transaction::rollback) explicitly for a clean WAL trail.
pub struct Transaction<'a, V: VectorStore> {
    /// Unique ID for this transaction, written to WAL at begin/commit/rollback.
    pub tx_id: Uuid,
    /// Mutable reference to the database — exclusively held during the transaction.
    db:  &'a mut NietzscheDB<V>,
    /// Buffered operations — applied atomically on commit.
    ops: Vec<TxOp>,
}

impl<'a, V: VectorStore> Transaction<'a, V> {
    pub(crate) fn new(tx_id: Uuid, db: &'a mut NietzscheDB<V>) -> Self {
        Self { tx_id, db, ops: Vec::new() }
    }

    // ── Buffer operations ─────────────────────────────

    /// Buffer a node insertion.
    pub fn insert_node(&mut self, node: Node) {
        self.ops.push(TxOp::InsertNode(node));
    }

    /// Buffer an edge insertion.
    pub fn insert_edge(&mut self, edge: Edge) {
        self.ops.push(TxOp::InsertEdge(edge));
    }

    /// Buffer a soft-delete (prune) of a node.
    pub fn prune_node(&mut self, id: Uuid) {
        self.ops.push(TxOp::PruneNode(id));
    }

    /// Buffer a hard-delete of a node.
    pub fn delete_node(&mut self, id: Uuid) {
        self.ops.push(TxOp::DeleteNode(id));
    }

    /// Buffer an edge hard-delete.
    pub fn delete_edge(&mut self, id: Uuid) {
        self.ops.push(TxOp::DeleteEdge(id));
    }

    /// Buffer an energy update for a node.
    pub fn update_energy(&mut self, node_id: Uuid, energy: f32) {
        self.ops.push(TxOp::UpdateEnergy { node_id, energy });
    }

    /// Number of buffered (unapplied) operations.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    // ── Commit ────────────────────────────────────────

    /// Commit all buffered operations.
    ///
    /// ## WAL protocol
    /// 1. `TxBegin(tx_id)` → WAL
    /// 2. Each op as `GraphWalEntry` → WAL
    /// 3. `TxCommitted(tx_id)` → WAL  ← point of no return
    /// 4. Apply each op to RocksDB + adjacency + vector store
    ///
    /// Steps 1–3 are crash-safe (WAL is flushed after each append).
    /// If the process crashes after step 3, `recover_from_wal` will
    /// re-apply the ops on next startup.
    pub fn commit(self) -> Result<TxReport, TxError> {
        let tx_id = self.tx_id;
        let ops   = self.ops;
        let db    = self.db;

        // ── Step 1: TxBegin ───────────────────────────
        db.wal_append(&GraphWalEntry::TxBegin(tx_id))?;

        // ── Step 2: log all ops ───────────────────────
        for op in &ops {
            db.wal_append(&op.to_wal_entry())?;
        }

        // ── Step 3: TxCommitted (point of no return) ──
        db.wal_append(&GraphWalEntry::TxCommitted(tx_id))?;

        // ── Step 4: apply to storage ──────────────────
        let mut report = TxReport::new(tx_id);
        for op in ops {
            apply_op(db, op, &mut report)?;
        }

        Ok(report)
    }

    // ── Rollback ──────────────────────────────────────

    /// Discard all buffered operations.
    ///
    /// Writes `TxRolledBack(tx_id)` to the WAL for audit trail.
    /// Since no ops were ever applied to storage, no undo is needed.
    pub fn rollback(self) -> Result<(), TxError> {
        self.db.wal_append(&GraphWalEntry::TxRolledBack(self.tx_id))?;
        Ok(())
    }
}

// ─────────────────────────────────────────────
// Internal: apply one op (no WAL write — already logged by commit)
// ─────────────────────────────────────────────

fn apply_op<V: VectorStore>(
    db:     &mut NietzscheDB<V>,
    op:     TxOp,
    report: &mut TxReport,
) -> Result<(), TxError> {
    report.ops_applied += 1;
    match op {
        TxOp::InsertNode(node) => {
            db.apply_insert_node(&node)?;
            report.nodes_inserted += 1;
        }
        TxOp::InsertEdge(edge) => {
            db.apply_insert_edge(&edge);
            report.edges_inserted += 1;
        }
        TxOp::PruneNode(id) => {
            db.apply_prune_node(id)?;
            report.nodes_pruned += 1;
        }
        TxOp::DeleteNode(id) => {
            db.apply_delete_node(id)?;
            report.nodes_deleted += 1;
        }
        TxOp::DeleteEdge(id) => {
            db.apply_delete_edge(id)?;
            report.edges_deleted += 1;
        }
        TxOp::UpdateEnergy { node_id, energy } => {
            db.apply_update_energy(node_id, energy)?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Edge, EdgeType, PoincareVector};
    use crate::db::MockVectorStore;
    use tempfile::TempDir;

    fn open_db(dir: &TempDir) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap()
    }

    fn node(x: f64) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, 0.0]),
            serde_json::json!({}),
        )
    }

    // ── Commit ────────────────────────────────

    #[test]
    fn commit_applies_all_ops() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.1);
        let b = node(0.3);
        let id_a = a.id;
        let id_b = b.id;
        let edge = Edge::new(id_a, id_b, EdgeType::Association, 0.8);

        let mut tx = db.begin_transaction();
        tx.insert_node(a);
        tx.insert_node(b);
        tx.insert_edge(edge);

        let report = tx.commit().unwrap();

        // Ops reported correctly
        assert_eq!(report.nodes_inserted, 2);
        assert_eq!(report.edges_inserted, 1);
        assert_eq!(report.ops_applied,    3);

        // Data persisted
        assert!(db.get_node(id_a).unwrap().is_some());
        assert!(db.get_node(id_b).unwrap().is_some());
        assert!(db.neighbors_out(id_a).contains(&id_b));
    }

    #[test]
    fn commit_empty_transaction_succeeds() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let tx = db.begin_transaction();
        let report = tx.commit().unwrap();

        assert_eq!(report.ops_applied, 0);
    }

    // ── Rollback ──────────────────────────────

    #[test]
    fn rollback_applies_nothing() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.2);
        let id_a = a.id;

        let mut tx = db.begin_transaction();
        tx.insert_node(a);
        assert_eq!(tx.op_count(), 1);

        tx.rollback().unwrap();

        // Node was NOT inserted
        assert!(db.get_node(id_a).unwrap().is_none());
    }

    #[test]
    fn rollback_after_multiple_ops() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.1);
        let b = node(0.2);
        let c = node(0.3);
        let ids = [a.id, b.id, c.id];

        let mut tx = db.begin_transaction();
        tx.insert_node(a);
        tx.insert_node(b);
        tx.insert_node(c);
        tx.rollback().unwrap();

        // None of the nodes were inserted
        for id in &ids {
            assert!(db.get_node(*id).unwrap().is_none(), "node {id} should not exist after rollback");
        }
    }

    // ── Mixed commit + rollback ────────────────

    #[test]
    fn two_transactions_independent() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        // Tx1: commits
        let a = node(0.1);
        let id_a = a.id;
        let mut tx1 = db.begin_transaction();
        tx1.insert_node(a);
        tx1.commit().unwrap();

        // Tx2: rolls back
        let b = node(0.9);
        let id_b = b.id;
        let mut tx2 = db.begin_transaction();
        tx2.insert_node(b);
        tx2.rollback().unwrap();

        assert!(db.get_node(id_a).unwrap().is_some(), "tx1 commit should persist");
        assert!(db.get_node(id_b).unwrap().is_none(), "tx2 rollback should not persist");
    }

    // ── WAL trail ─────────────────────────────

    #[test]
    fn commit_writes_txbegin_and_txcommitted_to_wal() {
        use crate::wal::GraphWal;

        let dir = TempDir::new().unwrap();
        {
            let mut db = open_db(&dir);
            let mut tx = db.begin_transaction();
            tx.insert_node(node(0.1));
            tx.commit().unwrap();
        }

        let entries = GraphWal::replay(dir.path()).unwrap();
        let has_begin     = entries.iter().any(|e| matches!(e, GraphWalEntry::TxBegin(_)));
        let has_committed = entries.iter().any(|e| matches!(e, GraphWalEntry::TxCommitted(_)));
        assert!(has_begin,     "WAL should contain TxBegin");
        assert!(has_committed, "WAL should contain TxCommitted");
    }

    #[test]
    fn rollback_writes_txrolledback_to_wal() {
        use crate::wal::GraphWal;

        let dir = TempDir::new().unwrap();
        {
            let mut db = open_db(&dir);
            let mut tx = db.begin_transaction();
            tx.insert_node(node(0.5));
            tx.rollback().unwrap();
        }

        let entries = GraphWal::replay(dir.path()).unwrap();
        let has_rolled_back = entries.iter()
            .any(|e| matches!(e, GraphWalEntry::TxRolledBack(_)));
        assert!(has_rolled_back, "WAL should contain TxRolledBack");

        // A TxBegin is NOT written on rollback (ops were buffered, never logged)
        let has_begin = entries.iter().any(|e| matches!(e, GraphWalEntry::TxBegin(_)));
        assert!(!has_begin, "WAL should NOT contain TxBegin for a rolled-back transaction");
    }

    // ── Prune inside transaction ───────────────

    #[test]
    fn prune_inside_transaction() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        // Insert a node outside a transaction first
        let a = node(0.2);
        let id_a = a.id;
        db.insert_node(a).unwrap();

        // Prune it inside a transaction
        let mut tx = db.begin_transaction();
        tx.prune_node(id_a);
        tx.commit().unwrap();

        let found = db.get_node(id_a).unwrap().unwrap();
        assert_eq!(found.energy, 0.0, "pruned node should have energy=0");
    }

    // ── Energy update ──────────────────────────

    #[test]
    fn update_energy_inside_transaction() {
        let dir = TempDir::new().unwrap();
        let mut db = open_db(&dir);

        let a = node(0.3);
        let id_a = a.id;
        db.insert_node(a).unwrap();

        let mut tx = db.begin_transaction();
        tx.update_energy(id_a, 0.55);
        tx.commit().unwrap();

        let found = db.get_node(id_a).unwrap().unwrap();
        assert!((found.energy - 0.55).abs() < 1e-6);
    }

    // ── Crash recovery ────────────────────────

    #[test]
    fn committed_data_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let node_id;

        // Commit a transaction
        {
            let mut db = open_db(&dir);
            let a = node(0.15);
            node_id = a.id;
            let mut tx = db.begin_transaction();
            tx.insert_node(a);
            tx.commit().unwrap();
        }

        // Re-open (simulates restart)
        let db2 = open_db(&dir);
        assert!(
            db2.get_node(node_id).unwrap().is_some(),
            "node should survive db reopen after committed transaction"
        );
    }

    #[test]
    fn rolled_back_data_absent_after_reopen() {
        let dir = TempDir::new().unwrap();
        let node_id;

        {
            let mut db = open_db(&dir);
            let a = node(0.15);
            node_id = a.id;
            let mut tx = db.begin_transaction();
            tx.insert_node(a);
            tx.rollback().unwrap();
        }

        let db2 = open_db(&dir);
        assert!(
            db2.get_node(node_id).unwrap().is_none(),
            "rolled-back node should be absent after reopen"
        );
    }
}
