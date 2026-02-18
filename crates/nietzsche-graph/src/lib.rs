//! # nietzsche-graph
//!
//! Hyperbolic graph engine for NietzscheDB.
//!
//! Provides the core data model, persistent storage, and in-memory index
//! for the knowledge graph:
//!
//! - [`model::PoincareVector`]     — points in the Poincaré ball (‖x‖ < 1.0)
//! - [`model::Node`]               — hyperbolic graph node with energy + Hausdorff fields
//! - [`model::Edge`]               — directed typed edge
//! - [`adjacency::AdjacencyIndex`] — lock-free bidirectional adjacency index
//! - [`storage::GraphStorage`]     — RocksDB persistence (5 column families)
//! - [`wal::GraphWal`]             — binary append-only Write-Ahead Log
//! - [`db::NietzscheDB`]           — dual-write coordinator (graph + vector store)
//! - [`db::VectorStore`]           — trait abstraction over HyperspaceDB / mock
//! - [`transaction::Transaction`]  — ACID saga transaction (Phase 7)

pub mod adjacency;
pub mod db;
pub mod error;
pub mod model;
pub mod storage;
pub mod transaction;
pub mod traversal;
pub mod wal;

pub use adjacency::AdjacencyIndex;
pub use db::{MockVectorStore, NietzscheDB, VectorStore};
pub use error::GraphError;
pub use model::{Edge, EdgeType, Node, NodeType, PoincareVector};
pub use storage::{GraphStorage, NodeIterator, EdgeIterator};
pub use transaction::{Transaction, TxError, TxOp, TxReport};
pub use traversal::{
    bfs, diffusion_walk, dijkstra, shortest_path, BfsConfig, DiffusionConfig, DijkstraConfig,
};
pub use wal::{GraphWal, GraphWalEntry};
