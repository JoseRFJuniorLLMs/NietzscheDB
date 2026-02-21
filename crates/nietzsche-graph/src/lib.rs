//! # nietzsche-graph
//!
//! Hyperbolic graph engine for NietzscheDB.
//!
//! Provides the core data model, persistent storage, and in-memory index
//! for the knowledge graph:
//!
//! - [`model::PoincareVector`]     — points in the Poincaré ball (‖x‖ < 1.0)
//! - [`model::Node`]               — hyperbolic graph node with energy + Hausdorff fields
//! - [`model::NodeMeta`]           — lightweight node metadata (no embedding)
//! - [`model::Edge`]               — directed typed edge
//! - [`adjacency::AdjacencyIndex`] — lock-free bidirectional adjacency index
//! - [`storage::GraphStorage`]     — RocksDB persistence (6 column families)
//! - [`wal::GraphWal`]             — binary append-only Write-Ahead Log
//! - [`db::NietzscheDB`]           — dual-write coordinator (graph + vector store)
//! - [`db::VectorStore`]           — trait abstraction over HyperspaceDB / mock
//! - [`transaction::Transaction`]  — ACID saga transaction (Phase 7)
//!
//! # Architectural Decisions — Committee 2026-02-19
//!
//! ## REJECTED: Binary Quantization (ITEM F)
//!
//! Binary Quantization (`sign(x)` → 1-bit per coordinate) is **INCOMPATIBLE**
//! with the hyperbolic geometry of NietzscheDB. The magnitude `‖x‖` encodes
//! hierarchical position (center = abstract/semantic, boundary = specific/episodic).
//! `sign(x)` discards this magnitude, destroying the fundamental property of the
//! Poincaré ball that justifies the entire architecture.
//!
//! - **Decision:** REJECTED unanimously (Claude + Grok + Committee)
//! - **Reference:** `risco_hiperbolico.md`, PARTE 4 (sections 4.1–4.5)
//! - **Date:** 2026-02-19
//! - **Constraint:** NEVER implement as primary or secondary HNSW metric.
//! - **Pre-filter only:** If ever needed, require dim ≥ 1536, oversampling ≥ 30×,
//!   and mandatory rescore with exact hyperbolic distance.

pub mod adjacency;
pub mod backup;
pub mod collection_manager;
pub mod encryption;
pub mod db;
pub mod embedded_vector_store;
pub mod error;
pub mod fulltext;
pub mod import_export;
pub mod model;
pub mod schema;
pub mod storage;
pub mod transaction;
pub mod traversal;
pub mod wal;

pub use adjacency::{AdjacencyIndex, AdjEntry};
pub use collection_manager::{CollectionConfig, CollectionInfo, CollectionManager};
pub use db::{BackpressureSignal, MetadataFilter, MockVectorStore, NietzscheDB, VectorStore};
pub use embedded_vector_store::{AnyVectorStore, EmbeddedVectorStore, VectorMetric};
pub use error::GraphError;
pub use model::{Edge, EdgeType, Node, NodeMeta, NodeType, PoincareVector, SparseVector};
pub use storage::{GraphStorage, NodeIterator, NodeMetaIterator, EdgeIterator};
pub use transaction::{Transaction, TxError, TxOp, TxReport};
pub use traversal::{
    bfs, diffusion_walk, dijkstra, shortest_path, BfsConfig, DiffusionConfig, DijkstraConfig,
};
pub use wal::{GraphWal, GraphWalEntry};
pub use backup::{BackupManager, BackupInfo};
pub use encryption::EncryptionConfig;
pub use fulltext::{FullTextIndex, FtsResult};
pub use schema::{SchemaValidator, SchemaConstraint, FieldType};
pub use import_export::{
    ImportResult, export_nodes_csv, export_edges_csv,
    export_nodes_jsonl, export_edges_jsonl,
    import_nodes_jsonl, import_edges_jsonl,
};
