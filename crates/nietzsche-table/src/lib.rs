//! # nietzsche-table
//!
//! Relational table store backed by SQLite for NietzscheDB.
//!
//! This crate provides an optional relational layer that coexists with
//! NietzscheDB's hyperbolic graph engine. Tables can reference graph nodes
//! through [`ColumnType::NodeRef`] columns, bridging the graph and relational
//! paradigms.
//!
//! ## Quick start
//!
//! ```no_run
//! use nietzsche_table::{TableStore, TableSchema, ColumnDef, ColumnType};
//!
//! let store = TableStore::open_memory().unwrap();
//!
//! let schema = TableSchema {
//!     name: "users".into(),
//!     columns: vec![
//!         ColumnDef { name: "id".into(), col_type: ColumnType::Integer, nullable: false, primary_key: true },
//!         ColumnDef { name: "name".into(), col_type: ColumnType::Text, nullable: false, primary_key: false },
//!         ColumnDef { name: "graph_node".into(), col_type: ColumnType::NodeRef, nullable: true, primary_key: false },
//!     ],
//! };
//!
//! store.create_table(&schema).unwrap();
//! ```

pub mod error;
pub mod schema;
pub mod store;

// Re-exports for convenience.
pub use error::TableError;
pub use schema::{ColumnDef, ColumnType, TableSchema};
pub use store::TableStore;

#[cfg(test)]
mod tests;
