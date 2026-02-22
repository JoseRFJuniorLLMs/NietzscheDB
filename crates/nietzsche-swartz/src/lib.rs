// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Swartz — Embedded SQL Layer for NietzscheDB
//!
//! Named in honor of **Aaron Swartz** (1986–2013), co-creator of RSS,
//! co-founder of Reddit, and tireless advocate for open access to information.
//!
//! The Swartz layer adds a full embedded relational SQL engine to NietzscheDB
//! using GlueSQL on the **same RocksDB instance** as the graph engine.
//!
//! ## Architecture
//!
//! ```text
//! SwartzEngine
//!   └── Glue<SwartzStore>
//!         └── SwartzStore (GlueSQL Store/StoreMut traits)
//!               └── GraphStorage (shared RocksDB via Arc<DB>)
//!                     ├── CF sql_schema — table definitions + auto-inc counters
//!                     └── CF sql_data   — row data (table_name\x00row_id → JSON)
//! ```
//!
//! ## Benefits
//!
//! - **Atomic backups**: `Checkpoint::new()` includes SQL tables automatically
//! - **Zero external deps**: no PostgreSQL/SQLite process needed
//! - **Graph↔SQL bridge**: `NodeRef` columns reference graph node UUIDs
//! - **Full SQL**: SELECT, INSERT, UPDATE, DELETE, JOIN, WHERE, GROUP BY, ORDER BY
//!
//! ## Usage
//!
//! ```ignore
//! use nietzsche_swartz::SwartzEngine;
//!
//! let engine = SwartzEngine::new(storage_arc);
//! engine.execute("CREATE TABLE moods (id INTEGER, label TEXT, score FLOAT)").await?;
//! engine.execute("INSERT INTO moods VALUES (1, 'calm', 0.85)").await?;
//! let rows = engine.query("SELECT * FROM moods WHERE score > 0.5").await?;
//! ```

pub mod engine;
pub mod error;
pub mod store;

pub use engine::SwartzEngine;
pub use error::SwartzError;
pub use store::SwartzStore;
