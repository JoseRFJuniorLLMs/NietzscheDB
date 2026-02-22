// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! High-level SQL engine for NietzscheDB's Swartz layer.
//!
//! Wraps GlueSQL's `Glue<SwartzStore>` to provide a simple API for
//! executing SQL statements and queries against RocksDB-backed tables.

use std::sync::Arc;

use gluesql::prelude::{Glue, Payload, Value};
use nietzsche_graph::storage::GraphStorage;
use tracing::debug;

use crate::error::SwartzError;
use crate::store::SwartzStore;

/// The Swartz SQL engine — an embedded relational database inside NietzscheDB.
///
/// Named in honor of Aaron Swartz, advocate for open access to information.
///
/// # Example
/// ```ignore
/// let engine = SwartzEngine::new(storage_arc);
/// engine.execute("CREATE TABLE moods (id INTEGER, label TEXT, score FLOAT)").await?;
/// engine.execute("INSERT INTO moods VALUES (1, 'calm', 0.85)").await?;
/// let rows = engine.query("SELECT * FROM moods WHERE score > 0.5").await?;
/// ```
pub struct SwartzEngine {
    glue: Glue<SwartzStore>,
}

impl SwartzEngine {
    /// Create a new Swartz engine backed by the given RocksDB GraphStorage.
    ///
    /// The engine shares the same RocksDB instance as the graph — no data
    /// duplication, atomic backups included.
    pub fn new(storage: Arc<GraphStorage>) -> Self {
        let store = SwartzStore::new(storage);
        Self {
            glue: Glue::new(store),
        }
    }

    /// Execute a SQL statement (CREATE TABLE, INSERT, UPDATE, DELETE, DROP TABLE).
    ///
    /// Returns the GlueSQL payloads describing what happened.
    pub async fn execute(&mut self, sql: &str) -> Result<Vec<Payload>, SwartzError> {
        debug!(sql = %sql, "[Swartz] Executing SQL");
        self.glue
            .execute(sql)
            .await
            .map_err(|e| SwartzError::Glue(e.to_string()))
    }

    /// Execute a SQL query (SELECT) and return rows as JSON values.
    ///
    /// Each row is a JSON object with column names as keys.
    pub async fn query(&mut self, sql: &str) -> Result<Vec<serde_json::Value>, SwartzError> {
        debug!(sql = %sql, "[Swartz] Querying SQL");
        let payloads = self
            .glue
            .execute(sql)
            .await
            .map_err(|e| SwartzError::Glue(e.to_string()))?;

        let mut results = Vec::new();
        for payload in payloads {
            match payload {
                Payload::Select { labels, rows } => {
                    for row in rows {
                        let mut obj = serde_json::Map::new();
                        for (label, value) in labels.iter().zip(row.iter()) {
                            obj.insert(label.clone(), glue_value_to_json(value));
                        }
                        results.push(serde_json::Value::Object(obj));
                    }
                }
                _ => {
                    // Non-SELECT payloads (e.g. Create, Insert) are ignored in query results
                }
            }
        }
        Ok(results)
    }

    /// Execute a SQL statement and return the number of affected rows.
    ///
    /// Useful for INSERT, UPDATE, DELETE operations.
    pub async fn exec_count(&mut self, sql: &str) -> Result<u64, SwartzError> {
        let payloads = self.execute(sql).await?;
        let mut total = 0u64;
        for payload in &payloads {
            match payload {
                Payload::Insert(n) => total += *n as u64,
                Payload::Delete(n) => total += *n as u64,
                Payload::Update(n) => total += *n as u64,
                Payload::Create => total += 1,
                Payload::DropTable(_) => total += 1,
                Payload::AlterTable => total += 1,
                _ => {}
            }
        }
        Ok(total)
    }

    /// List all SQL tables in this engine via direct storage scan.
    pub async fn list_tables(&self) -> Result<Vec<String>, SwartzError> {
        let items = self
            .glue
            .storage
            .storage
            .scan_sql_schemas()
            .map_err(SwartzError::Storage)?;
        Ok(items.into_iter().map(|(name, _)| name).collect())
    }
}

/// Convert a GlueSQL Value to a serde_json Value.
fn glue_value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::I8(n) => serde_json::json!(*n),
        Value::I16(n) => serde_json::json!(*n),
        Value::I32(n) => serde_json::json!(*n),
        Value::I64(n) => serde_json::json!(*n),
        Value::I128(n) => serde_json::json!(n.to_string()),
        Value::U8(n) => serde_json::json!(*n),
        Value::U16(n) => serde_json::json!(*n),
        Value::U32(n) => serde_json::json!(*n),
        Value::U64(n) => serde_json::json!(*n),
        Value::U128(n) => serde_json::json!(n.to_string()),
        Value::F32(n) => serde_json::json!(*n),
        Value::F64(n) => serde_json::json!(*n),
        Value::Str(s) => serde_json::Value::String(s.clone()),
        Value::Null => serde_json::Value::Null,
        other => serde_json::Value::String(format!("{:?}", other)),
    }
}
