// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
// SPDX-License-Identifier: AGPL-3.0-or-later

//! GlueSQL storage backend for NietzscheDB.
//!
//! Implements GlueSQL's `Store` and `StoreMut` traits on top of RocksDB
//! column families (`sql_schema` and `sql_data`), sharing the same RocksDB
//! instance used by the graph engine. Atomic backups via `Checkpoint::new()`
//! include all SQL tables automatically.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use gluesql::core::data::Schema;
use gluesql::core::store::{
    AlterTable, CustomFunction, CustomFunctionMut, DataRow, Index, IndexMut,
    Metadata, RowIter, Store, StoreMut, Transaction,
};
use gluesql::prelude::{Key, Result};
use nietzsche_graph::storage::GraphStorage;
use tracing::debug;

/// Persisted metadata for a table: the GlueSQL Schema plus an auto-increment counter.
#[derive(serde::Serialize, serde::Deserialize)]
struct TableMeta {
    schema_json: String,
    next_row_id: u64,
}

/// GlueSQL storage backend backed by NietzscheDB's RocksDB column families.
///
/// Two CFs are used:
/// - `sql_schema` — key: table_name → `TableMeta` (JSON)
/// - `sql_data`   — key: `table_name\x00row_id_be(8B)` → `DataRow` (JSON)
pub struct SwartzStore {
    pub(crate) storage: Arc<GraphStorage>,
}

impl SwartzStore {
    pub fn new(storage: Arc<GraphStorage>) -> Self {
        Self { storage }
    }

    fn load_meta(&self, table_name: &str) -> Result<Option<TableMeta>> {
        match self.storage.get_sql_schema(table_name) {
            Ok(Some(bytes)) => {
                let meta: TableMeta = serde_json::from_slice(&bytes)
                    .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
                Ok(Some(meta))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(gluesql::prelude::Error::StorageMsg(e.to_string())),
        }
    }

    fn save_meta(&self, table_name: &str, meta: &TableMeta) -> Result<()> {
        let bytes = serde_json::to_vec(meta)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
        self.storage
            .put_sql_schema(table_name, &bytes)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))
    }

    fn next_row_id(&self, table_name: &str) -> Result<u64> {
        let mut meta = self
            .load_meta(table_name)?
            .ok_or_else(|| {
                gluesql::prelude::Error::StorageMsg(format!("table '{}' not found", table_name))
            })?;
        let id = meta.next_row_id;
        meta.next_row_id += 1;
        self.save_meta(table_name, &meta)?;
        Ok(id)
    }

    fn serialize_row(row: &DataRow) -> Result<Vec<u8>> {
        serde_json::to_vec(row)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))
    }

    fn deserialize_row(bytes: &[u8]) -> Result<DataRow> {
        serde_json::from_slice(bytes)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))
    }

    /// Convert a Key to a u64 row_id for RocksDB storage.
    fn key_to_row_id(key: &Key) -> u64 {
        match key {
            Key::I64(v) => *v as u64,
            Key::U64(v) => *v,
            Key::I32(v) => *v as u64,
            Key::U32(v) => *v as u64,
            Key::I16(v) => *v as u64,
            Key::I8(v) => *v as u64,
            _ => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                format!("{:?}", key).hash(&mut hasher);
                hasher.finish()
            }
        }
    }

    fn row_id_to_key(row_id: u64) -> Key {
        Key::I64(row_id as i64)
    }
}

// ── GlueSQL Store trait (read operations) ────────────────────────────────

#[async_trait]
impl Store for SwartzStore {
    async fn fetch_all_schemas(&self) -> Result<Vec<Schema>> {
        let items = self
            .storage
            .scan_sql_schemas()
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;

        let mut schemas = Vec::with_capacity(items.len());
        for (_name, bytes) in items {
            let meta: TableMeta = serde_json::from_slice(&bytes)
                .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
            let schema: Schema = serde_json::from_str(&meta.schema_json)
                .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
            schemas.push(schema);
        }
        Ok(schemas)
    }

    async fn fetch_schema(&self, table_name: &str) -> Result<Option<Schema>> {
        match self.load_meta(table_name)? {
            Some(meta) => {
                let schema: Schema = serde_json::from_str(&meta.schema_json)
                    .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
                Ok(Some(schema))
            }
            None => Ok(None),
        }
    }

    async fn fetch_data(&self, table_name: &str, key: &Key) -> Result<Option<DataRow>> {
        let row_id = Self::key_to_row_id(key);
        match self.storage.get_sql_row(table_name, row_id) {
            Ok(Some(bytes)) => {
                let row = Self::deserialize_row(&bytes)?;
                Ok(Some(row))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(gluesql::prelude::Error::StorageMsg(e.to_string())),
        }
    }

    async fn scan_data<'a>(&'a self, table_name: &str) -> Result<RowIter<'a>> {
        let rows = self
            .storage
            .scan_sql_table(table_name)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;

        let items: Vec<Result<(Key, DataRow)>> = rows
            .into_iter()
            .map(|(row_id, bytes)| {
                let row = Self::deserialize_row(&bytes)?;
                let key = Self::row_id_to_key(row_id);
                Ok((key, row))
            })
            .collect();

        Ok(Box::pin(stream::iter(items)))
    }
}

// ── GlueSQL StoreMut trait (write operations) ────────────────────────────

#[async_trait]
impl StoreMut for SwartzStore {
    async fn insert_schema(&mut self, schema: &Schema) -> Result<()> {
        let table_name = schema.table_name.clone();
        let schema_json = serde_json::to_string(schema)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;

        let meta = TableMeta {
            schema_json,
            next_row_id: 1,
        };
        self.save_meta(&table_name, &meta)?;
        debug!(table = %table_name, "[Swartz] Schema created");
        Ok(())
    }

    async fn delete_schema(&mut self, table_name: &str) -> Result<()> {
        let deleted = self
            .storage
            .delete_sql_table_rows(table_name)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;

        self.storage
            .delete_sql_schema(table_name)
            .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;

        debug!(table = %table_name, rows_deleted = deleted, "[Swartz] Schema dropped");
        Ok(())
    }

    async fn append_data(&mut self, table_name: &str, rows: Vec<DataRow>) -> Result<()> {
        for row in rows {
            let row_id = self.next_row_id(table_name)?;
            let bytes = Self::serialize_row(&row)?;
            self.storage
                .put_sql_row(table_name, row_id, &bytes)
                .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
        }
        Ok(())
    }

    async fn insert_data(&mut self, table_name: &str, rows: Vec<(Key, DataRow)>) -> Result<()> {
        for (key, row) in rows {
            let row_id = Self::key_to_row_id(&key);
            let bytes = Self::serialize_row(&row)?;
            self.storage
                .put_sql_row(table_name, row_id, &bytes)
                .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
        }
        Ok(())
    }

    async fn delete_data(&mut self, table_name: &str, keys: Vec<Key>) -> Result<()> {
        for key in keys {
            let row_id = Self::key_to_row_id(&key);
            self.storage
                .delete_sql_row(table_name, row_id)
                .map_err(|e| gluesql::prelude::Error::StorageMsg(e.to_string()))?;
        }
        Ok(())
    }
}

// ── Empty/default trait impls required by Glue<T> ────────────────────────

impl AlterTable for SwartzStore {}
impl Transaction for SwartzStore {}
impl CustomFunction for SwartzStore {}
impl CustomFunctionMut for SwartzStore {}
impl Metadata for SwartzStore {}
impl Index for SwartzStore {}
impl IndexMut for SwartzStore {}
impl gluesql::core::store::Planner for SwartzStore {}
