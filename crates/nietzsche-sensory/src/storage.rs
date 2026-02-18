//! RocksDB storage for sensory data.
//!
//! Adds the `sensory` column family to the existing NietzscheDB storage.
//! Key: node UUID (16 bytes). Value: `SensoryMemory` serialized with bincode.
//!
//! This CF is **lazy-loaded** — graph traversals never touch it. Only
//! explicit `RECONSTRUCT` queries or energy-degradation ticks read from here.

use crate::types::SensoryMemory;
use rocksdb::{ColumnFamily, DB};
use uuid::Uuid;

/// Name of the RocksDB column family for sensory data.
pub const CF_SENSORY: &str = "sensory";

/// Sensory storage operations on top of an existing RocksDB instance.
///
/// This does **not** own the DB — it borrows from `GraphStorage`.
/// The caller is responsible for ensuring the `sensory` CF exists
/// when opening the database (pass `CF_SENSORY` to `DB::open_cf`).
pub struct SensoryStorage<'a> {
    db: &'a DB,
}

impl<'a> SensoryStorage<'a> {
    /// Wrap an existing RocksDB handle.
    ///
    /// # Panics
    ///
    /// Panics if the `sensory` column family does not exist.
    pub fn new(db: &'a DB) -> Self {
        debug_assert!(
            db.cf_handle(CF_SENSORY).is_some(),
            "missing column family '{CF_SENSORY}' — add it to DB::open_cf"
        );
        Self { db }
    }

    /// Store sensory memory for a node, overwriting any previous value.
    pub fn put(&self, node_id: &Uuid, sensory: &SensoryMemory) -> Result<(), SensoryStorageError> {
        let cf = self.cf()?;
        let key = node_id.as_bytes();
        let value = bincode::serialize(sensory)
            .map_err(|e| SensoryStorageError::Serialize(e.to_string()))?;
        self.db
            .put_cf(cf, key, value)
            .map_err(|e| SensoryStorageError::Rocksdb(e.to_string()))
    }

    /// Load sensory memory for a node. Returns `None` if no sensory data exists.
    pub fn get(&self, node_id: &Uuid) -> Result<Option<SensoryMemory>, SensoryStorageError> {
        let cf = self.cf()?;
        let key = node_id.as_bytes();
        match self.db.get_cf(cf, key) {
            Ok(Some(bytes)) => {
                let sm: SensoryMemory = bincode::deserialize(&bytes)
                    .map_err(|e| SensoryStorageError::Deserialize(e.to_string()))?;
                Ok(Some(sm))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(SensoryStorageError::Rocksdb(e.to_string())),
        }
    }

    /// Delete sensory memory for a node (e.g., when the node is pruned).
    pub fn delete(&self, node_id: &Uuid) -> Result<(), SensoryStorageError> {
        let cf = self.cf()?;
        let key = node_id.as_bytes();
        self.db
            .delete_cf(cf, key)
            .map_err(|e| SensoryStorageError::Rocksdb(e.to_string()))
    }

    /// Check if a node has sensory data without deserializing.
    pub fn exists(&self, node_id: &Uuid) -> Result<bool, SensoryStorageError> {
        let cf = self.cf()?;
        let key = node_id.as_bytes();
        match self.db.get_pinned_cf(cf, key) {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(e) => Err(SensoryStorageError::Rocksdb(e.to_string())),
        }
    }

    /// Iterate over all sensory entries. Used by the L-System degradation tick.
    ///
    /// Calls `f(node_id, sensory)` for each entry. If `f` returns an updated
    /// `SensoryMemory`, the entry is overwritten in place.
    pub fn scan_and_update<F>(&self, mut f: F) -> Result<usize, SensoryStorageError>
    where
        F: FnMut(Uuid, &SensoryMemory) -> Option<SensoryMemory>,
    {
        let cf = self.cf()?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        let mut updated = 0;

        for item in iter {
            let (key, value) = item.map_err(|e| SensoryStorageError::Rocksdb(e.to_string()))?;

            if key.len() != 16 {
                continue; // skip malformed keys
            }

            let node_id = Uuid::from_bytes(
                key.as_ref()
                    .try_into()
                    .map_err(|_| SensoryStorageError::Deserialize("invalid UUID key".into()))?,
            );

            let sm: SensoryMemory = bincode::deserialize(&value)
                .map_err(|e| SensoryStorageError::Deserialize(e.to_string()))?;

            if let Some(new_sm) = f(node_id, &sm) {
                let new_value = bincode::serialize(&new_sm)
                    .map_err(|e| SensoryStorageError::Serialize(e.to_string()))?;
                self.db
                    .put_cf(cf, key.as_ref(), new_value)
                    .map_err(|e| SensoryStorageError::Rocksdb(e.to_string()))?;
                updated += 1;
            }
        }

        Ok(updated)
    }

    /// Total number of sensory entries.
    pub fn count(&self) -> Result<usize, SensoryStorageError> {
        let cf = self.cf()?;
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        let mut n = 0;
        for item in iter {
            let _ = item.map_err(|e| SensoryStorageError::Rocksdb(e.to_string()))?;
            n += 1;
        }
        Ok(n)
    }

    fn cf(&self) -> Result<&ColumnFamily, SensoryStorageError> {
        self.db
            .cf_handle(CF_SENSORY)
            .ok_or(SensoryStorageError::MissingCF)
    }
}

/// Errors from sensory storage operations.
#[derive(Debug, thiserror::Error)]
pub enum SensoryStorageError {
    #[error("RocksDB error: {0}")]
    Rocksdb(String),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("deserialization error: {0}")]
    Deserialize(String),

    #[error("missing column family '{CF_SENSORY}' — was it registered at DB open?")]
    MissingCF,
}
