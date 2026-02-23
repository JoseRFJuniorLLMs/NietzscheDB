//! Multi-collection namespace for NietzscheDB.
//!
//! Each collection has its own isolated:
//! - HNSW vector index (own dimension + metric)
//! - RocksDB graph storage (5 CFs)
//! - Write-Ahead Log
//!
//! ## Storage layout
//! ```text
//! {base_dir}/
//!   collections/
//!     default/
//!       collection.json      ← CollectionConfig (JSON-persisted)
//!       rocksdb/             ← RocksDB column families
//!       graph.wal            ← GraphWal
//!       hnsw/                ← HNSW mmap chunks
//!     memories/
//!       collection.json
//!       ...
//! ```
//!
//! ## Usage
//! ```rust,ignore
//! let cm = CollectionManager::open(Path::new("/data/nietzsche"))?;
//! cm.create_collection(CollectionConfig {
//!     name:   "memories".into(),
//!     dim:    3072,
//!     metric: "cosine".into(),
//! })?;
//! let db = cm.get_or_default("memories").unwrap();
//! let mut guard = db.write().await;
//! guard.insert_node(node)?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::db::NietzscheDB;
use crate::embedded_vector_store::{AnyVectorStore, VectorMetric};
use crate::error::GraphError;

// ── Internal shared-DB alias ────────────────────────────────────────────────

type SharedDb = Arc<RwLock<NietzscheDB<AnyVectorStore>>>;

// ── Config + Info types ─────────────────────────────────────────────────────

/// Persisted per-collection configuration (stored as `collection.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Collection name (used as directory name and lookup key).
    pub name: String,
    /// Vector dimension (e.g. 3072 for Gemini, 192 for ECAPA-TDNN, 768 for BGE-M3).
    pub dim: usize,
    /// Distance metric: `"cosine"` | `"euclidean"` | `"poincare"` (default `"cosine"`).
    pub metric: String,
}

impl CollectionConfig {
    /// Map the string metric name to the typed [`VectorMetric`].
    pub fn vector_metric(&self) -> VectorMetric {
        match self.metric.to_lowercase().as_str() {
            "euclidean" | "l2"                       => VectorMetric::Euclidean,
            "poincare"  | "hyperbolic"               => VectorMetric::PoincareBall,
            "dotproduct" | "dot" | "inner_product"   => VectorMetric::DotProduct,
            _                                        => VectorMetric::Cosine,
        }
    }
}

/// Runtime info for a collection (returned by [`CollectionManager::list`]).
#[derive(Debug, Clone)]
pub struct CollectionInfo {
    pub name:       String,
    pub dim:        usize,
    pub metric:     String,
    pub node_count: usize,
    pub edge_count: usize,
}

// ── CollectionManager ────────────────────────────────────────────────────────

/// Manages multiple named NietzscheDB instances.
///
/// Each collection lives under `{base_dir}/collections/{name}/`.
/// The `"default"` collection is always present.
///
/// ## Thread safety
/// `DashMap` provides lock-free concurrent map access.
/// Each individual collection DB is protected by `tokio::sync::RwLock`:
/// read-only operations (KNN, graph queries) can proceed concurrently;
/// mutations (insert/delete/sleep) take an exclusive write lock.
pub struct CollectionManager {
    base_dir:    PathBuf,
    collections: DashMap<String, (CollectionConfig, SharedDb)>,
    /// Serialises `create_collection` calls to eliminate the TOCTOU race
    /// between the `contains_key` check and the `insert`.
    create_lock: Mutex<()>,
}

impl CollectionManager {
    // ── Construction ──────────────────────────────────────────────────────

    /// Open or create the collection manager at `base_dir`.
    ///
    /// Scans `{base_dir}/collections/` for existing collection directories
    /// (identified by the presence of `collection.json`), opens each DB,
    /// then ensures a `"default"` collection exists.
    pub fn open(base_dir: &Path) -> Result<Arc<Self>, GraphError> {
        let collections_dir = base_dir.join("collections");
        std::fs::create_dir_all(&collections_dir).map_err(|e| {
            GraphError::Storage(format!(
                "cannot create collections dir {}: {e}",
                collections_dir.display()
            ))
        })?;

        let cm = Arc::new(Self {
            base_dir:    base_dir.to_path_buf(),
            collections: DashMap::new(),
            create_lock: Mutex::new(()),
        });

        // Load existing collections from disk
        if let Ok(entries) = std::fs::read_dir(&collections_dir) {
            for entry in entries.flatten() {
                let col_dir = entry.path();
                if !col_dir.is_dir() {
                    continue;
                }
                let cfg_path = col_dir.join("collection.json");
                if !cfg_path.exists() {
                    continue;
                }

                let json = std::fs::read_to_string(&cfg_path).map_err(|e| {
                    GraphError::Storage(format!("read {}: {e}", cfg_path.display()))
                })?;
                let cfg: CollectionConfig = serde_json::from_str(&json).map_err(|e| {
                    GraphError::Storage(format!("parse {}: {e}", cfg_path.display()))
                })?;

                let db = Self::open_db(&col_dir, &cfg)?;
                cm.collections
                    .insert(cfg.name.clone(), (cfg, Arc::new(RwLock::new(db))));
            }
        }

        // Ensure "default" always exists
        if !cm.collections.contains_key("default") {
            let default_dim: usize = std::env::var("NIETZSCHE_VECTOR_DIM")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3072);
            let default_cfg = CollectionConfig {
                name:   "default".to_string(),
                dim:    default_dim,
                metric: "cosine".to_string(),
            };
            cm.create_collection_inner(&default_cfg)?;
        }

        Ok(cm)
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Create a new collection (idempotent — succeeds silently if already exists).
    ///
    /// Uses a creation mutex to eliminate the TOCTOU race between the
    /// `contains_key` check and the actual insert into the `DashMap`.
    pub fn create_collection(&self, cfg: CollectionConfig) -> Result<(), GraphError> {
        let _guard = self.create_lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.collections.contains_key(&cfg.name) {
            return Ok(()); // Already exists — idempotent
        }
        self.create_collection_inner(&cfg)
    }

    /// Drop a collection by name.
    ///
    /// The `"default"` collection cannot be dropped.
    ///
    /// Removes the in-memory entry **and** renames `collection.json` to
    /// `collection.json.dropped` so the collection is not re-discovered on
    /// the next [`CollectionManager::open`].  All raw data (RocksDB, HNSW)
    /// is kept on disk for manual recovery if needed.
    pub fn drop_collection(&self, name: &str) -> Result<(), GraphError> {
        if name == "default" {
            return Err(GraphError::Storage(
                "cannot drop the 'default' collection".to_string(),
            ));
        }

        // Rename collection.json so the collection is not re-opened on restart.
        let cfg_path     = self.collection_dir(name).join("collection.json");
        let dropped_path = self.collection_dir(name).join("collection.json.dropped");
        if cfg_path.exists() {
            std::fs::rename(&cfg_path, &dropped_path).map_err(|e| {
                GraphError::Storage(format!(
                    "failed to mark collection '{}' as dropped: {e}", name
                ))
            })?;
        }

        // Remove from in-memory map after the file operation succeeds.
        self.collections.remove(name);
        Ok(())
    }

    /// Get the shared DB for a named collection, or `None` if it doesn't exist.
    pub fn get(&self, name: &str) -> Option<SharedDb> {
        self.collections
            .get(name)
            .map(|entry| Arc::clone(&entry.value().1))
    }

    /// Get the shared DB for `name`, or the `"default"` collection if `name` is empty.
    ///
    /// Returns `None` only if neither the named collection nor `"default"` exists
    /// (which cannot happen after successful `open()`).
    pub fn get_or_default(&self, name: &str) -> Option<SharedDb> {
        let target = if name.is_empty() { "default" } else { name };
        self.get(target)
    }

    /// List all collections with best-effort runtime stats.
    ///
    /// Stats (node/edge counts) are obtained via `try_lock()` — collections
    /// currently locked by another task will report counts of `0`.
    pub fn list(&self) -> Vec<CollectionInfo> {
        let mut infos = Vec::with_capacity(self.collections.len());
        for entry in self.collections.iter() {
            let (cfg, db) = entry.value();
            let (node_count, edge_count) = match db.try_read() {
                Ok(guard) => (
                    guard.node_count().unwrap_or(0),
                    guard.edge_count().unwrap_or(0),
                ),
                Err(_) => (0, 0), // lock busy — skip counts
            };
            infos.push(CollectionInfo {
                name:       cfg.name.clone(),
                dim:        cfg.dim,
                metric:     cfg.metric.clone(),
                node_count,
                edge_count,
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        infos
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn create_collection_inner(&self, cfg: &CollectionConfig) -> Result<(), GraphError> {
        let dir = self.collection_dir(&cfg.name);
        std::fs::create_dir_all(&dir).map_err(|e| {
            GraphError::Storage(format!(
                "cannot create collection dir {}: {e}",
                dir.display()
            ))
        })?;

        // Persist config so next `open()` can reload it
        let json = serde_json::to_string_pretty(cfg).map_err(|e| {
            GraphError::Storage(format!("serialize CollectionConfig: {e}"))
        })?;
        std::fs::write(dir.join("collection.json"), json.as_bytes()).map_err(|e| {
            GraphError::Storage(format!("write collection.json: {e}"))
        })?;

        let db = Self::open_db(&dir, cfg)?;
        self.collections
            .insert(cfg.name.clone(), (cfg.clone(), Arc::new(RwLock::new(db))));
        Ok(())
    }

    fn collection_dir(&self, name: &str) -> PathBuf {
        self.base_dir.join("collections").join(name)
    }

    fn open_db(dir: &Path, cfg: &CollectionConfig) -> Result<NietzscheDB<AnyVectorStore>, GraphError> {
        let vs = AnyVectorStore::for_collection(dir, cfg.dim, cfg.vector_metric())
            .map_err(|e| GraphError::Storage(format!("vector store: {e}")))?;
        NietzscheDB::open(dir, vs)
    }
}
