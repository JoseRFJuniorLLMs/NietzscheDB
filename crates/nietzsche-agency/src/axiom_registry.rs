//! # AxiomRegistry — Civilizational Memory
//!
//! When a node crosses from Active → Mature, the AxiomRegistry records
//! an immutable **AxiomRecord** containing the cognitive state at the
//! moment of "Sagração" (consecration).
//!
//! ## Architecture: L1 + L2 Cache
//!
//! - **L1 (DashMap)**: O(1) lock-free reads for the hot-path
//!   (`is_axiom`, `get`, `count`). Populated on register and on
//!   cache-miss from L2.
//! - **L2 (RocksDB CF_META)**: Durable persistence. Records survive
//!   server restarts and are lazily loaded into L1 on access.
//!
//! This design removes disk I/O from the `MaturityEvaluator` hot-path,
//! which calls `is_axiom()` on every node per tick.
//!
//! ## Persistence
//!
//! Records are stored in CF_META under `axiom:<node_id>` keys as JSON.
//! Era snapshots are stored under `axiom_era:<epoch>`.

use dashmap::DashMap;
use uuid::Uuid;

use nietzsche_graph::GraphStorage;
use nietzsche_hyp_ops::poincare_distance;

use crate::centroid_guardian::CentroidGuardian;
use crate::error::AgencyError;
use crate::quantum::poincare_to_bloch;

// ─────────────────────────────────────────────
// AxiomRecord
// ─────────────────────────────────────────────

/// Immutable record of a node's consecration as an axiom.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AxiomRecord {
    /// Node ID.
    pub node_id: Uuid,
    /// Centroid epoch at time of consecration.
    pub birth_epoch: u64,
    /// Timestamp (Unix ms).
    pub birth_timestamp_ms: u64,
    /// Maturity score at consecration.
    pub maturity_score: f64,
    /// Poincaré distance to centroid at birth.
    pub centroid_distance: f64,
    /// Bloch state fidelity relative to centroid at birth.
    pub bloch_fidelity: f64,
    /// Embedding snapshot (f64, for geometric precision).
    pub embedding_snapshot: Vec<f64>,
    /// Centroid snapshot at birth (for rollback).
    pub centroid_snapshot: Vec<f64>,
}

// ─────────────────────────────────────────────
// EraSnapshot
// ─────────────────────────────────────────────

/// Snapshot of the civilizational state at a given epoch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EraSnapshot {
    pub epoch: u64,
    pub timestamp_ms: u64,
    pub axiom_count: usize,
    pub centroid: Vec<f64>,
}

// ─────────────────────────────────────────────
// AxiomRegistry
// ─────────────────────────────────────────────

/// Registry for tracking axiom consecrations and civilizational history.
///
/// Uses a `DashMap` L1 cache for O(1) hot-path access (`is_axiom`, `get`),
/// backed by RocksDB CF_META for durability.
/// Maximum number of axioms in the L1 cache before LRU eviction kicks in.
/// Prevents unbounded memory growth. At 49KB per record, 2048 = ~100MB cap.
const AXIOM_CACHE_CAP: usize = 2048;

pub struct AxiomRegistry {
    /// L1 cache: lock-free concurrent reads for MaturityEvaluator hot-path.
    cache: DashMap<Uuid, AxiomRecord>,
}

impl AxiomRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            cache: DashMap::new(),
        }
    }

    /// Evict oldest entries when cache exceeds AXIOM_CACHE_CAP.
    /// Uses birth_timestamp_ms as LRU proxy (oldest axiom evicted first).
    fn evict_if_over_cap(&self) {
        if self.cache.len() <= AXIOM_CACHE_CAP {
            return;
        }
        let to_evict = self.cache.len() - AXIOM_CACHE_CAP;
        // Collect (id, timestamp) pairs and sort by oldest
        let mut entries: Vec<(Uuid, u64)> = self.cache.iter()
            .map(|e| (*e.key(), e.value().birth_timestamp_ms))
            .collect();
        entries.sort_by_key(|&(_, ts)| ts);
        for (id, _) in entries.into_iter().take(to_evict) {
            self.cache.remove(&id);
        }
        tracing::debug!(
            evicted = to_evict,
            remaining = self.cache.len(),
            "AxiomRegistry: L1 cache eviction"
        );
    }

    /// Number of axioms currently in the L1 cache.
    pub fn count(&self) -> usize {
        self.cache.len()
    }

    /// Register a new axiom (Active → Mature transition).
    ///
    /// Writes to both L1 (DashMap) and L2 (RocksDB CF_META).
    /// Records the cognitive state at the moment of consecration:
    /// - Centroid distance and Bloch fidelity
    /// - Embedding and centroid snapshots for future rollback
    pub fn register(
        &self,
        storage: &GraphStorage,
        node_id: Uuid,
        embedding: &[f64],
        maturity_score: f64,
        guardian: &CentroidGuardian,
    ) -> Result<AxiomRecord, AgencyError> {
        let centroid = guardian.current_centroid();
        let epoch = guardian.epoch();

        // Compute centroid distance
        let centroid_distance = poincare_distance(embedding, centroid);

        // Compute Bloch fidelity
        let bloch_node = poincare_to_bloch(embedding, 1.0_f32);
        let bloch_centroid = poincare_to_bloch(centroid, 1.0_f32);
        let bloch_fidelity = bloch_node.fidelity(&bloch_centroid);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let record = AxiomRecord {
            node_id,
            birth_epoch: epoch,
            birth_timestamp_ms: now_ms,
            maturity_score,
            centroid_distance,
            bloch_fidelity,
            embedding_snapshot: embedding.to_vec(),
            centroid_snapshot: centroid.to_vec(),
        };

        // L1: Insert into DashMap (O(1), lock-free reads)
        self.cache.insert(node_id, record.clone());

        // Evict oldest entries if cache exceeds cap (prevents unbounded growth)
        self.evict_if_over_cap();

        // L2: Persist to RocksDB CF_META
        let key = format!("axiom:{}", node_id);
        let json = serde_json::to_vec(&record)
            .map_err(|e| AgencyError::Internal(format!("axiom serialize: {e}")))?;
        storage.put_meta(&key, &json)
            .map_err(|e| AgencyError::Internal(format!("axiom persist: {e}")))?;

        tracing::info!(
            node_id = %node_id,
            epoch,
            centroid_distance,
            bloch_fidelity,
            cache_size = self.cache.len(),
            "axiom registered (Sagração)"
        );

        Ok(record)
    }

    /// Retrieve an axiom record by node ID.
    ///
    /// Hot-path: checks L1 cache first (O(1)), falls back to L2 (RocksDB)
    /// and populates the cache on miss.
    pub fn get(&self, storage: &GraphStorage, node_id: Uuid) -> Result<Option<AxiomRecord>, AgencyError> {
        // L1: Check cache first
        if let Some(record) = self.cache.get(&node_id) {
            return Ok(Some(record.clone()));
        }

        // L2: Fallback to RocksDB
        let key = format!("axiom:{}", node_id);
        match storage.get_meta(&key)
            .map_err(|e| AgencyError::Internal(format!("axiom read: {e}")))?
        {
            Some(bytes) => {
                let record: AxiomRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| AgencyError::Internal(format!("axiom deserialize: {e}")))?;
                // Populate L1 on cache miss
                self.cache.insert(node_id, record.clone());
                self.evict_if_over_cap();
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// Iterate over all axiom records in the L1 cache.
    ///
    /// Used for semantic deduplication checks. O(N) but axiom count is small.
    pub fn iter_cache(&self) -> dashmap::iter::Iter<'_, Uuid, AxiomRecord> {
        self.cache.iter()
    }

    /// Check if a node is a registered axiom (O(1) hot-path via L1 cache).
    pub fn is_axiom(&self, node_id: Uuid) -> bool {
        self.cache.contains_key(&node_id)
    }

    /// Check if a node is a registered axiom with L2 fallback.
    ///
    /// Use this when the L1 cache might not be warmed (e.g., after restart).
    pub fn is_axiom_with_fallback(&self, storage: &GraphStorage, node_id: Uuid) -> bool {
        if self.cache.contains_key(&node_id) {
            return true;
        }
        // Try L2
        self.get(storage, node_id).ok().flatten().is_some()
    }

    /// Deregister an axiom (Mature → Active demotion).
    ///
    /// Removes from both L1 and L2.
    pub fn deregister(&self, storage: &GraphStorage, node_id: Uuid) -> Result<(), AgencyError> {
        // L1: Remove from cache
        self.cache.remove(&node_id);

        // L2: Remove from RocksDB
        let key = format!("axiom:{}", node_id);
        storage.delete_meta(&key)
            .map_err(|e| AgencyError::Internal(format!("axiom deregister: {e}")))?;
        tracing::info!(node_id = %node_id, "axiom deregistered");
        Ok(())
    }

    /// Record an era snapshot for civilizational history.
    pub fn snapshot_era(
        &self,
        storage: &GraphStorage,
        epoch: u64,
        centroid: &[f64],
    ) -> Result<EraSnapshot, AgencyError> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let snapshot = EraSnapshot {
            epoch,
            timestamp_ms: now_ms,
            axiom_count: self.cache.len(),
            centroid: centroid.to_vec(),
        };

        let key = format!("axiom_era:{}", epoch);
        let json = serde_json::to_vec(&snapshot)
            .map_err(|e| AgencyError::Internal(format!("era snapshot serialize: {e}")))?;
        storage.put_meta(&key, &json)
            .map_err(|e| AgencyError::Internal(format!("era snapshot persist: {e}")))?;

        tracing::debug!(epoch, axiom_count = snapshot.axiom_count, "era snapshot recorded");

        Ok(snapshot)
    }

    /// Retrieve an era snapshot by epoch.
    pub fn get_era(storage: &GraphStorage, epoch: u64) -> Result<Option<EraSnapshot>, AgencyError> {
        let key = format!("axiom_era:{}", epoch);
        match storage.get_meta(&key)
            .map_err(|e| AgencyError::Internal(format!("era read: {e}")))?
        {
            Some(bytes) => {
                let snapshot: EraSnapshot = serde_json::from_slice(&bytes)
                    .map_err(|e| AgencyError::Internal(format!("era deserialize: {e}")))?;
                Ok(Some(snapshot))
            }
            None => Ok(None),
        }
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn register_and_retrieve() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let guardian = CentroidGuardian::new(3);
        let registry = AxiomRegistry::new();

        let node_id = Uuid::new_v4();
        let embedding = vec![0.2, 0.1, 0.0];

        let record = registry.register(
            &storage, node_id, &embedding, 0.75, &guardian
        ).unwrap();

        assert_eq!(record.node_id, node_id);
        assert_eq!(record.birth_epoch, 0);
        assert!(record.centroid_distance > 0.0);
        assert!(record.bloch_fidelity >= 0.0 && record.bloch_fidelity <= 1.0);

        // Retrieve from L1 cache
        let retrieved = registry.get(&storage, node_id).unwrap().unwrap();
        assert_eq!(retrieved.node_id, node_id);
        assert_eq!(retrieved.maturity_score, 0.75);
    }

    #[test]
    fn is_axiom_hot_path() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let guardian = CentroidGuardian::new(3);
        let registry = AxiomRegistry::new();

        let node_id = Uuid::new_v4();
        assert!(!registry.is_axiom(node_id));

        registry.register(&storage, node_id, &[0.1, 0.0, 0.0], 0.5, &guardian).unwrap();
        assert!(registry.is_axiom(node_id)); // O(1) from DashMap
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn l2_fallback_populates_l1() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let guardian = CentroidGuardian::new(3);

        let node_id = Uuid::new_v4();

        // Register with one registry instance
        let registry1 = AxiomRegistry::new();
        registry1.register(&storage, node_id, &[0.1, 0.0, 0.0], 0.5, &guardian).unwrap();

        // Create a fresh registry (simulates restart — empty L1)
        let registry2 = AxiomRegistry::new();
        assert!(!registry2.is_axiom(node_id)); // Not in L1

        // L2 fallback should find it and populate L1
        assert!(registry2.is_axiom_with_fallback(&storage, node_id));
        assert!(registry2.is_axiom(node_id)); // Now in L1 too
    }

    #[test]
    fn deregister_removes_from_both() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let guardian = CentroidGuardian::new(3);
        let registry = AxiomRegistry::new();

        let node_id = Uuid::new_v4();
        registry.register(&storage, node_id, &[0.1, 0.0, 0.0], 0.5, &guardian).unwrap();
        assert!(registry.is_axiom(node_id));

        registry.deregister(&storage, node_id).unwrap();
        assert!(!registry.is_axiom(node_id));
        // Also gone from L2
        assert!(!registry.is_axiom_with_fallback(&storage, node_id));
    }

    #[test]
    fn era_snapshot() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let registry = AxiomRegistry::new();

        let centroid = vec![0.05, 0.0, 0.0];
        let snap = registry.snapshot_era(&storage, 42, &centroid).unwrap();
        assert_eq!(snap.epoch, 42);
        assert_eq!(snap.axiom_count, 0); // empty registry

        let retrieved = AxiomRegistry::get_era(&storage, 42).unwrap().unwrap();
        assert_eq!(retrieved.epoch, 42);
    }

    #[test]
    fn count_tracks_registrations() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let guardian = CentroidGuardian::new(3);
        let registry = AxiomRegistry::new();

        assert_eq!(registry.count(), 0);

        for i in 0..5 {
            let id = Uuid::new_v4();
            let x = 0.1 * (i as f64 + 1.0);
            registry.register(&storage, id, &[x, 0.0, 0.0], 0.5, &guardian).unwrap();
        }

        assert_eq!(registry.count(), 5);
    }
}
