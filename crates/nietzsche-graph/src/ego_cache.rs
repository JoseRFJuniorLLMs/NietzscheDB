// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XVII — Ego-Cache: depth-2 subgraph cache in RocksDB CF_EGO.
//!
//! Caches the "ego network" (depth-2 neighborhood) of frequently accessed
//! nodes as contiguous bincode blobs in a dedicated Column Family. This
//! reduces query latency from O(degree²) random reads to O(1) sequential
//! read + deserialize.
//!
//! ## Cache lifecycle
//!
//! 1. **Build**: `build_ego_entry()` collects depth-1 and depth-2 neighbors
//!    via `AdjacencyIndex`, reads their metadata, and serializes to bincode.
//!
//! 2. **Put**: `GraphStorage::put_ego()` writes the entry to `CF_EGO`.
//!
//! 3. **Get**: `GraphStorage::get_ego()` reads and deserializes.
//!
//! 4. **Invalidate**: On any mutation (insert/delete node/edge), the cache
//!    entries for affected nodes + their depth-1 neighbors are deleted.
//!
//! ## Design decisions
//!
//! - **Metadata only** (no embeddings): keeps entries ~100 bytes/node instead
//!   of ~24 KB. Embeddings can be fetched separately when needed.
//! - **RocksDB CF** instead of mmap files: leverages RocksDB's block cache,
//!   compression, and atomic writes.
//! - **Lazy invalidation**: entries are deleted on mutation rather than eagerly
//!   rebuilt. Rebuild happens on next cache miss.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::model::NodeMeta;
use crate::storage::GraphStorage;
use crate::error::GraphError;

// ─────────────────────────────────────────────
// Cache entry
// ─────────────────────────────────────────────

/// A cached ego-network (depth-2 neighborhood) for a single node.
///
/// Contains lightweight metadata for all neighbors up to depth 2,
/// plus edge information within the subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoCacheEntry {
    /// Root node ID.
    pub root_id: Uuid,
    /// Root node metadata.
    pub root_meta: NodeMeta,
    /// Depth-1 neighbors (directly connected).
    pub depth1: Vec<EgoNeighbor>,
    /// Depth-2 neighbors (connected to depth-1, excluding root and depth-1).
    pub depth2: Vec<EgoNeighbor>,
    /// Total edges within the ego subgraph.
    pub edge_count: usize,
    /// Unix timestamp (seconds) when this entry was built.
    pub built_at: i64,
}

/// A neighbor in the ego-network with edge metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoNeighbor {
    pub id: Uuid,
    pub energy: f32,
    pub depth: f32,
    pub node_type: String,
    /// Edge type connecting this neighbor to its parent in the ego tree.
    pub edge_type: String,
    /// Edge weight.
    pub weight: f32,
    /// Whether this neighbor is also connected to the root (for depth-2).
    pub connected_to_root: bool,
}

// ─────────────────────────────────────────────
// Build ego entry
// ─────────────────────────────────────────────

/// Build an ego-cache entry for a node by traversing depth-1 and depth-2.
///
/// This is a read-only operation using `AdjacencyIndex` (in-memory) and
/// `GraphStorage` (metadata reads from `CF_NODES`).
///
/// Returns `None` if the root node doesn't exist.
pub fn build_ego_entry(
    root_id: Uuid,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    max_depth1: usize,
    max_depth2: usize,
) -> Result<Option<EgoCacheEntry>, GraphError> {
    // 1. Get root metadata
    let root_meta = match storage.get_node_meta(&root_id)? {
        Some(m) => m,
        None => return Ok(None),
    };

    // 2. Collect depth-1 neighbors
    let mut depth1_ids: HashSet<Uuid> = HashSet::new();
    let mut depth1: Vec<EgoNeighbor> = Vec::new();

    let entries_out = adjacency.entries_out(&root_id);
    let entries_in = adjacency.entries_in(&root_id);

    for entry in entries_out.iter().chain(entries_in.iter()) {
        if depth1_ids.contains(&entry.neighbor_id) {
            continue;
        }
        if depth1.len() >= max_depth1 {
            break;
        }
        depth1_ids.insert(entry.neighbor_id);

        if let Ok(Some(meta)) = storage.get_node_meta(&entry.neighbor_id) {
            depth1.push(EgoNeighbor {
                id: meta.id,
                energy: meta.energy,
                depth: meta.depth,
                node_type: format!("{:?}", meta.node_type),
                edge_type: format!("{:?}", entry.edge_type),
                weight: entry.weight,
                connected_to_root: true,
            });
        }
    }

    // 3. Collect depth-2 neighbors (neighbors of depth-1, excluding root + depth-1)
    let mut depth2_ids: HashSet<Uuid> = HashSet::new();
    let mut depth2: Vec<EgoNeighbor> = Vec::new();
    let mut edge_count = entries_out.len() + entries_in.len();

    for d1_id in &depth1_ids {
        let d1_out = adjacency.entries_out(d1_id);
        let d1_in = adjacency.entries_in(d1_id);
        edge_count += d1_out.len() + d1_in.len();

        for entry in d1_out.iter().chain(d1_in.iter()) {
            let nid = entry.neighbor_id;
            // Skip root and depth-1 nodes
            if nid == root_id || depth1_ids.contains(&nid) || depth2_ids.contains(&nid) {
                continue;
            }
            if depth2.len() >= max_depth2 {
                break;
            }
            depth2_ids.insert(nid);

            if let Ok(Some(meta)) = storage.get_node_meta(&nid) {
                depth2.push(EgoNeighbor {
                    id: meta.id,
                    energy: meta.energy,
                    depth: meta.depth,
                    node_type: format!("{:?}", meta.node_type),
                    edge_type: format!("{:?}", entry.edge_type),
                    weight: entry.weight,
                    connected_to_root: false,
                });
            }
        }
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    Ok(Some(EgoCacheEntry {
        root_id,
        root_meta,
        depth1,
        depth2,
        edge_count,
        built_at: now,
    }))
}

// GraphStorage methods (put_ego, get_ego, delete_ego, invalidate_ego, ego_cache_count)
// are defined in storage.rs where they have access to private fields.

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ego_neighbor_serializes() {
        let neighbor = EgoNeighbor {
            id: Uuid::from_u128(42),
            energy: 0.8,
            depth: 0.3,
            node_type: "Semantic".into(),
            edge_type: "Association".into(),
            weight: 0.9,
            connected_to_root: true,
        };
        let bytes = bincode::serialize(&neighbor).unwrap();
        let back: EgoNeighbor = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.id, neighbor.id);
        assert_eq!(back.energy, 0.8);
        assert!(back.connected_to_root);
    }

    #[test]
    fn ego_entry_roundtrip() {
        let entry = EgoCacheEntry {
            root_id: Uuid::from_u128(1),
            root_meta: NodeMeta {
                id: Uuid::from_u128(1),
                depth: 0.5,
                content: serde_json::json!({"name": "test"}),
                node_type: crate::model::NodeType::Semantic,
                energy: 0.9,
                lsystem_generation: 0,
                hausdorff_local: 1.2,
                created_at: 1000,
                expires_at: None,
                metadata: Default::default(),
                valence: 0.0,
                arousal: 0.0,
                is_phantom: false,
            },
            depth1: vec![EgoNeighbor {
                id: Uuid::from_u128(2),
                energy: 0.5,
                depth: 0.4,
                node_type: "Episodic".into(),
                edge_type: "Association".into(),
                weight: 0.7,
                connected_to_root: true,
            }],
            depth2: vec![],
            edge_count: 1,
            built_at: 2000,
        };

        let bytes = bincode::serialize(&entry).unwrap();
        let back: EgoCacheEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back.root_id, entry.root_id);
        assert_eq!(back.depth1.len(), 1);
        assert_eq!(back.depth1[0].id, Uuid::from_u128(2));
        assert_eq!(back.edge_count, 1);
    }

    #[test]
    fn ego_entry_size_estimate() {
        // Verify that metadata-only entries are small
        let entry = EgoCacheEntry {
            root_id: Uuid::from_u128(1),
            root_meta: NodeMeta {
                id: Uuid::from_u128(1),
                depth: 0.5,
                content: serde_json::json!({"summary": "A brief concept"}),
                node_type: crate::model::NodeType::Semantic,
                energy: 0.9,
                lsystem_generation: 0,
                hausdorff_local: 1.0,
                created_at: 1000,
                expires_at: None,
                metadata: Default::default(),
                valence: 0.0,
                arousal: 0.0,
                is_phantom: false,
            },
            depth1: (0..20).map(|i| EgoNeighbor {
                id: Uuid::from_u128(i + 100),
                energy: 0.5,
                depth: 0.3,
                node_type: "Semantic".into(),
                edge_type: "Association".into(),
                weight: 0.5,
                connected_to_root: true,
            }).collect(),
            depth2: (0..50).map(|i| EgoNeighbor {
                id: Uuid::from_u128(i + 1000),
                energy: 0.3,
                depth: 0.6,
                node_type: "Episodic".into(),
                edge_type: "LSystemGenerated".into(),
                weight: 0.3,
                connected_to_root: false,
            }).collect(),
            edge_count: 150,
            built_at: 3000,
        };

        let bytes = bincode::serialize(&entry).unwrap();
        // 20 depth-1 + 50 depth-2 neighbors → should be well under 10 KB
        assert!(bytes.len() < 10_000, "ego entry too large: {} bytes", bytes.len());
    }
}
