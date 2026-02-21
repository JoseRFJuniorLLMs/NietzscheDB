//! Collective Unconscious — cross-collection archetype sharing.
//!
//! Archetypes are elite nodes that have been "published" for sharing
//! across collections. They serve as shadow references that other
//! collections can discover and optionally replicate.
//!
//! ## Storage
//!
//! Archetypes are stored in the local process memory (DashMap) and
//! can be propagated via the gossip protocol. Each archetype entry
//! contains the node metadata + embedding + source collection.

use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A shared archetype — an elite node published for cross-collection discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Archetype {
    /// The original node ID.
    pub node_id: Uuid,
    /// Source collection name.
    pub source_collection: String,
    /// Target collection (where it was shared to).
    pub target_collection: String,
    /// Node energy at time of sharing.
    pub energy: f64,
    /// Node depth at time of sharing.
    pub depth: f64,
    /// Content snapshot (JSON).
    pub content: serde_json::Value,
    /// Timestamp when shared.
    pub shared_at: i64,
}

/// Registry of shared archetypes, keyed by node_id.
#[derive(Debug, Clone)]
pub struct ArchetypeRegistry {
    archetypes: Arc<DashMap<Uuid, Archetype>>,
}

impl ArchetypeRegistry {
    pub fn new() -> Self {
        Self {
            archetypes: Arc::new(DashMap::new()),
        }
    }

    /// Register a new archetype.
    pub fn share(&self, archetype: Archetype) {
        self.archetypes.insert(archetype.node_id, archetype);
    }

    /// Get an archetype by node ID.
    pub fn get(&self, node_id: &Uuid) -> Option<Archetype> {
        self.archetypes.get(node_id).map(|e| e.value().clone())
    }

    /// List all archetypes.
    pub fn list(&self) -> Vec<Archetype> {
        self.archetypes.iter().map(|e| e.value().clone()).collect()
    }

    /// List archetypes for a specific target collection.
    pub fn list_for_collection(&self, collection: &str) -> Vec<Archetype> {
        self.archetypes.iter()
            .filter(|e| e.value().target_collection == collection)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Remove an archetype.
    pub fn remove(&self, node_id: &Uuid) -> Option<Archetype> {
        self.archetypes.remove(node_id).map(|(_, v)| v)
    }

    /// Number of registered archetypes.
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Merge archetypes from a peer (gossip protocol).
    pub fn merge_peer_archetypes(&self, peer_archetypes: Vec<Archetype>) {
        for arch in peer_archetypes {
            // Only insert if we don't have it or if peer version is newer
            let should_insert = match self.archetypes.get(&arch.node_id) {
                Some(existing) => arch.shared_at > existing.shared_at,
                None => true,
            };
            if should_insert {
                self.archetypes.insert(arch.node_id, arch);
            }
        }
    }
}

impl Default for ArchetypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_archetype(name: &str) -> Archetype {
        Archetype {
            node_id: Uuid::new_v4(),
            source_collection: "default".to_string(),
            target_collection: name.to_string(),
            energy: 0.95,
            depth: 0.3,
            content: serde_json::json!({"title": "elite node"}),
            shared_at: 1000,
        }
    }

    #[test]
    fn share_and_list() {
        let registry = ArchetypeRegistry::new();
        let arch = sample_archetype("memories");
        let id = arch.node_id;
        registry.share(arch);
        assert_eq!(registry.len(), 1);
        assert!(registry.get(&id).is_some());
    }

    #[test]
    fn list_for_collection_filters() {
        let registry = ArchetypeRegistry::new();
        registry.share(sample_archetype("memories"));
        registry.share(sample_archetype("dreams"));
        registry.share(sample_archetype("memories"));

        let mem_archetypes = registry.list_for_collection("memories");
        assert_eq!(mem_archetypes.len(), 2);
    }

    #[test]
    fn merge_peer_archetypes() {
        let registry = ArchetypeRegistry::new();
        let arch = sample_archetype("test");
        let id = arch.node_id;
        registry.share(arch.clone());

        // Peer has a newer version
        let mut newer = arch.clone();
        newer.shared_at = 2000;
        newer.energy = 0.99;
        registry.merge_peer_archetypes(vec![newer]);

        let loaded = registry.get(&id).unwrap();
        assert!((loaded.energy - 0.99).abs() < 1e-6);
    }

    #[test]
    fn remove_archetype() {
        let registry = ArchetypeRegistry::new();
        let arch = sample_archetype("test");
        let id = arch.node_id;
        registry.share(arch);
        assert_eq!(registry.len(), 1);
        registry.remove(&id);
        assert!(registry.is_empty());
    }
}
