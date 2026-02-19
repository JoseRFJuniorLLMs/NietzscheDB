//! ClusterRegistry — in-memory peer table with gossip-merge support.

use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::{ClusterError, ClusterNode, NodeHealth};

// ─────────────────────────────────────────────
// ClusterRegistry
// ─────────────────────────────────────────────

/// Thread-safe registry of every known cluster member.
///
/// The registry is the single source of truth for peer state inside a
/// NietzscheDB node.  External gossip messages call `merge_peer()`;
/// the health-checker calls `mark_health()`.
///
/// Cloning the registry is cheap — it shares the same underlying
/// `DashMap` via `Arc`.
#[derive(Clone, Debug)]
pub struct ClusterRegistry {
    nodes: Arc<DashMap<Uuid, ClusterNode>>,
    /// UUID of this process — used to skip self when routing.
    local_id: Uuid,
}

impl ClusterRegistry {
    /// Create a new, empty registry for this node.
    pub fn new(local_id: Uuid) -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
            local_id,
        }
    }

    // ── Registration ─────────────────────────────────────────

    /// Register or fully replace a peer entry.
    pub fn register(&self, node: ClusterNode) {
        self.nodes.insert(node.id, node);
    }

    /// Remove a peer from the registry entirely.
    pub fn remove(&self, id: &Uuid) -> Option<ClusterNode> {
        self.nodes.remove(id).map(|(_, n)| n)
    }

    // ── Gossip merge ─────────────────────────────────────────

    /// Merge a peer state received via gossip.
    ///
    /// The merge rule is **last-seen wins**: if the incoming `last_seen_ms`
    /// is strictly greater than the stored value we accept the whole record.
    /// This keeps the registry eventually consistent under concurrent updates.
    pub fn merge_peer(&self, incoming: ClusterNode) {
        self.nodes
            .entry(incoming.id)
            .and_modify(|existing| {
                if incoming.last_seen_ms > existing.last_seen_ms {
                    *existing = incoming.clone();
                }
            })
            .or_insert(incoming);
    }

    /// Merge a full gossip snapshot (e.g. received on join).
    pub fn merge_snapshot(&self, snapshot: Vec<ClusterNode>) {
        for peer in snapshot {
            self.merge_peer(peer);
        }
    }

    // ── Health updates ───────────────────────────────────────

    /// Update the health of a specific peer.
    pub fn mark_health(&self, id: &Uuid, health: NodeHealth) -> Result<(), ClusterError> {
        match self.nodes.get_mut(id) {
            Some(mut entry) => {
                entry.health = health;
                Ok(())
            }
            None => Err(ClusterError::NodeNotFound(*id)),
        }
    }

    /// Touch (refresh `last_seen_ms`) on a successful heartbeat ACK.
    pub fn touch(&self, id: &Uuid) -> Result<(), ClusterError> {
        match self.nodes.get_mut(id) {
            Some(mut entry) => {
                entry.touch();
                Ok(())
            }
            None => Err(ClusterError::NodeNotFound(*id)),
        }
    }

    // ── Queries ──────────────────────────────────────────────

    /// Returns `true` if the registry contains the given ID.
    pub fn contains(&self, id: &Uuid) -> bool {
        self.nodes.contains_key(id)
    }

    /// Snapshot of a single node (cloned).
    pub fn get(&self, id: &Uuid) -> Option<ClusterNode> {
        self.nodes.get(id).map(|r| r.clone())
    }

    /// Snapshot of all nodes (cloned), sorted by token for stability.
    pub fn all_nodes(&self) -> Vec<ClusterNode> {
        let mut nodes: Vec<ClusterNode> = self.nodes.iter().map(|r| r.clone()).collect();
        nodes.sort_by_key(|n| n.token);
        nodes
    }

    /// All *available* (Healthy or Degraded) nodes, sorted by token.
    pub fn available_nodes(&self) -> Vec<ClusterNode> {
        self.all_nodes()
            .into_iter()
            .filter(|n| n.is_available())
            .collect()
    }

    /// Available nodes excluding this process itself.
    pub fn remote_available_nodes(&self) -> Vec<ClusterNode> {
        self.available_nodes()
            .into_iter()
            .filter(|n| n.id != self.local_id)
            .collect()
    }

    /// Total registered peer count (including local, including unhealthy).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` when the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Export a gossip snapshot of all nodes for broadcasting to peers.
    pub fn export_snapshot(&self) -> Vec<ClusterNode> {
        self.all_nodes()
    }

    /// The UUID this registry considers its local identity.
    pub fn local_id(&self) -> Uuid {
        self.local_id
    }
}
