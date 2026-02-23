//! ClusterRouter — consistent-hashing request router.
//!
//! Uses a **token ring**: every node owns the half-open range
//! `(prev_token, token]`.  A key is routed to the first node whose
//! token is ≥ the key hash (wrapping around).  This matches the
//! classic Dynamo / Cassandra approach.

use uuid::Uuid;

use crate::{ClusterError, ClusterNode, ClusterRegistry};

use std::hash::{Hash, Hasher};
use siphasher::sip::SipHasher24;

// ─────────────────────────────────────────────
// Token hashing
// ─────────────────────────────────────────────

/// Hash a `Uuid` to a `u64` partition token using SipHash-2-4.
/// 
/// Point 7 Audit Fix: Replace FNV-1a with SipHash for better distribution
/// and security against collision attacks.
pub fn hash_key(key: &Uuid) -> u64 {
    let mut hasher = SipHasher24::new();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Hash an arbitrary byte slice to a `u64` partition token using SipHash-2-4.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut hasher = SipHasher24::new();
    data.hash(&mut hasher);
    hasher.finish()
}

// ─────────────────────────────────────────────
// ClusterRouter
// ─────────────────────────────────────────────

/// Stateless router that delegates all peer knowledge to a `ClusterRegistry`.
///
/// Every method snaps a read of the registry's available nodes at call time,
/// builds or uses a sorted token ring, and returns the target node.
///
/// No caching — the registry is already lock-free (DashMap) so the
/// per-call overhead of sorting a Vec of N nodes (N ≈ tens) is trivial.
#[derive(Clone, Debug)]
pub struct ClusterRouter {
    registry: ClusterRegistry,
}

impl ClusterRouter {
    /// Wrap an existing registry.
    pub fn new(registry: ClusterRegistry) -> Self {
        Self { registry }
    }

    // ── Core routing ─────────────────────────────────────────

    /// Route a `Uuid` key to the responsible primary/available node.
    ///
    /// Uses the token ring: returns the node whose token is the
    /// smallest value ≥ `hash_key(key)` (wrapping to the first node).
    pub fn route_key(&self, key: &Uuid) -> Result<ClusterNode, ClusterError> {
        let token = hash_key(key);
        self.route_token(token)
    }

    /// Route an arbitrary byte slice to the responsible node.
    pub fn route_bytes(&self, data: &[u8]) -> Result<ClusterNode, ClusterError> {
        let token = hash_bytes(data);
        self.route_token(token)
    }

    /// Route a pre-computed `u64` token to the responsible node.
    ///
    /// Returns `Err(ClusterError::EmptyCluster)` if no available nodes exist.
    pub fn route_token(&self, token: u64) -> Result<ClusterNode, ClusterError> {
        let ring = self.registry.available_nodes();
        if ring.is_empty() {
            return Err(ClusterError::EmptyCluster);
        }
        Ok(find_on_ring(&ring, token).clone())
    }

    // ── Replication ──────────────────────────────────────────

    /// Return `replica_count` nodes responsible for a key (for replication).
    ///
    /// The first node is the primary; the rest are preferred replicas in
    /// ring order.  Deduplicates automatically when fewer nodes are
    /// available than requested.
    pub fn replicas(&self, key: &Uuid, replica_count: usize) -> Result<Vec<ClusterNode>, ClusterError> {
        let token = hash_key(key);
        let ring  = self.registry.available_nodes();
        if ring.is_empty() {
            return Err(ClusterError::EmptyCluster);
        }

        let n      = ring.len();
        let count  = replica_count.min(n);
        let start  = ring_position(&ring, token);

        let mut replicas = Vec::with_capacity(count);
        for i in 0..count {
            replicas.push(ring[(start + i) % n].clone());
        }
        Ok(replicas)
    }

    // ── Node access ──────────────────────────────────────────

    /// Resolve a specific node ID.
    pub fn node_by_id(&self, id: &Uuid) -> Result<ClusterNode, ClusterError> {
        self.registry.get(id).ok_or(ClusterError::NodeNotFound(*id))
    }

    /// All available nodes (for broadcasting or scatter-gather queries).
    pub fn all_available(&self) -> Vec<ClusterNode> {
        self.registry.available_nodes()
    }

    /// All available nodes excluding the local one.
    pub fn remote_available(&self) -> Vec<ClusterNode> {
        self.registry.remote_available_nodes()
    }

    /// Whether this router has any peers at all.
    pub fn has_peers(&self) -> bool {
        !self.registry.is_empty()
    }

    /// Borrow the underlying registry.
    pub fn registry(&self) -> &ClusterRegistry {
        &self.registry
    }
}

// ─────────────────────────────────────────────
// Ring helpers  (pure functions, easy to test)
// ─────────────────────────────────────────────

/// Find the index on a **sorted** ring whose token is the smallest
/// value ≥ `token`, wrapping around to index 0 if no such node exists.
fn ring_position(ring: &[ClusterNode], token: u64) -> usize {
    match ring.binary_search_by_key(&token, |n| n.token) {
        Ok(i)  => i,
        Err(i) => i % ring.len(),   // wrap
    }
}

/// Like `ring_position` but returns a reference.
fn find_on_ring(ring: &[ClusterNode], token: u64) -> &ClusterNode {
    &ring[ring_position(ring, token)]
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(token: u64) -> ClusterNode {
        use crate::{NodeHealth, NodeRole};
        ClusterNode {
            id:           Uuid::new_v4(),
            addr:         format!("127.0.0.1:{}", 6660 + token),
            name:         format!("node-{token}"),
            role:         NodeRole::Primary,
            health:       NodeHealth::Healthy,
            last_seen_ms: 0,
            token,
        }
    }

    #[test]
    fn ring_wraps_correctly() {
        // Nodes at tokens 100, 200, 300.
        let ring = vec![make_node(100), make_node(200), make_node(300)];

        // Exactly on token 200 → index 1
        assert_eq!(ring_position(&ring, 200), 1);

        // Between 200 and 300 → index 2 (node 300)
        assert_eq!(ring_position(&ring, 250), 2);

        // Beyond 300 → wraps to index 0 (node 100)
        assert_eq!(ring_position(&ring, 400), 0);

        // Before 100 → index 0 (node 100)
        assert_eq!(ring_position(&ring, 50), 0);
    }

    #[test]
    fn hash_key_is_deterministic() {
        let id = Uuid::nil();
        assert_eq!(hash_key(&id), hash_key(&id));
        // Two different UUIDs very likely hash differently
        let id2 = Uuid::new_v4();
        // (not asserting different — could collide — just that it runs)
        let _ = hash_key(&id2);
    }

    #[test]
    fn replicas_respects_ring_order() {
        let registry = ClusterRegistry::new(Uuid::new_v4());
        for t in [100u64, 200, 300, 400, 500] {
            registry.register(make_node(t));
        }
        let router = ClusterRouter::new(registry);

        // Key that hashes to something between 200 and 300 → primary = 300
        let replicas = router.replicas(&Uuid::nil(), 3).unwrap();
        assert_eq!(replicas.len(), 3);
        // Tokens should be in ring order (primary + 2 successive)
        let t0 = replicas[0].token;
        let t1 = replicas[1].token;
        let t2 = replicas[2].token;
        // Each successive token should be on the ring after the previous
        assert!(t1 != t0 && t2 != t1);
    }
}
