//! ClusterNode — identity and health of a single NietzscheDB process.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────
// NodeRole
// ─────────────────────────────────────────────

/// The functional role of a cluster member.
///
/// In Phase G all nodes can serve reads and writes (`Primary`).
/// Future phases will differentiate read replicas, coordinators, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Authoritative shard — accepts reads and writes.
    Primary,
    /// Read-only replica — eventual consistency from primary.
    Replica,
    /// Coordinator-only — routes requests, holds no data.
    Coordinator,
}

impl std::fmt::Display for NodeRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeRole::Primary     => write!(f, "primary"),
            NodeRole::Replica     => write!(f, "replica"),
            NodeRole::Coordinator => write!(f, "coordinator"),
        }
    }
}

// ─────────────────────────────────────────────
// NodeHealth
// ─────────────────────────────────────────────

/// Coarse-grained health status of a cluster member.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealth {
    /// Node responded to the last heartbeat within the deadline.
    Healthy,
    /// Node is reachable but under pressure (high latency / CPU / disk).
    Degraded,
    /// Node missed ≥ 2 consecutive heartbeats.
    Unreachable,
}

impl std::fmt::Display for NodeHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeHealth::Healthy     => write!(f, "healthy"),
            NodeHealth::Degraded    => write!(f, "degraded"),
            NodeHealth::Unreachable => write!(f, "unreachable"),
        }
    }
}

// ─────────────────────────────────────────────
// ClusterNode
// ─────────────────────────────────────────────

/// Identity and runtime state of a single NietzscheDB cluster member.
///
/// Immutable fields: `id`, `addr`, `role`.
/// Mutable fields (updated by gossip): `health`, `last_seen_ms`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    /// Stable unique identifier for this node. Never changes.
    pub id: Uuid,

    /// gRPC listen address, e.g. `"10.0.1.42:6660"`.
    pub addr: String,

    /// Human-readable name, e.g. `"nietzsche-shard-0"`.
    pub name: String,

    /// Current functional role in the cluster.
    pub role: NodeRole,

    /// Last known health status.
    pub health: NodeHealth,

    /// Unix timestamp (milliseconds) of the last successful heartbeat.
    pub last_seen_ms: u64,

    /// Partition token for consistent hashing.
    /// Each node owns the token range `(prev_token, token]`.
    /// The router sorts all registered nodes by token and uses binary search.
    pub token: u64,
}

impl ClusterNode {
    /// Create a new cluster node with `Healthy` status and the current time.
    pub fn new(id: Uuid, name: impl Into<String>, addr: impl Into<String>, role: NodeRole, token: u64) -> Self {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id,
            addr: addr.into(),
            name: name.into(),
            role,
            health: NodeHealth::Healthy,
            last_seen_ms: now_ms,
            token,
        }
    }

    /// Returns `true` if this node can serve requests right now.
    pub fn is_available(&self) -> bool {
        matches!(self.health, NodeHealth::Healthy | NodeHealth::Degraded)
    }

    /// Update the heartbeat timestamp to now.
    pub fn touch(&mut self) {
        self.last_seen_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }
}
