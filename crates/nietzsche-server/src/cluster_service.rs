//! Cluster bootstrapping and heartbeat for NietzscheDB.
//!
//! Wraps `nietzsche-cluster` with startup wiring:
//!   - registers the local node in a fresh `ClusterRegistry`
//!   - parses seed peers from `NIETZSCHE_CLUSTER_SEEDS`
//!   - starts a tokio heartbeat task that keeps `last_seen_ms` fresh

use std::time::Duration;

use tracing::{info, warn};
use uuid::Uuid;

use nietzsche_cluster::{ClusterNode, ClusterRegistry, NodeRole};

/// Parse `NIETZSCHE_CLUSTER_ROLE` → `NodeRole`.
pub fn parse_role(s: &str) -> NodeRole {
    match s.to_lowercase().as_str() {
        "replica"     => NodeRole::Replica,
        "coordinator" => NodeRole::Coordinator,
        _             => NodeRole::Primary,
    }
}

/// FNV-1a hash of a byte slice → u64 partition token.
fn fnv1a(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME:  u64 = 1_099_511_628_211;
    let mut h = OFFSET;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

/// Parse a single seed entry.
///
/// Accepted formats:
///   `"name@host:port"` → (name, addr)
///   `"host:port"`      → ("seed", addr)
fn parse_seed(s: &str) -> (&str, &str) {
    if let Some(at) = s.find('@') {
        (&s[..at], &s[at + 1..])
    } else {
        ("seed", s)
    }
}

/// Build a `ClusterRegistry` pre-populated with the local node and any seeds.
pub fn build_registry(
    node_name: &str,
    local_grpc_addr: &str,
    role: NodeRole,
    seeds_csv: &str,
) -> (ClusterRegistry, Uuid) {
    let local_id = Uuid::new_v4();
    let registry = ClusterRegistry::new(local_id);

    // Register self
    let token = fnv1a(local_grpc_addr.as_bytes());
    let local_node = ClusterNode::new(local_id, node_name, local_grpc_addr, role, token);
    registry.register(local_node);
    info!(
        id    = %local_id,
        name  = node_name,
        addr  = local_grpc_addr,
        token = token,
        "cluster: local node registered"
    );

    // Register seeds
    for raw in seeds_csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, addr) = parse_seed(raw);
        let seed_id    = Uuid::new_v4();
        let seed_token = fnv1a(addr.as_bytes());
        let seed_node  = ClusterNode::new(seed_id, name, addr, NodeRole::Primary, seed_token);
        registry.register(seed_node);
        info!(name, addr, token = seed_token, "cluster: seed peer registered");
    }

    (registry, local_id)
}

/// Spawn a background task that refreshes `last_seen_ms` of the local node
/// every `interval` seconds, keeping it visibly alive in the registry.
pub fn start_heartbeat(registry: ClusterRegistry, local_id: Uuid, interval_secs: u64) {
    let interval = Duration::from_secs(interval_secs);
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            if let Err(e) = registry.touch(&local_id) {
                warn!(error = %e, "cluster heartbeat: failed to touch local node");
            }
        }
    });
}
