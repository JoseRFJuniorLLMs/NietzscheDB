//! Cluster bootstrapping and heartbeat for NietzscheDB.
//!
//! Wraps `nietzsche-cluster` with startup wiring:
//!   - registers the local node in a fresh `ClusterRegistry`
//!   - parses seed peers from `NIETZSCHE_CLUSTER_SEEDS`
//!   - starts a tokio heartbeat task that keeps `last_seen_ms` fresh

use std::time::Duration;

use tracing::{debug, info, warn};
use uuid::Uuid;

use nietzsche_cluster::{ClusterNode, ClusterRegistry, NodeHealth, NodeRole};
use nietzsche_api::proto::nietzsche::{
    self as pb,
    nietzsche_db_client::NietzscheDbClient,
};

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

/// Spawn a background gossip loop that exchanges cluster state with peers.
///
/// Every `interval_secs`, picks a random remote peer and calls `ExchangeGossip`
/// with the local snapshot. The response is merged back, achieving eventual
/// consistency across all cluster members.
pub fn start_gossip_loop(registry: ClusterRegistry, local_id: Uuid, interval_secs: u64) {
    let interval = Duration::from_secs(interval_secs);
    info!(interval_secs, "cluster: gossip loop started");

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;

            // Get remote peers (excluding self)
            let peers = registry.remote_available_nodes();
            if peers.is_empty() {
                continue;
            }

            // Pick a random peer (simple random gossip target selection)
            let idx = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as usize) % peers.len();
            let target = &peers[idx];
            let addr = target.addr.clone();

            // Export our snapshot to send
            let snapshot = registry.export_snapshot();
            let gossip_nodes: Vec<pb::ClusterNodeProto> = snapshot.into_iter().map(|n| {
                pb::ClusterNodeProto {
                    id:           n.id.to_string(),
                    addr:         n.addr,
                    name:         n.name,
                    role:         format!("{}", n.role),
                    health:       format!("{}", n.health),
                    last_seen_ms: n.last_seen_ms,
                    token:        n.token,
                }
            }).collect();

            let request = pb::GossipRequest {
                sender_id: local_id.to_string(),
                nodes:     gossip_nodes,
            };

            // Connect to the peer and exchange gossip
            let endpoint = format!("http://{}", addr);
            match NietzscheDbClient::connect(endpoint).await {
                Ok(mut client) => {
                    match client.exchange_gossip(request).await {
                        Ok(response) => {
                            let incoming: Vec<ClusterNode> = response.into_inner().nodes.iter().filter_map(|p| {
                                let id = Uuid::parse_str(&p.id).ok()?;
                                let role = match p.role.as_str() {
                                    "replica"     => NodeRole::Replica,
                                    "coordinator" => NodeRole::Coordinator,
                                    _             => NodeRole::Primary,
                                };
                                let health = match p.health.as_str() {
                                    "degraded"    => NodeHealth::Degraded,
                                    "unreachable" => NodeHealth::Unreachable,
                                    _             => NodeHealth::Healthy,
                                };
                                let mut node = ClusterNode::new(id, &p.name, &p.addr, role, p.token);
                                node.health = health;
                                node.last_seen_ms = p.last_seen_ms;
                                Some(node)
                            }).collect();

                            let count = incoming.len();
                            registry.merge_snapshot(incoming);
                            debug!(peer = %addr, merged = count, "gossip exchange ok");
                        }
                        Err(e) => {
                            warn!(peer = %addr, error = %e, "gossip exchange failed");
                            // Mark peer as unreachable
                            let _ = registry.mark_health(&target.id, NodeHealth::Unreachable);
                        }
                    }
                }
                Err(e) => {
                    warn!(peer = %addr, error = %e, "gossip connect failed");
                    registry.mark_health(&target.id, NodeHealth::Unreachable);
                }
            }
        }
    });
}
