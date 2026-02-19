//! # nietzsche-cluster
//!
//! **Phase G — Distributed Foundation**
//!
//! Provides the building blocks for a horizontally-scalable NietzscheDB cluster:
//!
//! - [`ClusterNode`] — identity and health of a single NietzscheDB process
//! - [`ClusterRegistry`] — live view of all cluster peers (gossip-updated)
//! - [`ClusterRouter`] — deterministic consistent-hashing request routing
//!
//! ## Design principles
//!
//! - **No Raft in this phase**: each shard is authoritative for its partition.
//!   Consistency is eventual; the registry is best-effort.
//! - **Deterministic routing**: `ClusterRouter` uses consistent hashing so
//!   the same collection key always maps to the same node (stable without
//!   coordination).
//! - **Gossip-ready**: `ClusterRegistry` exposes `merge_peer_view()` so a
//!   future gossip loop can propagate node state changes cluster-wide.

pub mod node;
pub mod registry;
pub mod router;
pub mod error;

pub use node::{ClusterNode, NodeRole, NodeHealth};
pub use registry::ClusterRegistry;
pub use router::ClusterRouter;
pub use error::ClusterError;
