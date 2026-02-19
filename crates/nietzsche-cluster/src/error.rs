//! Error types for the cluster layer.

use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ClusterError {
    #[error("cluster is empty — no nodes registered")]
    EmptyCluster,

    #[error("node '{0}' not found in registry")]
    NodeNotFound(Uuid),

    #[error("no healthy nodes available for collection '{0}'")]
    NoHealthyNode(String),

    #[error("routing failed: {0}")]
    RoutingError(String),
}
