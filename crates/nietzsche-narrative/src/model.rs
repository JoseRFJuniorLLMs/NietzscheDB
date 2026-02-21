use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A narrative report summarizing graph evolution over a time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeReport {
    /// The collection this narrative was generated from.
    pub collection: String,
    /// Time window in hours that was analyzed.
    pub window_hours: u64,
    /// When the narrative was generated (Unix timestamp).
    pub generated_at: i64,
    /// Total nodes in the collection.
    pub total_nodes: usize,
    /// Total edges in the collection.
    pub total_edges: usize,
    /// Key narrative events detected.
    pub events: Vec<NarrativeEvent>,
    /// Graph-level statistics.
    pub statistics: NarrativeStats,
    /// The story arc summary.
    pub summary: String,
}

/// A single narrative event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEvent {
    pub event_type: NarrativeEventType,
    pub description: String,
    pub node_ids: Vec<Uuid>,
    pub significance: f64,
}

/// Types of narrative events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NarrativeEventType {
    /// A cluster of related nodes formed.
    ClusterFormation,
    /// An elite node (high energy) emerged.
    EliteEmergence,
    /// Energy cascade — propagation through edges.
    EnergyCascade,
    /// A node or cluster decayed significantly.
    Decay,
    /// A bridge node connecting two communities appeared.
    BridgeNode,
    /// Temporal pattern — recurrence detected.
    TemporalRecurrence,
}

/// Graph-level statistics for narrative context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeStats {
    pub mean_energy: f64,
    pub max_energy: f64,
    pub min_energy: f64,
    pub mean_depth: f64,
    pub node_type_counts: Vec<(String, usize)>,
}
