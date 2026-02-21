use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A dream event detected during speculative exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamEvent {
    pub event_type: DreamEventType,
    pub node_id:    Uuid,
    pub energy:     f64,
    pub depth:      f64,
    pub description: String,
}

/// Types of dream events detected during exploration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DreamEventType {
    /// Two clusters collided during diffusion.
    ClusterCollision,
    /// A node experienced an energy spike above threshold.
    EnergySpike,
    /// Curvature anomaly detected (Hausdorff dimension shift).
    CurvatureAnomaly,
    /// Temporal pattern recurrence detected.
    Recurrence,
}

/// A dream session — speculative exploration results pending approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamSession {
    pub id:          String,
    pub seed_node:   Uuid,
    pub depth:       usize,
    pub noise:       f64,
    pub events:      Vec<DreamEvent>,
    pub created_at:  i64,
    pub status:      DreamStatus,
    /// Nodes discovered during the dream (new positions).
    pub dream_nodes: Vec<DreamNodeDelta>,
}

/// Status of a dream session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DreamStatus {
    Pending,
    Applied,
    Rejected,
}

/// A node modification proposed by the dream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamNodeDelta {
    pub node_id:    Uuid,
    pub old_energy: f64,
    pub new_energy: f64,
    pub event_type: Option<DreamEventType>,
}
