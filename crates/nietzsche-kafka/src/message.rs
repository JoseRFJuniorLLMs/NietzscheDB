use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A mutation event consumed from a Kafka topic.
///
/// Each variant maps to a single atomic operation against the NietzscheDB
/// hyperbolic graph.  Messages are expected to arrive as JSON with a `"type"`
/// discriminator field (serde internally tagged enum):
///
/// ```json
/// { "type": "InsertNode", "content": {"text": "hello"}, "energy": 0.9 }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GraphMutation {
    /// Insert a node into the graph.
    ///
    /// If `id` is `None`, a new UUIDv4 is generated.
    /// `energy` defaults to 1.0 if omitted.
    /// `embedding` defaults to the origin of the Poincare ball if omitted.
    InsertNode {
        id: Option<Uuid>,
        content: serde_json::Value,
        energy: Option<f32>,
        embedding: Option<Vec<f32>>,
    },

    /// Delete a node by ID.
    DeleteNode { id: Uuid },

    /// Insert a directed edge between two existing nodes.
    ///
    /// `weight` defaults to 1.0.
    /// `edge_type` defaults to `"Association"`.
    InsertEdge {
        from: Uuid,
        to: Uuid,
        weight: Option<f32>,
        edge_type: Option<String>,
    },

    /// Delete an edge by ID.
    DeleteEdge { id: Uuid },

    /// Update the energy level of an existing node.
    SetEnergy { id: Uuid, energy: f32 },

    /// Merge additional fields into a node's JSON content.
    SetContent { id: Uuid, fields: serde_json::Value },
}
