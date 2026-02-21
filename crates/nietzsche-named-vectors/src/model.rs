use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// A named vector attached to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedVector {
    pub node_id: Uuid,
    pub name: String,
    pub coordinates: Vec<f32>,
    pub metric: VectorMetric,
}

/// Which distance metric this named vector uses.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum VectorMetric {
    /// Poincare ball distance (hyperbolic)
    Poincare,
    /// Cosine similarity
    Cosine,
    /// Euclidean distance
    Euclidean,
}

impl Default for VectorMetric {
    fn default() -> Self { VectorMetric::Poincare }
}

impl NamedVector {
    pub fn new(node_id: Uuid, name: impl Into<String>, coordinates: Vec<f32>, metric: VectorMetric) -> Self {
        Self { node_id, name: name.into(), coordinates, metric }
    }

    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
}
