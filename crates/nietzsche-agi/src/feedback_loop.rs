//! # Feedback Loop — re-insertion of conclusions into the graph
//!
//! After a synthesis produces a new node, the feedback loop prepares:
//!
//! 1. The synthesis Node (ready for `NietzscheDB::insert_node()`)
//! 2. Edges from source nodes to the synthesis node
//! 3. Optional Minkowski causality on the edges
//!
//! The caller (server) is responsible for actually inserting into the DB.
//! This keeps the AGI crate as a **pure computation library** with no
//! mutable storage access.

use uuid::Uuid;
use nietzsche_graph::{Edge, EdgeType, Node, CausalType};
use nietzsche_hyp_ops::minkowski;

use crate::representation::SynthesisNode;

/// Convert hyp-ops CausalType to graph CausalType (separate enum definitions).
fn convert_causal(hyp: nietzsche_hyp_ops::minkowski::CausalType) -> CausalType {
    match hyp {
        nietzsche_hyp_ops::minkowski::CausalType::Timelike => CausalType::Timelike,
        nietzsche_hyp_ops::minkowski::CausalType::Spacelike => CausalType::Spacelike,
        nietzsche_hyp_ops::minkowski::CausalType::Lightlike => CausalType::Lightlike,
    }
}

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the feedback loop.
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    /// Weight assigned to edges from source → synthesis nodes.
    /// Default: 0.8 (strong connection to parent concepts).
    pub synthesis_edge_weight: f32,

    /// Whether to compute Minkowski causality for new edges.
    /// Default: true
    pub compute_causality: bool,

    /// Default causal speed parameter.
    /// Default: 1.0
    pub causal_speed: f64,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            synthesis_edge_weight: 0.8,
            compute_causality: true,
            causal_speed: 1.0,
        }
    }
}

// ─────────────────────────────────────────────
// FeedbackResult — what the caller needs to insert
// ─────────────────────────────────────────────

/// Result of the feedback loop: the node and edges to be inserted.
///
/// The caller is responsible for inserting these into the graph via
/// `NietzscheDB::insert_node()` and `NietzscheDB::insert_edge()`.
#[derive(Debug, Clone)]
pub struct FeedbackResult {
    /// The synthesis node to insert.
    pub node: Node,
    /// Edges from source nodes to the synthesis node.
    pub edges: Vec<Edge>,
    /// AGI metadata (for logging/tracking).
    pub synthesis_meta: SynthesisNode,
}

// ─────────────────────────────────────────────
// SourceNodeInfo — data needed to create edges
// ─────────────────────────────────────────────

/// Minimal info about a source node for edge creation.
///
/// The caller extracts this from the graph before calling the feedback loop.
#[derive(Debug, Clone)]
pub struct SourceNodeInfo {
    pub id: Uuid,
    /// f32 embedding coordinates (from PoincareVector::coords).
    pub coords: Vec<f32>,
    /// Unix timestamp (seconds).
    pub created_at: i64,
}

// ─────────────────────────────────────────────
// FeedbackLoop
// ─────────────────────────────────────────────

/// Prepares synthesis results for insertion into the graph.
///
/// **Pure computation** — does not access the graph directly.
pub struct FeedbackLoop {
    config: FeedbackConfig,
}

impl FeedbackLoop {
    pub fn new(config: FeedbackConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(FeedbackConfig::default())
    }

    /// Prepare a synthesis result for insertion into the graph.
    ///
    /// # Arguments
    /// - `node`: the graph-ready Node (from `FrechetSynthesizer::to_graph_node`)
    /// - `synthesis`: the AGI metadata
    /// - `source_infos`: info about each source node (for edge creation)
    ///
    /// # Returns
    /// A [`FeedbackResult`] containing the node and edges to insert.
    pub fn prepare(
        &self,
        node: Node,
        synthesis: SynthesisNode,
        source_infos: &[SourceNodeInfo],
    ) -> FeedbackResult {
        let mut edges = Vec::with_capacity(source_infos.len());

        for source in source_infos {
            let mut edge = Edge::new(
                source.id,
                synthesis.node_id,
                EdgeType::Hierarchical,
                self.config.synthesis_edge_weight,
            );

            // Compute Minkowski causality if enabled
            if self.config.compute_causality {
                let interval = minkowski::minkowski_interval(
                    &source.coords,
                    &node.embedding.coords,
                    source.created_at,
                    node.meta.created_at,
                    self.config.causal_speed,
                );
                let hyp_causal = minkowski::classify(interval, 1e-6);

                edge.minkowski_interval = interval as f32;
                edge.causal_type = convert_causal(hyp_causal);
            }

            edges.push(edge);
        }

        tracing::info!(
            synthesis_id = %synthesis.node_id,
            source_count = source_infos.len(),
            edges_created = edges.len(),
            "feedback loop: prepared for insertion"
        );

        FeedbackResult {
            node,
            edges,
            synthesis_meta: synthesis,
        }
    }

    pub fn config(&self) -> &FeedbackConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{NodeMeta, NodeType, PoincareVector};
    use crate::rationale::RationaleBuilder;

    fn make_node() -> Node {
        let meta = NodeMeta {
            id: Uuid::new_v4(),
            depth: 0.2,
            content: serde_json::json!({"type": "synthesis"}),
            node_type: NodeType::Semantic,
            energy: 1.0,
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: 1000,
            expires_at: None,
            metadata: std::collections::HashMap::new(),
            valence: 0.0,
            arousal: 0.0,
            is_phantom: false,
        };
        Node {
            meta,
            embedding: PoincareVector {
                coords: vec![0.15, 0.1, 0.0],
                dim: 3,
            },
        }
    }

    fn make_synthesis(node_id: Uuid) -> SynthesisNode {
        let rationale = RationaleBuilder::new()
            .path(vec![Uuid::new_v4(), Uuid::new_v4()])
            .hop_gcs(vec![0.9])
            .radial_gradient(-0.1)
            .build(0.5, 0.05);
        SynthesisNode::new(
            node_id,
            crate::rationale::InferenceType::DialecticalSynthesis,
            vec![Uuid::new_v4(), Uuid::new_v4()],
            rationale,
        )
    }

    #[test]
    fn test_prepare_creates_edges() {
        let fl = FeedbackLoop::with_defaults();
        let node = make_node();
        let node_id = node.meta.id;
        let synthesis = make_synthesis(node_id);

        let sources = vec![
            SourceNodeInfo {
                id: synthesis.source_nodes[0],
                coords: vec![0.3, 0.0, 0.0],
                created_at: 500,
            },
            SourceNodeInfo {
                id: synthesis.source_nodes[1],
                coords: vec![0.0, 0.3, 0.0],
                created_at: 600,
            },
        ];

        let result = fl.prepare(node, synthesis, &sources);
        assert_eq!(result.edges.len(), 2);

        // All edges point to the synthesis node
        for edge in &result.edges {
            assert_eq!(edge.to, node_id);
            assert_eq!(edge.edge_type, EdgeType::Hierarchical);
            assert_eq!(edge.weight, 0.8);
        }
    }

    #[test]
    fn test_prepare_with_causality() {
        let fl = FeedbackLoop::with_defaults();
        let node = make_node();
        let synthesis = make_synthesis(node.meta.id);

        let sources = vec![SourceNodeInfo {
            id: synthesis.source_nodes[0],
            coords: vec![0.3, 0.0, 0.0],
            created_at: 500, // Before the synthesis node
        }];

        let result = fl.prepare(node, synthesis, &sources);
        // With dt > 0 and close embeddings, should be Timelike (causal)
        assert_eq!(result.edges.len(), 1);
        // The exact causal type depends on the interval calculation
    }
}
