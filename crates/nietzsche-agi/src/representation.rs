// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # Representation — AGI node types and wrappers
//!
//! Extends the base NietzscheDB [`Node`] with AGI-specific metadata:
//! - What inference produced this node
//! - The Rationale that justifies its existence
//! - Cluster membership information
//! - Synthesis lineage (which nodes were combined)

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::rationale::{InferenceType, Rationale};

// ─────────────────────────────────────────────
// SynthesisNode — AGI wrapper around graph Node
// ─────────────────────────────────────────────

/// An AGI-aware node that extends the base graph node with inference metadata.
///
/// A SynthesisNode is created by the [`FrechetSynthesizer`](crate::synthesis::FrechetSynthesizer)
/// or the [`InferenceEngine`](crate::inference_engine::InferenceEngine) and carries
/// full provenance information about how and why it was created.
///
/// # Storage
/// The base node (with embedding) is stored in NietzscheDB as usual.
/// The AGI metadata is stored in the node's `content` JSON under the `_agi` key,
/// keeping the graph format backward-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisNode {
    /// ID of the node in the graph.
    pub node_id: Uuid,

    /// What type of inference produced this node.
    pub inference_type: InferenceType,

    /// IDs of the source nodes that were synthesized to produce this one.
    /// For Generalization/Specialization: typically 2 nodes (start, end).
    /// For DialecticalSynthesis: ≥2 thesis nodes.
    pub source_nodes: Vec<Uuid>,

    /// The cluster ID this node belongs to (if cluster analysis has been run).
    pub cluster_id: Option<u32>,

    /// The Rationale that justifies this node's existence.
    /// `None` for nodes that pre-date the AGI layer.
    pub rationale: Option<Rationale>,

    /// How many times this node has been referenced by subsequent inferences.
    /// Used by [`RelevanceDecay`](crate::relevance_decay::RelevanceDecay) to
    /// boost frequently-used synthesis nodes.
    pub reference_count: u64,

    /// Generation counter: how many levels of synthesis deep this node is.
    /// 0 = original data, 1 = first synthesis, 2 = synthesis of syntheses, etc.
    pub synthesis_depth: u32,
}

impl SynthesisNode {
    /// Create a new SynthesisNode from a synthesis operation.
    pub fn new(
        node_id: Uuid,
        inference_type: InferenceType,
        source_nodes: Vec<Uuid>,
        rationale: Rationale,
    ) -> Self {
        Self {
            node_id,
            inference_type,
            source_nodes,
            cluster_id: None,
            rationale: Some(rationale),
            reference_count: 0,
            synthesis_depth: 0,
        }
    }

    /// Create a legacy wrapper for a pre-existing node (no AGI metadata).
    pub fn wrap_legacy(node_id: Uuid) -> Self {
        Self {
            node_id,
            inference_type: InferenceType::StructuralBridge,
            source_nodes: Vec::new(),
            cluster_id: None,
            rationale: None,
            reference_count: 0,
            synthesis_depth: 0,
        }
    }

    /// Increment the reference count (called when another inference uses this node).
    pub fn bump_reference(&mut self) {
        self.reference_count = self.reference_count.saturating_add(1);
    }

    /// Returns the fidelity of the underlying rationale, or 0.0 for legacy nodes.
    pub fn fidelity(&self) -> f64 {
        self.rationale.as_ref().map_or(0.0, |r| r.fidelity)
    }

    /// Serialize the AGI metadata to a JSON value suitable for storing
    /// in the node's `content._agi` field.
    pub fn to_agi_content(&self) -> serde_json::Value {
        serde_json::json!({
            "inference_type": self.inference_type.to_string(),
            "source_nodes": self.source_nodes.iter().map(|u| u.to_string()).collect::<Vec<_>>(),
            "cluster_id": self.cluster_id,
            "reference_count": self.reference_count,
            "synthesis_depth": self.synthesis_depth,
            "fidelity": self.fidelity(),
        })
    }
}

// ─────────────────────────────────────────────
// NodeDepthInfo — lightweight depth snapshot
// ─────────────────────────────────────────────

/// Lightweight snapshot of a node's position in the Poincaré ball.
///
/// Used during trajectory analysis where we only need the embedding norm
/// (depth) and cluster membership, not the full node.
#[derive(Debug, Clone, Copy)]
pub struct NodeDepthInfo {
    /// Node ID.
    pub id: Uuid,
    /// ‖embedding‖ — the radial position in the Poincaré ball.
    /// Low = abstract (center), High = concrete (boundary).
    pub depth: f64,
    /// Cluster ID (if known).
    pub cluster_id: Option<u32>,
}

// ─────────────────────────────────────────────
// AGI Content Keys
// ─────────────────────────────────────────────

/// Key in node content JSON where AGI metadata is stored.
pub const AGI_CONTENT_KEY: &str = "_agi";

/// Key in node content JSON for the inference type.
pub const AGI_INFERENCE_TYPE_KEY: &str = "inference_type";

/// Key in node content JSON for source node IDs.
pub const AGI_SOURCE_NODES_KEY: &str = "source_nodes";

/// Extract AGI metadata from a node's content JSON, if present.
pub fn extract_agi_metadata(content: &serde_json::Value) -> Option<SynthesisNode> {
    let agi = content.get(AGI_CONTENT_KEY)?;
    let node_id = content
        .get("id")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse::<Uuid>().ok())
        .unwrap_or_else(Uuid::new_v4);

    let inference_type_str = agi.get("inference_type")?.as_str()?;
    let inference_type = match inference_type_str {
        "Generalization" => InferenceType::Generalization,
        "Specialization" => InferenceType::Specialization,
        "DialecticalSynthesis" => InferenceType::DialecticalSynthesis,
        "StructuralBridge" => InferenceType::StructuralBridge,
        "AnalogicalMapping" => InferenceType::AnalogicalMapping,
        "LogicalRupture" => InferenceType::LogicalRupture,
        _ => return None,
    };

    let source_nodes = agi
        .get("source_nodes")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str()?.parse::<Uuid>().ok())
                .collect()
        })
        .unwrap_or_default();

    let cluster_id = agi.get("cluster_id").and_then(|v| v.as_u64()).map(|v| v as u32);
    let reference_count = agi.get("reference_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let synthesis_depth = agi.get("synthesis_depth").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    Some(SynthesisNode {
        node_id,
        inference_type,
        source_nodes,
        cluster_id,
        rationale: None, // Rationale is not stored inline in content
        reference_count,
        synthesis_depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_node_to_agi_content() {
        let node = SynthesisNode::wrap_legacy(Uuid::new_v4());
        let content = node.to_agi_content();
        assert_eq!(content["inference_type"], "StructuralBridge");
        assert_eq!(content["reference_count"], 0);
        assert_eq!(content["synthesis_depth"], 0);
    }

    #[test]
    fn test_extract_roundtrip() {
        let id = Uuid::new_v4();
        let src1 = Uuid::new_v4();
        let src2 = Uuid::new_v4();

        let mut node = SynthesisNode::wrap_legacy(id);
        node.inference_type = InferenceType::DialecticalSynthesis;
        node.source_nodes = vec![src1, src2];
        node.cluster_id = Some(42);
        node.reference_count = 7;
        node.synthesis_depth = 2;

        let agi_content = node.to_agi_content();
        let mut content = serde_json::json!({
            "id": id.to_string(),
        });
        content[AGI_CONTENT_KEY] = agi_content;

        let extracted = extract_agi_metadata(&content).expect("should parse");
        assert_eq!(extracted.inference_type, InferenceType::DialecticalSynthesis);
        assert_eq!(extracted.source_nodes.len(), 2);
        assert_eq!(extracted.cluster_id, Some(42));
        assert_eq!(extracted.reference_count, 7);
        assert_eq!(extracted.synthesis_depth, 2);
    }
}
