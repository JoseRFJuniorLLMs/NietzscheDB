// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XI.2 — Concept Path API.
//!
//! Transforms raw graph routes into **annotated semantic paths** with
//! per-hop metadata, distance metrics, and human-readable explanations.
//!
//! Built on top of the Hyperbolic Greedy Router ([`crate::traversal`]).
//!
//! ## Two entry points
//!
//! - [`concept_path`]: path between two known nodes (concept → concept).
//! - [`concept_path_from_embedding`]: path from a start node toward a
//!   target embedding (query → closest reachable concept).
//!
//! ## Path explanation
//!
//! Each hop in the returned [`ConceptPath`] carries a [`PathHop`] with:
//! - Node type, energy, depth (radial position)
//! - Poincaré distance to previous hop and to target
//! - Content summary extracted from the node's JSON payload
//!
//! This enables semantic explanations like:
//! ```text
//! Mathematics (Concept, r=0.21)
//!   ──0.42──▸ Information Theory (Semantic, r=0.48)
//!   ──0.31──▸ Algorithms (Semantic, r=0.63)
//!   ──0.27──▸ Computer Science (Concept, r=0.35)
//! ```

use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::model::{NodeType, PoincareVector};
use crate::storage::GraphStorage;
use crate::traversal::{greedy_route, greedy_route_to_embedding, GreedyRouteConfig};

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// A single hop in a concept path.
#[derive(Debug, Clone)]
pub struct PathHop {
    /// Node ID at this hop.
    pub node_id: Uuid,
    /// Node type (Episodic / Semantic / Concept / DreamSnapshot).
    pub node_type: NodeType,
    /// Energy level of this node.
    pub energy: f32,
    /// Radial depth ‖embedding‖ — position in the Poincaré ball.
    pub depth: f64,
    /// L-System generation that created this node.
    pub generation: u32,
    /// Content summary (first non-empty string field or truncated JSON).
    pub summary: String,
    /// Poincaré distance from the previous hop (0.0 for the first hop).
    pub distance_from_prev: f64,
    /// Poincaré distance from this hop to the final target.
    pub distance_to_target: f64,
}

/// An annotated semantic path through the hyperbolic graph.
#[derive(Debug, Clone)]
pub struct ConceptPath {
    /// Ordered hops from source to destination.
    pub hops: Vec<PathHop>,
    /// Total Poincaré distance along the path.
    pub total_distance: f64,
    /// Number of edges traversed.
    pub hop_count: usize,
    /// Whether the path reached the exact target node.
    pub reached_target: bool,
    /// Whether A* fallback was used.
    pub used_fallback: bool,
    /// Poincaré distance from the last hop to the target (0.0 if reached).
    pub residual_distance: f64,
}

// ─────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────

/// Find an annotated concept path between two nodes.
///
/// Uses the hyperbolic greedy router (with optional A* fallback) and
/// enriches each hop with node metadata for semantic explanation.
pub fn concept_path(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    source: Uuid,
    target: Uuid,
    config: &GreedyRouteConfig,
) -> Result<ConceptPath, GraphError> {
    let route = greedy_route(storage, adjacency, source, target, config)?;

    // Load target embedding for distance-to-target computation
    let target_emb = storage.get_embedding(&target)?;

    annotate_path(storage, &route.path, target_emb.as_ref(), route.reached_target, route.used_fallback)
}

/// Find an annotated concept path from a start node toward a target embedding.
///
/// The target is specified as coordinates in the Poincaré ball (e.g. from
/// a text→embedding pipeline). The path terminates at the closest reachable
/// node. `reached_target` is always false since there is no specific node
/// to reach.
pub fn concept_path_from_embedding(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    start: Uuid,
    target_coords: &[f32],
    config: &GreedyRouteConfig,
) -> Result<ConceptPath, GraphError> {
    let route = greedy_route_to_embedding(storage, adjacency, start, target_coords, config)?;

    let target_emb = PoincareVector::new(target_coords.to_vec());

    annotate_path(storage, &route.path, Some(&target_emb), false, false)
}

/// Format a [`ConceptPath`] as a human-readable explanation string.
///
/// ```text
/// [1] Mathematics (Concept, r=0.21, E=0.85)
///   ──0.42──▸
/// [2] Information Theory (Semantic, r=0.48, E=0.72)
///   ──0.31──▸
/// [3] Computer Science (Concept, r=0.35, E=0.91)
/// ```
pub fn explain_path(path: &ConceptPath) -> String {
    let mut out = String::new();
    for (i, hop) in path.hops.iter().enumerate() {
        out.push_str(&format!(
            "[{}] {} ({:?}, r={:.2}, E={:.2}, gen={})\n",
            i + 1,
            hop.summary,
            hop.node_type,
            hop.depth,
            hop.energy,
            hop.generation,
        ));
        if i < path.hops.len() - 1 {
            out.push_str(&format!(
                "  ──{:.3}──▸\n",
                path.hops[i + 1].distance_from_prev,
            ));
        }
    }
    if !path.reached_target && path.residual_distance > 0.0 {
        out.push_str(&format!(
            "  ⚠ target not reached (residual distance: {:.4})\n",
            path.residual_distance,
        ));
    }
    out
}

// ─────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────

/// Enrich a raw path with per-hop metadata.
fn annotate_path(
    storage: &GraphStorage,
    path: &[Uuid],
    target_emb: Option<&PoincareVector>,
    reached_target: bool,
    used_fallback: bool,
) -> Result<ConceptPath, GraphError> {
    let mut hops = Vec::with_capacity(path.len());
    let mut total_distance = 0.0;
    let mut prev_emb: Option<PoincareVector> = None;

    for &node_id in path {
        let meta = storage.get_node_meta(&node_id)?
            .ok_or_else(|| GraphError::Storage(format!("node {node_id} missing in concept path")))?;

        let emb = storage.get_embedding(&node_id)?
            .ok_or_else(|| GraphError::Storage(format!("embedding {node_id} missing in concept path")))?;

        let depth = emb.norm();
        let summary = extract_summary(&meta.content);

        let distance_from_prev = match &prev_emb {
            Some(pe) => {
                let d = pe.distance(&emb);
                total_distance += d;
                d
            }
            None => 0.0,
        };

        let distance_to_target = match target_emb {
            Some(te) => emb.distance(te),
            None => 0.0,
        };

        hops.push(PathHop {
            node_id,
            node_type: meta.node_type,
            energy: meta.energy,
            depth,
            generation: meta.lsystem_generation,
            summary,
            distance_from_prev,
            distance_to_target,
        });

        prev_emb = Some(emb);
    }

    let residual_distance = hops.last()
        .map(|h| h.distance_to_target)
        .unwrap_or(f64::INFINITY);

    Ok(ConceptPath {
        hop_count: if hops.is_empty() { 0 } else { hops.len() - 1 },
        hops,
        total_distance,
        reached_target,
        used_fallback,
        residual_distance,
    })
}

/// Public version of `extract_summary` with configurable max length.
/// Used by `hyperbolic_export` and other modules.
pub fn extract_summary_public(content: &serde_json::Value, max_len: usize) -> String {
    const PREFERRED_FIELDS: &[&str] = &[
        "title", "name", "label", "text", "summary", "description",
        "node_label", "topic", "concept",
    ];

    if let Some(obj) = content.as_object() {
        for &field in PREFERRED_FIELDS {
            if let Some(serde_json::Value::String(s)) = obj.get(field) {
                if !s.is_empty() {
                    return truncate(s, max_len);
                }
            }
        }
        for (key, val) in obj {
            if key.starts_with('_') { continue; }
            if let serde_json::Value::String(s) = val {
                if !s.is_empty() {
                    return truncate(s, max_len);
                }
            }
        }
    }
    let raw = content.to_string();
    truncate(&raw, max_len.min(60))
}

/// Extract a human-readable summary from node content JSON.
///
/// Tries common field names (`title`, `name`, `label`, `text`, `summary`,
/// `description`), falls back to first string field, then truncated JSON.
fn extract_summary(content: &serde_json::Value) -> String {
    const PREFERRED_FIELDS: &[&str] = &[
        "title", "name", "label", "text", "summary", "description",
        "node_label", "topic", "concept",
    ];

    if let Some(obj) = content.as_object() {
        // Try preferred fields first
        for &field in PREFERRED_FIELDS {
            if let Some(serde_json::Value::String(s)) = obj.get(field) {
                if !s.is_empty() {
                    return truncate(s, 80);
                }
            }
        }

        // Fall back to first non-empty string field
        for (key, val) in obj {
            if key.starts_with('_') { continue; }
            if let serde_json::Value::String(s) = val {
                if !s.is_empty() {
                    return truncate(s, 80);
                }
            }
        }
    }

    // Last resort: truncated JSON
    let raw = content.to_string();
    truncate(&raw, 60)
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        // Find a valid UTF-8 boundary at or before `max`
        let mut end = max.min(s.len());
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::AdjacencyIndex;
    use crate::model::{Edge, Node, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_storage(dir: &TempDir) -> GraphStorage {
        let p = dir.path().join("rocksdb");
        GraphStorage::open(p.to_str().unwrap()).unwrap()
    }

    fn named_node(x: f64, name: &str) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, 0.0]),
            serde_json::json!({ "name": name }),
        )
    }

    fn build_chain(dir: &TempDir) -> (GraphStorage, AdjacencyIndex, Vec<Node>) {
        let storage = open_storage(dir);
        let adjacency = AdjacencyIndex::new();

        let nodes = vec![
            named_node(0.1, "Mathematics"),
            named_node(0.3, "Information Theory"),
            named_node(0.5, "Algorithms"),
            named_node(0.7, "Computer Science"),
        ];

        for n in &nodes { storage.put_node(n).unwrap(); }

        for i in 0..nodes.len() - 1 {
            let edge = Edge::association(nodes[i].id, nodes[i + 1].id, 0.9);
            storage.put_edge(&edge).unwrap();
            adjacency.add_edge(&edge);
        }

        (storage, adjacency, nodes)
    }

    #[test]
    fn concept_path_linear_chain() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let path = concept_path(
            &storage, &adjacency,
            nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(path.reached_target);
        assert_eq!(path.hop_count, 3);
        assert_eq!(path.hops.len(), 4);
        assert!(path.total_distance > 0.0);
        assert_eq!(path.residual_distance, 0.0);

        // Verify summaries
        assert_eq!(path.hops[0].summary, "Mathematics");
        assert_eq!(path.hops[3].summary, "Computer Science");

        // First hop should have zero distance_from_prev
        assert_eq!(path.hops[0].distance_from_prev, 0.0);
        // Subsequent hops should have positive distance
        assert!(path.hops[1].distance_from_prev > 0.0);
    }

    #[test]
    fn concept_path_has_decreasing_distance_to_target() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let path = concept_path(
            &storage, &adjacency,
            nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        // Each hop should get closer to the target (greedy property)
        for w in path.hops.windows(2) {
            assert!(
                w[1].distance_to_target <= w[0].distance_to_target,
                "hop {} (d={:.4}) should be closer than hop {} (d={:.4})",
                w[1].summary, w[1].distance_to_target,
                w[0].summary, w[0].distance_to_target,
            );
        }
    }

    #[test]
    fn concept_path_from_embedding_works() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let target = vec![0.65_f32, 0.0]; // near "Computer Science" at 0.7

        let path = concept_path_from_embedding(
            &storage, &adjacency,
            nodes[0].id, &target,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(!path.reached_target); // embedding target — never "reached"
        assert!(path.hop_count > 0);
        assert!(path.residual_distance < 1.0);
    }

    #[test]
    fn explain_path_produces_readable_output() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let path = concept_path(
            &storage, &adjacency,
            nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        let explanation = explain_path(&path);

        assert!(explanation.contains("Mathematics"));
        assert!(explanation.contains("Computer Science"));
        assert!(explanation.contains("[1]"));
        assert!(explanation.contains("[4]"));
        assert!(explanation.contains("──"));
    }

    #[test]
    fn concept_path_trivial_same_node() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let path = concept_path(
            &storage, &adjacency,
            nodes[0].id, nodes[0].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        assert!(path.reached_target);
        assert_eq!(path.hop_count, 0);
        assert_eq!(path.hops.len(), 1);
        assert_eq!(path.hops[0].summary, "Mathematics");
    }

    #[test]
    fn extract_summary_preferred_fields() {
        let content = serde_json::json!({
            "data": "ignored",
            "name": "Test Concept",
            "value": 42
        });
        assert_eq!(extract_summary(&content), "Test Concept");

        let content2 = serde_json::json!({
            "title": "My Title",
            "name": "My Name"
        });
        assert_eq!(extract_summary(&content2), "My Title"); // title > name
    }

    #[test]
    fn extract_summary_fallback_to_json() {
        let content = serde_json::json!({ "x": 42, "y": true });
        let summary = extract_summary(&content);
        assert!(!summary.is_empty());
    }

    #[test]
    fn depth_reflects_radial_position() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_chain(&dir);

        let path = concept_path(
            &storage, &adjacency,
            nodes[0].id, nodes[3].id,
            &GreedyRouteConfig::default(),
        ).unwrap();

        // Nodes at x=0.1, 0.3, 0.5, 0.7 → depths should increase
        for w in path.hops.windows(2) {
            assert!(w[1].depth > w[0].depth,
                "{} (r={:.2}) should be deeper than {} (r={:.2})",
                w[1].summary, w[1].depth, w[0].summary, w[0].depth);
        }
    }
}
