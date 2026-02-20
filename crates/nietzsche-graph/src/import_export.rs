//! Import/Export: CSV and JSONL streaming for nodes and edges.

use std::io::{BufRead, Write};
use uuid::Uuid;

use crate::error::GraphError;
use crate::model::{Edge, EdgeType, Node, NodeType, PoincareVector};
use crate::storage::GraphStorage;

/// Result of an import operation.
#[derive(Debug, Default)]
pub struct ImportResult {
    pub imported: usize,
    pub errors: usize,
    pub error_messages: Vec<String>,
}

// ── Export ───────────────────────────────────────────────────────────────────

/// Export all nodes as CSV lines (header + data).
///
/// Columns: `id,node_type,energy,depth,hausdorff,created_at,content_json`
pub fn export_nodes_csv<W: Write>(
    storage: &GraphStorage,
    writer: &mut W,
) -> Result<usize, GraphError> {
    writeln!(writer, "id,node_type,energy,depth,hausdorff,created_at,content_json")
        .map_err(|e| GraphError::Storage(format!("csv write: {e}")))?;

    let mut count = 0;
    for result in storage.iter_nodes_meta() {
        let meta = result?;
        let content_str = serde_json::to_string(&meta.content).unwrap_or_default();
        // Escape CSV: wrap content in quotes, double any internal quotes
        let escaped = content_str.replace('"', "\"\"");
        writeln!(
            writer,
            "{},{:?},{},{},{},{},\"{}\"",
            meta.id, meta.node_type, meta.energy, meta.depth,
            meta.hausdorff_local, meta.created_at, escaped,
        ).map_err(|e| GraphError::Storage(format!("csv write: {e}")))?;
        count += 1;
    }
    Ok(count)
}

/// Export all edges as CSV lines (header + data).
///
/// Columns: `id,from,to,edge_type,weight`
pub fn export_edges_csv<W: Write>(
    storage: &GraphStorage,
    writer: &mut W,
) -> Result<usize, GraphError> {
    writeln!(writer, "id,from,to,edge_type,weight")
        .map_err(|e| GraphError::Storage(format!("csv write: {e}")))?;

    let mut count = 0;
    for result in storage.iter_edges() {
        let edge = result?;
        writeln!(
            writer,
            "{},{},{},{:?},{}",
            edge.id, edge.from, edge.to, edge.edge_type, edge.weight,
        ).map_err(|e| GraphError::Storage(format!("csv write: {e}")))?;
        count += 1;
    }
    Ok(count)
}

/// Export all nodes as JSONL (one JSON object per line).
pub fn export_nodes_jsonl<W: Write>(
    storage: &GraphStorage,
    writer: &mut W,
) -> Result<usize, GraphError> {
    let mut count = 0;
    for result in storage.iter_nodes() {
        let node = result?;
        let (meta, embedding) = node.into_parts();
        let json = serde_json::json!({
            "id": meta.id.to_string(),
            "node_type": format!("{:?}", meta.node_type),
            "energy": meta.energy,
            "depth": meta.depth,
            "hausdorff": meta.hausdorff_local,
            "created_at": meta.created_at,
            "content": meta.content,
            "embedding": embedding.coords_f64(),
        });
        serde_json::to_writer(&mut *writer, &json)
            .map_err(|e| GraphError::Storage(format!("jsonl write: {e}")))?;
        writeln!(writer)
            .map_err(|e| GraphError::Storage(format!("jsonl write: {e}")))?;
        count += 1;
    }
    Ok(count)
}

/// Export all edges as JSONL.
pub fn export_edges_jsonl<W: Write>(
    storage: &GraphStorage,
    writer: &mut W,
) -> Result<usize, GraphError> {
    let mut count = 0;
    for result in storage.iter_edges() {
        let edge = result?;
        let json = serde_json::json!({
            "id": edge.id.to_string(),
            "from": edge.from.to_string(),
            "to": edge.to.to_string(),
            "edge_type": format!("{:?}", edge.edge_type),
            "weight": edge.weight,
        });
        serde_json::to_writer(&mut *writer, &json)
            .map_err(|e| GraphError::Storage(format!("jsonl write: {e}")))?;
        writeln!(writer)
            .map_err(|e| GraphError::Storage(format!("jsonl write: {e}")))?;
        count += 1;
    }
    Ok(count)
}

// ── Import ───────────────────────────────────────────────────────────────────

/// Import nodes from JSONL format.
///
/// Each line should be a JSON object with at least `embedding` (array of floats).
/// Optional fields: `id`, `node_type`, `energy`, `content`.
pub fn import_nodes_jsonl<R: BufRead>(
    storage: &GraphStorage,
    reader: R,
) -> Result<ImportResult, GraphError> {
    let mut result = ImportResult::default();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("read error: {e}"));
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let obj: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("json parse: {e}"));
                continue;
            }
        };

        let id = obj.get("id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::new_v4);

        let embedding_coords: Vec<f64> = obj.get("embedding")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_else(|| vec![0.0; 4]);

        let embedding = PoincareVector::from_f64(embedding_coords);

        let content = obj.get("content").cloned().unwrap_or(serde_json::Value::Null);

        let node_type = match obj.get("node_type").and_then(|v| v.as_str()) {
            Some("Episodic") => NodeType::Episodic,
            Some("Concept") => NodeType::Concept,
            Some("DreamSnapshot") => NodeType::DreamSnapshot,
            _ => NodeType::Semantic,
        };

        let energy = obj.get("energy")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);

        let mut node = Node::new(id, embedding, content);
        node.meta.node_type = node_type;
        node.energy = energy;

        match storage.put_node(&node) {
            Ok(_) => result.imported += 1,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("put node {}: {e}", id));
            }
        }
    }

    Ok(result)
}

/// Import edges from JSONL format.
///
/// Each line should be a JSON object with `from`, `to` (UUID strings).
/// Optional: `id`, `edge_type`, `weight`.
pub fn import_edges_jsonl<R: BufRead>(
    storage: &GraphStorage,
    reader: R,
) -> Result<ImportResult, GraphError> {
    let mut result = ImportResult::default();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("read error: {e}"));
                continue;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }

        let obj: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("json parse: {e}"));
                continue;
            }
        };

        let from = match obj.get("from").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()) {
            Some(u) => u,
            None => {
                result.errors += 1;
                result.error_messages.push("missing or invalid 'from' UUID".into());
                continue;
            }
        };
        let to = match obj.get("to").and_then(|v| v.as_str()).and_then(|s| Uuid::parse_str(s).ok()) {
            Some(u) => u,
            None => {
                result.errors += 1;
                result.error_messages.push("missing or invalid 'to' UUID".into());
                continue;
            }
        };

        let id = obj.get("id")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::new_v4);

        let edge_type = match obj.get("edge_type").and_then(|v| v.as_str()) {
            Some("Hierarchical") => EdgeType::Hierarchical,
            Some("LSystemGenerated") => EdgeType::LSystemGenerated,
            Some("Pruned") => EdgeType::Pruned,
            _ => EdgeType::Association,
        };

        let weight = obj.get("weight")
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0);

        let mut edge = Edge::new(from, to, edge_type, weight);
        edge.id = id;

        match storage.put_edge(&edge) {
            Ok(_) => result.imported += 1,
            Err(e) => {
                result.errors += 1;
                result.error_messages.push(format!("put edge {}: {e}", id));
            }
        }
    }

    Ok(result)
}
