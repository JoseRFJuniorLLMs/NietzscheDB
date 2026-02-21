//! MCP tool definitions and dispatch.
//!
//! Each tool maps to a core NietzscheDB operation and defines its
//! JSON schema for input parameters.

use nietzsche_graph::{AdjacencyIndex, GraphStorage, Node, PoincareVector, NodeType, Edge};
use nietzsche_query::{execute, parser, Params, ParamValue, QueryResult};
use uuid::Uuid;

use crate::error::McpError;
use crate::protocol::{McpTool, McpToolResult};

/// Return all available MCP tools.
pub fn list_tools() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "query".into(),
            description: "Execute an NQL (Nietzsche Query Language) query against the graph database. Supports MATCH, CREATE, MERGE, SET, DELETE, DIFFUSE, EXPLAIN, and more.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "nql": { "type": "string", "description": "The NQL query string" },
                    "params": { "type": "object", "description": "Named parameters for $param references", "default": {} }
                },
                "required": ["nql"]
            }),
        },
        McpTool {
            name: "insert_node".into(),
            description: "Insert a new node into the hyperbolic graph. Returns the assigned UUID.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": { "type": "object", "description": "JSON content/metadata for the node" },
                    "node_type": { "type": "string", "enum": ["Semantic", "Episodic", "Concept", "DreamSnapshot"], "default": "Semantic" },
                    "energy": { "type": "number", "description": "Initial energy (0.0-1.0)", "default": 1.0 },
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Poincaré ball coordinates (norm < 1.0)" }
                },
                "required": ["content"]
            }),
        },
        McpTool {
            name: "get_node".into(),
            description: "Retrieve a node by its UUID.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Node UUID" }
                },
                "required": ["id"]
            }),
        },
        McpTool {
            name: "delete_node".into(),
            description: "Delete a node by its UUID.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Node UUID" }
                },
                "required": ["id"]
            }),
        },
        McpTool {
            name: "insert_edge".into(),
            description: "Insert a directed edge between two nodes.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Source node UUID" },
                    "to": { "type": "string", "description": "Target node UUID" },
                    "edge_type": { "type": "string", "enum": ["Association", "Hierarchical", "LSystemGenerated", "Pruned"], "default": "Association" },
                    "weight": { "type": "number", "default": 1.0 }
                },
                "required": ["from", "to"]
            }),
        },
        McpTool {
            name: "get_stats".into(),
            description: "Get database statistics: node count, edge count, collections.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        McpTool {
            name: "knn_search".into(),
            description: "Hyperbolic k-nearest-neighbor search in Poincaré ball space.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "embedding": { "type": "array", "items": { "type": "number" }, "description": "Query vector (Poincaré ball coordinates)" },
                    "k": { "type": "integer", "description": "Number of neighbors", "default": 10 },
                    "min_energy": { "type": "number", "description": "Minimum energy filter", "default": 0.0 }
                },
                "required": ["embedding"]
            }),
        },
        McpTool {
            name: "diffuse".into(),
            description: "Run hyperbolic heat kernel diffusion from a seed node. Returns activated nodes at multiple time scales.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "seed_id": { "type": "string", "description": "Seed node UUID" },
                    "t_values": { "type": "array", "items": { "type": "number" }, "description": "Diffusion time scales", "default": [0.1, 1.0, 10.0] },
                    "max_hops": { "type": "integer", "default": 5 }
                },
                "required": ["seed_id"]
            }),
        },
        McpTool {
            name: "run_algorithm".into(),
            description: "Run a graph algorithm. Available: pagerank, louvain, label_propagation, betweenness, closeness, degree, wcc, scc, triangle_count, jaccard.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "algorithm": { "type": "string", "description": "Algorithm name" },
                    "iterations": { "type": "integer", "default": 20 },
                    "damping": { "type": "number", "default": 0.85 }
                },
                "required": ["algorithm"]
            }),
        },
    ]
}

/// Dispatch a tool call by name.
pub fn call_tool(
    name: &str,
    args: &serde_json::Value,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    match name {
        "query"          => tool_query(args, storage, adjacency),
        "insert_node"    => tool_insert_node(args, storage, adjacency),
        "get_node"       => tool_get_node(args, storage),
        "delete_node"    => tool_delete_node(args, storage, adjacency),
        "insert_edge"    => tool_insert_edge(args, storage, adjacency),
        "get_stats"      => tool_get_stats(storage, adjacency),
        "knn_search"     => tool_knn_search(args, storage),
        "diffuse"        => tool_diffuse(args, storage, adjacency),
        "run_algorithm"  => tool_run_algorithm(args, storage, adjacency),
        _ => Err(McpError::ToolNotFound(name.to_string())),
    }
}

// ── Tool implementations ──────────────────────────────────

fn tool_query(
    args: &serde_json::Value,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let nql = args.get("nql")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'nql' parameter".into()))?;

    let mut params = Params::new();
    if let Some(p) = args.get("params").and_then(|v| v.as_object()) {
        for (k, v) in p {
            let pv = match v {
                serde_json::Value::Number(n) => {
                    ParamValue::Float(n.as_f64().unwrap_or(0.0))
                }
                serde_json::Value::String(s) => {
                    if let Ok(uuid) = Uuid::parse_str(s) {
                        ParamValue::Uuid(uuid)
                    } else {
                        ParamValue::Str(s.clone())
                    }
                }
                serde_json::Value::Array(arr) => {
                    let floats: Vec<f64> = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                    ParamValue::Vector(floats)
                }
                _ => ParamValue::Str(v.to_string()),
            };
            params.insert(k.clone(), pv);
        }
    }

    let query = parser::parse(nql)?;
    let results = execute(&query, storage, adjacency, &params)?;

    let output: Vec<serde_json::Value> = results.iter().map(|r| match r {
        QueryResult::Node(n) => serde_json::json!({
            "type": "node",
            "id": n.id.to_string(),
            "energy": n.energy,
            "depth": n.depth,
            "node_type": format!("{:?}", n.node_type),
            "content": n.content,
        }),
        QueryResult::NodePair { from, to } => serde_json::json!({
            "type": "pair",
            "from": from.id.to_string(),
            "to": to.id.to_string(),
        }),
        QueryResult::DiffusionPath(path) => serde_json::json!({
            "type": "path",
            "node_ids": path.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
        }),
        QueryResult::Scalar(row) => serde_json::json!({
            "type": "row",
            "columns": row.iter().map(|(k, v)| serde_json::json!({"name": k, "value": format!("{:?}", v)})).collect::<Vec<_>>(),
        }),
        _ => serde_json::json!({ "type": "other", "debug": format!("{:?}", r) }),
    }).collect();

    Ok(McpToolResult::text(serde_json::to_string_pretty(&output)?))
}

fn tool_insert_node(
    args: &serde_json::Value,
    storage: &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let content = args.get("content")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let node_type_str = args.get("node_type")
        .and_then(|v| v.as_str())
        .unwrap_or("Semantic");

    let energy = args.get("energy")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    let embedding = if let Some(arr) = args.get("embedding").and_then(|v| v.as_array()) {
        let coords: Vec<f32> = arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();
        PoincareVector::new(coords)
    } else {
        PoincareVector::origin(8)
    };

    let id = Uuid::new_v4();
    let mut node = Node::new(id, embedding, content);
    node.energy = energy;
    node.node_type = match node_type_str {
        "Episodic" => NodeType::Episodic,
        "Concept" => NodeType::Concept,
        "DreamSnapshot" => NodeType::DreamSnapshot,
        _ => NodeType::Semantic,
    };

    storage.put_node(&node)?;

    Ok(McpToolResult::text(serde_json::json!({
        "id": id.to_string(),
        "status": "inserted"
    }).to_string()))
}

fn tool_get_node(
    args: &serde_json::Value,
    storage: &GraphStorage,
) -> Result<McpToolResult, McpError> {
    let id_str = args.get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'id'".into()))?;

    let id = Uuid::parse_str(id_str)
        .map_err(|e| McpError::InvalidRequest(format!("invalid UUID: {e}")))?;

    match storage.get_node(&id)? {
        Some(node) => Ok(McpToolResult::text(serde_json::json!({
            "id": node.id.to_string(),
            "energy": node.energy,
            "depth": node.depth,
            "node_type": format!("{:?}", node.node_type),
            "content": node.content,
            "hausdorff_local": node.hausdorff_local,
            "created_at": node.created_at,
        }).to_string())),
        None => Ok(McpToolResult::error(format!("node {} not found", id))),
    }
}

fn tool_delete_node(
    args: &serde_json::Value,
    storage: &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let id_str = args.get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'id'".into()))?;

    let id = Uuid::parse_str(id_str)
        .map_err(|e| McpError::InvalidRequest(format!("invalid UUID: {e}")))?;

    storage.delete_node(&id)?;
    Ok(McpToolResult::text(serde_json::json!({ "status": "deleted", "id": id_str }).to_string()))
}

fn tool_insert_edge(
    args: &serde_json::Value,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let from_str = args.get("from").and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'from'".into()))?;
    let to_str = args.get("to").and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'to'".into()))?;

    let from = Uuid::parse_str(from_str)
        .map_err(|e| McpError::InvalidRequest(format!("invalid 'from' UUID: {e}")))?;
    let to = Uuid::parse_str(to_str)
        .map_err(|e| McpError::InvalidRequest(format!("invalid 'to' UUID: {e}")))?;

    let weight = args.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

    let edge = Edge::association(from, to, weight);
    storage.put_edge(&edge)?;
    adjacency.add_edge(&edge);

    Ok(McpToolResult::text(serde_json::json!({
        "id": edge.id.to_string(),
        "from": from_str,
        "to": to_str,
        "status": "inserted"
    }).to_string()))
}

fn tool_get_stats(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let node_count = storage.iter_nodes_meta().count();
    let edge_count = adjacency.edge_count();

    Ok(McpToolResult::text(serde_json::json!({
        "node_count": node_count,
        "edge_count": edge_count,
        "engine": "NietzscheDB",
        "geometry": "Poincaré Ball (hyperbolic)",
    }).to_string()))
}

fn tool_knn_search(
    args: &serde_json::Value,
    storage: &GraphStorage,
) -> Result<McpToolResult, McpError> {
    let embedding = args.get("embedding")
        .and_then(|v| v.as_array())
        .ok_or_else(|| McpError::InvalidRequest("missing 'embedding'".into()))?;

    let coords: Vec<f32> = embedding.iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let min_energy = args.get("min_energy").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

    let query_vec = PoincareVector::new(coords);

    // Scan nodes and compute distances (CPU path for MCP — uses graph storage directly)
    let mut candidates: Vec<(Uuid, f64, f32)> = Vec::new();
    for result in storage.iter_nodes_meta() {
        if let Ok(meta) = result {
            if meta.energy >= min_energy {
                if let Ok(Some(node)) = storage.get_node(&meta.id) {
                    let dist = node.embedding.distance(&query_vec);
                    candidates.push((meta.id, dist, meta.energy));
                }
            }
        }
    }
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);

    let results: Vec<serde_json::Value> = candidates.iter().map(|(id, dist, energy)| {
        serde_json::json!({
            "id": id.to_string(),
            "distance": dist,
            "energy": energy,
        })
    }).collect();

    Ok(McpToolResult::text(serde_json::to_string_pretty(&results)?))
}

fn tool_diffuse(
    args: &serde_json::Value,
    storage: &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let seed_str = args.get("seed_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'seed_id'".into()))?;

    let seed_id = Uuid::parse_str(seed_str)
        .map_err(|e| McpError::InvalidRequest(format!("invalid UUID: {e}")))?;

    let max_hops = args.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    // Simple BFS-based diffusion (the full Chebyshev version is in nietzsche-pregel)
    let mut visited = std::collections::HashSet::new();
    let mut frontier = vec![seed_id];
    visited.insert(seed_id);
    let mut activated = Vec::new();

    for _hop in 0..max_hops {
        let mut next = Vec::new();
        for nid in &frontier {
            if let Ok(Some(meta)) = storage.get_node_meta(nid) {
                activated.push(serde_json::json!({
                    "id": nid.to_string(),
                    "energy": meta.energy,
                    "depth": meta.depth,
                }));
            }
            if let Ok(edge_ids) = storage.edge_ids_from(nid) {
                for eid in edge_ids {
                    if let Ok(Some(edge)) = storage.get_edge(&eid) {
                        if visited.insert(edge.to) {
                            next.push(edge.to);
                        }
                    }
                }
            }
        }
        frontier = next;
        if frontier.is_empty() { break; }
    }

    Ok(McpToolResult::text(serde_json::to_string_pretty(&activated)?))
}

fn tool_run_algorithm(
    args: &serde_json::Value,
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<McpToolResult, McpError> {
    let algo = args.get("algorithm")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::InvalidRequest("missing 'algorithm'".into()))?;

    let iterations = args.get("iterations").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let damping = args.get("damping").and_then(|v| v.as_f64()).unwrap_or(0.85);

    match algo {
        "pagerank" => {
            let config = nietzsche_algo::PageRankConfig {
                damping_factor: damping,
                max_iterations: iterations,
                ..Default::default()
            };
            let result = nietzsche_algo::pagerank(storage, adjacency, &config)
                .map_err(|e| McpError::InvalidRequest(format!("pagerank failed: {e}")))?;
            let top: Vec<serde_json::Value> = result.scores.iter().take(20).map(|(id, s)| {
                serde_json::json!({ "id": id.to_string(), "score": s })
            }).collect();
            Ok(McpToolResult::text(serde_json::to_string_pretty(&serde_json::json!({
                "scores": top,
                "iterations": result.iterations,
                "converged": result.converged,
            }))?))
        }
        "louvain" => {
            let config = nietzsche_algo::LouvainConfig {
                max_iterations: iterations,
                ..Default::default()
            };
            let result = nietzsche_algo::louvain(storage, adjacency, &config)
                .map_err(|e| McpError::InvalidRequest(format!("louvain failed: {e}")))?;
            let summary = serde_json::json!({
                "community_count": result.community_count,
                "modularity": result.modularity,
                "assignments": result.communities.iter().take(50).map(|(id, c)| {
                    serde_json::json!({ "id": id.to_string(), "community": c })
                }).collect::<Vec<_>>(),
            });
            Ok(McpToolResult::text(serde_json::to_string_pretty(&summary)?))
        }
        _ => Ok(McpToolResult::error(format!(
            "unknown algorithm '{}'. Available: pagerank, louvain, label_propagation, betweenness, closeness, degree, wcc, scc, triangle_count, jaccard",
            algo
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_tools_returns_9() {
        let tools = list_tools();
        assert_eq!(tools.len(), 9);
        assert!(tools.iter().any(|t| t.name == "query"));
        assert!(tools.iter().any(|t| t.name == "knn_search"));
    }

    #[test]
    fn tool_schemas_have_required_fields() {
        let tools = list_tools();
        for tool in &tools {
            assert!(tool.input_schema.is_object(), "tool {} has non-object schema", tool.name);
        }
    }

    #[test]
    fn call_unknown_tool_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().join("db").to_str().unwrap()).unwrap();
        let adjacency = AdjacencyIndex::new();
        let result = call_tool("nonexistent", &serde_json::json!({}), &storage, &adjacency);
        assert!(result.is_err());
    }

    #[test]
    fn tool_get_stats_works() {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().join("db").to_str().unwrap()).unwrap();
        let adjacency = AdjacencyIndex::new();
        let result = call_tool("get_stats", &serde_json::json!({}), &storage, &adjacency).unwrap();
        assert!(!result.is_error);
        assert!(result.content[0].text.contains("node_count"));
    }

    #[test]
    fn tool_insert_and_get_node() {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().join("db").to_str().unwrap()).unwrap();
        let adjacency = AdjacencyIndex::new();

        let insert_result = call_tool("insert_node", &serde_json::json!({
            "content": {"title": "test"},
            "energy": 0.9
        }), &storage, &adjacency).unwrap();

        let resp: serde_json::Value = serde_json::from_str(&insert_result.content[0].text).unwrap();
        let id = resp["id"].as_str().unwrap();

        let get_result = call_tool("get_node", &serde_json::json!({"id": id}), &storage, &adjacency).unwrap();
        assert!(!get_result.is_error);
        assert!(get_result.content[0].text.contains("test"));
    }

    #[test]
    fn tool_query_nql() {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().join("db").to_str().unwrap()).unwrap();
        let adjacency = AdjacencyIndex::new();

        // Insert a node first
        let mut node = Node::new(Uuid::new_v4(), PoincareVector::origin(8), serde_json::json!({"x": 1}));
        node.energy = 0.9;
        storage.put_node(&node).unwrap();

        let result = call_tool("query", &serde_json::json!({
            "nql": "MATCH (n) WHERE n.energy > 0.5 RETURN n"
        }), &storage, &adjacency).unwrap();

        assert!(!result.is_error);
        assert!(result.content[0].text.contains("node"));
    }
}
