//! HTTP dashboard server — REST API + embedded web UI.
//!
//! Exposes a JSON REST API on top of the shared NietzscheDB instance and
//! serves a single-page HTML dashboard that includes graph visualization,
//! CRUD forms, NQL console and stats panel.
//!
//! Endpoints:
//!   GET  /                    → dashboard HTML
//!   GET  /api/stats           → node/edge counts + version
//!   GET  /api/health          → liveness probe
//!   GET  /api/graph           → all nodes + edges (scan, limited to 500)
//!   GET  /api/node/:id        → single node
//!   POST /api/node            → insert node
//!   DELETE /api/node/:id      → delete node
//!   POST /api/edge            → insert edge
//!   DELETE /api/edge/:id      → delete edge
//!   POST /api/query           → NQL query
//!   POST /api/sleep           → trigger sleep cycle

use std::sync::Arc;
use std::net::SocketAddr;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

use nietzsche_graph::{
    Edge, EdgeType, MockVectorStore, NietzscheDB, Node, NodeType, PoincareVector,
};
use nietzsche_query::{parse as nql_parse, execute as nql_execute, Params, QueryResult};
use nietzsche_sleep::{SleepConfig, SleepCycle};

use crate::html::DASHBOARD_HTML;

// ── Shared state ─────────────────────────────────────────────────────────────

type Db = Arc<Mutex<NietzscheDB<MockVectorStore>>>;

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn serve(db: Db, port: u16) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    let app = Router::new()
        .route("/", get(root))
        .route("/api/stats", get(stats))
        .route("/api/health", get(health))
        .route("/api/graph", get(graph))
        .route("/api/node/:id", get(get_node))
        .route("/api/node", post(insert_node))
        .route("/api/node/:id", delete(delete_node))
        .route("/api/edge", post(insert_edge))
        .route("/api/edge/:id", delete(delete_edge))
        .route("/api/query", post(query_nql))
        .route("/api/sleep", post(trigger_sleep))
        .layer(CorsLayer::permissive())
        .with_state(db);

    info!(%addr, "NietzscheDB dashboard listening");

    if let Err(e) = axum::serve(
        tokio::net::TcpListener::bind(addr).await.unwrap(),
        app,
    )
    .await
    {
        warn!(error = %e, "dashboard server error");
    }
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn root() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

// GET /api/stats
#[derive(Serialize)]
struct StatsResponse {
    node_count: usize,
    edge_count: usize,
    version: &'static str,
}

async fn stats(State(db): State<Db>) -> impl IntoResponse {
    let db = db.lock().await;
    Json(StatsResponse {
        node_count: db.node_count().unwrap_or(0),
        edge_count: db.edge_count().unwrap_or(0),
        version: env!("CARGO_PKG_VERSION"),
    })
}

// GET /api/health
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

// GET /api/graph — scan all nodes + edges (max 500 each)
#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<NodeJson>,
    edges: Vec<EdgeJson>,
}

async fn graph(State(db): State<Db>) -> impl IntoResponse {
    let db = db.lock().await;
    let nodes: Vec<NodeJson> = db
        .storage()
        .scan_nodes()
        .unwrap_or_default()
        .into_iter()
        .take(500)
        .map(NodeJson::from)
        .collect();
    let edges: Vec<EdgeJson> = db
        .storage()
        .scan_edges()
        .unwrap_or_default()
        .into_iter()
        .take(500)
        .map(EdgeJson::from)
        .collect();
    Json(GraphResponse { nodes, edges })
}

// GET /api/node/:id
async fn get_node(
    State(db): State<Db>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let db = db.lock().await;
    match db.get_node(uuid) {
        Ok(Some(n)) => Json(NodeJson::from(n)).into_response(),
        Ok(None)    => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"not found"}))).into_response(),
        Err(e)      => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/node
#[derive(Deserialize)]
struct InsertNodeRequest {
    id:        Option<String>,
    node_type: Option<String>,
    energy:    Option<f32>,
    content:   Option<serde_json::Value>,
    embedding: Option<Vec<f64>>,
}

async fn insert_node(
    State(db): State<Db>,
    Json(req): Json<InsertNodeRequest>,
) -> impl IntoResponse {
    let id = req.id
        .as_deref()
        .and_then(|s| s.parse::<Uuid>().ok())
        .unwrap_or_else(Uuid::new_v4);

    let node_type = match req.node_type.as_deref().unwrap_or("Semantic") {
        "Episodic"      => NodeType::Episodic,
        "Concept"       => NodeType::Concept,
        "DreamSnapshot" => NodeType::DreamSnapshot,
        _               => NodeType::Semantic,
    };

    let embedding = req.embedding
        .map(PoincareVector::new)
        .unwrap_or_else(|| PoincareVector::origin(4));

    let content = req.content.unwrap_or(serde_json::Value::Object(Default::default()));
    let mut node = Node::new(id, embedding, content);
    node.node_type = node_type;
    if let Some(e) = req.energy { node.energy = e; }

    let mut db = db.lock().await;
    match db.insert_node(node) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/node/:id
async fn delete_node(
    State(db): State<Db>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let mut db = db.lock().await;
    match db.delete_node(uuid) {
        Ok(_)  => Json(serde_json::json!({"deleted": uuid.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/edge
#[derive(Deserialize)]
struct InsertEdgeRequest {
    from:      String,
    to:        String,
    edge_type: Option<String>,
    weight:    Option<f32>,
}

async fn insert_edge(
    State(db): State<Db>,
    Json(req): Json<InsertEdgeRequest>,
) -> impl IntoResponse {
    let from = match req.from.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid from uuid"}))).into_response(),
    };
    let to = match req.to.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid to uuid"}))).into_response(),
    };
    let edge_type = match req.edge_type.as_deref().unwrap_or("Association") {
        "Hierarchical"     => EdgeType::Hierarchical,
        "LSystemGenerated" => EdgeType::LSystemGenerated,
        "Pruned"           => EdgeType::Pruned,
        _                  => EdgeType::Association,
    };
    let weight = req.weight.unwrap_or(1.0);
    let mut edge = Edge::new(from, to, edge_type, weight);

    let mut db = db.lock().await;
    let id = edge.id;
    match db.insert_edge(edge) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/edge/:id
async fn delete_edge(
    State(db): State<Db>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let mut db = db.lock().await;
    match db.delete_edge(uuid) {
        Ok(_)  => Json(serde_json::json!({"deleted": uuid.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/query  — NQL via nietzsche-query executor
#[derive(Deserialize)]
struct QueryRequest { nql: String }

async fn query_nql(
    State(db): State<Db>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let query = match nql_parse(&req.nql) {
        Ok(q)  => q,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };
    let db = db.lock().await;
    let params = Params::new();
    match nql_execute(&query, db.storage(), db.adjacency(), &params) {
        Ok(results) => {
            let nodes: Vec<NodeJson> = results.into_iter().filter_map(|r| {
                if let QueryResult::Node(n) = r { Some(NodeJson::from(n)) } else { None }
            }).collect();
            Json(serde_json::json!({"nodes": nodes, "error": null})).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/sleep
#[derive(Deserialize)]
struct SleepRequest {
    noise:               Option<f64>,
    adam_steps:          Option<usize>,
    hausdorff_threshold: Option<f32>,
}

async fn trigger_sleep(
    State(db): State<Db>,
    Json(req): Json<SleepRequest>,
) -> impl IntoResponse {
    let cfg = SleepConfig {
        noise:               req.noise.unwrap_or(0.02),
        adam_steps:          req.adam_steps.unwrap_or(10),
        adam_lr:             5e-3,
        hausdorff_threshold: req.hausdorff_threshold.unwrap_or(0.15),
    };
    let mut db  = db.lock().await;
    let seed: u64 = rand::random();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
    match SleepCycle::run(&cfg, &mut *db, &mut rng) {
        Ok(r) => Json(serde_json::json!({
            "hausdorff_before":  r.hausdorff_before,
            "hausdorff_after":   r.hausdorff_after,
            "hausdorff_delta":   r.hausdorff_delta,
            "committed":         r.committed,
            "nodes_perturbed":   r.nodes_perturbed,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── JSON DTOs ─────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct NodeJson {
    id:         String,
    node_type:  String,
    energy:     f32,
    depth:      f32,
    hausdorff:  f32,
    created_at: i64,
    content:    serde_json::Value,
}

impl From<Node> for NodeJson {
    fn from(n: Node) -> Self {
        Self {
            id:         n.id.to_string(),
            node_type:  format!("{:?}", n.node_type),
            energy:     n.energy,
            depth:      n.depth,
            hausdorff:  n.hausdorff_local,
            created_at: n.created_at,
            content:    n.content,
        }
    }
}

#[derive(Serialize)]
struct EdgeJson {
    id:        String,
    from:      String,
    to:        String,
    edge_type: String,
    weight:    f32,
}

impl From<Edge> for EdgeJson {
    fn from(e: Edge) -> Self {
        Self {
            id:        e.id.to_string(),
            from:      e.from.to_string(),
            to:        e.to.to_string(),
            edge_type: format!("{:?}", e.edge_type),
            weight:    e.weight,
        }
    }
}
