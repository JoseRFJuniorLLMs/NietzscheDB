//! HTTP dashboard server — REST API + embedded web UI.
//!
//! Exposes a JSON REST API backed by the shared [`CollectionManager`] and
//! serves a single-page HTML dashboard for graph visualization, CRUD forms,
//! NQL console, and stats panel.
//!
//! All data-plane endpoints target the `"default"` collection.  A `?collection=`
//! query parameter can be added in the future (Fase B+).
//!
//! Endpoints:
//!   GET  /                        → dashboard HTML
//!   GET  /api/stats               → total node/edge counts + version
//!   GET  /api/health              → liveness probe
//!   GET  /api/collections         → list all collections
//!   GET  /api/graph               → default collection: all nodes + edges (≤500 each)
//!   GET  /api/node/:id            → single node (default collection)
//!   POST /api/node                → insert node (default collection)
//!   DELETE /api/node/:id          → delete node (default collection)
//!   POST /api/edge                → insert edge (default collection)
//!   DELETE /api/edge/:id          → delete edge (default collection)
//!   POST /api/query               → NQL query (default collection)
//!   POST /api/sleep               → trigger sleep cycle (default collection)
//!   GET  /api/agency/health        → all persisted HealthReports
//!   GET  /api/agency/health/latest → most recent HealthReport
//!   GET  /api/agency/counterfactual/remove/:id → simulate removing node
//!   POST /api/agency/counterfactual/add        → simulate adding node

use std::sync::Arc;
use std::net::SocketAddr;

use axum::{
    extract::{Extension, Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

use nietzsche_cluster::ClusterRegistry;
use nietzsche_graph::{
    CollectionManager,
    Edge, EdgeType, Node, NodeType, PoincareVector,
};
use nietzsche_query::{parse as nql_parse, execute as nql_execute, Params, QueryResult};
use nietzsche_sleep::{SleepConfig, SleepCycle};

use crate::html::DASHBOARD_HTML;
use crate::metrics::OperationMetrics;

// ── Shared state ──────────────────────────────────────────────────────────────

type AppState = (Arc<CollectionManager>, Arc<OperationMetrics>);

// ── Entry point ───────────────────────────────────────────────────────────────

pub async fn serve(
    db: Arc<CollectionManager>,
    ops: Arc<OperationMetrics>,
    port: u16,
    cluster: Option<ClusterRegistry>,
) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let state: AppState = (db, ops);

    let app = Router::new()
        .route("/", get(root))
        .route("/metrics", get(prometheus_metrics))
        .route("/api/stats", get(stats))
        .route("/api/status", get(stats))   // alias for dashboard compatibility
        .route("/api/health", get(health))
        .route("/api/collections", get(list_collections))
        .route("/api/graph", get(graph))
        .route("/api/node/:id", get(get_node))
        .route("/api/node", post(insert_node))
        .route("/api/node/:id", delete(delete_node))
        .route("/api/edge", post(insert_edge))
        .route("/api/edge/:id", delete(delete_edge))
        .route("/api/query", post(query_nql))
        .route("/api/sleep", post(trigger_sleep))
        .route("/api/batch/nodes", post(batch_insert_nodes))
        .route("/api/batch/edges", post(batch_insert_edges))
        // Graph algorithms
        .route("/api/algo/pagerank", get(algo_pagerank))
        .route("/api/algo/louvain", get(algo_louvain))
        .route("/api/algo/labelprop", get(algo_labelprop))
        .route("/api/algo/betweenness", get(algo_betweenness))
        .route("/api/algo/closeness", get(algo_closeness))
        .route("/api/algo/degree", get(algo_degree))
        .route("/api/algo/wcc", get(algo_wcc))
        .route("/api/algo/scc", get(algo_scc))
        .route("/api/algo/triangles", get(algo_triangles))
        .route("/api/algo/jaccard", get(algo_jaccard))
        // Backup/Restore
        .route("/api/backup", post(create_backup))
        .route("/api/backup", get(list_backups))
        // Full-Text Search
        .route("/api/search", get(fulltext_search))
        // Export
        .route("/api/export/nodes", get(export_nodes))
        .route("/api/export/edges", get(export_edges))
        // Agency
        .route("/api/agency/health", get(agency_health))
        .route("/api/agency/health/latest", get(agency_health_latest))
        .route("/api/agency/counterfactual/remove/:id", get(agency_cf_remove))
        .route("/api/agency/counterfactual/add", post(agency_cf_add))
        .route("/api/agency/desires", get(agency_desires))
        .route("/api/agency/desires/:id/fulfill", post(agency_fulfill_desire))
        .route("/api/agency/observer", get(agency_observer))
        .route("/api/agency/evolution", get(agency_evolution))
        .route("/api/agency/narrative", get(agency_narrative))
        .route("/api/agency/quantum/map", post(agency_quantum_map))
        .route("/api/agency/quantum/fidelity", post(agency_quantum_fidelity))
        // Cluster
        .route("/api/cluster/status", get(cluster_status))
        .route("/api/cluster/ring", get(cluster_ring))
        .layer(Extension(cluster))
        .layer(CorsLayer::permissive())
        .with_state(state);

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

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Get the default collection, or return 500 if somehow unavailable.
macro_rules! default_col {
    ($cm:expr) => {
        match $cm.get_or_default("default") {
            Some(db) => db,
            None => return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "default collection unavailable"})),
            ).into_response(),
        }
    };
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn root() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

// GET /metrics — Prometheus text exposition format
async fn prometheus_metrics(State((cm, ops)): State<AppState>) -> impl IntoResponse {
    let body = ops.to_prometheus(&cm);
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
}

// GET /api/stats — aggregate across all collections
#[derive(Serialize)]
struct StatsResponse {
    node_count: usize,
    edge_count: usize,
    collections: usize,
    version: &'static str,
}

async fn stats(State((cm, _ops)): State<AppState>) -> impl IntoResponse {
    let infos = cm.list();
    let (nodes, edges) = infos.iter().fold((0, 0), |(n, e), i| {
        (n + i.node_count, e + i.edge_count)
    });
    Json(StatsResponse {
        node_count:  nodes,
        edge_count:  edges,
        collections: infos.len(),
        version:     env!("CARGO_PKG_VERSION"),
    })
}

// GET /api/health
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

// GET /api/collections
#[derive(Serialize)]
struct CollectionJson {
    name:       String,
    dim:        usize,
    metric:     String,
    node_count: usize,
    edge_count: usize,
}

async fn list_collections(State((cm, _ops)): State<AppState>) -> impl IntoResponse {
    let list: Vec<CollectionJson> = cm.list().into_iter().map(|i| CollectionJson {
        name:       i.name,
        dim:        i.dim,
        metric:     i.metric,
        node_count: i.node_count,
        edge_count: i.edge_count,
    }).collect();
    Json(list)
}

// GET /api/graph — default collection, max 500 nodes + edges
#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<NodeJson>,
    edges: Vec<EdgeJson>,
}

async fn graph(State((cm, _ops)): State<AppState>) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
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
    Json(GraphResponse { nodes, edges }).into_response()
}

// GET /api/node/:id
async fn get_node(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = default_col!(cm);
    let db = shared.read().await;
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
    State((cm, _ops)): State<AppState>,
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
        .map(PoincareVector::from_f64)
        .unwrap_or_else(|| PoincareVector::origin(4));

    let content = req.content.unwrap_or(serde_json::Value::Object(Default::default()));
    let mut node = Node::new(id, embedding, content);
    node.node_type = node_type;
    if let Some(e) = req.energy { node.energy = e; }

    let shared = default_col!(cm);
    let mut db = shared.write().await;
    match db.insert_node(node) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/node/:id
async fn delete_node(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = default_col!(cm);
    let mut db = shared.write().await;
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
    State((cm, _ops)): State<AppState>,
    Json(req): Json<InsertEdgeRequest>,
) -> impl IntoResponse {
    let from = match req.from.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid from uuid"}))).into_response(),
    };
    let to = match req.to.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid to uuid"}))).into_response(),
    };
    let edge_type = match req.edge_type.as_deref().unwrap_or("Association") {
        "Hierarchical"     => EdgeType::Hierarchical,
        "LSystemGenerated" => EdgeType::LSystemGenerated,
        "Pruned"           => EdgeType::Pruned,
        _                  => EdgeType::Association,
    };
    let edge = Edge::new(from, to, edge_type, req.weight.unwrap_or(1.0));
    let id   = edge.id;

    let shared = default_col!(cm);
    let mut db = shared.write().await;
    match db.insert_edge(edge) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/edge/:id
async fn delete_edge(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = default_col!(cm);
    let mut db = shared.write().await;
    match db.delete_edge(uuid) {
        Ok(_)  => Json(serde_json::json!({"deleted": uuid.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/query
#[derive(Deserialize)]
struct QueryRequest { nql: String }

async fn query_nql(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let query = match nql_parse(&req.nql) {
        Ok(q)  => q,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };
    let shared = default_col!(cm);
    let db     = shared.read().await;
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
    State((cm, _ops)): State<AppState>,
    Json(req): Json<SleepRequest>,
) -> impl IntoResponse {
    let cfg = SleepConfig {
        noise:               req.noise.unwrap_or(0.02),
        adam_steps:          req.adam_steps.unwrap_or(10),
        adam_lr:             5e-3,
        hausdorff_threshold: req.hausdorff_threshold.unwrap_or(0.15),
    };
    let shared   = default_col!(cm);
    let mut db   = shared.write().await;
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

// POST /api/batch/nodes
#[derive(Deserialize)]
struct BatchInsertNodesRequest {
    nodes: Vec<InsertNodeRequest>,
}

async fn batch_insert_nodes(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<BatchInsertNodesRequest>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let mut db = shared.write().await;

    let mut nodes_vec = Vec::with_capacity(req.nodes.len());
    let mut ids = Vec::with_capacity(req.nodes.len());

    for nr in req.nodes {
        let id = nr.id
            .as_deref()
            .and_then(|s| s.parse::<Uuid>().ok())
            .unwrap_or_else(Uuid::new_v4);

        let node_type = match nr.node_type.as_deref().unwrap_or("Semantic") {
            "Episodic"      => NodeType::Episodic,
            "Concept"       => NodeType::Concept,
            "DreamSnapshot" => NodeType::DreamSnapshot,
            _               => NodeType::Semantic,
        };
        let embedding = nr.embedding
            .map(PoincareVector::from_f64)
            .unwrap_or_else(|| PoincareVector::origin(4));
        let content = nr.content.unwrap_or(serde_json::Value::Object(Default::default()));
        let mut node = Node::new(id, embedding, content);
        node.node_type = node_type;
        if let Some(e) = nr.energy { node.energy = e; }
        ids.push(id.to_string());
        nodes_vec.push(node);
    }

    match db.insert_nodes_bulk(nodes_vec) {
        Ok(_)  => Json(serde_json::json!({"inserted": ids.len(), "node_ids": ids})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/batch/edges
#[derive(Deserialize)]
struct BatchInsertEdgesRequest {
    edges: Vec<InsertEdgeRequest>,
}

async fn batch_insert_edges(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<BatchInsertEdgesRequest>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let mut db = shared.write().await;

    let mut edges_vec = Vec::with_capacity(req.edges.len());
    let mut ids = Vec::with_capacity(req.edges.len());

    for er in req.edges {
        let from = match er.from.parse::<Uuid>() {
            Ok(u) => u,
            Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid from uuid"}))).into_response(),
        };
        let to = match er.to.parse::<Uuid>() {
            Ok(u) => u,
            Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid to uuid"}))).into_response(),
        };
        let edge_type = match er.edge_type.as_deref().unwrap_or("Association") {
            "Hierarchical"     => EdgeType::Hierarchical,
            "LSystemGenerated" => EdgeType::LSystemGenerated,
            "Pruned"           => EdgeType::Pruned,
            _                  => EdgeType::Association,
        };
        let edge = Edge::new(from, to, edge_type, er.weight.unwrap_or(1.0));
        ids.push(edge.id.to_string());
        edges_vec.push(edge);
    }

    match db.insert_edges_bulk(edges_vec) {
        Ok(_)  => Json(serde_json::json!({"inserted": ids.len(), "edge_ids": ids})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Algorithm endpoints ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AlgoParams {
    #[allow(dead_code)]
    collection: Option<String>,
    damping: Option<f64>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    sample: Option<usize>,
    direction: Option<String>,
    top_k: Option<usize>,
    threshold: Option<f64>,
}

// GET /api/algo/pagerank
async fn algo_pagerank(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let config = nietzsche_algo::PageRankConfig {
        damping_factor: p.damping.unwrap_or(0.85),
        max_iterations: p.iterations.unwrap_or(20),
        convergence_threshold: 1e-7,
    };
    match nietzsche_algo::pagerank(db.storage(), db.adjacency(), &config) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "pagerank",
            "iterations": r.iterations,
            "converged": r.converged,
            "duration_ms": r.duration_ms,
            "scores": r.scores.iter().take(100).map(|(id, s)| {
                serde_json::json!({"node_id": id.to_string(), "score": s})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/louvain
async fn algo_louvain(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let config = nietzsche_algo::LouvainConfig {
        max_iterations: p.iterations.unwrap_or(10),
        resolution: p.resolution.unwrap_or(1.0),
    };
    match nietzsche_algo::louvain(db.storage(), db.adjacency(), &config) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "louvain",
            "community_count": r.community_count,
            "modularity": r.modularity,
            "iterations": r.iterations,
            "duration_ms": r.duration_ms,
            "communities": r.communities.iter().take(500).map(|(id, c)| {
                serde_json::json!({"node_id": id.to_string(), "community_id": c})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/labelprop
async fn algo_labelprop(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let max_iter = p.iterations.unwrap_or(10);
    match nietzsche_algo::label_propagation(db.storage(), db.adjacency(), max_iter) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "label_propagation",
            "community_count": r.community_count,
            "iterations": r.iterations,
            "duration_ms": r.duration_ms,
            "labels": r.labels.iter().take(500).map(|(id, c)| {
                serde_json::json!({"node_id": id.to_string(), "community_id": c})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/betweenness
async fn algo_betweenness(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let sample = p.sample.filter(|&s| s > 0);
    match nietzsche_algo::betweenness_centrality(db.storage(), db.adjacency(), sample) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "betweenness_centrality",
            "duration_ms": r.duration_ms,
            "scores": r.scores.iter().take(100).map(|(id, s)| {
                serde_json::json!({"node_id": id.to_string(), "score": s})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/closeness
async fn algo_closeness(
    State((cm, _ops)): State<AppState>,
    Query(_p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_algo::closeness_centrality(db.storage(), db.adjacency()) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "closeness_centrality",
            "scores": r.iter().take(100).map(|(id, s)| {
                serde_json::json!({"node_id": id.to_string(), "score": s})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/degree
async fn algo_degree(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let direction = match p.direction.as_deref() {
        Some("in")  => nietzsche_algo::Direction::In,
        Some("out") => nietzsche_algo::Direction::Out,
        _           => nietzsche_algo::Direction::Both,
    };
    let node_ids: Vec<Uuid> = db.storage()
        .scan_nodes_meta()
        .unwrap_or_default()
        .into_iter()
        .map(|n| n.id)
        .collect();
    let r = nietzsche_algo::degree_centrality(db.adjacency(), direction, &node_ids);
    Json(serde_json::json!({
        "algorithm": "degree_centrality",
        "scores": r.iter().take(100).map(|(id, s)| {
            serde_json::json!({"node_id": id.to_string(), "score": s})
        }).collect::<Vec<_>>(),
    })).into_response()
}

// GET /api/algo/wcc
async fn algo_wcc(
    State((cm, _ops)): State<AppState>,
    Query(_p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_algo::weakly_connected_components(db.storage(), db.adjacency()) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "wcc",
            "component_count": r.component_count,
            "largest_component_size": r.largest_component_size,
            "duration_ms": r.duration_ms,
            "components": r.components.iter().take(500).map(|(id, c)| {
                serde_json::json!({"node_id": id.to_string(), "component_id": c})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/scc
async fn algo_scc(
    State((cm, _ops)): State<AppState>,
    Query(_p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_algo::strongly_connected_components(db.storage(), db.adjacency()) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "scc",
            "component_count": r.component_count,
            "largest_component_size": r.largest_component_size,
            "duration_ms": r.duration_ms,
            "components": r.components.iter().take(500).map(|(id, c)| {
                serde_json::json!({"node_id": id.to_string(), "component_id": c})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/triangles
async fn algo_triangles(
    State((cm, _ops)): State<AppState>,
    Query(_p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_algo::triangle_count(db.storage(), db.adjacency()) {
        Ok(count) => Json(serde_json::json!({
            "algorithm": "triangle_count",
            "count": count,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/jaccard
async fn algo_jaccard(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let top_k = p.top_k.unwrap_or(100);
    let threshold = p.threshold.unwrap_or(0.0);
    match nietzsche_algo::jaccard_similarity(db.storage(), db.adjacency(), top_k, threshold) {
        Ok(r) => Json(serde_json::json!({
            "algorithm": "jaccard_similarity",
            "pairs": r.iter().map(|p| {
                serde_json::json!({"node_a": p.node_a.to_string(), "node_b": p.node_b.to_string(), "score": p.score})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Backup endpoints ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct BackupParams {
    label: Option<String>,
}

// POST /api/backup
async fn create_backup(
    State((cm, _ops)): State<AppState>,
    Json(p): Json<BackupParams>,
) -> impl IntoResponse {
    let label = p.label.as_deref().unwrap_or("manual");
    let shared = default_col!(cm);
    let db = shared.read().await;

    let backup_dir = std::env::var("NIETZSCHE_BACKUP_DIR")
        .unwrap_or_else(|_| "backups".to_string());
    match nietzsche_graph::BackupManager::new(&backup_dir) {
        Ok(mgr) => match mgr.create_backup(db.storage(), label) {
            Ok(info) => Json(serde_json::json!({
                "label": info.label,
                "path": info.path.to_string_lossy(),
                "created_at": info.created_at,
                "size_bytes": info.size_bytes,
            })).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/backup
async fn list_backups() -> impl IntoResponse {
    let backup_dir = std::env::var("NIETZSCHE_BACKUP_DIR")
        .unwrap_or_else(|_| "backups".to_string());
    match nietzsche_graph::BackupManager::new(&backup_dir) {
        Ok(mgr) => match mgr.list_backups() {
            Ok(backups) => Json(serde_json::json!({
                "backups": backups.iter().map(|b| {
                    serde_json::json!({
                        "label": b.label,
                        "path": b.path.to_string_lossy(),
                        "created_at": b.created_at,
                        "size_bytes": b.size_bytes,
                    })
                }).collect::<Vec<_>>(),
            })).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Full-Text Search endpoint ──────────────────────────────────────────────

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    limit: Option<usize>,
}

// GET /api/search?q=text&limit=10
async fn fulltext_search(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<SearchParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let limit = p.limit.unwrap_or(10);
    let fts = nietzsche_graph::FullTextIndex::new(db.storage());
    match fts.search(&p.q, limit) {
        Ok(results) => Json(serde_json::json!({
            "query": p.q,
            "results": results.iter().map(|r| {
                serde_json::json!({"node_id": r.node_id.to_string(), "score": r.score})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Export endpoints ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ExportParams {
    format: Option<String>, // "csv" or "jsonl", default "jsonl"
}

// GET /api/export/nodes
async fn export_nodes(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<ExportParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let mut buf = Vec::new();

    let (content_type, result) = match p.format.as_deref() {
        Some("csv") => {
            let r = nietzsche_graph::export_nodes_csv(db.storage(), &mut buf);
            ("text/csv", r)
        }
        _ => {
            let r = nietzsche_graph::export_nodes_jsonl(db.storage(), &mut buf);
            ("application/x-ndjson", r)
        }
    };

    match result {
        Ok(_) => (
            [(axum::http::header::CONTENT_TYPE, content_type)],
            buf,
        ).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/export/edges
async fn export_edges(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<ExportParams>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    let mut buf = Vec::new();

    let (content_type, result) = match p.format.as_deref() {
        Some("csv") => {
            let r = nietzsche_graph::export_edges_csv(db.storage(), &mut buf);
            ("text/csv", r)
        }
        _ => {
            let r = nietzsche_graph::export_edges_jsonl(db.storage(), &mut buf);
            ("application/x-ndjson", r)
        }
    };

    match result {
        Ok(_) => (
            [(axum::http::header::CONTENT_TYPE, content_type)],
            buf,
        ).into_response(),
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
        let (meta, _embedding) = n.into_parts();
        Self {
            id:         meta.id.to_string(),
            node_type:  format!("{:?}", meta.node_type),
            energy:     meta.energy,
            depth:      meta.depth,
            hausdorff:  meta.hausdorff_local,
            created_at: meta.created_at,
            content:    meta.content,
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

// ── Cluster endpoints ─────────────────────────────────────────────────────────

// GET /api/cluster/status — full peer list with health
async fn cluster_status(
    Extension(cluster): Extension<Option<ClusterRegistry>>,
) -> impl IntoResponse {
    match cluster {
        None => Json(serde_json::json!({
            "enabled": false,
            "message": "cluster mode disabled (set NIETZSCHE_CLUSTER_ENABLED=true)"
        })).into_response(),
        Some(reg) => {
            let nodes = reg.all_nodes();
            Json(serde_json::json!({
                "enabled":    true,
                "local_id":   reg.local_id().to_string(),
                "node_count": nodes.len(),
                "nodes": nodes.iter().map(|n| serde_json::json!({
                    "id":           n.id.to_string(),
                    "name":         n.name,
                    "addr":         n.addr,
                    "role":         n.role.to_string(),
                    "health":       n.health.to_string(),
                    "last_seen_ms": n.last_seen_ms,
                    "token":        n.token,
                })).collect::<Vec<_>>(),
            })).into_response()
        }
    }
}

// GET /api/cluster/ring — consistent hash ring summary
async fn cluster_ring(
    Extension(cluster): Extension<Option<ClusterRegistry>>,
) -> impl IntoResponse {
    match cluster {
        None => Json(serde_json::json!({"enabled": false, "ring": []})).into_response(),
        Some(reg) => {
            let nodes = reg.all_nodes();
            Json(serde_json::json!({
                "enabled": true,
                "ring": nodes.iter().map(|n| serde_json::json!({
                    "token":  n.token,
                    "name":   n.name,
                    "addr":   n.addr,
                    "health": n.health.to_string(),
                })).collect::<Vec<_>>(),
            })).into_response()
        }
    }
}

// ── Agency endpoints ────────────────────────────────────────────────────────

// GET /api/agency/health — all persisted HealthReports
async fn agency_health(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::list_health_reports(db.storage()) {
        Ok(reports) => Json(serde_json::json!({
            "count": reports.len(),
            "reports": reports,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/health/latest — most recent HealthReport
async fn agency_health_latest(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::get_latest_health_report(db.storage()) {
        Ok(Some(report)) => Json(serde_json::json!(report)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "no health reports yet"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/counterfactual/remove/:id — simulate removing a node
async fn agency_cf_remove(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid uuid"}))).into_response(),
    };
    let shared = default_col!(cm);
    let db = shared.read().await;
    let config = nietzsche_agency::AgencyConfig::default();
    match nietzsche_agency::CounterfactualEngine::what_if_remove(db.storage(), db.adjacency(), uuid, &config) {
        Ok(result) => Json(serde_json::json!({
            "operation": "remove_node",
            "node_id": uuid.to_string(),
            "mean_energy_delta": result.mean_energy_delta,
            "affected_radius": result.affected_radius,
            "impact_scores": result.impact_scores.iter().take(50).map(|(id, score)| {
                serde_json::json!({"node_id": id.to_string(), "score": score})
            }).collect::<Vec<_>>(),
            "energy_changes": result.energy_changes.iter().take(50).map(|ec| {
                serde_json::json!({"node_id": ec.node_id.to_string(), "before": ec.before, "after": ec.after})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/agency/counterfactual/add — simulate adding a node
#[derive(Deserialize)]
struct CfAddRequest {
    energy:     Option<f32>,
    depth:      Option<f32>,
    connect_to: Vec<String>,
}

async fn agency_cf_add(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<CfAddRequest>,
) -> impl IntoResponse {
    let connect_to: Vec<Uuid> = match req.connect_to.iter().map(|s| s.parse::<Uuid>()).collect::<Result<Vec<_>, _>>() {
        Ok(ids) => ids,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid uuid in connect_to"}))).into_response(),
    };

    let new_meta = nietzsche_graph::NodeMeta {
        id: Uuid::new_v4(),
        depth: req.depth.unwrap_or(0.3),
        content: serde_json::json!({"source": "counterfactual_simulation"}),
        node_type: nietzsche_graph::NodeType::Semantic,
        energy: req.energy.unwrap_or(0.7),
        lsystem_generation: 0,
        hausdorff_local: 1.0,
        created_at: 0,
        expires_at: None,
        metadata: std::collections::HashMap::new(),
        is_phantom: false,
    };

    let shared = default_col!(cm);
    let db = shared.read().await;
    let config = nietzsche_agency::AgencyConfig::default();
    match nietzsche_agency::CounterfactualEngine::what_if_add(db.storage(), db.adjacency(), new_meta, connect_to, &config) {
        Ok(result) => Json(serde_json::json!({
            "operation": "add_node",
            "mean_energy_delta": result.mean_energy_delta,
            "affected_radius": result.affected_radius,
            "impact_scores": result.impact_scores.iter().take(50).map(|(id, score)| {
                serde_json::json!({"node_id": id.to_string(), "score": score})
            }).collect::<Vec<_>>(),
            "energy_changes": result.energy_changes.iter().take(50).map(|ec| {
                serde_json::json!({"node_id": ec.node_id.to_string(), "before": ec.before, "after": ec.after})
            }).collect::<Vec<_>>(),
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/desires — list all pending desire signals
async fn agency_desires(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::list_pending_desires(db.storage()) {
        Ok(desires) => Json(serde_json::json!({
            "count": desires.len(),
            "desires": desires,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/agency/desires/:id/fulfill — mark a desire as fulfilled
async fn agency_fulfill_desire(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid uuid"}))).into_response(),
    };
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::fulfill_desire(db.storage(), uuid) {
        Ok(true) => Json(serde_json::json!({"fulfilled": uuid.to_string()})).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "desire not found"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/observer — get the Observer Identity meta-node
async fn agency_observer(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::ObserverIdentity::get_id(db.storage()) {
        Ok(Some(id)) => {
            match db.storage().get_node_meta(&id) {
                Ok(Some(meta)) => Json(serde_json::json!({
                    "observer_id": id.to_string(),
                    "energy": meta.energy,
                    "depth": meta.depth,
                    "hausdorff_local": meta.hausdorff_local,
                    "is_observer": nietzsche_agency::ObserverIdentity::is_observer(&meta),
                    "content": meta.content,
                })).into_response(),
                Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "observer node not found in graph"}))).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
            }
        }
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "observer identity not yet created — agency engine must tick first"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/evolution — current evolution state and strategy
async fn agency_evolution(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match nietzsche_agency::RuleEvolution::load_state(db.storage()) {
        Ok(state) => Json(serde_json::json!({
            "generation": state.generation,
            "last_strategy": state.last_strategy,
            "fitness_history": state.fitness_history,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/narrative — latest generated narrative
async fn agency_narrative(
    State((cm, _ops)): State<AppState>,
) -> impl IntoResponse {
    let shared = default_col!(cm);
    let db = shared.read().await;
    match db.storage().get_meta("agency:latest_narrative") {
        Ok(Some(bytes)) => {
            let summary: String = serde_json::from_slice(&bytes).unwrap_or_default();
            Json(serde_json::json!({"narrative": summary})).into_response()
        }
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "no narrative yet — agency engine must tick first"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/agency/quantum/map — map Poincaré embeddings to Bloch states
#[derive(Deserialize)]
struct QuantumMapRequest {
    nodes: Vec<QuantumNodeInput>,
}

#[derive(Deserialize)]
struct QuantumNodeInput {
    embedding: Vec<f64>,
    energy: f32,
}

async fn agency_quantum_map(
    Json(req): Json<QuantumMapRequest>,
) -> impl IntoResponse {
    let states: Vec<_> = req.nodes.iter().map(|n| {
        let state = nietzsche_agency::poincare_to_bloch(&n.embedding, n.energy);
        serde_json::json!({
            "theta": state.theta,
            "phi": state.phi,
            "purity": state.purity,
            "bloch_vector": state.vector,
        })
    }).collect();
    Json(serde_json::json!({"states": states})).into_response()
}

// POST /api/agency/quantum/fidelity — compute fidelity between two groups
#[derive(Deserialize)]
struct FidelityRequest {
    group_a: Vec<QuantumNodeInput>,
    group_b: Vec<QuantumNodeInput>,
}

async fn agency_quantum_fidelity(
    Json(req): Json<FidelityRequest>,
) -> impl IntoResponse {
    let a: Vec<_> = req.group_a.iter().map(|n| nietzsche_agency::poincare_to_bloch(&n.embedding, n.energy)).collect();
    let b: Vec<_> = req.group_b.iter().map(|n| nietzsche_agency::poincare_to_bloch(&n.embedding, n.energy)).collect();
    let proxy = nietzsche_agency::entanglement_proxy(&a, &b);
    Json(serde_json::json!({
        "entanglement_proxy": proxy,
        "group_a_size": a.len(),
        "group_b_size": b.len(),
    })).into_response()
}
