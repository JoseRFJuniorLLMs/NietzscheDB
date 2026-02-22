//! HTTP dashboard server — REST API + embedded web UI.
//!
//! Exposes a JSON REST API backed by the shared [`CollectionManager`] and
//! serves a single-page HTML dashboard for graph visualization, CRUD forms,
//! NQL console, and stats panel.
//!
//! All data-plane endpoints accept `?collection=name` (default: `"default"`).
//! `/api/graph` also accepts `&limit=N` (default 500, max 5000).
//!
//! Endpoints:
//!   GET  /                        → dashboard HTML
//!   GET  /api/stats               → total node/edge counts + version
//!   GET  /api/health              → liveness probe
//!   GET  /api/collections         → list all collections
//!   GET  /api/graph               → nodes + edges (≤limit, default 500, max 5000)
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
    Edge, EdgeType, Node, NodeMeta, NodeType, PoincareVector,
    SchemaValidator, SchemaConstraint, FieldType,
};
use nietzsche_query::{parse as nql_parse, execute as nql_execute, Params, QueryResult};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use nietzsche_hyp_ops::{riemann, klein, minkowski, manifold};
use ordered_float::OrderedFloat;

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
        // Schema management
        .route("/api/schemas", get(list_schemas))
        .route("/api/schemas", post(set_schema))
        .route("/api/schemas/:node_type", delete(delete_schema))
        // Reasoning (Multi-Manifold)
        .route("/api/reasoning/synthesis", post(reasoning_synthesis))
        .route("/api/reasoning/synthesis-multi", post(reasoning_synthesis_multi))
        .route("/api/reasoning/causal-neighbors", post(reasoning_causal_neighbors))
        .route("/api/reasoning/causal-chain", post(reasoning_causal_chain))
        .route("/api/reasoning/klein-path", post(reasoning_klein_path))
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

/// Resolve a collection by optional name, falling back to "default".
macro_rules! resolve_col {
    ($cm:expr, $col:expr) => {{
        let col_name = $col.as_deref().unwrap_or("default");
        match $cm.get_or_default(col_name) {
            Some(db) => db,
            None => return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({"error": format!("collection '{}' not found", col_name)})),
            ).into_response(),
        }
    }};
    ($cm:expr) => {
        resolve_col!($cm, Option::<String>::None)
    };
}

/// Query param for endpoints that only need an optional collection.
#[derive(Deserialize)]
struct CollectionQuery {
    collection: Option<String>,
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

// GET /api/graph?collection=name&limit=1000
#[derive(Deserialize)]
struct GraphParams {
    collection: Option<String>,
    limit:      Option<usize>,
}

#[derive(Serialize)]
struct GraphResponse {
    nodes: Vec<NodeJson>,
    edges: Vec<EdgeJson>,
}

async fn graph(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<GraphParams>,
) -> impl IntoResponse {
    let col_name = p.collection.as_deref().unwrap_or("default");
    let max = p.limit.unwrap_or(500).min(5000);
    let shared = match cm.get_or_default(col_name) {
        Some(db) => db,
        None => return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("collection '{}' not found", col_name)})),
        ).into_response(),
    };
    let db = shared.read().await;
    // Use scan_nodes_meta (no embedding join) — faster and works for V0-migrated nodes
    // that may not have separate embeddings in CF_EMBEDDINGS.
    let nodes: Vec<NodeJson> = match db.storage().scan_nodes_meta() {
        Ok(ns) => ns.into_iter().take(max).map(NodeJson::from).collect(),
        Err(e) => {
            warn!(collection = col_name, error = %e, "scan_nodes failed");
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("scan_nodes failed: {}", e)}))).into_response();
        }
    };
    let edges: Vec<EdgeJson> = match db.storage().scan_edges() {
        Ok(es) => es.into_iter().take(max).map(EdgeJson::from).collect(),
        Err(e) => {
            warn!(collection = col_name, error = %e, "scan_edges failed");
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("scan_edges failed: {}", e)}))).into_response();
        }
    };
    Json(GraphResponse { nodes, edges }).into_response()
}

// GET /api/node/:id?collection=
async fn get_node(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = resolve_col!(cm, cq.collection);
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
    id:         Option<String>,
    node_type:  Option<String>,
    energy:     Option<f32>,
    content:    Option<serde_json::Value>,
    embedding:  Option<Vec<f64>>,
    collection: Option<String>,
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

    let shared = resolve_col!(cm, req.collection);
    let mut db = shared.write().await;
    match db.insert_node(node) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/node/:id?collection=
async fn delete_node(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = resolve_col!(cm, cq.collection);
    let mut db = shared.write().await;
    match db.delete_node(uuid) {
        Ok(_)  => Json(serde_json::json!({"deleted": uuid.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/edge
#[derive(Deserialize)]
struct InsertEdgeRequest {
    from:       String,
    to:         String,
    edge_type:  Option<String>,
    weight:     Option<f32>,
    collection: Option<String>,
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

    let shared = resolve_col!(cm, req.collection);
    let mut db = shared.write().await;
    match db.insert_edge(edge) {
        Ok(_)  => Json(serde_json::json!({"id": id.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/edge/:id?collection=
async fn delete_edge(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid"}))).into_response(),
    };
    let shared = resolve_col!(cm, cq.collection);
    let mut db = shared.write().await;
    match db.delete_edge(uuid) {
        Ok(_)  => Json(serde_json::json!({"deleted": uuid.to_string()})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/query
#[derive(Deserialize)]
struct QueryRequest {
    nql: String,
    collection: Option<String>,
}

async fn query_nql(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> impl IntoResponse {
    let query = match nql_parse(&req.nql) {
        Ok(q)  => q,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };
    let shared = resolve_col!(cm, req.collection);
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
    collection:          Option<String>,
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
        ..Default::default()
    };
    let shared   = resolve_col!(cm, req.collection);
    let mut db   = shared.write().await;
    let seed: u64 = rand::random();
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);
    match SleepCycle::run(&cfg, &mut *db, &mut rng) {
        Ok(r) => Json(serde_json::json!({
            "hausdorff_before":    r.hausdorff_before,
            "hausdorff_after":     r.hausdorff_after,
            "hausdorff_delta":     r.hausdorff_delta,
            "semantic_drift_avg":  r.semantic_drift_avg,
            "semantic_drift_max":  r.semantic_drift_max,
            "committed":           r.committed,
            "nodes_perturbed":     r.nodes_perturbed,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/batch/nodes
#[derive(Deserialize)]
struct BatchInsertNodesRequest {
    nodes: Vec<InsertNodeRequest>,
    collection: Option<String>,
}

async fn batch_insert_nodes(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<BatchInsertNodesRequest>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, req.collection);
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
    collection: Option<String>,
}

async fn batch_insert_edges(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<BatchInsertEdgesRequest>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, req.collection);
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
    collection: Option<String>,
    damping: Option<f64>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    sample: Option<usize>,
    direction: Option<String>,
    top_k: Option<usize>,
    threshold: Option<f64>,
}

// GET /api/algo/pagerank?collection=
async fn algo_pagerank(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/louvain?collection=
async fn algo_louvain(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/labelprop?collection=
async fn algo_labelprop(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/betweenness?collection=
async fn algo_betweenness(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/closeness?collection=
async fn algo_closeness(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/degree?collection=
async fn algo_degree(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
    let db = shared.read().await;
    let direction = match p.direction.as_deref() {
        Some("in")  => nietzsche_algo::Direction::In,
        Some("out") => nietzsche_algo::Direction::Out,
        _           => nietzsche_algo::Direction::Both,
    };
    let node_ids: Vec<Uuid> = match db.storage().scan_nodes_meta() {
        Ok(metas) => metas.into_iter().map(|n| n.id).collect(),
        Err(e) => {
            warn!(error = %e, "scan_nodes_meta failed for degree centrality");
            return (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("scan_nodes_meta failed: {}", e)}))).into_response();
        }
    };
    let r = nietzsche_algo::degree_centrality(db.adjacency(), direction, &node_ids);
    Json(serde_json::json!({
        "algorithm": "degree_centrality",
        "scores": r.iter().take(100).map(|(id, s)| {
            serde_json::json!({"node_id": id.to_string(), "score": s})
        }).collect::<Vec<_>>(),
    })).into_response()
}

// GET /api/algo/wcc?collection=
async fn algo_wcc(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/scc?collection=
async fn algo_scc(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/algo/triangles?collection=
async fn algo_triangles(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
    let db = shared.read().await;
    match nietzsche_algo::triangle_count(db.storage(), db.adjacency()) {
        Ok(count) => Json(serde_json::json!({
            "algorithm": "triangle_count",
            "count": count,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/algo/jaccard?collection=
async fn algo_jaccard(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<AlgoParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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
    label:      Option<String>,
    collection: Option<String>,
}

// POST /api/backup
async fn create_backup(
    State((cm, _ops)): State<AppState>,
    Json(p): Json<BackupParams>,
) -> impl IntoResponse {
    let label = p.label.as_deref().unwrap_or("manual");
    let shared = resolve_col!(cm, p.collection);
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
    q:          String,
    limit:      Option<usize>,
    collection: Option<String>,
}

// GET /api/search?q=text&limit=10&collection=
async fn fulltext_search(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<SearchParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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
    format:     Option<String>, // "csv" or "jsonl", default "jsonl"
    collection: Option<String>,
}

// GET /api/export/nodes?collection=
async fn export_nodes(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<ExportParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// GET /api/export/edges?collection=
async fn export_edges(
    State((cm, _ops)): State<AppState>,
    Query(p): Query<ExportParams>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, p.collection);
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

// ── Schema endpoints ─────────────────────────────────────────────────────────

// GET /api/schemas?collection=
async fn list_schemas(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    match SchemaValidator::load_all(db.storage()) {
        Ok(validator) => {
            let schemas: Vec<serde_json::Value> = validator.list_constraints().iter().map(|c| {
                serde_json::json!({
                    "node_type": c.node_type,
                    "required_fields": c.required_fields,
                    "field_types": c.field_types.iter().map(|(k, v)| {
                        serde_json::json!({"field_name": k, "field_type": format!("{:?}", v)})
                    }).collect::<Vec<_>>(),
                })
            }).collect();
            Json(serde_json::json!({"schemas": schemas})).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/schemas
#[derive(Deserialize)]
struct SetSchemaRequest {
    node_type:       String,
    required_fields: Vec<String>,
    field_types:     Vec<SchemaFieldJson>,
    collection:      Option<String>,
}

#[derive(Deserialize)]
struct SchemaFieldJson {
    field_name: String,
    field_type: String,
}

async fn set_schema(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<SetSchemaRequest>,
) -> impl IntoResponse {
    let mut ft = std::collections::HashMap::new();
    for f in &req.field_types {
        let ftype = match f.field_type.to_lowercase().as_str() {
            "string" | "str"     => FieldType::String,
            "number" | "float" | "int" => FieldType::Number,
            "bool" | "boolean"   => FieldType::Bool,
            "array" | "list"     => FieldType::Array,
            "object" | "map"     => FieldType::Object,
            _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("unknown field type '{}' — use string|number|bool|array|object", f.field_type)
            }))).into_response(),
        };
        ft.insert(f.field_name.clone(), ftype);
    }
    let constraint = SchemaConstraint {
        node_type: req.node_type.clone(),
        required_fields: req.required_fields,
        field_types: ft,
    };

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;
    let validator = SchemaValidator::new();
    match validator.save_constraint(db.storage(), &constraint) {
        Ok(_)  => Json(serde_json::json!({"status": "ok", "node_type": req.node_type})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// DELETE /api/schemas/:node_type?collection=
async fn delete_schema(
    State((cm, _ops)): State<AppState>,
    Path(node_type): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    let validator = SchemaValidator::new();
    match validator.delete_constraint(db.storage(), &node_type) {
        Ok(_)  => Json(serde_json::json!({"status": "ok", "deleted": node_type})).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// ── Reasoning endpoints (Multi-Manifold) ────────────────────────────────────

// POST /api/reasoning/synthesis
#[derive(Deserialize)]
struct SynthesisRequest {
    node_id_a:  String,
    node_id_b:  String,
    collection: Option<String>,
}

async fn reasoning_synthesis(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<SynthesisRequest>,
) -> impl IntoResponse {
    let id_a = match req.node_id_a.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid node_id_a"}))).into_response(),
    };
    let id_b = match req.node_id_b.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid node_id_b"}))).into_response(),
    };

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;

    let emb_a = match db.storage().get_embedding(&id_a) {
        Ok(Some(e)) => e, Ok(None) => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"node_id_a embedding not found"}))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };
    let emb_b = match db.storage().get_embedding(&id_b) {
        Ok(Some(e)) => e, Ok(None) => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"node_id_b embedding not found"}))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };

    let coords_a = emb_a.coords_f64();
    let coords_b = emb_b.coords_f64();

    let synthesis_raw = match riemann::synthesis(&coords_a, &coords_b) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("synthesis failed: {:?}", e)}))).into_response(),
    };
    let synthesis_coords = manifold::sanitize_poincare_f64(&synthesis_raw);

    // find nearest existing node
    let synthesis_pv = PoincareVector::from_f64(synthesis_coords.clone());
    let (nearest_id, nearest_dist) = match db.knn(&synthesis_pv, 1) {
        Ok(results) if !results.is_empty() => (results[0].0.to_string(), results[0].1),
        Ok(_) => ("none".to_string(), f64::INFINITY),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };

    Json(serde_json::json!({
        "synthesis_coords": synthesis_coords,
        "nearest_node_id": nearest_id,
        "nearest_distance": nearest_dist,
    })).into_response()
}

// POST /api/reasoning/synthesis-multi
#[derive(Deserialize)]
struct SynthesisMultiRequest {
    node_ids:   Vec<String>,
    collection: Option<String>,
}

async fn reasoning_synthesis_multi(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<SynthesisMultiRequest>,
) -> impl IntoResponse {
    if req.node_ids.len() < 2 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"need at least 2 node_ids"}))).into_response();
    }
    let ids: Vec<Uuid> = match req.node_ids.iter().map(|s| s.parse::<Uuid>()).collect::<Result<Vec<_>, _>>() {
        Ok(v) => v, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid uuid in node_ids"}))).into_response(),
    };

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;

    let mut all_coords: Vec<Vec<f64>> = Vec::with_capacity(ids.len());
    for id in &ids {
        match db.storage().get_embedding(id) {
            Ok(Some(e)) => all_coords.push(e.coords_f64()),
            Ok(None) => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": format!("embedding not found for {}", id)}))).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
        }
    }

    let refs: Vec<&[f64]> = all_coords.iter().map(|v| v.as_slice()).collect();
    let synthesis_raw = match riemann::synthesis_multi(&refs) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("synthesis_multi failed: {:?}", e)}))).into_response(),
    };
    let synthesis_coords = manifold::sanitize_poincare_f64(&synthesis_raw);

    let synthesis_pv = PoincareVector::from_f64(synthesis_coords.clone());
    let (nearest_id, nearest_dist) = match db.knn(&synthesis_pv, 1) {
        Ok(results) if !results.is_empty() => (results[0].0.to_string(), results[0].1),
        Ok(_) => ("none".to_string(), f64::INFINITY),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };

    Json(serde_json::json!({
        "synthesis_coords": synthesis_coords,
        "nearest_node_id": nearest_id,
        "nearest_distance": nearest_dist,
    })).into_response()
}

// POST /api/reasoning/causal-neighbors
#[derive(Deserialize)]
struct CausalNeighborsRequest {
    node_id:    String,
    direction:  Option<String>, // "future" | "past" | "both"
    collection: Option<String>,
}

async fn reasoning_causal_neighbors(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<CausalNeighborsRequest>,
) -> impl IntoResponse {
    let node_id = match req.node_id.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid node_id"}))).into_response(),
    };
    let direction = req.direction.as_deref().unwrap_or("both");

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;

    // get origin meta + embedding
    let origin_meta = match db.storage().get_node_meta(&node_id) {
        Ok(Some(m)) => m, Ok(None) => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"node not found"}))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };
    let origin_emb = match db.storage().get_embedding(&node_id) {
        Ok(Some(e)) => e, Ok(None) => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"origin embedding not found"}))).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    };

    let mut edges_out = Vec::new();

    // collect neighbors from adjacency
    let mut adj_entries = Vec::new();
    if direction != "future" {
        for entry in db.adjacency().entries_in(&node_id) {
            adj_entries.push((entry.edge_id, entry.neighbor_id, entry.edge_type, false));
        }
    }
    if direction != "past" {
        for entry in db.adjacency().entries_out(&node_id) {
            adj_entries.push((entry.edge_id, entry.neighbor_id, entry.edge_type, true));
        }
    }

    for (edge_id, neighbor_id, edge_type, is_outgoing) in adj_entries {
        let neighbor_meta = match db.storage().get_node_meta(&neighbor_id) {
            Ok(Some(m)) => m, _ => continue,
        };
        let neighbor_emb = match db.storage().get_embedding(&neighbor_id) {
            Ok(Some(e)) => e, _ => continue,
        };

        let (interval, causal_type) = minkowski::compute_edge_causality(
            &origin_emb.coords, &neighbor_emb.coords,
            origin_meta.created_at, neighbor_meta.created_at,
            minkowski::DEFAULT_CAUSAL_SPEED, 1e-6,
        );

        let (from_id, to_id) = if is_outgoing {
            (node_id, neighbor_id)
        } else {
            (neighbor_id, node_id)
        };

        edges_out.push(serde_json::json!({
            "edge_id": edge_id.to_string(),
            "from_node_id": from_id.to_string(),
            "to_node_id": to_id.to_string(),
            "minkowski_interval": interval,
            "causal_type": format!("{:?}", causal_type),
            "edge_type": format!("{:?}", edge_type),
        }));
    }

    Json(serde_json::json!({"edges": edges_out})).into_response()
}

// POST /api/reasoning/causal-chain
#[derive(Deserialize)]
struct CausalChainRequest {
    node_id:    String,
    max_depth:  Option<u32>,
    direction:  Option<String>, // "future" | "past"
    collection: Option<String>,
}

async fn reasoning_causal_chain(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<CausalChainRequest>,
) -> impl IntoResponse {
    let start_id = match req.node_id.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid node_id"}))).into_response(),
    };
    let max_depth = req.max_depth.unwrap_or(10).min(50) as usize;
    let is_past = req.direction.as_deref().unwrap_or("past") != "future";

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;

    let mut chain_ids: Vec<String> = vec![start_id.to_string()];
    let mut chain_edges: Vec<serde_json::Value> = Vec::new();
    let mut visited = std::collections::HashSet::new();
    visited.insert(start_id);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((start_id, 0usize));

    while let Some((current_id, depth)) = queue.pop_front() {
        if depth >= max_depth { continue; }

        let current_meta = match db.storage().get_node_meta(&current_id) {
            Ok(Some(m)) => m, _ => continue,
        };
        let current_emb = match db.storage().get_embedding(&current_id) {
            Ok(Some(e)) => e, _ => continue,
        };

        let entries = if is_past {
            db.adjacency().entries_in(&current_id)
        } else {
            db.adjacency().entries_out(&current_id)
        };

        for entry in entries {
            if visited.contains(&entry.neighbor_id) { continue; }

            let neighbor_meta = match db.storage().get_node_meta(&entry.neighbor_id) {
                Ok(Some(m)) => m, _ => continue,
            };
            let neighbor_emb = match db.storage().get_embedding(&entry.neighbor_id) {
                Ok(Some(e)) => e, _ => continue,
            };

            let (interval, causal_type) = minkowski::compute_edge_causality(
                &current_emb.coords, &neighbor_emb.coords,
                current_meta.created_at, neighbor_meta.created_at,
                minkowski::DEFAULT_CAUSAL_SPEED, 1e-6,
            );

            // only follow timelike edges in the causal chain
            if matches!(causal_type, minkowski::CausalType::Timelike) {
                visited.insert(entry.neighbor_id);
                chain_ids.push(entry.neighbor_id.to_string());

                let (from_id, to_id) = if is_past {
                    (entry.neighbor_id, current_id)
                } else {
                    (current_id, entry.neighbor_id)
                };

                chain_edges.push(serde_json::json!({
                    "edge_id": entry.edge_id.to_string(),
                    "from_node_id": from_id.to_string(),
                    "to_node_id": to_id.to_string(),
                    "minkowski_interval": interval,
                    "causal_type": format!("{:?}", causal_type),
                    "edge_type": format!("{:?}", entry.edge_type),
                }));

                queue.push_back((entry.neighbor_id, depth + 1));
            }
        }
    }

    Json(serde_json::json!({
        "chain_ids": chain_ids,
        "edges": chain_edges,
    })).into_response()
}

// POST /api/reasoning/klein-path
#[derive(Deserialize)]
struct KleinPathRequest {
    start_node_id: String,
    goal_node_id:  String,
    collection:    Option<String>,
}

async fn reasoning_klein_path(
    State((cm, _ops)): State<AppState>,
    Json(req): Json<KleinPathRequest>,
) -> impl IntoResponse {
    let start_id = match req.start_node_id.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid start_node_id"}))).into_response(),
    };
    let goal_id = match req.goal_node_id.parse::<Uuid>() {
        Ok(u) => u, Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid goal_node_id"}))).into_response(),
    };

    let shared = resolve_col!(cm, req.collection);
    let db = shared.read().await;

    // Dijkstra in Klein space
    let mut dist: std::collections::HashMap<Uuid, f64> = std::collections::HashMap::new();
    let mut prev: std::collections::HashMap<Uuid, Uuid> = std::collections::HashMap::new();
    let mut heap = std::collections::BinaryHeap::new();
    let max_explored = 2000usize;

    dist.insert(start_id, 0.0);
    heap.push(std::cmp::Reverse((OrderedFloat(0.0f64), start_id)));

    let mut found = false;
    let mut explored = 0usize;

    while let Some(std::cmp::Reverse((OrderedFloat(cost), current))) = heap.pop() {
        if current == goal_id { found = true; break; }
        explored += 1;
        if explored > max_explored { break; }
        if cost > *dist.get(&current).unwrap_or(&f64::INFINITY) { continue; }

        let current_emb = match db.storage().get_embedding(&current) {
            Ok(Some(e)) => e, _ => continue,
        };
        let current_klein = match klein::to_klein(&current_emb.coords_f64()) {
            Ok(k) => k, Err(_) => continue,
        };

        for entry in db.adjacency().entries_out(&current) {
            let neighbor_emb = match db.storage().get_embedding(&entry.neighbor_id) {
                Ok(Some(e)) => e, _ => continue,
            };
            let neighbor_klein = match klein::to_klein(&neighbor_emb.coords_f64()) {
                Ok(k) => k, Err(_) => continue,
            };

            let edge_cost = klein::klein_distance(&current_klein, &neighbor_klein);
            let new_dist = cost + edge_cost;

            if new_dist < *dist.get(&entry.neighbor_id).unwrap_or(&f64::INFINITY) {
                dist.insert(entry.neighbor_id, new_dist);
                prev.insert(entry.neighbor_id, current);
                heap.push(std::cmp::Reverse((OrderedFloat(new_dist), entry.neighbor_id)));
            }
        }
    }

    if found {
        // reconstruct path
        let mut path = Vec::new();
        let mut cur = goal_id;
        while cur != start_id {
            path.push(cur.to_string());
            cur = match prev.get(&cur) {
                Some(&p) => p,
                None => break,
            };
        }
        path.push(start_id.to_string());
        path.reverse();

        let total_cost = dist.get(&goal_id).copied().unwrap_or(0.0);
        Json(serde_json::json!({
            "found": true,
            "path": path,
            "cost": total_cost,
            "hops": path.len() - 1,
        })).into_response()
    } else {
        Json(serde_json::json!({
            "found": false,
            "path": [],
            "cost": 0.0,
            "hops": 0,
        })).into_response()
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
        Self::from(meta)
    }
}

impl From<NodeMeta> for NodeJson {
    fn from(meta: NodeMeta) -> Self {
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

// GET /api/agency/health?collection=
async fn agency_health(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    match nietzsche_agency::list_health_reports(db.storage()) {
        Ok(reports) => Json(serde_json::json!({
            "count": reports.len(),
            "reports": reports,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/health/latest?collection=
async fn agency_health_latest(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    match nietzsche_agency::get_latest_health_report(db.storage()) {
        Ok(Some(report)) => Json(serde_json::json!(report)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "no health reports yet"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/counterfactual/remove/:id?collection=
async fn agency_cf_remove(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u)  => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid uuid"}))).into_response(),
    };
    let shared = resolve_col!(cm, cq.collection);
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
    collection: Option<String>,
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
        valence: 0.0,
        arousal: 0.0,
        is_phantom: false,
    };

    let shared = resolve_col!(cm, req.collection);
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

// GET /api/agency/desires?collection=
async fn agency_desires(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    match nietzsche_agency::list_pending_desires(db.storage()) {
        Ok(desires) => Json(serde_json::json!({
            "count": desires.len(),
            "desires": desires,
        })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// POST /api/agency/desires/:id/fulfill?collection=
async fn agency_fulfill_desire(
    State((cm, _ops)): State<AppState>,
    Path(id): Path<String>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let uuid = match id.parse::<Uuid>() {
        Ok(u) => u,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid uuid"}))).into_response(),
    };
    let shared = resolve_col!(cm, cq.collection);
    let db = shared.read().await;
    match nietzsche_agency::fulfill_desire(db.storage(), uuid) {
        Ok(true) => Json(serde_json::json!({"fulfilled": uuid.to_string()})).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "desire not found"}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))).into_response(),
    }
}

// GET /api/agency/observer?collection=
async fn agency_observer(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
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

// GET /api/agency/evolution?collection=
async fn agency_evolution(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
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

// GET /api/agency/narrative?collection=
async fn agency_narrative(
    State((cm, _ops)): State<AppState>,
    Query(cq): Query<CollectionQuery>,
) -> impl IntoResponse {
    let shared = resolve_col!(cm, cq.collection);
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
