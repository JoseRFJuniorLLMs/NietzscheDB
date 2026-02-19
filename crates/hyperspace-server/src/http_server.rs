use crate::manager::CollectionManager;
use axum::{
    extract::{Extension, Path, Query, Request, State},
    http::{StatusCode, Uri},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use hyperspace_core::SearchParams;
use nietzsche_api::pb::nietzsche_db_client::NietzscheDbClient;
use nietzsche_api::pb::QueryRequest;
use rust_embed::RustEmbed;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use tonic::transport::Channel;
use std::time::Instant;
use sysinfo::{Pid, System};
#[cfg(not(windows))]
use tikv_jemalloc_ctl::epoch;
use tower_http::cors::CorsLayer;

#[derive(RustEmbed)]
#[folder = "../../dashboard/dist"]
struct FrontendAssets;

// API Key validation middleware
#[derive(Clone)]
pub struct RequestContext {
    pub user_id: String,
    pub is_admin: bool,
}

async fn validate_api_key(
    State(expected_hash): State<Option<String>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 1. Check Trusted Header (SaaS / Kong)
    // Clone header value to avoid holding immutable borrow
    let user_id_header = request
        .headers()
        .get("x-hyperspace-user-id")
        .and_then(|v| v.to_str().ok())
        .map(std::string::ToString::to_string);

    if let Some(uid) = user_id_header {
        request.extensions_mut().insert(RequestContext {
            user_id: uid,
            is_admin: false,
        });
        return Ok(next.run(request).await);
    }

    // Auth is skipped for static files (except if we want to enforce user context?)
    if !request.uri().path().starts_with("/api/") && request.uri().path() != "/metrics" {
        return Ok(next.run(request).await);
    }

    if let Some(expected) = expected_hash {
        match request.headers().get("x-api-key") {
            Some(key) => {
                if let Ok(key_str) = key.to_str() {
                    let mut hasher = Sha256::new();
                    hasher.update(key_str.as_bytes());
                    let hash = hex::encode(hasher.finalize());

                    if hash == expected {
                        request.extensions_mut().insert(RequestContext {
                            user_id: "default_admin".to_string(),
                            is_admin: true,
                        });
                        return Ok(next.run(request).await);
                    }
                }
                Err(StatusCode::UNAUTHORIZED)
            }
            None => Err(StatusCode::UNAUTHORIZED),
        }
    } else {
        // No Auth configured (Dev mode)
        request.extensions_mut().insert(RequestContext {
            user_id: "anonymous".to_string(),
            is_admin: true,
        });
        Ok(next.run(request).await)
    }
}

#[derive(Clone, serde::Serialize)]
pub struct EmbeddingInfo {
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub dimension: usize,
}

pub async fn start_http_server(
    manager: Arc<CollectionManager>,
    port: u16,
    embedding_info: Option<EmbeddingInfo>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Get API key hash if set
    let api_key_hash = std::env::var("HYPERSPACE_API_KEY").ok().map(|key| {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hex::encode(hasher.finalize())
    });

    let start_time = Arc::new(Instant::now());
    let embedding_state = Arc::new(embedding_info);

    let app = Router::new()
        .route(
            "/api/collections",
            get(list_collections).post(create_collection),
        )
        .route(
            "/api/collections/{name}",
            get(get_collection_digest).delete(delete_collection),
        )
        .route("/api/collections/{name}/insert", post(insert_vector))
        .route("/api/collections/{name}/stats", get(get_stats))
        .route("/api/collections/{name}/digest", get(get_collection_digest))
        .route("/api/collections/{name}/peek", get(peek_collection))
        .route("/api/collections/{name}/search", post(search_collection))
        .route("/api/status", get(get_status))
        .route("/api/cluster/status", get(get_cluster_status))
        .route("/api/metrics", get(get_metrics))
        .route("/metrics", get(get_prometheus_metrics))
        .route("/api/logs", get(get_logs))
        .route(
            "/api/collections/{name}/rebuild",
            post(rebuild_collection_http),
        )
        .route("/api/admin/vacuum", post(trigger_vacuum_http))
        .route("/api/admin/usage", get(get_usage_report_http))
        .route("/api/nietzsche/graph", get(get_nietzsche_graph))
        .layer(middleware::from_fn_with_state(
            api_key_hash.clone(),
            validate_api_key,
        ))
        .fallback(static_handler)
        .layer(CorsLayer::permissive())
        .with_state((manager, start_time, embedding_state));

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!("HTTP Dashboard listening on http://{addr}");
    if api_key_hash.is_some() {
        println!("🔒 Dashboard API Key Auth Enabled");
    } else {
        println!("⚠️  Dashboard API Key Auth Disabled");
    }

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    axum::serve(listener, app)
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    Ok(())
}

async fn static_handler(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');

    if path.is_empty() || path == "index.html" {
        return index_html().await;
    }

    match FrontendAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [(axum::http::header::CONTENT_TYPE, mime.as_ref())],
                content.data,
            )
                .into_response()
        }
        None => {
            if path.starts_with("api") {
                (StatusCode::NOT_FOUND, "API Route Not Found").into_response()
            } else {
                // SPA fallback
                index_html().await
            }
        }
    }
}

async fn index_html() -> Response {
    match FrontendAssets::get("index.html") {
        Some(content) => Html(content.data).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            "Dashboard not built. Run `npm run build` in dashboard/",
        )
            .into_response(),
    }
}

// Handlers

#[derive(serde::Serialize)]
struct CollectionSummary {
    name: String,
    count: usize,
    dimension: usize,
    metric: String,
    indexing_queue: u64,
}

async fn get_cluster_status(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
) -> Json<crate::manager::ClusterState> {
    let state = manager.cluster_state.read().await;
    Json(state.clone())
}

async fn list_collections(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> Json<Vec<CollectionSummary>> {
    let names = manager.list(&ctx.user_id);
    let mut summaries = Vec::new();
    for name in names {
        if let Some(col) = manager.get(&ctx.user_id, &name).await {
            summaries.push(CollectionSummary {
                name: name.clone(),
                count: col.count(),
                dimension: col.dimension(),
                metric: col.metric_name().to_string(),
                indexing_queue: col.queue_size(),
            });
        }
    }
    Json(summaries)
}
#[derive(serde::Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: u32,
    metric: String,
}

#[derive(serde::Deserialize)]
struct InsertPayload {
    vector: Vec<f64>,
    id: u32,
    metadata: Option<HashMap<String, String>>,
}

async fn create_collection(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
    Json(payload): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    match manager
        .create_collection(
            &ctx.user_id,
            &payload.name,
            payload.dimension,
            &payload.metric,
        )
        .await
    {
        Ok(()) => StatusCode::CREATED.into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
}

async fn insert_vector(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
    Json(payload): Json<InsertPayload>,
) -> impl IntoResponse {
    if let Some(col) = manager.get(&ctx.user_id, &name).await {
        let clock = manager.cluster_state.read().await.logical_clock;
        let meta = payload.metadata.unwrap_or_default();

        match col
            .insert(
                &payload.vector,
                payload.id,
                meta,
                clock,
                hyperspace_core::Durability::Default,
            )
            .await
        {
            Ok(()) => StatusCode::OK.into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        }
    } else {
        (StatusCode::NOT_FOUND, "Collection not found").into_response()
    }
}

async fn delete_collection(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    match manager.delete_collection(&ctx.user_id, &name).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (StatusCode::NOT_FOUND, e).into_response(),
    }
}

async fn get_stats(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if let Some(col) = manager.get(&ctx.user_id, &name).await {
        Json(serde_json::json!({
            "count": col.count(),
            "dimension": col.dimension(),
            "metric": col.metric_name(),
            "quantization": format!("{:?}", col.quantization_mode()),
            "indexing_queue": col.queue_size(),
        }))
        .into_response()
    } else {
        (StatusCode::NOT_FOUND, "Collection not found").into_response()
    }
}

async fn get_collection_digest(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if let Some(col) = manager.get(&ctx.user_id, &name).await {
        let clock = manager.cluster_state.read().await.logical_clock;
        let digest =
            crate::sync::CollectionDigest::new(name.clone(), clock, col.count(), col.buckets());
        Json(digest).into_response()
    } else {
        (StatusCode::NOT_FOUND, "Collection not found").into_response()
    }
}

async fn get_status(
    State((_, start_time, embedding)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
) -> Json<serde_json::Value> {
    let dim = std::env::var("HS_DIMENSION").unwrap_or("1024".to_string());
    let metric = std::env::var("HS_METRIC").unwrap_or("l2".to_string());
    let quantization = std::env::var("HS_QUANTIZATION_LEVEL").unwrap_or("scalar".to_string());
    let uptime_secs = start_time.elapsed().as_secs();
    let uptime_str = if uptime_secs < 60 {
        format!("{uptime_secs}s")
    } else if uptime_secs < 3600 {
        format!("{}m {}s", uptime_secs / 60, uptime_secs % 60)
    } else {
        format!("{}h {}m", uptime_secs / 3600, (uptime_secs % 3600) / 60)
    };

    Json(serde_json::json!({
        "status": "ONLINE",
        "version": "2.0.0",
        "uptime": uptime_str,
        "config": {
            "dimension": dim,
            "metric": metric,
            "quantization": quantization,
        },
        "embedding": embedding.as_ref()
    }))
}

async fn get_metrics(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if !ctx.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }

    let total_vecs = manager.total_vector_count();

    // Calculate disk usage from data directory
    let disk_usage_bytes = calculate_dir_size("./data").unwrap_or(0);
    let disk_usage_mb = (disk_usage_bytes as f64 / 1_048_576.0).round() as u64;

    // Get real system metrics
    let sys = System::new_all();
    let current_pid = Pid::from_u32(std::process::id());

    let (ram_usage_mb, cpu_usage_percent) = if let Some(process) = sys.process(current_pid) {
        let ram = (process.memory() as f64 / 1_048_576.0).round() as u64;
        let cpu = process.cpu_usage().round() as u64;
        (ram, cpu)
    } else {
        (0, 0)
    };

    let (active_count, idle_count) = manager.get_collection_counts();

    Json(serde_json::json!({
        "total_vectors": total_vecs,
        "active_collections": active_count,
        "idle_collections": idle_count,
        "total_collections": active_count + idle_count,
        "ram_usage_mb": ram_usage_mb,
        "cpu_usage_percent": cpu_usage_percent,
        "disk_usage_mb": disk_usage_mb,
    }))
    .into_response()
}

async fn get_prometheus_metrics(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if !ctx.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }

    let (active, idle) = manager.get_collection_counts();
    let total_vecs = manager.total_vector_count();

    // System stats
    let mut sys = System::new_all();
    sys.refresh_all();
    let pid = Pid::from_u32(std::process::id());
    let (ram_mb, cpu_percent) = if let Some(proc) = sys.process(pid) {
        (
            (proc.memory() as f64 / 1_048_576.0).round() as u64,
            proc.cpu_usage().round() as u64,
        )
    } else {
        (0, 0)
    };

    let disk_mb = calculate_dir_size("./data").unwrap_or(0) / 1_048_576;

    let body = format!(
        "# HELP hyperspace_active_collections Number of collections in memory\n\
         # TYPE hyperspace_active_collections gauge\n\
         hyperspace_active_collections {active}\n\
         # HELP hyperspace_idle_collections Number of collections unloaded to disk\n\
         # TYPE hyperspace_idle_collections gauge\n\
         hyperspace_idle_collections {idle}\n\
         # HELP hyperspace_total_vectors Total number of vectors in active collections\n\
         # TYPE hyperspace_total_vectors gauge\n\
         hyperspace_total_vectors {total_vecs}\n\
         # HELP hyperspace_ram_usage_mb Memory usage in MB\n\
         # TYPE hyperspace_ram_usage_mb gauge\n\
         hyperspace_ram_usage_mb {ram_mb}\n\
         # HELP hyperspace_disk_usage_mb Disk usage in MB\n\
         # TYPE hyperspace_disk_usage_mb gauge\n\
         hyperspace_disk_usage_mb {disk_mb}\n\
         # HELP hyperspace_cpu_usage_percent CPU usage percent\n\
         # TYPE hyperspace_cpu_usage_percent gauge\n\
         hyperspace_cpu_usage_percent {cpu_percent}\n"
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    )
        .into_response()
}

fn calculate_dir_size(path: &str) -> std::io::Result<u64> {
    let mut total_size = 0u64;

    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += calculate_dir_size(&entry.path().to_string_lossy())?;
            }
        }
    }

    Ok(total_size)
}

#[derive(serde::Deserialize)]
struct PeekParams {
    limit: Option<usize>,
}

async fn peek_collection(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
    Query(params): Query<PeekParams>,
) -> impl IntoResponse {
    let limit = params.limit.unwrap_or(50).min(100);
    if let Some(col) = manager.get(&ctx.user_id, &name).await {
        let items = col.peek(limit);
        Json(items).into_response()
    } else {
        (StatusCode::NOT_FOUND, "Collection not found").into_response()
    }
}

#[derive(serde::Deserialize)]
struct SearchReq {
    vector: Vec<f64>,
    top_k: Option<usize>,
}

fn default_ef_search() -> usize {
    static DEFAULT_EF_SEARCH: OnceLock<usize> = OnceLock::new();
    *DEFAULT_EF_SEARCH.get_or_init(|| {
        std::env::var("HS_HNSW_EF_SEARCH")
            .unwrap_or_else(|_| "100".to_string())
            .parse()
            .unwrap_or(100)
    })
}

async fn search_collection(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
    Json(payload): Json<SearchReq>,
) -> impl IntoResponse {
    let k = payload.top_k.unwrap_or(10);
    if let Some(col) = manager.get(&ctx.user_id, &name).await {
        let dummy_params = SearchParams {
            top_k: k,
            ef_search: default_ef_search(),
            hybrid_query: None,
            hybrid_alpha: None,
        };
        match col
            .search(&payload.vector, &HashMap::new(), &[], &dummy_params)
            .await
        {
            Ok(res) => {
                let mapped: Vec<serde_json::Value> = res
                    .iter()
                    .map(|(id, dist, meta)| {
                        serde_json::json!({
                            "id": id,
                            "distance": dist,
                            "metadata": meta
                        })
                    })
                    .collect();
                Json(mapped).into_response()
            }
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
        }
    } else {
        (StatusCode::NOT_FOUND, "Collection not found").into_response()
    }
}

async fn get_logs() -> Json<Vec<String>> {
    Json(vec![
        "[SYSTEM] Hyperspace DB v1.2.0 Online".into(),
        "[INFO] Control Plane: HTTP :50050".into(),
        "[INFO] Data Plane: gRPC :50051".into(),
    ])
}

async fn rebuild_collection_http(
    Path(name): Path<String>,
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    match manager.rebuild_collection(&ctx.user_id, &name).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
    }
}

async fn trigger_vacuum_http(
    State((_manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if !ctx.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }

    #[cfg(not(windows))]
    {
        // 1. Refresh jemalloc statistics
        if let Err(e) = epoch::advance() {
            eprintln!("Failed to advance jemalloc epoch: {e}");
        }

        // 2. Perform global purge via mallctl
        // In jemalloc 5.x, "arena.4096.purge" purges all arenas.
        // SAFETY: Calling jemalloc purge is safe here as it only triggers memory return to OS.
        if let Err(e) = unsafe { tikv_jemalloc_ctl::raw::update(b"arena.4096.purge\0", ()) } {
            eprintln!("Failed to purge jemalloc arenas: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"status": "Error", "message": format!("Purge failed: {e}")})),
            )
                .into_response();
        }

        return Json(serde_json::json!({
            "status": "Success",
            "message": "System memory purged and returned to OS"
        }))
        .into_response();
    }

    #[cfg(windows)]
    Json(serde_json::json!({
        "status": "Unavailable",
        "message": "Memory purge via jemalloc is not supported on Windows"
    }))
    .into_response()
}

async fn get_usage_report_http(
    State((manager, _, _)): State<(
        Arc<CollectionManager>,
        Arc<Instant>,
        Arc<Option<EmbeddingInfo>>,
    )>,
    Extension(ctx): Extension<RequestContext>,
) -> impl IntoResponse {
    if !ctx.is_admin {
        return (StatusCode::FORBIDDEN, "Admin access required").into_response();
    }
    let report = manager.get_usage_report();
    Json(report).into_response()
}

// ─────────────────────────────────────────────────────────────────────────────
// NietzscheDB graph proxy  (GET /api/nietzsche/graph)
// ─────────────────────────────────────────────────────────────────────────────

/// Lazy-connected gRPC client shared across requests (connect_lazy = no TCP until
/// first RPC; cloning is cheap — shares the underlying connection pool).
static NIETZSCHE: OnceLock<NietzscheDbClient<Channel>> = OnceLock::new();

fn nietzsche_client() -> NietzscheDbClient<Channel> {
    NIETZSCHE
        .get_or_init(|| {
            let addr = std::env::var("NIETZSCHE_ADDR")
                .unwrap_or_else(|_| "http://[::1]:50051".to_string());
            let uri: tonic::transport::Uri =
                addr.parse().expect("NIETZSCHE_ADDR must be a valid URI");
            NietzscheDbClient::new(Channel::builder(uri).connect_lazy())
        })
        .clone()
}

#[derive(serde::Deserialize)]
struct GraphQueryParams {
    collection: Option<String>,
    node_limit: Option<u32>,
    edge_limit: Option<u32>,
}

#[derive(serde::Serialize)]
struct CosmoNode {
    id:         String,
    label:      String,
    energy:     f32,
    node_type:  String,
    color:      String,
    x:          f64,
    y:          f64,
    created_at: i64,
}

#[derive(serde::Serialize)]
struct CosmoLink {
    source: String,
    target: String,
}

#[derive(serde::Serialize)]
struct CosmoGraph {
    nodes:     Vec<CosmoNode>,
    links:     Vec<CosmoLink>,
    reachable: bool,
}

fn node_type_color(t: &str) -> &'static str {
    match t {
        "Semantic"      => "#6366f1",
        "Episodic"      => "#06b6d4",
        "Concept"       => "#f59e0b",
        "DreamSnapshot" => "#8b5cf6",
        _               => "#64748b",
    }
}

async fn get_nietzsche_graph(
    Query(params): Query<GraphQueryParams>,
) -> impl IntoResponse {
    let collection = params.collection.unwrap_or_default();
    let node_limit = params.node_limit.unwrap_or(500);
    let edge_limit = params.edge_limit.unwrap_or(2000);

    let mut client = nietzsche_client();

    // ── Fetch nodes ──────────────────────────────────────────────────────────
    let node_nql = format!("MATCH (n) RETURN n LIMIT {node_limit}");
    let node_resp = match client
        .query(QueryRequest {
            nql:        node_nql,
            params:     Default::default(),
            collection: collection.clone(),
        })
        .await
    {
        Ok(r)  => r.into_inner(),
        Err(e) => {
            return Json(serde_json::json!({
                "nodes": [], "links": [], "reachable": false,
                "error": e.to_string()
            }))
            .into_response();
        }
    };

    let nodes: Vec<CosmoNode> = node_resp
        .nodes
        .into_iter()
        .map(|n| {
            let (x, y) = n
                .embedding
                .as_ref()
                .filter(|e| e.coords.len() >= 2)
                .map(|e| (e.coords[0], e.coords[1]))
                .unwrap_or((0.0, 0.0));

            let label = serde_json::from_slice::<serde_json::Value>(&n.content)
                .ok()
                .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(String::from))
                .unwrap_or_else(|| {
                    let short = if n.id.len() >= 8 { &n.id[..8] } else { &n.id };
                    format!("{} · {}", n.node_type, short)
                });

            CosmoNode {
                color:      node_type_color(&n.node_type).to_string(),
                id:         n.id,
                label,
                energy:     n.energy,
                node_type:  n.node_type,
                x,
                y,
                created_at: n.created_at,
            }
        })
        .collect();

    // ── Fetch edges (node pairs from path pattern) ───────────────────────────
    let edge_nql = format!("MATCH (a)-[]->(b) RETURN a, b LIMIT {edge_limit}");
    let edge_resp = client
        .query(QueryRequest {
            nql:        edge_nql,
            params:     Default::default(),
            collection,
        })
        .await
        .map(|r| r.into_inner())
        .unwrap_or_default();

    let links: Vec<CosmoLink> = edge_resp
        .node_pairs
        .into_iter()
        .filter_map(|pair| {
            let src = pair.from?.id;
            let dst = pair.to?.id;
            if src.is_empty() || dst.is_empty() {
                return None;
            }
            Some(CosmoLink { source: src, target: dst })
        })
        .collect();

    Json(CosmoGraph { nodes, links, reachable: true }).into_response()
}
