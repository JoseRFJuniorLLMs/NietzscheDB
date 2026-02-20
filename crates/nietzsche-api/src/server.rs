//! gRPC service implementation for NietzscheDB.
//!
//! [`NietzscheServer`] wraps an [`Arc<CollectionManager>`] and dispatches every
//! data-plane RPC to the collection named in the `collection` field of the
//! request (empty string → `"default"`).
//!
//! ## Multi-collection routing
//! ```text
//! RPC request { collection: "memories", ... }
//!       │
//!       ▼
//! CollectionManager::get_or_default("memories")
//!       │  returns Arc<RwLock<NietzscheDB<AnyVectorStore>>>
//!       ▼
//! db.read().await  →   &NietzscheDB  (concurrent reads)
//! db.write().await →   &mut NietzscheDB (exclusive mutations)
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use nietzsche_graph::{
    CollectionConfig, CollectionManager,
    Edge, EdgeType, GraphError, Node, NodeType, PoincareVector,
    traversal::{BfsConfig, DijkstraConfig},
    bfs, dijkstra,
};
use nietzsche_pregel::{DiffusionConfig, DiffusionEngine};
use nietzsche_query::{ParamValue, Params, execute, parse};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use nietzsche_zaratustra::{ZaratustraConfig, ZaratustraEngine};
use nietzsche_sensory::{
    Modality, OriginalShape, QuantLevel,
    encoder::build_sensory_memory,
    storage::SensoryStorage,
};

use crate::cdc::{CdcBroadcaster, CdcEventType};
use crate::proto::nietzsche::{
    self,
    nietzsche_db_server::{NietzscheDb, NietzscheDbServer},
};
use crate::validation::{
    parse_uuid as val_uuid, validate_embedding, validate_energy, validate_k,
    validate_nql, validate_sleep_params, validate_source_count, validate_t_values,
};

pub use nietzsche::nietzsche_db_server::NietzscheDbServer as TonicServer;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn graph_err(e: GraphError) -> Status {
    Status::internal(e.to_string())
}

fn parse_uuid(s: &str, field: &str) -> Result<Uuid, Status> {
    val_uuid(s, field)
}

fn parse_edge_type(s: &str) -> EdgeType {
    match s {
        "Hierarchical"     => EdgeType::Hierarchical,
        "LSystemGenerated" => EdgeType::LSystemGenerated,
        "Pruned"           => EdgeType::Pruned,
        _                  => EdgeType::Association,
    }
}

fn parse_node_type(s: &str) -> NodeType {
    match s {
        "Episodic"      => NodeType::Episodic,
        "Concept"       => NodeType::Concept,
        "DreamSnapshot" => NodeType::DreamSnapshot,
        _               => NodeType::Semantic,
    }
}

fn node_to_proto(node: Node) -> nietzsche::NodeResponse {
    let (meta, embedding) = node.into_parts();
    let content_bytes = serde_json::to_vec(&meta.content).unwrap_or_default();
    nietzsche::NodeResponse {
        found:           true,
        id:              meta.id.to_string(),
        embedding:       Some(nietzsche::PoincareVector {
            coords: embedding.coords_f64(),
            dim:    embedding.dim as u32,
        }),
        energy:          meta.energy,
        depth:           meta.depth,
        hausdorff_local: meta.hausdorff_local,
        created_at:      meta.created_at,
        content:         content_bytes,
        node_type:       format!("{:?}", meta.node_type),
        expires_at:      meta.expires_at.unwrap_or(0),
    }
}

fn not_found() -> nietzsche::NodeResponse {
    nietzsche::NodeResponse { found: false, ..Default::default() }
}

fn ok_status() -> nietzsche::StatusResponse {
    nietzsche::StatusResponse { status: "ok".into(), error: String::new() }
}

/// Resolve `collection` field: empty → `"default"`.
#[inline]
fn col(s: &str) -> &str {
    if s.is_empty() { "default" } else { s }
}

/// Look up a collection, returning `Status::not_found` if missing.
macro_rules! get_col {
    ($cm:expr, $name:expr) => {{
        match $cm.get_or_default(col($name)) {
            Some(db) => db,
            None => return Err(Status::not_found(format!(
                "collection '{}' not found", col($name)
            ))),
        }
    }};
}

/// Marshal proto `QueryParamValue` map into executor `Params`.
fn marshal_params(
    proto: &HashMap<String, nietzsche::QueryParamValue>,
) -> Result<Params, Status> {
    let mut params = Params::new();
    for (key, pv) in proto {
        let val = match &pv.value {
            Some(nietzsche::query_param_value::Value::StringVal(s)) =>
                ParamValue::Str(s.clone()),
            Some(nietzsche::query_param_value::Value::FloatVal(f)) =>
                ParamValue::Float(*f),
            Some(nietzsche::query_param_value::Value::IntVal(i)) =>
                ParamValue::Int(*i),
            Some(nietzsche::query_param_value::Value::UuidVal(s)) => {
                let u = Uuid::parse_str(s).map_err(|_| {
                    Status::invalid_argument(format!("param '{key}': invalid UUID '{s}'"))
                })?;
                ParamValue::Uuid(u)
            }
            Some(nietzsche::query_param_value::Value::VecVal(v)) =>
                ParamValue::Vector(v.coords.clone()),
            None => return Err(Status::invalid_argument(
                format!("param '{key}' has no value set")
            )),
        };
        params.insert(key.clone(), val);
    }
    Ok(params)
}

// ── Sensory helpers ───────────────────────────────────────────────────────────

fn parse_modality(modality: &str, meta: &serde_json::Value) -> Result<Modality, Status> {
    match modality {
        "text" => Ok(Modality::Text {
            token_count: meta.get("token_count").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            language:    meta.get("language").and_then(|v| v.as_str()).unwrap_or("en").to_string(),
        }),
        "audio" => Ok(Modality::Audio {
            sample_rate: meta.get("sample_rate").and_then(|v| v.as_u64()).unwrap_or(16000) as u32,
            duration_ms: meta.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            channels:    meta.get("channels").and_then(|v| v.as_u64()).unwrap_or(1) as u8,
        }),
        "image" => Ok(Modality::Image {
            width:    meta.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            height:   meta.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
            channels: meta.get("channels").and_then(|v| v.as_u64()).unwrap_or(3) as u8,
        }),
        "fused" => Ok(Modality::Fused { components: vec![] }),
        other => Err(Status::invalid_argument(format!("unknown modality '{other}'; expected text|audio|image|fused"))),
    }
}

// ── NietzscheServer ───────────────────────────────────────────────────────────

/// gRPC service handler backed by a [`CollectionManager`].
///
/// Each RPC extracts the `collection` field (empty → `"default"`) and
/// dispatches to the corresponding isolated `NietzscheDB<AnyVectorStore>`.
pub struct NietzscheServer {
    cm: Arc<CollectionManager>,
    cdc: Arc<CdcBroadcaster>,
}

impl NietzscheServer {
    pub fn new(cm: Arc<CollectionManager>, cdc: Arc<CdcBroadcaster>) -> Self {
        Self { cm, cdc }
    }

    /// Wrap this server in a tonic [`NietzscheDbServer`].
    pub fn into_service(self) -> NietzscheDbServer<Self> {
        NietzscheDbServer::new(self)
    }

    /// Start listening on `addr`.
    pub async fn serve(
        self,
        addr: std::net::SocketAddr,
    ) -> Result<(), tonic::transport::Error> {
        tracing::info!("NietzscheDB gRPC listening on {addr}");
        tonic::transport::Server::builder()
            .add_service(self.into_service())
            .serve(addr)
            .await
    }
}

// ── gRPC trait implementation ─────────────────────────────────────────────────

#[tonic::async_trait]
impl NietzscheDb for NietzscheServer {

    // ── Collection management ─────────────────────────────────────────────

    async fn create_collection(
        &self,
        req: Request<nietzsche::CreateCollectionRequest>,
    ) -> Result<Response<nietzsche::CreateCollectionResponse>, Status> {
        let r = req.into_inner();
        if r.collection.is_empty() {
            return Err(Status::invalid_argument("collection name must not be empty"));
        }
        let already_exists = self.cm.get(&r.collection).is_some();

        let dim    = if r.dim > 0 { r.dim as usize } else { 3072 };
        let metric = if r.metric.is_empty() { "cosine".to_string() } else { r.metric.clone() };

        let cfg = CollectionConfig { name: r.collection.clone(), dim, metric };
        self.cm.create_collection(cfg).map_err(graph_err)?;

        Ok(Response::new(nietzsche::CreateCollectionResponse {
            created:    !already_exists,
            collection: r.collection,
        }))
    }

    async fn drop_collection(
        &self,
        req: Request<nietzsche::DropCollectionRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r = req.into_inner();
        self.cm.drop_collection(&r.collection).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
    }

    async fn list_collections(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::ListCollectionsResponse>, Status> {
        let collections = self.cm.list()
            .into_iter()
            .map(|i| nietzsche::CollectionInfoProto {
                collection: i.name,
                dim:        i.dim as u32,
                metric:     i.metric,
                node_count: i.node_count as u64,
                edge_count: i.edge_count as u64,
            })
            .collect();
        Ok(Response::new(nietzsche::ListCollectionsResponse { collections }))
    }

    // ── Node CRUD ─────────────────────────────────────────────────────────

    #[instrument(skip(self, req), fields(node_id))]
    async fn insert_node(
        &self,
        req: Request<nietzsche::InsertNodeRequest>,
    ) -> Result<Response<nietzsche::NodeResponse>, Status> {
        let r = req.into_inner();

        let id = if r.id.is_empty() { Uuid::new_v4() } else { parse_uuid(&r.id, "id")? };

        let emb_proto = r.embedding.ok_or_else(|| {
            Status::invalid_argument("embedding is required")
        })?;
        validate_embedding(&emb_proto)?;
        if r.energy != 0.0 { validate_energy(r.energy)?; }

        let embedding = PoincareVector::from_f64(emb_proto.coords);
        let content: serde_json::Value = if r.content.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&r.content)
                .map_err(|e| Status::invalid_argument(format!("invalid content JSON: {e}")))?
        };

        let mut node = Node::new(id, embedding, content);
        if r.energy > 0.0 { node.energy = r.energy; }
        if r.expires_at > 0 { node.meta.expires_at = Some(r.expires_at); }
        debug!(node_id = %id, collection = %col(&r.collection), "inserting node");

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.insert_node(node.clone()).map_err(graph_err)?;
        self.cdc.publish(CdcEventType::InsertNode, id, col(&r.collection));

        Ok(Response::new(node_to_proto(node)))
    }

    async fn get_node(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::NodeResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.id, "id")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;
        match db.get_node(id).map_err(graph_err)? {
            Some(node) => Ok(Response::new(node_to_proto(node))),
            None       => Ok(Response::new(not_found())),
        }
    }

    async fn delete_node(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.id, "id")?;

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.delete_node(id).map_err(graph_err)?;
        self.cdc.publish(CdcEventType::DeleteNode, id, col(&r.collection));
        Ok(Response::new(ok_status()))
    }

    async fn update_energy(
        &self,
        req: Request<nietzsche::UpdateEnergyRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.node_id, "node_id")?;

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.update_energy(id, r.energy).map_err(graph_err)?;
        self.cdc.publish(CdcEventType::UpdateNode, id, col(&r.collection));
        Ok(Response::new(ok_status()))
    }

    // ── Edge CRUD ─────────────────────────────────────────────────────────

    async fn insert_edge(
        &self,
        req: Request<nietzsche::InsertEdgeRequest>,
    ) -> Result<Response<nietzsche::EdgeResponse>, Status> {
        let r = req.into_inner();

        let id   = if r.id.is_empty() { Uuid::new_v4() } else { parse_uuid(&r.id, "id")? };
        let from = parse_uuid(&r.from, "from")?;
        let to   = parse_uuid(&r.to,   "to")?;

        let mut edge = Edge::new(from, to, parse_edge_type(&r.edge_type), r.weight as f32);
        edge.id       = id;
        edge.metadata = HashMap::new();

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.insert_edge(edge).map_err(graph_err)?;
        self.cdc.publish(CdcEventType::InsertEdge, id, col(&r.collection));

        Ok(Response::new(nietzsche::EdgeResponse { success: true, id: id.to_string() }))
    }

    async fn delete_edge(
        &self,
        req: Request<nietzsche::EdgeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.id, "id")?;

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.delete_edge(id).map_err(graph_err)?;
        self.cdc.publish(CdcEventType::DeleteEdge, id, col(&r.collection));
        Ok(Response::new(ok_status()))
    }

    // ── MERGE (upsert semantic) ──────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn merge_node(
        &self,
        req: Request<nietzsche::MergeNodeRequest>,
    ) -> Result<Response<nietzsche::MergeNodeResponse>, Status> {
        let r = req.into_inner();

        let match_keys: serde_json::Value = if r.match_keys.is_empty() {
            return Err(Status::invalid_argument("match_keys must not be empty"));
        } else {
            serde_json::from_slice(&r.match_keys)
                .map_err(|e| Status::invalid_argument(format!("invalid match_keys JSON: {e}")))?
        };

        let on_create: serde_json::Value = if r.on_create_set.is_empty() {
            serde_json::Value::Object(Default::default())
        } else {
            serde_json::from_slice(&r.on_create_set)
                .map_err(|e| Status::invalid_argument(format!("invalid on_create_set JSON: {e}")))?
        };

        let on_match: serde_json::Value = if r.on_match_set.is_empty() {
            serde_json::Value::Object(Default::default())
        } else {
            serde_json::from_slice(&r.on_match_set)
                .map_err(|e| Status::invalid_argument(format!("invalid on_match_set JSON: {e}")))?
        };

        let node_type_str = if r.node_type.is_empty() { "Semantic" } else { &r.node_type };

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;

        // Try to find an existing node by type + content keys
        let existing = db.find_node_by_content(node_type_str, &match_keys)
            .map_err(graph_err)?;

        match existing {
            Some(node) => {
                // ON MATCH: update content with on_match_set
                let node_id = node.id;
                if on_match.as_object().map_or(false, |o| !o.is_empty()) {
                    db.update_node_content(node_id, &on_match).map_err(graph_err)?;
                }
                // Re-fetch updated node
                let updated = db.get_node(node_id).map_err(graph_err)?
                    .unwrap_or(node);
                debug!(node_id = %node_id, collection = %col(&r.collection), "MergeNode: matched existing");
                Ok(Response::new(nietzsche::MergeNodeResponse {
                    created: false,
                    node_id: node_id.to_string(),
                    node:    Some(node_to_proto(updated)),
                }))
            }
            None => {
                // ON CREATE: build new node with match_keys + on_create_set merged
                let id = Uuid::new_v4();

                // Merge match_keys and on_create_set into content
                let mut content = serde_json::Map::new();
                if let Some(mk) = match_keys.as_object() {
                    for (k, v) in mk { content.insert(k.clone(), v.clone()); }
                }
                if let Some(oc) = on_create.as_object() {
                    for (k, v) in oc { content.insert(k.clone(), v.clone()); }
                }

                // Build embedding — use provided or zero vector (origin of Poincaré ball)
                let embedding = if let Some(emb_proto) = r.embedding {
                    validate_embedding(&emb_proto)?;
                    PoincareVector::from_f64(emb_proto.coords)
                } else {
                    PoincareVector::from_f64(vec![0.0; 3072])
                };

                let mut node = Node::new(id, embedding, serde_json::Value::Object(content));
                node.meta.node_type = parse_node_type(node_type_str);
                if r.energy > 0.0 { node.energy = r.energy; }

                db.insert_node(node.clone()).map_err(graph_err)?;
                debug!(node_id = %id, collection = %col(&r.collection), "MergeNode: created new");

                Ok(Response::new(nietzsche::MergeNodeResponse {
                    created: true,
                    node_id: id.to_string(),
                    node:    Some(node_to_proto(node)),
                }))
            }
        }
    }

    #[instrument(skip(self, req))]
    async fn merge_edge(
        &self,
        req: Request<nietzsche::MergeEdgeRequest>,
    ) -> Result<Response<nietzsche::MergeEdgeResponse>, Status> {
        let r = req.into_inner();

        let from = parse_uuid(&r.from_node_id, "from_node_id")?;
        let to   = parse_uuid(&r.to_node_id, "to_node_id")?;
        let edge_type_str = if r.edge_type.is_empty() { "Association" } else { &r.edge_type };

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;

        // Try to find an existing edge by (from, to, type)
        let existing = db.find_edge(from, to, edge_type_str)
            .map_err(graph_err)?;

        match existing {
            Some(edge) => {
                // ON MATCH: update edge metadata if provided
                // Edge metadata updates would go here if edges had mutable content.
                // For now, we just return the existing edge ID.
                debug!(edge_id = %edge.id, "MergeEdge: matched existing");
                Ok(Response::new(nietzsche::MergeEdgeResponse {
                    created: false,
                    edge_id: edge.id.to_string(),
                }))
            }
            None => {
                // ON CREATE: insert new edge
                let id = Uuid::new_v4();
                let mut edge = Edge::new(from, to, parse_edge_type(edge_type_str), 1.0);
                edge.id = id;
                edge.metadata = HashMap::new();

                // Apply on_create_set as edge metadata
                if !r.on_create_set.is_empty() {
                    let props: serde_json::Value = serde_json::from_slice(&r.on_create_set)
                        .map_err(|e| Status::invalid_argument(format!("invalid on_create_set JSON: {e}")))?;
                    if let Some(obj) = props.as_object() {
                        for (k, v) in obj {
                            edge.metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

                db.insert_edge(edge).map_err(graph_err)?;
                debug!(edge_id = %id, "MergeEdge: created new");

                Ok(Response::new(nietzsche::MergeEdgeResponse {
                    created: true,
                    edge_id: id.to_string(),
                }))
            }
        }
    }

    // ── NQL query ─────────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn query(
        &self,
        req: Request<nietzsche::QueryRequest>,
    ) -> Result<Response<nietzsche::QueryResponse>, Status> {
        let inner = req.into_inner();
        validate_nql(&inner.nql)?;

        let ast    = parse(&inner.nql).map_err(|e| {
            Status::invalid_argument(format!("NQL parse error: {e}"))
        })?;
        let params = marshal_params(&inner.params)?;

        let shared  = get_col!(self.cm, &inner.collection);
        let mut db  = shared.read().await;
        let results = execute(&ast, db.storage(), db.adjacency(), &params)
            .map_err(|e| Status::internal(e.to_string()))?;

        use nietzsche_query::{QueryResult, ScalarValue};
        let mut nodes       = Vec::new();
        let mut node_pairs  = Vec::new();
        let mut path_ids    = Vec::new();
        let mut explain     = String::new();
        let mut scalar_rows = Vec::new();

        for r in results {
            match r {
                QueryResult::Node(n) => nodes.push(node_to_proto(n)),
                QueryResult::NodePair { from, to } => node_pairs.push(nietzsche::NodePair {
                    from: Some(node_to_proto(from)),
                    to:   Some(node_to_proto(to)),
                }),
                QueryResult::DiffusionPath(ids) =>
                    path_ids.extend(ids.iter().map(|u| u.to_string())),
                QueryResult::ReconstructRequest { node_id, modality, quality } => {
                    path_ids.push(format!("reconstruct:{}:{}:{}",
                        node_id,
                        modality.unwrap_or_default(),
                        quality.unwrap_or_default(),
                    ));
                }
                QueryResult::ExplainPlan(plan) => { explain = plan; }
                // Phase C: INVOKE ZARATUSTRA — engine is invoked server-side;
                // the response carries a status message in path_ids for now.
                QueryResult::InvokeZaratustraRequest { collection, cycles, alpha, decay } => {
                    path_ids.push(format!(
                        "zaratustra:col={}:cycles={}:alpha={:.4}:decay={:.4}",
                        collection.unwrap_or_else(|| "default".into()),
                        cycles.unwrap_or(1),
                        alpha.unwrap_or(0.1),
                        decay.unwrap_or(0.05),
                    ));
                }
                // Phase D: MERGE intents — the executor returned an upsert
                // request; we need write access to perform the actual mutation.
                // Since we already hold a read lock, we collect these intents
                // and execute them in a second pass with a write lock.
                QueryResult::MergeNodeRequest { node_type, match_keys, on_create_set, on_match_set } => {
                    // Drop read lock, acquire write, perform merge, re-acquire read
                    drop(db);
                    let mut db_w = shared.write().await;
                    let node_type_str = node_type.as_deref().unwrap_or("Semantic");
                    let existing = db_w.find_node_by_content(node_type_str, &match_keys)
                        .map_err(graph_err)?;
                    match existing {
                        Some(node) => {
                            // ON MATCH: update content
                            if on_match_set.as_object().map_or(false, |o| !o.is_empty()) {
                                db_w.update_node_content(node.id, &on_match_set)
                                    .map_err(graph_err)?;
                            }
                            nodes.push(node_to_proto(node));
                        }
                        None => {
                            // ON CREATE: build new node with merged content
                            let mut content = match_keys.clone();
                            if let (serde_json::Value::Object(base), serde_json::Value::Object(extra))
                                = (&mut content, &on_create_set)
                            {
                                for (k, v) in extra {
                                    base.insert(k.clone(), v.clone());
                                }
                            }
                            let nt = parse_node_type(node_type_str);
                            let embedding = PoincareVector::new(vec![0.0_f32; 3072]);
                            let mut node = Node::new(Uuid::new_v4(), embedding, content);
                            node.node_type = nt;
                            db_w.insert_node(node.clone()).map_err(graph_err)?;
                            nodes.push(node_to_proto(node));
                        }
                    }
                    drop(db_w);
                    db = shared.read().await;
                }
                QueryResult::MergeEdgeRequest { from_match, to_match, edge_type, on_create_set, on_match_set } => {
                    drop(db);
                    let mut db_w = shared.write().await;
                    // Find 'from' and 'to' nodes by content match
                    let from_node = db_w.find_node_by_content("Semantic", &from_match)
                        .map_err(graph_err)?;
                    let to_node = db_w.find_node_by_content("Semantic", &to_match)
                        .map_err(graph_err)?;
                    match (from_node, to_node) {
                        (Some(f), Some(t)) => {
                            let et_str = edge_type.as_deref().unwrap_or("Association");
                            let et = parse_edge_type(et_str);
                            let existing = db_w.find_edge(f.id, t.id, et_str)
                                .map_err(graph_err)?;
                            match existing {
                                Some(mut edge) => {
                                    // ON MATCH: update metadata
                                    if let serde_json::Value::Object(updates) = &on_match_set {
                                        for (k, v) in updates {
                                            edge.metadata.insert(k.clone(), v.clone());
                                        }
                                    }
                                    path_ids.push(format!("merge:edge:matched:{}", edge.id));
                                }
                                None => {
                                    let mut metadata = HashMap::new();
                                    if let serde_json::Value::Object(props) = &on_create_set {
                                        for (k, v) in props {
                                            metadata.insert(k.clone(), v.clone());
                                        }
                                    }
                                    let mut edge = Edge::new(f.id, t.id, et, 1.0);
                                    edge.metadata = metadata;
                                    db_w.insert_edge(edge.clone()).map_err(graph_err)?;
                                    path_ids.push(format!("merge:edge:created:{}", edge.id));
                                }
                            }
                        }
                        _ => {
                            return Err(Status::not_found(
                                "MERGE edge: one or both endpoint nodes not found"
                            ));
                        }
                    }
                    drop(db_w);
                    db = shared.read().await;
                }
                // Phase F: transaction control — acknowledged; actual Tx state
                // is managed by the connection session layer (future phases).
                QueryResult::TxBegin    => { path_ids.push("tx:begin".into()); }
                QueryResult::TxCommit   => { path_ids.push("tx:commit".into()); }
                QueryResult::TxRollback => { path_ids.push("tx:rollback".into()); }
                QueryResult::Scalar(row) => {
                    let entries = row.into_iter().map(|(col_name, val)| {
                        let mut entry = nietzsche::ScalarEntry {
                            column:  col_name,
                            is_null: false,
                            ..Default::default()
                        };
                        match val {
                            ScalarValue::Float(f) => entry.value = Some(
                                nietzsche::scalar_entry::Value::FloatVal(f)
                            ),
                            ScalarValue::Int(i) => entry.value = Some(
                                nietzsche::scalar_entry::Value::IntVal(i)
                            ),
                            ScalarValue::Str(s) => entry.value = Some(
                                nietzsche::scalar_entry::Value::StringVal(s)
                            ),
                            ScalarValue::Bool(b) => entry.value = Some(
                                nietzsche::scalar_entry::Value::BoolVal(b)
                            ),
                            ScalarValue::Null => entry.is_null = true,
                        }
                        entry
                    }).collect();
                    scalar_rows.push(nietzsche::ScalarRow { entries });
                }
            }
        }

        Ok(Response::new(nietzsche::QueryResponse {
            nodes,
            node_pairs,
            path_ids,
            error: String::new(),
            explain,
            scalar_rows,
        }))
    }

    // ── KNN search ────────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn knn_search(
        &self,
        req: Request<nietzsche::KnnRequest>,
    ) -> Result<Response<nietzsche::KnnResponse>, Status> {
        let r     = req.into_inner();
        validate_k(r.k)?;
        let query = PoincareVector::from_f64(r.query_coords);
        let k     = r.k as usize;

        let shared  = get_col!(self.cm, &r.collection);
        let db      = shared.read().await;
        let results = db.knn(&query, k).map_err(graph_err)?;

        Ok(Response::new(nietzsche::KnnResponse {
            results: results.into_iter()
                .map(|(id, dist)| nietzsche::KnnResult {
                    id:       id.to_string(),
                    distance: dist,
                })
                .collect(),
        }))
    }

    // ── Traversal ─────────────────────────────────────────────────────────

    async fn bfs(
        &self,
        req: Request<nietzsche::TraversalRequest>,
    ) -> Result<Response<nietzsche::TraversalResponse>, Status> {
        let r     = req.into_inner();
        let start = parse_uuid(&r.start_node_id, "start_node_id")?;

        let config = BfsConfig {
            max_depth:  if r.max_depth > 0 { r.max_depth as usize } else { 10 },
            max_nodes:  if r.max_nodes > 0 { r.max_nodes as usize } else { 1_000 },
            energy_min: r.energy_min,
        };

        let shared  = get_col!(self.cm, &r.collection);
        let db      = shared.read().await;
        let visited = bfs(db.storage(), db.adjacency(), start, &config)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::TraversalResponse {
            visited_ids: visited.iter().map(|u| u.to_string()).collect(),
            costs:       Vec::new(),
        }))
    }

    async fn dijkstra(
        &self,
        req: Request<nietzsche::TraversalRequest>,
    ) -> Result<Response<nietzsche::TraversalResponse>, Status> {
        let r     = req.into_inner();
        let start = parse_uuid(&r.start_node_id, "start_node_id")?;

        let config = DijkstraConfig {
            max_nodes:    if r.max_nodes > 0 { r.max_nodes as usize } else { 1_000 },
            energy_min:   r.energy_min,
            max_distance: if r.max_cost > 0.0 { r.max_cost } else { f64::INFINITY },
        };

        let shared = get_col!(self.cm, &r.collection);
        let db     = shared.read().await;
        let costs  = dijkstra(db.storage(), db.adjacency(), start, &config)
            .map_err(graph_err)?;

        let mut entries: Vec<(Uuid, f64)> = costs.into_iter().collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(Response::new(nietzsche::TraversalResponse {
            visited_ids: entries.iter().map(|(u, _)| u.to_string()).collect(),
            costs:       entries.iter().map(|(_, c)| *c).collect(),
        }))
    }

    // ── Diffusion ─────────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn diffuse(
        &self,
        req: Request<nietzsche::DiffusionRequest>,
    ) -> Result<Response<nietzsche::DiffusionResponse>, Status> {
        let r = req.into_inner();
        validate_source_count(r.source_ids.len())?;

        let sources: Vec<Uuid> = r.source_ids.iter()
            .map(|s| parse_uuid(s, "source_ids"))
            .collect::<Result<_, _>>()?;

        let t_values = if r.t_values.is_empty() {
            vec![0.1, 1.0, 10.0]
        } else {
            validate_t_values(&r.t_values)?;
            r.t_values
        };

        let k_chebyshev = if r.k_chebyshev > 0 { r.k_chebyshev as usize } else { 10 };
        let config      = DiffusionConfig { k_chebyshev, ..Default::default() };

        let shared  = get_col!(self.cm, &r.collection);
        let db      = shared.read().await;
        let engine  = DiffusionEngine::new(config);
        let results = engine
            .diffuse(db.storage(), db.adjacency(), &sources, &t_values)
            .map_err(|e| Status::internal(e.to_string()))?;

        let scales = results.into_iter()
            .map(|dr| nietzsche::DiffusionScale {
                t:        dr.t,
                node_ids: dr.activated.iter().map(|(u, _)| u.to_string()).collect(),
                scores:   dr.activated.iter().map(|(_, s)| *s).collect(),
            })
            .collect();

        Ok(Response::new(nietzsche::DiffusionResponse { scales }))
    }

    // ── Sleep cycle ───────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn trigger_sleep(
        &self,
        req: Request<nietzsche::SleepRequest>,
    ) -> Result<Response<nietzsche::SleepResponse>, Status> {
        let r         = req.into_inner();
        let noise_val = if r.noise > 0.0   { r.noise }   else { 0.02 };
        let lr_val    = if r.adam_lr > 0.0 { r.adam_lr } else { 5e-3 };
        validate_sleep_params(noise_val, lr_val)?;

        let config = SleepConfig {
            noise:               noise_val,
            adam_steps:          if r.adam_steps > 0 { r.adam_steps as usize } else { 10 },
            adam_lr:             lr_val,
            hausdorff_threshold: if r.hausdorff_threshold > 0.0 {
                r.hausdorff_threshold
            } else {
                0.15
            },
        };
        warn!(
            noise      = config.noise,
            adam_steps = config.adam_steps,
            collection = %col(&r.collection),
            "triggering sleep cycle — collection will be locked"
        );

        let seed = if r.rng_seed == 0 { rand::random::<u64>() } else { r.rng_seed };
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);

        let shared = get_col!(self.cm, &r.collection);
        let col_name = col(&r.collection).to_string();
        let mut db = shared.write().await;
        let report = SleepCycle::run(&config, &mut *db, &mut rng)
            .map_err(|e| Status::internal(e.to_string()))?;
        self.cdc.publish(CdcEventType::SleepCycle, Uuid::nil(), &col_name);

        Ok(Response::new(nietzsche::SleepResponse {
            hausdorff_before: report.hausdorff_before,
            hausdorff_after:  report.hausdorff_after,
            hausdorff_delta:  report.hausdorff_delta,
            committed:        report.committed,
            nodes_perturbed:  report.nodes_perturbed as u32,
            snapshot_nodes:   report.snapshot_nodes  as u32,
        }))
    }

    // ── Zaratustra (Phase Z) ──────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn invoke_zaratustra(
        &self,
        req: Request<nietzsche::ZaratustraRequest>,
    ) -> Result<Response<nietzsche::ZaratustraResponse>, Status> {
        let r = req.into_inner();

        let mut cfg = ZaratustraConfig::from_env();
        if r.alpha > 0.0 { cfg.alpha = r.alpha; }
        if r.decay > 0.0 { cfg.decay = r.decay; }
        let cycles = if r.cycles == 0 { 1 } else { r.cycles as usize };

        let engine = ZaratustraEngine::new(cfg);
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let mut last_report = None;
        for _ in 0..cycles {
            let report = engine
                .run_cycle(db.storage(), db.adjacency())
                .map_err(|e| Status::internal(format!("Zaratustra error: {e}")))?;
            last_report = Some(report);
        }

        let report = last_report.unwrap();
        let elite_ids: Vec<String> = report
            .ubermensch
            .elite_node_ids
            .iter()
            .map(|u| u.to_string())
            .collect();

        self.cdc.publish(CdcEventType::Zaratustra, Uuid::nil(), col(&r.collection));
        warn!(
            collection = %col(&r.collection),
            cycles = cycles,
            duration_ms = report.duration_ms,
            elite_count = report.ubermensch.elite_count,
            echoes_created = report.eternal_recurrence.echoes_created,
            "Zaratustra cycle completed"
        );

        Ok(Response::new(nietzsche::ZaratustraResponse {
            nodes_updated:      report.will_to_power.nodes_updated,
            mean_energy_before: report.will_to_power.mean_energy_before,
            mean_energy_after:  report.will_to_power.mean_energy_after,
            total_energy_delta: report.will_to_power.total_energy_delta,
            echoes_created:     report.eternal_recurrence.echoes_created,
            echoes_evicted:     report.eternal_recurrence.echoes_evicted,
            total_echoes:       report.eternal_recurrence.total_echoes,
            elite_count:        report.ubermensch.elite_count,
            elite_threshold:    report.ubermensch.energy_threshold,
            mean_elite_energy:  report.ubermensch.mean_elite_energy,
            mean_base_energy:   report.ubermensch.mean_base_energy,
            elite_node_ids:     elite_ids,
            duration_ms:        report.duration_ms,
            cycles_run:         cycles as u32,
        }))
    }

    // ── Batch CRUD ──────────────────────────────────────────────────────

    async fn batch_insert_nodes(
        &self,
        req: Request<nietzsche::BatchInsertNodesRequest>,
    ) -> Result<Response<nietzsche::BatchInsertNodesResponse>, Status> {
        let r = req.into_inner();
        let col_name = col(&r.collection);
        let shared = get_col!(self.cm, col_name);
        let mut db = shared.write().await;

        let mut nodes_vec = Vec::with_capacity(r.nodes.len());
        let mut ids = Vec::with_capacity(r.nodes.len());

        for nr in &r.nodes {
            let id = if nr.id.is_empty() { Uuid::new_v4() } else { parse_uuid(&nr.id, "id")? };
            let emb_proto = nr.embedding.as_ref().ok_or_else(|| {
                Status::invalid_argument("each node requires an embedding")
            })?;
            validate_embedding(emb_proto)?;
            let embedding = PoincareVector::from_f64(emb_proto.coords.clone());
            let content: serde_json::Value = if nr.content.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::from_slice(&nr.content)
                    .map_err(|e| Status::invalid_argument(format!("invalid content JSON: {e}")))?
            };
            let mut node = Node::new(id, embedding, content);
            node.meta.node_type = parse_node_type(&nr.node_type);
            if nr.energy > 0.0 { node.energy = nr.energy; }
            ids.push(id.to_string());
            nodes_vec.push(node);
        }

        let count = ids.len() as u32;
        db.insert_nodes_bulk(nodes_vec).map_err(graph_err)?;
        self.cdc.publish(
            CdcEventType::BatchInsertNodes { count },
            Uuid::nil(),
            col_name,
        );

        Ok(Response::new(nietzsche::BatchInsertNodesResponse {
            inserted: count,
            node_ids: ids,
        }))
    }

    async fn batch_insert_edges(
        &self,
        req: Request<nietzsche::BatchInsertEdgesRequest>,
    ) -> Result<Response<nietzsche::BatchInsertEdgesResponse>, Status> {
        let r = req.into_inner();
        let col_name = col(&r.collection);
        let shared = get_col!(self.cm, col_name);
        let mut db = shared.write().await;

        let mut edges_vec = Vec::with_capacity(r.edges.len());
        let mut ids = Vec::with_capacity(r.edges.len());

        for er in &r.edges {
            let id   = if er.id.is_empty() { Uuid::new_v4() } else { parse_uuid(&er.id, "id")? };
            let from = parse_uuid(&er.from, "from")?;
            let to   = parse_uuid(&er.to, "to")?;
            let mut edge = Edge::new(from, to, parse_edge_type(&er.edge_type), er.weight as f32);
            edge.id = id;
            edge.metadata = HashMap::new();
            ids.push(id.to_string());
            edges_vec.push(edge);
        }

        let count = ids.len() as u32;
        db.insert_edges_bulk(edges_vec).map_err(graph_err)?;
        self.cdc.publish(
            CdcEventType::BatchInsertEdges { count },
            Uuid::nil(),
            col_name,
        );

        Ok(Response::new(nietzsche::BatchInsertEdgesResponse {
            inserted: count,
            edge_ids: ids,
        }))
    }

    // ── Graph Algorithms ───────────────────────────────────────────────

    async fn run_page_rank(
        &self,
        req: Request<nietzsche::PageRankRequest>,
    ) -> Result<Response<nietzsche::AlgorithmScoreResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let config = nietzsche_algo::PageRankConfig {
            damping_factor: if r.damping_factor > 0.0 { r.damping_factor } else { 0.85 },
            max_iterations: if r.max_iterations > 0 { r.max_iterations as usize } else { 20 },
            convergence_threshold: if r.convergence_threshold > 0.0 { r.convergence_threshold } else { 1e-7 },
        };

        let result = nietzsche_algo::pagerank(db.storage(), db.adjacency(), &config)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmScoreResponse {
            scores: result.scores.into_iter().map(|(id, s)| nietzsche::NodeScore {
                node_id: id.to_string(), score: s,
            }).collect(),
            duration_ms: result.duration_ms,
            iterations: result.iterations as u32,
        }))
    }

    async fn run_louvain(
        &self,
        req: Request<nietzsche::LouvainRequest>,
    ) -> Result<Response<nietzsche::AlgorithmCommunityResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let config = nietzsche_algo::LouvainConfig {
            max_iterations: if r.max_iterations > 0 { r.max_iterations as usize } else { 10 },
            resolution: if r.resolution > 0.0 { r.resolution } else { 1.0 },
        };

        let result = nietzsche_algo::louvain(db.storage(), db.adjacency(), &config)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmCommunityResponse {
            assignments: result.communities.into_iter().map(|(id, c)| nietzsche::NodeCommunity {
                node_id: id.to_string(), community_id: c,
            }).collect(),
            community_count: result.community_count as u64,
            largest_size: 0,
            modularity: result.modularity,
            duration_ms: result.duration_ms,
            iterations: result.iterations as u32,
        }))
    }

    async fn run_label_prop(
        &self,
        req: Request<nietzsche::LabelPropRequest>,
    ) -> Result<Response<nietzsche::AlgorithmCommunityResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let max_iter = if r.max_iterations > 0 { r.max_iterations as usize } else { 10 };
        let result = nietzsche_algo::label_propagation(db.storage(), db.adjacency(), max_iter)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmCommunityResponse {
            assignments: result.labels.into_iter().map(|(id, c)| nietzsche::NodeCommunity {
                node_id: id.to_string(), community_id: c,
            }).collect(),
            community_count: result.community_count as u64,
            largest_size: 0,
            modularity: 0.0,
            duration_ms: result.duration_ms,
            iterations: result.iterations as u32,
        }))
    }

    async fn run_betweenness(
        &self,
        req: Request<nietzsche::BetweennessRequest>,
    ) -> Result<Response<nietzsche::AlgorithmScoreResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let sample = if r.sample_size > 0 { Some(r.sample_size as usize) } else { None };
        let result = nietzsche_algo::betweenness_centrality(db.storage(), db.adjacency(), sample)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmScoreResponse {
            scores: result.scores.into_iter().map(|(id, s)| nietzsche::NodeScore {
                node_id: id.to_string(), score: s,
            }).collect(),
            duration_ms: result.duration_ms,
            iterations: 0,
        }))
    }

    async fn run_closeness(
        &self,
        req: Request<nietzsche::ClosenessRequest>,
    ) -> Result<Response<nietzsche::AlgorithmScoreResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let result = nietzsche_algo::closeness_centrality(db.storage(), db.adjacency())
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmScoreResponse {
            scores: result.into_iter().map(|(id, s)| nietzsche::NodeScore {
                node_id: id.to_string(), score: s,
            }).collect(),
            duration_ms: 0,
            iterations: 0,
        }))
    }

    async fn run_degree_centrality(
        &self,
        req: Request<nietzsche::DegreeCentralityRequest>,
    ) -> Result<Response<nietzsche::AlgorithmScoreResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let direction = match r.direction.as_str() {
            "in"  => nietzsche_algo::Direction::In,
            "out" => nietzsche_algo::Direction::Out,
            _     => nietzsche_algo::Direction::Both,
        };

        let node_ids: Vec<Uuid> = db.storage()
            .scan_nodes_meta()
            .map_err(graph_err)?
            .into_iter()
            .map(|n| n.id)
            .collect();

        let result = nietzsche_algo::degree_centrality(db.adjacency(), direction, &node_ids);

        Ok(Response::new(nietzsche::AlgorithmScoreResponse {
            scores: result.into_iter().map(|(id, s)| nietzsche::NodeScore {
                node_id: id.to_string(), score: s,
            }).collect(),
            duration_ms: 0,
            iterations: 0,
        }))
    }

    async fn run_wcc(
        &self,
        req: Request<nietzsche::WccRequest>,
    ) -> Result<Response<nietzsche::AlgorithmCommunityResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let result = nietzsche_algo::weakly_connected_components(db.storage(), db.adjacency())
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmCommunityResponse {
            assignments: result.components.into_iter().map(|(id, c)| nietzsche::NodeCommunity {
                node_id: id.to_string(), community_id: c,
            }).collect(),
            community_count: result.component_count as u64,
            largest_size: result.largest_component_size as u64,
            modularity: 0.0,
            duration_ms: result.duration_ms,
            iterations: 0,
        }))
    }

    async fn run_scc(
        &self,
        req: Request<nietzsche::SccRequest>,
    ) -> Result<Response<nietzsche::AlgorithmCommunityResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let result = nietzsche_algo::strongly_connected_components(db.storage(), db.adjacency())
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::AlgorithmCommunityResponse {
            assignments: result.components.into_iter().map(|(id, c)| nietzsche::NodeCommunity {
                node_id: id.to_string(), community_id: c,
            }).collect(),
            community_count: result.component_count as u64,
            largest_size: result.largest_component_size as u64,
            modularity: 0.0,
            duration_ms: result.duration_ms,
            iterations: 0,
        }))
    }

    async fn run_a_star(
        &self,
        req: Request<nietzsche::AStarRequest>,
    ) -> Result<Response<nietzsche::AStarResponse>, Status> {
        let r = req.into_inner();
        let start = parse_uuid(&r.start_node_id, "start_node_id")?;
        let goal  = parse_uuid(&r.goal_node_id, "goal_node_id")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let result = nietzsche_algo::astar(db.storage(), db.adjacency(), start, goal)
            .map_err(graph_err)?;

        match result {
            Some((path, cost)) => Ok(Response::new(nietzsche::AStarResponse {
                found: true,
                path: path.into_iter().map(|id| id.to_string()).collect(),
                cost,
            })),
            None => Ok(Response::new(nietzsche::AStarResponse {
                found: false, path: vec![], cost: 0.0,
            })),
        }
    }

    async fn run_triangle_count(
        &self,
        req: Request<nietzsche::TriangleCountRequest>,
    ) -> Result<Response<nietzsche::TriangleCountResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let count = nietzsche_algo::triangle_count(db.storage(), db.adjacency())
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::TriangleCountResponse { count }))
    }

    async fn run_jaccard_similarity(
        &self,
        req: Request<nietzsche::JaccardRequest>,
    ) -> Result<Response<nietzsche::SimilarityResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let top_k = if r.top_k > 0 { r.top_k as usize } else { 100 };
        let threshold = r.threshold;

        let result = nietzsche_algo::jaccard_similarity(
            db.storage(), db.adjacency(), top_k, threshold
        ).map_err(graph_err)?;

        Ok(Response::new(nietzsche::SimilarityResponse {
            pairs: result.into_iter().map(|p| nietzsche::SimilarityPairProto {
                node_a: p.node_a.to_string(),
                node_b: p.node_b.to_string(),
                score: p.score,
            }).collect(),
        }))
    }

    // ── Backup/Restore ───────────────────────────────────────────────────

    async fn create_backup(
        &self,
        req: Request<nietzsche::CreateBackupRequest>,
    ) -> Result<Response<nietzsche::BackupResponse>, Status> {
        let r = req.into_inner();
        let label = if r.label.is_empty() { "manual" } else { &r.label };

        let shared = get_col!(self.cm, "default");
        let db = shared.read().await;

        let backup_dir = std::env::var("NIETZSCHE_BACKUP_DIR")
            .unwrap_or_else(|_| "backups".to_string());
        let mgr = nietzsche_graph::BackupManager::new(&backup_dir)
            .map_err(graph_err)?;
        let info = mgr.create_backup(db.storage(), label)
            .map_err(graph_err)?;

        Ok(Response::new(nietzsche::BackupResponse {
            label: info.label,
            path: info.path.to_string_lossy().to_string(),
            created_at: info.created_at,
            size_bytes: info.size_bytes,
        }))
    }

    async fn list_backups(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::ListBackupsResponse>, Status> {
        let backup_dir = std::env::var("NIETZSCHE_BACKUP_DIR")
            .unwrap_or_else(|_| "backups".to_string());
        let mgr = nietzsche_graph::BackupManager::new(&backup_dir)
            .map_err(graph_err)?;
        let backups = mgr.list_backups().map_err(graph_err)?;

        Ok(Response::new(nietzsche::ListBackupsResponse {
            backups: backups.into_iter().map(|b| nietzsche::BackupInfoProto {
                label: b.label,
                path: b.path.to_string_lossy().to_string(),
                created_at: b.created_at,
                size_bytes: b.size_bytes,
            }).collect(),
        }))
    }

    async fn restore_backup(
        &self,
        req: Request<nietzsche::RestoreBackupRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r = req.into_inner();
        if r.backup_path.is_empty() || r.target_path.is_empty() {
            return Err(Status::invalid_argument("backup_path and target_path required"));
        }

        let backup_dir = std::env::var("NIETZSCHE_BACKUP_DIR")
            .unwrap_or_else(|_| "backups".to_string());
        let mgr = nietzsche_graph::BackupManager::new(&backup_dir)
            .map_err(graph_err)?;
        mgr.restore_backup(
            std::path::Path::new(&r.backup_path),
            std::path::Path::new(&r.target_path),
        ).map_err(graph_err)?;

        Ok(Response::new(ok_status()))
    }

    // ── Full-Text Search ─────────────────────────────────────────────────

    async fn full_text_search(
        &self,
        req: Request<nietzsche::FullTextSearchRequest>,
    ) -> Result<Response<nietzsche::FullTextSearchResponse>, Status> {
        let r = req.into_inner();
        if r.query.is_empty() {
            return Err(Status::invalid_argument("query must not be empty"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let limit = if r.limit > 0 { r.limit as usize } else { 10 };
        let fts = nietzsche_graph::FullTextIndex::new(db.storage());
        let results = fts.search(&r.query, limit).map_err(graph_err)?;

        Ok(Response::new(nietzsche::FullTextSearchResponse {
            results: results.into_iter().map(|r| nietzsche::FtsResultProto {
                node_id: r.node_id.to_string(),
                score: r.score,
            }).collect(),
        }))
    }

    // ── Change Data Capture ───────────────────────────────────────────────

    type SubscribeCDCStream = std::pin::Pin<
        Box<dyn futures_core::Stream<Item = Result<nietzsche::CdcEvent, Status>> + Send + 'static>,
    >;

    async fn subscribe_cdc(
        &self,
        req: Request<nietzsche::CdcRequest>,
    ) -> Result<Response<Self::SubscribeCDCStream>, Status> {
        let r = req.into_inner();
        let from_lsn = r.from_lsn;
        let filter_col = if r.collection.is_empty() { None } else { Some(r.collection) };

        let mut rx = self.cdc.subscribe();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        // Skip events before from_lsn
                        if event.lsn < from_lsn {
                            continue;
                        }
                        // Filter by collection if requested
                        if let Some(ref col) = filter_col {
                            if &event.collection != col {
                                continue;
                            }
                        }
                        let batch_count = match &event.event_type {
                            crate::cdc::CdcEventType::BatchInsertNodes { count } => *count,
                            crate::cdc::CdcEventType::BatchInsertEdges { count } => *count,
                            _ => 0,
                        };
                        yield Ok(nietzsche::CdcEvent {
                            lsn:          event.lsn,
                            event_type:   event.event_type.as_str().to_string(),
                            timestamp_ms: event.timestamp_ms,
                            entity_id:    event.entity_id.to_string(),
                            collection:   event.collection,
                            batch_count,
                        });
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "CDC subscriber lagged — some events dropped");
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // ── Admin ─────────────────────────────────────────────────────────────

    async fn get_stats(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::StatsResponse>, Status> {
        // Aggregate node/edge counts across all collections
        let infos = self.cm.list();
        let (node_count, edge_count) = infos
            .iter()
            .fold((0u64, 0u64), |(n, e), i| {
                (n + i.node_count as u64, e + i.edge_count as u64)
            });

        Ok(Response::new(nietzsche::StatsResponse {
            node_count,
            edge_count,
            version:       env!("CARGO_PKG_VERSION").to_string(),
            sensory_count: 0,
        }))
    }

    async fn health_check(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        Ok(Response::new(ok_status()))
    }

    // ── Sensory Compression Layer (Phase 11) ──────────────────────────────

    async fn insert_sensory(
        &self,
        req: Request<nietzsche::InsertSensoryRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;
        let col_name = col(&r.collection).to_string();

        // Parse modality from string + JSON meta bytes
        let modality_meta: serde_json::Value = if r.modality_meta.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_slice(&r.modality_meta)
                .map_err(|e| Status::invalid_argument(format!("modality_meta JSON: {e}")))?
        };

        let modality = parse_modality(&r.modality, &modality_meta)?;
        let original_shape: OriginalShape = if r.original_shape.is_empty() {
            OriginalShape::Text { tokens: 0 }
        } else {
            serde_json::from_slice(&r.original_shape)
                .map_err(|e| Status::invalid_argument(format!("original_shape JSON: {e}")))?
        };

        let latent_f32: Vec<f32> = r.latent.iter().map(|&f| f as f32).collect();
        let sm = build_sensory_memory(
            &latent_f32,
            modality,
            original_shape,
            r.original_bytes as usize,
            r.encoder_version,
        );

        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        {
            let graph_storage = db.storage();
            let sensory = SensoryStorage::new(graph_storage.db_handle());
            sensory.put(&node_id, &sm)
                .map_err(|e| Status::internal(format!("sensory put: {e}")))?;
        }

        debug!(collection = %col_name, node_id = %node_id, modality = %r.modality, dim = sm.latent.dim, "InsertSensory ok");
        Ok(Response::new(ok_status()))
    }

    async fn get_sensory(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::SensoryResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.id, "id")?;
        let col_name = col(&r.collection).to_string();

        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let graph_storage = db.storage();
        let sensory = SensoryStorage::new(graph_storage.db_handle());
        let sm = sensory.get(&node_id)
            .map_err(|e| Status::internal(format!("sensory get: {e}")))?;

        match sm {
            None => Ok(Response::new(nietzsche::SensoryResponse {
                found: false, ..Default::default()
            })),
            Some(sm) => {
                let quant_level_str = match sm.latent.quant_level {
                    QuantLevel::F32  => "f32",
                    QuantLevel::F16  => "f16",
                    QuantLevel::Int8 => "int8",
                    QuantLevel::PQ   => "pq",
                    QuantLevel::Gone => "gone",
                }.to_string();
                let modality_str = match &sm.modality {
                    Modality::Text { .. }  => "text",
                    Modality::Audio { .. } => "audio",
                    Modality::Image { .. } => "image",
                    Modality::Fused { .. } => "fused",
                }.to_string();
                Ok(Response::new(nietzsche::SensoryResponse {
                    found:                  true,
                    node_id:                node_id.to_string(),
                    modality:               modality_str,
                    dim:                    sm.latent.dim,
                    quant_level:            quant_level_str,
                    reconstruction_quality: sm.reconstruction_quality,
                    compression_ratio:      sm.compression_ratio,
                    encoder_version:        sm.encoder_version,
                    byte_size:              sm.latent.byte_size() as u32,
                }))
            }
        }
    }

    async fn reconstruct(
        &self,
        req: Request<nietzsche::ReconstructRequest>,
    ) -> Result<Response<nietzsche::ReconstructResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;
        let col_name = col(&r.collection);

        let shared = get_col!(self.cm, col_name);
        let db = shared.read().await;
        let graph_storage = db.storage();
        let sensory = SensoryStorage::new(graph_storage.db_handle());
        let sm = sensory.get(&node_id)
            .map_err(|e| Status::internal(format!("sensory get: {e}")))?;

        match sm {
            None => Ok(Response::new(nietzsche::ReconstructResponse {
                found: false, ..Default::default()
            })),
            Some(sm) => {
                let f32_latent = sm.latent.as_f32().unwrap_or_default();
                let modality_str = match &sm.modality {
                    Modality::Text { .. }  => "text",
                    Modality::Audio { .. } => "audio",
                    Modality::Image { .. } => "image",
                    Modality::Fused { .. } => "fused",
                }.to_string();
                let original_shape_bytes = serde_json::to_vec(&sm.original_shape)
                    .unwrap_or_default();
                Ok(Response::new(nietzsche::ReconstructResponse {
                    found:          true,
                    node_id:        node_id.to_string(),
                    latent:         f32_latent,
                    modality:       modality_str,
                    quality:        sm.reconstruction_quality,
                    original_shape: original_shape_bytes,
                }))
            }
        }
    }

    async fn degrade_sensory(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.id, "id")?;
        let col_name = col(&r.collection).to_string();

        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;

        // Get current node energy to drive degradation
        let energy = db.get_node_fast(node_id)
            .map_err(graph_err)?
            .map(|n| n.energy)
            .unwrap_or(0.0);

        let graph_storage = db.storage();
        let sensory = SensoryStorage::new(graph_storage.db_handle());
        let mut sm = match sensory.get(&node_id)
            .map_err(|e| Status::internal(format!("sensory get: {e}")))?
        {
            None => return Err(Status::not_found(format!("no sensory data for {node_id}"))),
            Some(sm) => sm,
        };

        let new_quality = sm.latent.degrade(energy);
        sm.reconstruction_quality = new_quality;
        sensory.put(&node_id, &sm)
            .map_err(|e| Status::internal(format!("sensory put after degrade: {e}")))?;

        debug!(collection = %col_name, node_id = %node_id, energy, new_quality, "DegradeSensory ok");
        Ok(Response::new(ok_status()))
    }
}
