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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

use nietzsche_graph::{
    AdjEntry, BackpressureSignal, CausalType, CollectionConfig, CollectionManager,
    Edge, EdgeType, GraphError, MetadataFilter, Node, NodeType, PoincareVector,
    traversal::{BfsConfig, DijkstraConfig},
    bfs, dijkstra,
};
use nietzsche_hyp_ops::{klein, riemann, minkowski, manifold};
use nietzsche_pregel::{DiffusionConfig, DiffusionEngine};
use nietzsche_query::{ParamValue, Params, execute, parse};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use nietzsche_zaratustra::{ZaratustraConfig, ZaratustraEngine};
use nietzsche_neural::{REGISTRY, ModelMetadata};
use nietzsche_sensory::{
    Modality, OriginalShape, QuantLevel,
    encoder::build_sensory_memory,
    storage::SensoryStorage,
};
use nietzsche_gnn::{GnnEngine, NeighborSampler};
use nietzsche_mcts::{MctsAdvisor, MctsConfig};

use nietzsche_cluster::{ClusterNode, ClusterRegistry, NodeHealth, NodeRole};
use crate::cdc::{CdcBroadcaster, CdcEventType};
use crate::rbac::{require_admin, require_writer};
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

/// Convert proto KnnFilter list → MetadataFilter (AND semantics).
fn proto_filters_to_metadata_filter(filters: &[nietzsche::KnnFilter]) -> MetadataFilter {
    if filters.is_empty() {
        return MetadataFilter::None;
    }
    let subs: Vec<MetadataFilter> = filters.iter().filter_map(|f| {
        f.condition.as_ref().map(|c| match c {
            nietzsche::knn_filter::Condition::MatchFilter(m) => MetadataFilter::Eq {
                field: m.field.clone(),
                value: m.value.clone(),
            },
            nietzsche::knn_filter::Condition::RangeFilter(r) => MetadataFilter::Range {
                field: r.field.clone(),
                gte: r.gte,
                lte: r.lte,
            },
        })
    }).collect();
    if subs.len() == 1 {
        subs.into_iter().next().unwrap()
    } else {
        MetadataFilter::And(subs)
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
        backpressure:    None, // populated only by write RPCs
    }
}

fn not_found() -> nietzsche::NodeResponse {
    nietzsche::NodeResponse { found: false, ..Default::default() }
}

/// Convert a graph-layer [`BackpressureSignal`] to the proto representation.
fn bp_to_proto(bp: &BackpressureSignal) -> nietzsche::BackpressureSignal {
    nietzsche::BackpressureSignal {
        accept:            bp.accept,
        reason:            bp.reason.clone(),
        suggested_delay_ms: bp.suggested_delay_ms,
    }
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
    /// Optional cluster registry for gossip protocol. `None` if cluster mode disabled.
    cluster_registry: Option<ClusterRegistry>,
    /// Archetype registry for Collective Unconscious.
    archetype_registry: nietzsche_cluster::ArchetypeRegistry,
    /// Speculative exploration engine.
    dream_engine: nietzsche_dream::DreamEngine,
}

impl NietzscheServer {
    pub fn new(cm: Arc<CollectionManager>, cdc: Arc<CdcBroadcaster>) -> Self {
        Self {
            cm,
            cdc,
            cluster_registry: None,
            archetype_registry: nietzsche_cluster::ArchetypeRegistry::new(),
            dream_engine: nietzsche_dream::DreamEngine::new(nietzsche_dream::DreamConfig::from_env()),
        }
    }

    /// Set the cluster registry for gossip support.
    pub fn with_cluster_registry(mut self, registry: ClusterRegistry) -> Self {
        self.cluster_registry = Some(registry);
        self
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
        require_admin(&req)?;
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
        require_admin(&req)?;
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
        require_writer(&req)?;
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
        let bp = db.check_backpressure();
        self.cdc.publish(CdcEventType::InsertNode, id, col(&r.collection));

        let mut resp = node_to_proto(node);
        resp.backpressure = Some(bp_to_proto(&bp));
        Ok(Response::new(resp))
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
        require_writer(&req)?;
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
        require_writer(&req)?;
        let r  = req.into_inner();
        let id = parse_uuid(&r.node_id, "node_id")?;
        if !r.energy.is_finite() || r.energy < 0.0 || r.energy > 1.0 {
            return Err(Status::invalid_argument(format!(
                "energy must be a finite value in [0.0, 1.0], got {}", r.energy
            )));
        }

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
        require_writer(&req)?;
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
        require_writer(&req)?;
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
        require_writer(&req)?;
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
        require_writer(&req)?;
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
                if !r.on_match_set.is_empty() {
                    let on_match: serde_json::Value = serde_json::from_slice(&r.on_match_set)
                        .map_err(|e| Status::invalid_argument(format!("invalid on_match_set JSON: {e}")))?;
                    if on_match.as_object().map_or(false, |o| !o.is_empty()) {
                        db.update_edge_metadata(edge.id, &on_match).map_err(graph_err)?;
                    }
                }
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

    // ── Edge metadata increment (D.2) ──────────────────────────────────

    #[instrument(skip(self, req))]
    async fn increment_edge_meta(
        &self,
        req: Request<nietzsche::IncrementEdgeMetaRequest>,
    ) -> Result<Response<nietzsche::IncrementEdgeMetaResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();

        let edge_id = parse_uuid(&r.edge_id, "edge_id")?;

        if r.field.is_empty() {
            return Err(Status::invalid_argument("field must not be empty"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;

        let new_value = db.increment_edge_metadata(edge_id, &r.field, r.delta)
            .map_err(graph_err)?;

        debug!(edge_id = %edge_id, field = %r.field, delta = r.delta, new_value, "IncrementEdgeMeta");

        Ok(Response::new(nietzsche::IncrementEdgeMetaResponse {
            new_value,
        }))
    }

    // ── NQL query ─────────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn query(
        &self,
        req: Request<nietzsche::QueryRequest>,
    ) -> Result<Response<nietzsche::QueryResponse>, Status> {
        // NQL can execute mutations (CREATE/SET/DELETE/MERGE/DAEMON) — require writer
        require_writer(&req)?;
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
        let mut nodes        = Vec::new();
        let mut node_pairs   = Vec::new();
        let mut path_ids     = Vec::new();
        let mut explain      = String::new();
        let mut scalar_rows  = Vec::new();
        let mut dream_result = None;

        for r in results {
            match r {
                QueryResult::Node(n) => {
                    let id = n.id;
                    nodes.push(node_to_proto(n));
                    
                    // Energy Backprop (Recall Boost): 
                    // Successful retrieval strengthens the memory trace.
                    // We drop the read lock and acquire a write lock briefly.
                    drop(db);
                    {
                        let mut db_w = shared.write().await;
                        if let Ok(Some(mut meta)) = db_w.storage().get_node_meta(&id) {
                            meta.energy = (meta.energy + 0.01).min(1.0);
                            let _ = db_w.storage().put_node_meta(&meta);
                        }
                    }
                    db = shared.read().await;
                }
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
                // NQL CREATE: create a new node from properties
                QueryResult::CreateNodeRequest { node_type, properties } => {
                    drop(db);
                    let mut db_w = shared.write().await;
                    let nt_str = node_type.as_deref().unwrap_or("Semantic");
                    let nt = parse_node_type(nt_str);
                    let embedding = PoincareVector::new(vec![0.0_f32; 3072]);

                    // Extract TTL from properties → set expires_at
                    let mut props_clone = properties.clone();
                    let ttl_secs = props_clone.as_object_mut()
                        .and_then(|m| m.remove("ttl"))
                        .and_then(|v| v.as_f64());

                    let mut node = Node::new(Uuid::new_v4(), embedding, props_clone);
                    node.node_type = nt;
                    if let Some(ttl) = ttl_secs {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        node.expires_at = Some((now + ttl as u64) as i64);
                    }
                    db_w.insert_node(node.clone()).map_err(graph_err)?;
                    nodes.push(node_to_proto(node));
                    drop(db_w);
                    db = shared.read().await;
                }
                // NQL MATCH…SET: update matched nodes
                QueryResult::SetRequest { updates } => {
                    drop(db);
                    let mut db_w = shared.write().await;
                    for (nid, assignments) in &updates {
                        db_w.update_node_content(*nid, assignments)
                            .map_err(graph_err)?;
                    }
                    path_ids.push(format!("set:updated:{}", updates.len()));
                    drop(db_w);
                    db = shared.read().await;
                }
                // NQL MATCH…DELETE: delete matched nodes (+ edges if detach)
                QueryResult::DeleteRequest { matched_ids, detach: _ } => {
                    drop(db);
                    let mut db_w = shared.write().await;
                    for nid in &matched_ids {
                        // delete_node already removes incident edges
                        db_w.delete_node(*nid).map_err(graph_err)?;
                    }
                    path_ids.push(format!("delete:removed:{}", matched_ids.len()));
                    drop(db_w);
                    db = shared.read().await;
                }
                // ── DAEMON Agents (Wiederkehr) ────────────────────────
                QueryResult::CreateDaemonRequest {
                    name, on_pattern, when_cond, then_action, every, energy,
                } => {
                    let interval_secs = match &every {
                        nietzsche_query::Expr::MathFunc {
                            func: nietzsche_query::MathFunc::Interval,
                            args,
                        } => {
                            if let Some(nietzsche_query::MathFuncArg::Str(s)) = args.first() {
                                nietzsche_wiederkehr::parse_interval_str(s)
                            } else { 3600.0 }
                        }
                        nietzsche_query::Expr::Float(f) => *f,
                        nietzsche_query::Expr::Int(i)   => *i as f64,
                        _ => 3600.0,
                    };
                    let def = nietzsche_wiederkehr::DaemonDef {
                        name:          name.clone(),
                        on_pattern,
                        when_cond,
                        then_action,
                        every,
                        energy:        energy.unwrap_or(1.0),
                        last_run:      0.0,
                        interval_secs,
                    };
                    nietzsche_wiederkehr::put_daemon(db.storage(), &def)
                        .map_err(|e| Status::internal(format!("daemon store error: {e}")))?;
                    path_ids.push(format!("daemon:created:{}", name));
                }
                QueryResult::DropDaemonRequest { name } => {
                    nietzsche_wiederkehr::delete_daemon(db.storage(), &name)
                        .map_err(|e| Status::internal(format!("daemon store error: {e}")))?;
                    path_ids.push(format!("daemon:dropped:{}", name));
                }
                QueryResult::ShowDaemonsRequest => {
                    let daemons = nietzsche_wiederkehr::list_daemons(db.storage())
                        .map_err(|e| Status::internal(format!("daemon store error: {e}")))?;
                    for d in daemons {
                        let row = vec![
                            ("name".into(),          ScalarValue::Str(d.name)),
                            ("energy".into(),        ScalarValue::Float(d.energy)),
                            ("interval_secs".into(), ScalarValue::Float(d.interval_secs)),
                            ("last_run".into(),      ScalarValue::Float(d.last_run)),
                        ];
                        scalar_rows.push(nietzsche::ScalarRow {
                            entries: row.into_iter().map(|(col_name, val)| {
                                let mut entry = nietzsche::ScalarEntry {
                                    column: col_name,
                                    is_null: false,
                                    ..Default::default()
                                };
                                match val {
                                    ScalarValue::Float(f) => entry.value = Some(
                                        nietzsche::scalar_entry::Value::FloatVal(f)
                                    ),
                                    ScalarValue::Str(s) => entry.value = Some(
                                        nietzsche::scalar_entry::Value::StringVal(s)
                                    ),
                                    _ => entry.is_null = true,
                                }
                                entry
                            }).collect(),
                        });
                    }
                }
                // ── Dream Queries (Phase 15.2) ────────────────────────
                QueryResult::DreamFromRequest { seed_param, seed_alias: _, depth, noise } => {
                    let seed_id = seed_param.as_ref()
                        .and_then(|p| params.get(p))
                        .and_then(|v| if let ParamValue::Uuid(u) = v { Some(*u) } else { None })
                        .ok_or_else(|| Status::invalid_argument("DREAM FROM requires a $param UUID"))?;
                    
                    let session = self.dream_engine.dream_from(db.storage(), seed_id, depth, noise)
                        .map_err(|e| Status::internal(format!("dream error: {e}")))?;
                    
                    dream_result = Some(dream_session_to_proto(session.clone()));
                    
                    scalar_rows.push(nietzsche::ScalarRow {
                        entries: vec![
                            nietzsche::ScalarEntry { column: "dream_id".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(session.id)) },
                            nietzsche::ScalarEntry { column: "status".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(format!("{:?}", session.status))) },
                            nietzsche::ScalarEntry { column: "events".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(session.events.len() as i64)) },
                            nietzsche::ScalarEntry { column: "deltas".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(session.dream_nodes.len() as i64)) },
                        ],
                    });
                }
                QueryResult::ApplyDreamRequest { dream_id } => {
                    drop(db);
                    let mut db_w = shared.write().await;
                    let session = self.dream_engine.apply_dream(db_w.storage(), &dream_id)
                        .map_err(|e| Status::internal(format!("dream apply error: {e}")))?;
                    
                    dream_result = Some(dream_session_to_proto(session));
                    path_ids.push(format!("dream:applied:{}", dream_id));
                    drop(db_w);
                    db = shared.read().await;
                }
                QueryResult::RejectDreamRequest { dream_id } => {
                    let session = self.dream_engine.reject_dream(db.storage(), &dream_id)
                        .map_err(|e| Status::internal(format!("dream reject error: {e}")))?;
                    
                    dream_result = Some(dream_session_to_proto(session));
                    path_ids.push(format!("dream:rejected:{}", dream_id));
                }
                QueryResult::ShowDreamsRequest => {
                    let dreams = nietzsche_dream::list_dreams(db.storage())
                        .map_err(|e| Status::internal(format!("dream store error: {e}")))?;
                    for d in dreams {
                        scalar_rows.push(nietzsche::ScalarRow {
                            entries: vec![
                                nietzsche::ScalarEntry { column: "dream_id".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(d.id)) },
                                nietzsche::ScalarEntry { column: "status".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(format!("{:?}", d.status))) },
                                nietzsche::ScalarEntry { column: "seed_node".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(d.seed_node.to_string())) },
                                nietzsche::ScalarEntry { column: "events".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(d.events.len() as i64)) },
                            ],
                        });
                    }
                }
                // ── Synesthesia (Phase 15.3) ────────────────────────
                QueryResult::TranslateRequest { node_id, node_alias: _, from_modality, to_modality, quality: _ } => {
                    let nid = node_id.ok_or_else(|| Status::invalid_argument("TRANSLATE requires a node UUID"))?;
                    let graph_storage = db.storage();
                    let sensory = SensoryStorage::new(graph_storage.db_handle());
                    let sm = sensory.get(&nid)
                        .map_err(|e| Status::internal(format!("sensory get: {e}")))?
                        .ok_or_else(|| Status::not_found(format!("no sensory data for {nid}")))?;
                    let target_mod = parse_modality(&to_modality, &serde_json::json!({}))?;
                    let result = nietzsche_sensory::translate_modality(&sm, target_mod);
                    sensory.put(&nid, &result.translated)
                        .map_err(|e| Status::internal(format!("sensory put: {e}")))?;
                    path_ids.push(format!(
                        "translate:{}:{}->{}:loss={:.4}",
                        nid, from_modality, to_modality, result.quality_loss
                    ));
                }
                // ── Counterfactual (Phase 15.4) ────────────────────────
                QueryResult::CounterfactualRequest { overlays, inner_results } => {
                    path_ids.push(format!("counterfactual:overlays={}", overlays));
                    for inner in inner_results {
                        match inner {
                            QueryResult::Node(n) => nodes.push(node_to_proto(n)),
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
                            _ => {}
                        }
                    }
                }
                // ── Collective Unconscious (Phase 15.6) ────────────────
                QueryResult::ShowArchetypesRequest => {
                    let archetypes = self.archetype_registry.list();
                    for a in archetypes {
                        scalar_rows.push(nietzsche::ScalarRow {
                            entries: vec![
                                nietzsche::ScalarEntry { column: "node_id".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(a.node_id.to_string())) },
                                nietzsche::ScalarEntry { column: "source_collection".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(a.source_collection.clone())) },
                                nietzsche::ScalarEntry { column: "target_collection".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(a.target_collection.clone())) },
                                nietzsche::ScalarEntry { column: "energy".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::FloatVal(a.energy)) },
                            ],
                        });
                    }
                }
                QueryResult::ShareArchetypeRequest { node_id: nid, target_collection } => {
                    let meta = db.storage().get_node_meta(&nid)
                        .map_err(graph_err)?
                        .ok_or_else(|| Status::not_found(format!("node {} not found", nid)))?;
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64;
                    let archetype = nietzsche_cluster::Archetype {
                        node_id:           nid,
                        source_collection: col(&inner.collection).to_string(),
                        target_collection: target_collection.clone(),
                        energy:            meta.energy as f64,
                        depth:             meta.depth as f64,
                        content:           meta.content.clone(),
                        shared_at:         now,
                    };
                    self.archetype_registry.share(archetype);
                    path_ids.push(format!("archetype:shared:{}:{}", nid, target_collection));
                }
                // ── Narrative Engine (Phase 15.7) ────────────────────────
                QueryResult::NarrateRequest { collection, window_hours, format: fmt } => {
                    let col_name_str = collection.as_deref().unwrap_or("default");
                    let engine = nietzsche_narrative::NarrativeEngine::new(
                        nietzsche_narrative::NarrativeConfig::default(),
                    );
                    let report = engine.narrate(db.storage(), col_name_str, window_hours)
                        .map_err(|e| Status::internal(format!("narrative error: {e}")))?;
                    let fmt_str = fmt.as_deref().unwrap_or("text");
                    if fmt_str == "json" {
                        let json = serde_json::to_string(&report).unwrap_or_default();
                        scalar_rows.push(nietzsche::ScalarRow {
                            entries: vec![
                                nietzsche::ScalarEntry { column: "narrative".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(json)) },
                            ],
                        });
                    } else {
                        scalar_rows.push(nietzsche::ScalarRow {
                            entries: vec![
                                nietzsche::ScalarEntry { column: "summary".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::StringVal(report.summary)) },
                                nietzsche::ScalarEntry { column: "total_nodes".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(report.total_nodes as i64)) },
                                nietzsche::ScalarEntry { column: "total_edges".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(report.total_edges as i64)) },
                                nietzsche::ScalarEntry { column: "events".into(), is_null: false, value: Some(nietzsche::scalar_entry::Value::IntVal(report.events.len() as i64)) },
                            ],
                        });
                    }
                }
                // Phase F: transaction control — acknowledged; actual Tx state
                // is managed by the connection session layer (future phases).
                QueryResult::TxBegin    => { path_ids.push("tx:begin".into()); }
                QueryResult::TxCommit   => { path_ids.push("tx:commit".into()); }
                QueryResult::TxRollback => { path_ids.push("tx:rollback".into()); }
                QueryResult::PsychoanalyzeResult { node_id, lineage } => {
                    path_ids.push(format!("psychoanalyze:{}:{}", node_id, lineage));
                }
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
            dream_result,
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

        // Convert proto filters → MetadataFilter
        let filter = proto_filters_to_metadata_filter(&r.filters);
        let results = db.knn_filtered(&query, k, &filter).map_err(graph_err)?;

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
        require_admin(&req)?;
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
            ..Default::default()
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
            hausdorff_before:   report.hausdorff_before,
            hausdorff_after:    report.hausdorff_after,
            hausdorff_delta:    report.hausdorff_delta,
            committed:          report.committed,
            nodes_perturbed:    report.nodes_perturbed as u32,
            snapshot_nodes:     report.snapshot_nodes  as u32,
            semantic_drift_avg: report.semantic_drift_avg,
            semantic_drift_max: report.semantic_drift_max,
        }))
    }

    // ── Zaratustra (Phase Z) ──────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn invoke_zaratustra(
        &self,
        req: Request<nietzsche::ZaratustraRequest>,
    ) -> Result<Response<nietzsche::ZaratustraResponse>, Status> {
        require_admin(&req)?;
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
        require_writer(&req)?;
        let r = req.into_inner();
        const MAX_BATCH_SIZE: usize = 10_000;
        if r.nodes.len() > MAX_BATCH_SIZE {
            return Err(Status::invalid_argument(format!(
                "batch size {} exceeds maximum {MAX_BATCH_SIZE}", r.nodes.len()
            )));
        }
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
        let bp = db.check_backpressure();
        self.cdc.publish(
            CdcEventType::BatchInsertNodes { count },
            Uuid::nil(),
            col_name,
        );

        Ok(Response::new(nietzsche::BatchInsertNodesResponse {
            inserted: count,
            node_ids: ids,
            backpressure: Some(bp_to_proto(&bp)),
        }))
    }

    async fn batch_insert_edges(
        &self,
        req: Request<nietzsche::BatchInsertEdgesRequest>,
    ) -> Result<Response<nietzsche::BatchInsertEdgesResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        const MAX_BATCH_SIZE: usize = 10_000;
        if r.edges.len() > MAX_BATCH_SIZE {
            return Err(Status::invalid_argument(format!(
                "batch size {} exceeds maximum {MAX_BATCH_SIZE}", r.edges.len()
            )));
        }
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
        require_admin(&req)?;
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
        require_admin(&req)?;
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

    // ── Hybrid BM25 + ANN Search ────────────────────────────────────────

    async fn hybrid_search(
        &self,
        req: Request<nietzsche::HybridSearchRequest>,
    ) -> Result<Response<nietzsche::KnnResponse>, Status> {
        let r = req.into_inner();
        if r.text_query.is_empty() && r.query_coords.is_empty() {
            return Err(Status::invalid_argument("text_query or query_coords must be provided"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let k = if r.k > 0 { r.k as usize } else { 10 };
        let text_weight = if r.text_weight > 0.0 { r.text_weight } else { 1.0 };
        let vector_weight = if r.vector_weight > 0.0 { r.vector_weight } else { 1.0 };

        let coords: Vec<f32> = r.query_coords.iter().map(|&v| v as f32).collect();
        let dim = coords.len();
        let query_vec = nietzsche_graph::PoincareVector { coords, dim };
        let results = db.hybrid_search(
            &r.text_query,
            &query_vec,
            k,
            text_weight,
            vector_weight,
        ).map_err(graph_err)?;

        Ok(Response::new(nietzsche::KnnResponse {
            results: results.into_iter().map(|(id, score)| nietzsche::KnnResult {
                id:       id.to_string(),
                distance: score,
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
        require_writer(&req)?;
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
        require_writer(&req)?;
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

    // ── ListStore (RPUSH/LRANGE) ────────────────────────────────────────

    async fn list_r_push(
        &self,
        req: Request<nietzsche::ListPushRequest>,
    ) -> Result<Response<nietzsche::ListPushResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let new_seq = db.storage().list_rpush(&node_id, &r.list_name, &r.value)
            .map_err(graph_err)?;
        Ok(Response::new(nietzsche::ListPushResponse { new_length: new_seq }))
    }

    async fn list_l_range(
        &self,
        req: Request<nietzsche::ListRangeRequest>,
    ) -> Result<Response<nietzsche::ListRangeResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let values = db.storage().list_lrange(&node_id, &r.list_name, r.start, r.stop)
            .map_err(graph_err)?;
        Ok(Response::new(nietzsche::ListRangeResponse { values }))
    }

    async fn list_len(
        &self,
        req: Request<nietzsche::ListLenRequest>,
    ) -> Result<Response<nietzsche::ListLenResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let length = db.storage().list_len(&node_id, &r.list_name)
            .map_err(graph_err)?;
        Ok(Response::new(nietzsche::ListLenResponse { length }))
    }

    // ── Cluster gossip (Phase G) ──────────────────────────────────────────

    async fn exchange_gossip(
        &self,
        req: Request<nietzsche::GossipRequest>,
    ) -> Result<Response<nietzsche::GossipResponse>, Status> {
        require_writer(&req)?;
        let registry = self.cluster_registry.as_ref()
            .ok_or_else(|| Status::failed_precondition("cluster mode not enabled"))?;

        let r = req.into_inner();

        // Parse incoming peers and merge them
        let incoming: Vec<ClusterNode> = r.nodes.iter().filter_map(|p| {
            let id = Uuid::parse_str(&p.id).ok()?;
            let role = match p.role.as_str() {
                "replica"     => NodeRole::Replica,
                "coordinator" => NodeRole::Coordinator,
                _             => NodeRole::Primary,
            };
            let health = match p.health.as_str() {
                "degraded"    => NodeHealth::Degraded,
                "unreachable" => NodeHealth::Unreachable,
                _             => NodeHealth::Healthy,
            };
            let mut node = ClusterNode::new(id, &p.name, &p.addr, role, p.token);
            node.health = health;
            node.last_seen_ms = p.last_seen_ms;
            Some(node)
        }).collect();

        let merged = incoming.len();
        registry.merge_snapshot(incoming);

        // Export our current snapshot as the response
        let snapshot = registry.export_snapshot();
        let response_nodes: Vec<nietzsche::ClusterNodeProto> = snapshot.into_iter().map(|n| {
            nietzsche::ClusterNodeProto {
                id:           n.id.to_string(),
                addr:         n.addr,
                name:         n.name,
                role:         format!("{}", n.role),
                health:       format!("{}", n.health),
                last_seen_ms: n.last_seen_ms,
                token:        n.token,
            }
        }).collect();

        debug!(merged_peers = merged, returned_peers = response_nodes.len(), "ExchangeGossip ok");
        Ok(Response::new(nietzsche::GossipResponse { nodes: response_nodes }))
    }

    // ── Schema Validation ─────────────────────────────────────────────────

    async fn set_schema(
        &self,
        req: Request<nietzsche::SetSchemaRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_admin(&req)?;
        let r = req.into_inner();
        if r.node_type.is_empty() {
            return Err(Status::invalid_argument("node_type must not be empty"));
        }

        let mut field_types = std::collections::HashMap::new();
        for ft in &r.field_types {
            let ftype = match ft.field_type.as_str() {
                "string" => nietzsche_graph::FieldType::String,
                "number" => nietzsche_graph::FieldType::Number,
                "bool"   => nietzsche_graph::FieldType::Bool,
                "array"  => nietzsche_graph::FieldType::Array,
                "object" => nietzsche_graph::FieldType::Object,
                other    => return Err(Status::invalid_argument(
                    format!("unknown field type '{}'; expected string|number|bool|array|object", other)
                )),
            };
            field_types.insert(ft.field_name.clone(), ftype);
        }

        let constraint = nietzsche_graph::SchemaConstraint {
            node_type: r.node_type.clone(),
            required_fields: r.required_fields,
            field_types,
        };

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;

        // Persist to CF_META and register in memory
        db.set_schema_constraint(constraint).map_err(graph_err)?;

        Ok(Response::new(nietzsche::StatusResponse {
            status: "ok".into(),
            error: String::new(),
        }))
    }

    async fn get_schema(
        &self,
        req: Request<nietzsche::GetSchemaRequest>,
    ) -> Result<Response<nietzsche::GetSchemaResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let response = match db.schema_validator().and_then(|v| v.get_constraint(&r.node_type)) {
            Some(c) => nietzsche::GetSchemaResponse {
                node_type: c.node_type.clone(),
                required_fields: c.required_fields.clone(),
                field_types: c.field_types.iter().map(|(k, v)| nietzsche::SchemaFieldType {
                    field_name: k.clone(),
                    field_type: v.to_string(),
                }).collect(),
                found: true,
            },
            None => nietzsche::GetSchemaResponse {
                node_type: r.node_type,
                required_fields: vec![],
                field_types: vec![],
                found: false,
            },
        };

        Ok(Response::new(response))
    }

    async fn list_schemas(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::ListSchemasResponse>, Status> {
        // List schemas from all collections would be complex;
        // for now, list from the default collection.
        let shared = get_col!(self.cm, &"");
        let db = shared.read().await;

        let schemas = match db.schema_validator() {
            Some(v) => v.list_constraints().into_iter().map(|c| {
                nietzsche::GetSchemaResponse {
                    node_type: c.node_type.clone(),
                    required_fields: c.required_fields.clone(),
                    field_types: c.field_types.iter().map(|(k, v)| nietzsche::SchemaFieldType {
                        field_name: k.clone(),
                        field_type: v.to_string(),
                    }).collect(),
                    found: true,
                }
            }).collect(),
            None => vec![],
        };

        Ok(Response::new(nietzsche::ListSchemasResponse { schemas }))
    }

    // ── Secondary Indexes (Phase E) ──────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn create_index(
        &self,
        req: Request<nietzsche::CreateIndexRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();

        if r.field.is_empty() {
            return Err(Status::invalid_argument("field must not be empty"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.create_index(&r.field).map_err(graph_err)?;

        info!(field = %r.field, collection = %col(&r.collection), "CreateIndex");
        Ok(Response::new(ok_status()))
    }

    #[instrument(skip(self, req))]
    async fn drop_index(
        &self,
        req: Request<nietzsche::DropIndexRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();

        if r.field.is_empty() {
            return Err(Status::invalid_argument("field must not be empty"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.write().await;
        db.drop_index(&r.field).map_err(graph_err)?;

        info!(field = %r.field, collection = %col(&r.collection), "DropIndex");
        Ok(Response::new(ok_status()))
    }

    #[instrument(skip(self, req))]
    async fn list_indexes(
        &self,
        req: Request<nietzsche::ListIndexesRequest>,
    ) -> Result<Response<nietzsche::ListIndexesResponse>, Status> {
        let r = req.into_inner();
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;
        let fields = db.list_indexes();

        Ok(Response::new(nietzsche::ListIndexesResponse { fields }))
    }

    // ── Cache (Phase C — Redis replacement) ──────────────────────────────

    #[instrument(skip(self, req))]
    async fn cache_set(
        &self,
        req: Request<nietzsche::CacheSetRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        // Store value in CF_META with "cache:" prefix
        let cache_key = format!("cache:{}", r.key);
        // If TTL > 0, prepend expiry timestamp to value
        let stored_value = if r.ttl_secs > 0 {
            let expires_at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() + r.ttl_secs;
            let mut buf = expires_at.to_be_bytes().to_vec();
            buf.extend_from_slice(&r.value);
            buf
        } else {
            let mut buf = 0u64.to_be_bytes().to_vec(); // 0 = no expiry
            buf.extend_from_slice(&r.value);
            buf
        };
        db.storage().put_meta(&cache_key, &stored_value).map_err(graph_err)?;
        info!(key = %r.key, collection = %col_name, "CacheSet");
        Ok(Response::new(ok_status()))
    }

    #[instrument(skip(self, req))]
    async fn cache_get(
        &self,
        req: Request<nietzsche::CacheGetRequest>,
    ) -> Result<Response<nietzsche::CacheGetResponse>, Status> {
        let r = req.into_inner();
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let cache_key = format!("cache:{}", r.key);
        match db.storage().get_meta(&cache_key).map_err(graph_err)? {
            Some(stored) if stored.len() >= 8 => {
                let expires_at = u64::from_be_bytes(stored[..8].try_into().unwrap());
                if expires_at > 0 {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    if now > expires_at {
                        // Expired — lazy delete
                        let _ = db.storage().delete_meta(&cache_key);
                        return Ok(Response::new(nietzsche::CacheGetResponse {
                            found: false,
                            value: vec![],
                        }));
                    }
                }
                Ok(Response::new(nietzsche::CacheGetResponse {
                    found: true,
                    value: stored[8..].to_vec(),
                }))
            }
            _ => Ok(Response::new(nietzsche::CacheGetResponse {
                found: false,
                value: vec![],
            })),
        }
    }

    #[instrument(skip(self, req))]
    async fn cache_del(
        &self,
        req: Request<nietzsche::CacheDelRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;
        let cache_key = format!("cache:{}", r.key);
        db.storage().delete_meta(&cache_key).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
    }

    #[instrument(skip(self, req))]
    async fn reap_expired(
        &self,
        req: Request<nietzsche::ReapExpiredRequest>,
    ) -> Result<Response<nietzsche::ReapExpiredResponse>, Status> {
        require_admin(&req)?;
        let r = req.into_inner();
        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let mut db = shared.write().await;
        let reaped = db.reap_expired().map_err(graph_err)?;
        info!(reaped = reaped, collection = %col_name, "ReapExpired");
        Ok(Response::new(nietzsche::ReapExpiredResponse {
            reaped_count: reaped as u64,
        }))
    }

    // ── Multi-Manifold Operations (Klein · Riemann · Minkowski) ──────────

    /// Dialectical synthesis of two concepts via the Riemann sphere.
    ///
    /// Projects both node embeddings to the sphere, computes the midpoint,
    /// and returns the result as a new Poincaré ball point that is more
    /// abstract (closer to the origin) than either input.
    #[instrument(skip(self, req))]
    async fn synthesis(
        &self,
        req: Request<nietzsche::SynthesisRequest>,
    ) -> Result<Response<nietzsche::SynthesisResponse>, Status> {
        let r = req.into_inner();
        let id_a = parse_uuid(&r.node_id_a, "node_id_a")?;
        let id_b = parse_uuid(&r.node_id_b, "node_id_b")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        // Fetch embeddings
        let emb_a = db.storage().get_embedding(&id_a).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {id_a} not found")))?;
        let emb_b = db.storage().get_embedding(&id_b).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {id_b} not found")))?;

        // Promote f32 → f64 for manifold math
        let coords_a = emb_a.coords_f64();
        let coords_b = emb_b.coords_f64();

        // Riemann synthesis: Poincaré → Sphere → midpoint → Poincaré (shallower)
        let synthesis_raw = riemann::synthesis(&coords_a, &coords_b)
            .map_err(|e| Status::internal(format!("synthesis failed: {e}")))?;
        // Post-query sanitize: ensure ‖x‖ < 1.0 after inter-manifold projection
        let synthesis_coords = manifold::sanitize_poincare_f64(&synthesis_raw);

        // Find nearest existing node to the synthesis point
        let synthesis_pv = PoincareVector::from_f64(synthesis_coords.clone());
        let nearest = db.knn(&synthesis_pv, 1).map_err(graph_err)?;
        let (nearest_id, nearest_dist) = nearest.first()
            .map(|(id, d)| (id.to_string(), *d))
            .unwrap_or_default();

        info!(
            node_a = %id_a, node_b = %id_b,
            nearest = %nearest_id, distance = nearest_dist,
            "Synthesis complete"
        );

        Ok(Response::new(nietzsche::SynthesisResponse {
            synthesis_coords,
            nearest_node_id: nearest_id,
            nearest_distance: nearest_dist,
        }))
    }

    /// Multi-point synthesis: find the unifying concept of N nodes.
    #[instrument(skip(self, req))]
    async fn synthesis_multi(
        &self,
        req: Request<nietzsche::SynthesisMultiRequest>,
    ) -> Result<Response<nietzsche::SynthesisResponse>, Status> {
        let r = req.into_inner();
        if r.node_ids.len() < 2 {
            return Err(Status::invalid_argument("synthesis_multi requires at least 2 node IDs"));
        }

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        // Fetch all embeddings (f64)
        let mut coords_list: Vec<Vec<f64>> = Vec::with_capacity(r.node_ids.len());
        for id_str in &r.node_ids {
            let id = parse_uuid(id_str, "node_ids")?;
            let emb = db.storage().get_embedding(&id).map_err(graph_err)?
                .ok_or_else(|| Status::not_found(format!("node {id} not found")))?;
            coords_list.push(emb.coords_f64());
        }

        let refs: Vec<&[f64]> = coords_list.iter().map(|v| v.as_slice()).collect();
        let synthesis_raw = riemann::synthesis_multi(&refs)
            .map_err(|e| Status::internal(format!("synthesis_multi failed: {e}")))?;
        // Post-query sanitize: ensure ‖x‖ < 1.0 after inter-manifold projection
        let synthesis_coords = manifold::sanitize_poincare_f64(&synthesis_raw);

        // Find nearest existing node
        let synthesis_pv = PoincareVector::from_f64(synthesis_coords.clone());
        let nearest = db.knn(&synthesis_pv, 1).map_err(graph_err)?;
        let (nearest_id, nearest_dist) = nearest.first()
            .map(|(id, d)| (id.to_string(), *d))
            .unwrap_or_default();

        info!(
            count = r.node_ids.len(),
            nearest = %nearest_id, distance = nearest_dist,
            "SynthesisMulti complete"
        );

        Ok(Response::new(nietzsche::SynthesisResponse {
            synthesis_coords,
            nearest_node_id: nearest_id,
            nearest_distance: nearest_dist,
        }))
    }

    /// Return only causally-connected (timelike) neighbors of a node.
    ///
    /// Uses the Minkowski spacetime interval stored on each edge to filter
    /// by ds² < 0 (timelike) and direction (future/past cone).
    #[instrument(skip(self, req))]
    async fn causal_neighbors(
        &self,
        req: Request<nietzsche::CausalNeighborsRequest>,
    ) -> Result<Response<nietzsche::CausalNeighborsResponse>, Status> {
        let r = req.into_inner();
        let node_id = parse_uuid(&r.node_id, "node_id")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        // Get origin node meta for timestamp
        let origin_meta = db.get_node_meta(node_id).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {node_id} not found")))?;
        let origin_ts = origin_meta.created_at;

        // Determine direction filter
        let direction = match r.direction.to_lowercase().as_str() {
            "future" => "future",
            "past" => "past",
            _ => "both",
        };

        // Collect edges from adjacency (both directions)
        let entries_out = db.adjacency().entries_out(&node_id);
        let entries_in = db.adjacency().entries_in(&node_id);

        let mut causal_edges = Vec::new();

        // Helper: check an adjacency entry, load the edge, filter by causality
        let process_entry = |entry: &AdjEntry, is_outgoing: bool| -> Option<nietzsche::CausalEdge> {
            let edge = db.storage().get_edge(&entry.edge_id).ok()??;

            // Check causal type
            if !matches!(edge.causal_type, CausalType::Timelike) {
                // If edge has pre-computed causality and it's not timelike, skip.
                // For legacy edges (Unknown), compute on-the-fly.
                if !matches!(edge.causal_type, CausalType::Unknown) {
                    return None;
                }
                // Legacy edge: compute Minkowski interval on-the-fly
                let target_id = if is_outgoing { edge.to } else { edge.from };
                let target_meta = db.get_node_meta(target_id).ok()??;
                let emb_origin = db.storage().get_embedding(&node_id).ok()??;
                let emb_target = db.storage().get_embedding(&target_id).ok()??;
                let interval = minkowski::minkowski_interval(
                    &emb_origin.coords, &emb_target.coords,
                    origin_ts, target_meta.created_at,
                    minkowski::DEFAULT_CAUSAL_SPEED,
                );
                if !minkowski::is_timelike(interval) {
                    return None;
                }
            }

            // Direction filter using timestamps
            let target_id = if is_outgoing { edge.to } else { edge.from };
            let target_meta = db.get_node_meta(target_id).ok()??;
            match direction {
                "future" if target_meta.created_at <= origin_ts => return None,
                "past" if target_meta.created_at >= origin_ts => return None,
                _ => {}
            }

            Some(nietzsche::CausalEdge {
                edge_id: edge.id.to_string(),
                from_node_id: edge.from.to_string(),
                to_node_id: edge.to.to_string(),
                minkowski_interval: edge.minkowski_interval as f64,
                causal_type: format!("{}", edge.causal_type),
                edge_type: format!("{:?}", edge.edge_type),
            })
        };

        for entry in &entries_out {
            if let Some(ce) = process_entry(entry, true) {
                causal_edges.push(ce);
            }
        }
        for entry in &entries_in {
            if let Some(ce) = process_entry(entry, false) {
                causal_edges.push(ce);
            }
        }

        info!(
            node = %node_id, direction = %direction,
            causal_count = causal_edges.len(),
            "CausalNeighbors"
        );

        Ok(Response::new(nietzsche::CausalNeighborsResponse {
            edges: causal_edges,
        }))
    }

    /// Recursive causal chain traversal following only timelike edges.
    ///
    /// Starting from a node, follows the causal chain (ds² < 0) in the
    /// requested direction up to max_depth. Used for "WHY did X happen?"
    /// queries — returns the chain of events that provably led to X.
    #[instrument(skip(self, req))]
    async fn causal_chain(
        &self,
        req: Request<nietzsche::CausalChainRequest>,
    ) -> Result<Response<nietzsche::CausalChainResponse>, Status> {
        let r = req.into_inner();
        let start_id = parse_uuid(&r.node_id, "node_id")?;
        let max_depth = if r.max_depth > 0 { r.max_depth as usize } else { 10 };
        let is_past = r.direction.to_lowercase() != "future";

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let mut chain_ids: Vec<String> = vec![start_id.to_string()];
        let mut chain_edges: Vec<nietzsche::CausalEdge> = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_id);

        let mut queue: VecDeque<(Uuid, usize)> = VecDeque::new();
        queue.push_back((start_id, 0));

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let current_meta = match db.get_node_meta(current_id).map_err(graph_err)? {
                Some(m) => m,
                None => continue,
            };

            // Get adjacency entries in the desired direction
            let entries = if is_past {
                db.adjacency().entries_in(&current_id)
            } else {
                db.adjacency().entries_out(&current_id)
            };

            for entry in &entries {
                if visited.contains(&entry.neighbor_id) {
                    continue;
                }

                // Load the full edge to check causality
                let edge = match db.storage().get_edge(&entry.edge_id).ok().flatten() {
                    Some(e) => e,
                    None => continue,
                };

                // Check timelike: either pre-computed or compute on-the-fly
                let is_causal = match edge.causal_type {
                    CausalType::Timelike => true,
                    CausalType::Spacelike | CausalType::Lightlike => false,
                    CausalType::Unknown => {
                        // Legacy edge: compute Minkowski interval
                        let neighbor_meta = match db.get_node_meta(entry.neighbor_id).ok().flatten() {
                            Some(m) => m,
                            None => continue,
                        };
                        let emb_cur = match db.storage().get_embedding(&current_id).ok().flatten() {
                            Some(e) => e,
                            None => continue,
                        };
                        let emb_nbr = match db.storage().get_embedding(&entry.neighbor_id).ok().flatten() {
                            Some(e) => e,
                            None => continue,
                        };
                        let interval = minkowski::minkowski_interval(
                            &emb_cur.coords, &emb_nbr.coords,
                            current_meta.created_at, neighbor_meta.created_at,
                            minkowski::DEFAULT_CAUSAL_SPEED,
                        );
                        minkowski::is_timelike(interval)
                    }
                };

                if !is_causal {
                    continue;
                }

                // Direction check on timestamps
                let neighbor_meta = match db.get_node_meta(entry.neighbor_id).ok().flatten() {
                    Some(m) => m,
                    None => continue,
                };
                if is_past && neighbor_meta.created_at > current_meta.created_at {
                    continue;
                }
                if !is_past && neighbor_meta.created_at < current_meta.created_at {
                    continue;
                }

                visited.insert(entry.neighbor_id);
                chain_ids.push(entry.neighbor_id.to_string());
                chain_edges.push(nietzsche::CausalEdge {
                    edge_id: edge.id.to_string(),
                    from_node_id: edge.from.to_string(),
                    to_node_id: edge.to.to_string(),
                    minkowski_interval: edge.minkowski_interval as f64,
                    causal_type: format!("{}", edge.causal_type),
                    edge_type: format!("{:?}", edge.edge_type),
                });
                queue.push_back((entry.neighbor_id, depth + 1));
            }
        }

        info!(
            start = %start_id,
            direction = if is_past { "past" } else { "future" },
            chain_len = chain_ids.len(),
            "CausalChain"
        );

        Ok(Response::new(nietzsche::CausalChainResponse {
            chain_ids,
            edges: chain_edges,
        }))
    }

    /// Find the shortest path between two nodes using Klein model geodesics.
    ///
    /// Projects all embeddings to the Klein model (where geodesics are
    /// straight lines), runs Dijkstra with Klein distances, then returns
    /// the path as Poincaré-space node IDs.
    #[instrument(skip(self, req))]
    async fn klein_path(
        &self,
        req: Request<nietzsche::KleinPathRequest>,
    ) -> Result<Response<nietzsche::KleinPathResponse>, Status> {
        let r = req.into_inner();
        let start_id = parse_uuid(&r.start_node_id, "start_node_id")?;
        let goal_id = parse_uuid(&r.goal_node_id, "goal_node_id")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        // Verify both nodes exist
        db.get_node_meta(start_id).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("start node {start_id} not found")))?;
        db.get_node_meta(goal_id).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("goal node {goal_id} not found")))?;

        // Klein-projected Dijkstra: BFS with priority queue using Klein distance
        let mut dist_map: HashMap<Uuid, f64> = HashMap::new();
        let mut prev_map: HashMap<Uuid, Uuid> = HashMap::new();
        let mut heap = std::collections::BinaryHeap::new();

        dist_map.insert(start_id, 0.0);
        // BinaryHeap is max-heap; use Reverse for min-heap behavior
        heap.push(std::cmp::Reverse((ordered_float::OrderedFloat(0.0_f64), start_id)));

        let mut found = false;

        while let Some(std::cmp::Reverse((ordered_float::OrderedFloat(cost), current))) = heap.pop() {
            if current == goal_id {
                found = true;
                break;
            }

            if cost > *dist_map.get(&current).unwrap_or(&f64::INFINITY) {
                continue; // outdated entry
            }

            // Get current node's embedding → project to Klein
            let emb_cur = match db.storage().get_embedding(&current).ok().flatten() {
                Some(e) => e,
                None => continue,
            };
            let coords_cur_f64 = emb_cur.coords_f64();
            let klein_cur = match klein::to_klein(&coords_cur_f64) {
                Ok(k) => k,
                Err(_) => continue,
            };

            // Explore neighbors
            let neighbors_out = db.adjacency().entries_out(&current);
            for entry in &neighbors_out {
                let emb_nbr = match db.storage().get_embedding(&entry.neighbor_id).ok().flatten() {
                    Some(e) => e,
                    None => continue,
                };
                let coords_nbr_f64 = emb_nbr.coords_f64();
                let klein_nbr = match klein::to_klein(&coords_nbr_f64) {
                    Ok(k) => k,
                    Err(_) => continue,
                };

                // Klein distance (same as hyperbolic distance, straight-line geodesic)
                let edge_cost = klein::klein_distance(&klein_cur, &klein_nbr);
                let new_cost = cost + edge_cost;

                if new_cost < *dist_map.get(&entry.neighbor_id).unwrap_or(&f64::INFINITY) {
                    dist_map.insert(entry.neighbor_id, new_cost);
                    prev_map.insert(entry.neighbor_id, current);
                    heap.push(std::cmp::Reverse((ordered_float::OrderedFloat(new_cost), entry.neighbor_id)));
                }
            }
        }

        if !found {
            return Ok(Response::new(nietzsche::KleinPathResponse {
                found: false,
                path: vec![],
                cost: 0.0,
            }));
        }

        // Reconstruct path
        let mut path = vec![goal_id.to_string()];
        let mut cursor = goal_id;
        while let Some(&prev) = prev_map.get(&cursor) {
            path.push(prev.to_string());
            cursor = prev;
            if cursor == start_id { break; }
        }
        path.reverse();

        let total_cost = dist_map.get(&goal_id).copied().unwrap_or(0.0);

        info!(
            start = %start_id, goal = %goal_id,
            path_len = path.len(), cost = total_cost,
            "KleinPath found"
        );

        Ok(Response::new(nietzsche::KleinPathResponse {
            found: true,
            path,
            cost: total_cost,
        }))
    }

    /// Check if a point C lies on the geodesic between A and B in Klein space.
    ///
    /// Projects all three node embeddings to the Klein model and performs
    /// an O(1) collinearity + betweenness check.
    #[instrument(skip(self, req))]
    async fn is_on_shortest_path(
        &self,
        req: Request<nietzsche::ShortestPathCheckRequest>,
    ) -> Result<Response<nietzsche::ShortestPathCheckResponse>, Status> {
        let r = req.into_inner();
        let id_a = parse_uuid(&r.node_id_a, "node_id_a")?;
        let id_b = parse_uuid(&r.node_id_b, "node_id_b")?;
        let id_c = parse_uuid(&r.node_id_c, "node_id_c")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        // Fetch embeddings
        let emb_a = db.storage().get_embedding(&id_a).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {id_a} not found")))?;
        let emb_b = db.storage().get_embedding(&id_b).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {id_b} not found")))?;
        let emb_c = db.storage().get_embedding(&id_c).map_err(graph_err)?
            .ok_or_else(|| Status::not_found(format!("node {id_c} not found")))?;

        // Project to Klein
        let klein_a = klein::to_klein(&emb_a.coords_f64())
            .map_err(|e| Status::internal(format!("Klein projection failed for A: {e}")))?;
        let klein_b = klein::to_klein(&emb_b.coords_f64())
            .map_err(|e| Status::internal(format!("Klein projection failed for B: {e}")))?;
        let klein_c = klein::to_klein(&emb_c.coords_f64())
            .map_err(|e| Status::internal(format!("Klein projection failed for C: {e}")))?;

        let on_path = klein::is_on_shortest_path(&klein_a, &klein_b, &klein_c, 1e-7);
        let distance = klein::klein_distance(&klein_a, &klein_b);

        info!(
            a = %id_a, b = %id_b, c = %id_c,
            on_path = on_path, distance = distance,
            "IsOnShortestPath"
        );

        Ok(Response::new(nietzsche::ShortestPathCheckResponse {
            on_path,
            distance,
        }))
    }

    // ── Swartz SQL Layer ──────────────────────────────────────────────────

    /// Execute a SQL SELECT query and return rows.
    #[instrument(skip(self, req))]
    async fn sql_query(
        &self,
        req: Request<nietzsche::SqlRequest>,
    ) -> Result<Response<nietzsche::SqlResultSet>, Status> {
        let r = req.into_inner();
        if r.sql.is_empty() {
            return Err(Status::invalid_argument("sql must not be empty"));
        }

        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;

        let mut engine = nietzsche_swartz::SwartzEngine::new(db.storage_arc());

        // query() returns Vec<serde_json::Value> — each is a JSON object with column names as keys
        let json_rows = engine
            .query(&r.sql)
            .await
            .map_err(|e| Status::internal(format!("Swartz SQL error: {e}")))?;

        // Build column definitions from the first row's keys
        let mut columns = Vec::new();
        if let Some(first) = json_rows.first() {
            if let Some(obj) = first.as_object() {
                columns = obj
                    .keys()
                    .map(|k| nietzsche::SqlColumn {
                        name: k.clone(),
                        r#type: "TEXT".to_string(),
                    })
                    .collect();
            }
        }

        // Convert JSON rows to proto SqlRow (each value JSON-encoded as bytes)
        let rows: Vec<nietzsche::SqlRow> = json_rows
            .iter()
            .map(|row| {
                let values = if let Some(obj) = row.as_object() {
                    // Maintain column order
                    columns
                        .iter()
                        .map(|c| {
                            let v = obj.get(&c.name).unwrap_or(&serde_json::Value::Null);
                            serde_json::to_vec(v).unwrap_or_default()
                        })
                        .collect()
                } else {
                    vec![serde_json::to_vec(row).unwrap_or_default()]
                };
                nietzsche::SqlRow { values }
            })
            .collect();

        debug!(
            sql = %r.sql, collection = %col_name,
            result_rows = rows.len(),
            "[Swartz] SqlQuery"
        );

        Ok(Response::new(nietzsche::SqlResultSet {
            columns,
            rows,
            affected_rows: 0,
        }))
    }

    /// Execute a SQL DDL/DML statement (CREATE TABLE, INSERT, UPDATE, DELETE, DROP TABLE).
    #[instrument(skip(self, req))]
    async fn sql_exec(
        &self,
        req: Request<nietzsche::SqlRequest>,
    ) -> Result<Response<nietzsche::SqlExecResult>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        if r.sql.is_empty() {
            return Err(Status::invalid_argument("sql must not be empty"));
        }

        let col_name = col(&r.collection).to_string();
        let shared = get_col!(self.cm, &col_name);
        let db = shared.read().await;

        let mut engine = nietzsche_swartz::SwartzEngine::new(db.storage_arc());

        let affected = engine
            .exec_count(&r.sql)
            .await
            .map_err(|e| Status::internal(format!("Swartz SQL error: {e}")))?;

        // CDC publish for SQL writes (entity_id = nil UUID for SQL ops)
        let sql_upper = r.sql.trim().to_uppercase();
        if sql_upper.starts_with("INSERT")
            || sql_upper.starts_with("UPDATE")
            || sql_upper.starts_with("DELETE")
            || sql_upper.starts_with("CREATE")
            || sql_upper.starts_with("DROP")
            || sql_upper.starts_with("ALTER")
        {
            self.cdc.publish(
                CdcEventType::SqlExec,
                Uuid::nil(),
                &col_name,
            );
        }

        info!(
            sql = %r.sql, collection = %col_name,
            affected_rows = affected,
            "[Swartz] SqlExec"
        );

        Ok(Response::new(nietzsche::SqlExecResult {
            affected_rows: affected,
            success: true,
            message: String::new(),
        }))
    }

    // ── Neural Foundation (Phase 1) ───────────────────────────────────────

    async fn load_model(
        &self,
        req: Request<nietzsche::LoadModelRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_admin(&req)?;
        let r = req.into_inner();
        let meta = ModelMetadata {
            name: r.name,
            path: r.path.into(),
            version: r.version,
            input_shape: vec![], // TODO: extract or provide in request
            output_shape: vec![],
        };

        REGISTRY.load_model(meta)
            .map_err(|e| Status::internal(format!("Failed to load model: {e}")))?;

        Ok(Response::new(ok_status()))
    }

    async fn list_models(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::ListModelsResponse>, Status> {
        let models = REGISTRY.list_models().into_iter().map(|m| {
            nietzsche::ModelMeta {
                name: m.name,
                path: m.path.to_string_lossy().to_string(),
                version: m.version,
            }
        }).collect();

        Ok(Response::new(nietzsche::ListModelsResponse { models }))
    }

    async fn unload_model(
        &self,
        req: Request<nietzsche::ModelNameRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        require_admin(&req)?;
        let r = req.into_inner();
        REGISTRY.unload_model(&r.name);
        Ok(Response::new(ok_status()))
    }

    async fn gnn_infer(
        &self,
        req: Request<nietzsche::GnnInferRequest>,
    ) -> Result<Response<nietzsche::GnnInferResponse>, Status> {
        let r = req.into_inner();
        let engine = GnnEngine::new(&r.model_name);

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let sampler = NeighborSampler::new(db.storage(), db.adjacency());
        
        let mut results = Vec::new();
        for id_str in r.node_ids {
            let id = parse_uuid(&id_str, "node_id")?;
            let subgraph = sampler.sample_k_hop(id, 2)
                .map_err(|e| Status::internal(e.to_string()))?;
            
            let predictions = engine.predict(&subgraph).await
                .map_err(|e| Status::internal(e.to_string()))?;
            
            for p in predictions {
                results.push(nietzsche::PoincareVector {
                    coords: p.embedding_delta.into_iter().map(|x| x as f64).collect(),
                    dim: 0, // indicates this might be a score or delta
                });
            }
        }

        Ok(Response::new(nietzsche::GnnInferResponse { embeddings: results }))
    }

    async fn mcts_search(
        &self,
        req: Request<nietzsche::MctsRequest>,
    ) -> Result<Response<nietzsche::MctsResponse>, Status> {
        let r = req.into_inner();
        let start_node = parse_uuid(&r.start_node_id, "start_node_id")?;
        
        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let config = MctsConfig {
            iterations: r.simulations as usize,
            exploration_constant: 1.41,
        };

        let advisor = MctsAdvisor::new(&*db, &r.model_name, config);
        let intents = advisor.advise(start_node).await
            .map_err(|e| Status::internal(e.to_string()))?;

        if let Some(best) = intents.first() {
            let best_id = best.target_node;
            
            // Record hit for EvolutionDaemon
            drop(db);
            {
                let mut db_w = shared.write().await;
                if let Ok(Some(mut node)) = db_w.storage().get_node(&best_id) {
                    let hits = node.content.get("mcts_hits").and_then(|v| v.as_u64()).unwrap_or(0);
                    let mut content = node.content.clone();
                    if let Some(obj) = content.as_object_mut() {
                        obj.insert("mcts_hits".to_string(), serde_json::Value::from(hits + 1));
                    }
                    node.content = content;
                    let _ = db_w.storage().put_node(&node);
                }
            }

            Ok(Response::new(nietzsche::MctsResponse {
                best_action_id: best_id.to_string(),
                value: best.confidence as f64,
            }))
        } else {
            Ok(Response::new(nietzsche::MctsResponse {
                best_action_id: String::new(),
                value: 0.0,
            }))
        }
    }

    // ── DreamerV3 Speculative Simulation (Phase 4) ───────────────────────

    async fn dream_from(
        &self,
        req: Request<nietzsche::DreamRequest>,
    ) -> Result<Response<nietzsche::DreamSessionProto>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();
        let seed_id = val_uuid(&r.seed_id, "seed_id")?;
        let depth = if r.depth > 0 { Some(r.depth as usize) } else { None };
        let noise = if r.noise > 0.0 { Some(r.noise) } else { None };

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let session = self.dream_engine.dream_from(db.storage(), seed_id, depth, noise)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(dream_session_to_proto(session)))
    }

    async fn apply_dream(
        &self,
        req: Request<nietzsche::DreamIdRequest>,
    ) -> Result<Response<nietzsche::DreamSessionProto>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();

        let shared = get_col!(self.cm, &r.collection);
        // We need write lock to apply changes to the graph
        let mut db = shared.write().await;

        let session = self.dream_engine.apply_dream(db.storage(), &r.dream_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(dream_session_to_proto(session)))
    }

    async fn reject_dream(
        &self,
        req: Request<nietzsche::DreamIdRequest>,
    ) -> Result<Response<nietzsche::DreamSessionProto>, Status> {
        require_writer(&req)?;
        let r = req.into_inner();

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.read().await;

        let session = self.dream_engine.reject_dream(db.storage(), &r.dream_id)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(dream_session_to_proto(session)))
    }
 
    // ── Wiederkehr Daemons (Phase 8) ─────────────────────────────────────
 
    async fn create_daemon(
        &self,
        _req: Request<nietzsche::CreateDaemonRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        // Placeholder for Phase 8
        Err(Status::unimplemented("CreateDaemon is not yet implemented (Phase 8)"))
    }
 
    async fn list_daemons(
        &self,
        _req: Request<nietzsche::ListDaemonsRequest>,
    ) -> Result<Response<nietzsche::ListDaemonsResponse>, Status> {
        // Placeholder for Phase 8
        Ok(Response::new(nietzsche::ListDaemonsResponse {
            daemons: vec![],
        }))
    }
 
    async fn drop_daemon(
        &self,
        _req: Request<nietzsche::DropDaemonRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        // Placeholder for Phase 8
        Err(Status::unimplemented("DropDaemon is not yet implemented (Phase 8)"))
    }
}

// ── Helpers (Dream) ──────────────────────────────────────────────────────────

fn dream_session_to_proto(s: nietzsche_dream::DreamSession) -> nietzsche::DreamSessionProto {
    nietzsche::DreamSessionProto {
        id:          s.id,
        seed_node:   s.seed_node.to_string(),
        depth:       s.depth as u32,
        noise:       s.noise,
        created_at:  s.created_at,
        status:      format!("{:?}", s.status).to_lowercase(),
        events:      s.events.into_iter().map(|e| nietzsche::DreamEventProto {
            event_type:  format!("{:?}", e.event_type),
            node_id:     e.node_id.to_string(),
            energy:      e.energy,
            depth:       e.depth,
            description: e.description,
        }).collect(),
        node_deltas: s.dream_nodes.into_iter().map(|d| nietzsche::DreamNodeDeltaProto {
            node_id:    d.node_id.to_string(),
            old_energy: d.old_energy,
            new_energy: d.new_energy,
            event_type: d.event_type.map(|et| format!("{:?}", et)).unwrap_or_default(),
        }).collect(),
    }
}
