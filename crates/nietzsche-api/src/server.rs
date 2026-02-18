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
//!       │  returns Arc<Mutex<NietzscheDB<AnyVectorStore>>>
//!       ▼
//! db.lock().await   →   NietzscheDB method
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use nietzsche_graph::{
    CollectionConfig, CollectionManager,
    Edge, EdgeType, GraphError, Node, PoincareVector,
    traversal::{BfsConfig, DijkstraConfig},
    bfs, dijkstra,
};
use nietzsche_pregel::{DiffusionConfig, DiffusionEngine};
use nietzsche_query::{ParamValue, Params, execute, parse};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use nietzsche_zaratustra::{ZaratustraConfig, ZaratustraEngine};

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

fn node_to_proto(node: Node) -> nietzsche::NodeResponse {
    let content_bytes = serde_json::to_vec(&node.content).unwrap_or_default();
    nietzsche::NodeResponse {
        found:           true,
        id:              node.id.to_string(),
        embedding:       Some(nietzsche::PoincareVector {
            coords: node.embedding.coords,
            dim:    node.embedding.dim as u32,
        }),
        energy:          node.energy,
        depth:           node.depth,
        hausdorff_local: node.hausdorff_local,
        created_at:      node.created_at,
        content:         content_bytes,
        node_type:       format!("{:?}", node.node_type),
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

// ── NietzscheServer ───────────────────────────────────────────────────────────

/// gRPC service handler backed by a [`CollectionManager`].
///
/// Each RPC extracts the `collection` field (empty → `"default"`) and
/// dispatches to the corresponding isolated `NietzscheDB<AnyVectorStore>`.
pub struct NietzscheServer {
    cm: Arc<CollectionManager>,
}

impl NietzscheServer {
    pub fn new(cm: Arc<CollectionManager>) -> Self {
        Self { cm }
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

        let embedding = PoincareVector::new(emb_proto.coords);
        let content: serde_json::Value = if r.content.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&r.content)
                .map_err(|e| Status::invalid_argument(format!("invalid content JSON: {e}")))?
        };

        let mut node = Node::new(id, embedding, content);
        if r.energy > 0.0 { node.energy = r.energy; }
        debug!(node_id = %id, collection = %col(&r.collection), "inserting node");

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.lock().await;
        db.insert_node(node.clone()).map_err(graph_err)?;

        Ok(Response::new(node_to_proto(node)))
    }

    async fn get_node(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::NodeResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.id, "id")?;

        let shared = get_col!(self.cm, &r.collection);
        let db = shared.lock().await;
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
        let mut db = shared.lock().await;
        db.delete_node(id).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
    }

    async fn update_energy(
        &self,
        req: Request<nietzsche::UpdateEnergyRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.node_id, "node_id")?;

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.lock().await;
        db.update_energy(id, r.energy).map_err(graph_err)?;
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
        let mut db = shared.lock().await;
        db.insert_edge(edge).map_err(graph_err)?;

        Ok(Response::new(nietzsche::EdgeResponse { success: true, id: id.to_string() }))
    }

    async fn delete_edge(
        &self,
        req: Request<nietzsche::EdgeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r  = req.into_inner();
        let id = parse_uuid(&r.id, "id")?;

        let shared = get_col!(self.cm, &r.collection);
        let mut db = shared.lock().await;
        db.delete_edge(id).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
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
        let db      = shared.lock().await;
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
        let query = PoincareVector::new(r.query_coords);
        let k     = r.k as usize;

        let shared  = get_col!(self.cm, &r.collection);
        let db      = shared.lock().await;
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
        let db      = shared.lock().await;
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
        let db     = shared.lock().await;
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
        let db      = shared.lock().await;
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
        let mut db = shared.lock().await;
        let report = SleepCycle::run(&config, &mut *db, &mut rng)
            .map_err(|e| Status::internal(e.to_string()))?;

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
        let db = shared.lock().await;

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

    // ── Sensory Compression Layer (Phase 11) — stubs ──────────────────────

    async fn insert_sensory(
        &self,
        _req: Request<nietzsche::InsertSensoryRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        Err(Status::unimplemented("InsertSensory not yet available in this build"))
    }

    async fn get_sensory(
        &self,
        _req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::SensoryResponse>, Status> {
        Err(Status::unimplemented("GetSensory not yet available in this build"))
    }

    async fn reconstruct(
        &self,
        _req: Request<nietzsche::ReconstructRequest>,
    ) -> Result<Response<nietzsche::ReconstructResponse>, Status> {
        Err(Status::unimplemented("Reconstruct not yet available in this build"))
    }

    async fn degrade_sensory(
        &self,
        _req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        Err(Status::unimplemented("DegradeSensory not yet available in this build"))
    }
}
