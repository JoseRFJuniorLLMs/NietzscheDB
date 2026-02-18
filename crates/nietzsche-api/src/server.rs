//! gRPC server implementation for the NietzscheDB API.
//!
//! Implements the `NietzscheDb` service generated from `proto/nietzsche.proto`.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use nietzsche_api::server::NietzscheServer;
//! use nietzsche_graph::{MockVectorStore, NietzscheDB};
//! use std::{path::Path, sync::Arc};
//! use tokio::sync::Mutex;
//!
//! let db = NietzscheDB::open(Path::new("/data/nietzsche"), MockVectorStore::default())?;
//! let server = NietzscheServer::new(Arc::new(Mutex::new(db)));
//! server.serve("[::1]:50051".parse()?).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tokio::sync::Mutex;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use nietzsche_graph::{
    Edge, EdgeType, GraphError, NietzscheDB, Node, PoincareVector, VectorStore,
    traversal::{BfsConfig, DijkstraConfig},
    bfs, dijkstra,
};
use nietzsche_pregel::{DiffusionConfig, DiffusionEngine};
use nietzsche_query::{Params, execute, parse};
use nietzsche_sleep::{SleepConfig, SleepCycle};

use crate::proto::nietzsche::{
    self,
    nietzsche_db_server::{NietzscheDb, NietzscheDbServer},
};
use crate::validation::{
    parse_uuid as val_uuid, validate_embedding, validate_energy, validate_k,
    validate_nql, validate_sleep_params, validate_source_count, validate_t_values,
};

// Re-export the tonic server wrapper for convenience
pub use nietzsche::nietzsche_db_server::NietzscheDbServer as TonicServer;

// ─────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────

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

fn err_status(msg: impl ToString) -> nietzsche::StatusResponse {
    nietzsche::StatusResponse { status: "error".into(), error: msg.to_string() }
}

// ─────────────────────────────────────────────
// NietzscheServer
// ─────────────────────────────────────────────

/// gRPC service handler wrapping a shared `NietzscheDB<V>`.
///
/// The inner DB is protected by a `tokio::sync::Mutex` so multiple concurrent
/// RPCs can access it safely.  For production use, consider a reader-writer
/// lock or sharding by node ID range.
pub struct NietzscheServer<V: VectorStore + 'static> {
    db: Arc<Mutex<NietzscheDB<V>>>,
}

impl<V: VectorStore + 'static> NietzscheServer<V> {
    pub fn new(db: Arc<Mutex<NietzscheDB<V>>>) -> Self {
        Self { db }
    }

    /// Wrap this server in a tonic [`NietzscheDbServer`] ready to be passed
    /// to `tonic::transport::Server::add_service`.
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

// ─────────────────────────────────────────────
// gRPC trait implementation
// ─────────────────────────────────────────────

#[tonic::async_trait]
impl<V: VectorStore + Send + Sync + 'static> NietzscheDb for NietzscheServer<V> {

    // ── Node CRUD ────────────────────────────────────────────────────────

    #[instrument(skip(self, req), fields(node_id))]
    async fn insert_node(
        &self,
        req: Request<nietzsche::InsertNodeRequest>,
    ) -> Result<Response<nietzsche::NodeResponse>, Status> {
        let r = req.into_inner();

        // Resolve ID (auto-generate if empty)
        let id = if r.id.is_empty() {
            Uuid::new_v4()
        } else {
            parse_uuid(&r.id, "id")?
        };

        let emb_proto = r.embedding.ok_or_else(|| {
            Status::invalid_argument("embedding is required")
        })?;

        // Validate embedding before accepting
        validate_embedding(&emb_proto)?;
        if r.energy != 0.0 { validate_energy(r.energy)?; }

        let coords: Vec<f64> = emb_proto.coords;
        let embedding = PoincareVector::new(coords);

        let content: serde_json::Value = if r.content.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&r.content)
                .map_err(|e| Status::invalid_argument(format!("invalid content JSON: {e}")))?
        };

        let mut node = Node::new(id, embedding, content);
        if r.energy > 0.0 { node.energy = r.energy; }
        debug!(node_id = %id, "inserting node");

        let mut db = self.db.lock().await;
        db.insert_node(node.clone()).map_err(graph_err)?;

        Ok(Response::new(node_to_proto(node)))
    }

    async fn get_node(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::NodeResponse>, Status> {
        let id = parse_uuid(&req.into_inner().id, "id")?;
        let db = self.db.lock().await;
        match db.get_node(id).map_err(graph_err)? {
            Some(node) => Ok(Response::new(node_to_proto(node))),
            None       => Ok(Response::new(not_found())),
        }
    }

    async fn delete_node(
        &self,
        req: Request<nietzsche::NodeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let id = parse_uuid(&req.into_inner().id, "id")?;
        let mut db = self.db.lock().await;
        db.delete_node(id).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
    }

    async fn update_energy(
        &self,
        req: Request<nietzsche::UpdateEnergyRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let r = req.into_inner();
        let id = parse_uuid(&r.node_id, "node_id")?;
        let mut db = self.db.lock().await;
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
        let to   = parse_uuid(&r.to, "to")?;

        let edge = Edge {
            id,
            from,
            to,
            edge_type: parse_edge_type(&r.edge_type),
            weight:    r.weight,
            metadata:  HashMap::new(),
        };

        let mut db = self.db.lock().await;
        db.insert_edge(edge).map_err(graph_err)?;

        Ok(Response::new(nietzsche::EdgeResponse {
            success: true,
            id:      id.to_string(),
        }))
    }

    async fn delete_edge(
        &self,
        req: Request<nietzsche::EdgeIdRequest>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        let id = parse_uuid(&req.into_inner().id, "id")?;
        let mut db = self.db.lock().await;
        db.delete_edge(id).map_err(graph_err)?;
        Ok(Response::new(ok_status()))
    }

    // ── NQL query ─────────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn query(
        &self,
        req: Request<nietzsche::QueryRequest>,
    ) -> Result<Response<nietzsche::QueryResponse>, Status> {
        let nql = req.into_inner().nql;
        validate_nql(&nql)?;

        let ast = parse(&nql).map_err(|e| {
            Status::invalid_argument(format!("NQL parse error: {e}"))
        })?;

        let db = self.db.lock().await;
        let params = Params::new();

        let results = execute(&ast, db.storage(), db.adjacency(), &params)
            .map_err(|e| Status::internal(e.to_string()))?;

        use nietzsche_query::QueryResult;
        let mut nodes      = Vec::new();
        let mut node_pairs = Vec::new();
        let mut path_ids   = Vec::new();

        for r in results {
            match r {
                QueryResult::Node(n)              => nodes.push(node_to_proto(n)),
                QueryResult::NodePair { from, to } => node_pairs.push(nietzsche::NodePair {
                    from: Some(node_to_proto(from)),
                    to:   Some(node_to_proto(to)),
                }),
                QueryResult::DiffusionPath(ids)   =>
                    path_ids.extend(ids.iter().map(|u| u.to_string())),
            }
        }

        Ok(Response::new(nietzsche::QueryResponse {
            nodes,
            node_pairs,
            path_ids,
            error: String::new(),
        }))
    }

    // ── KNN search ───────────────────────────────────────────────────────

    #[instrument(skip(self, req))]
    async fn knn_search(
        &self,
        req: Request<nietzsche::KnnRequest>,
    ) -> Result<Response<nietzsche::KnnResponse>, Status> {
        let r = req.into_inner();
        validate_k(r.k)?;
        let query = PoincareVector::new(r.query_coords);
        let k     = r.k as usize;

        let db      = self.db.lock().await;
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
            max_depth:  if r.max_depth  > 0 { r.max_depth  as usize } else { 10 },
            max_nodes:  if r.max_nodes  > 0 { r.max_nodes  as usize } else { 1_000 },
            energy_min: r.energy_min,
        };

        let db      = self.db.lock().await;
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

        let db      = self.db.lock().await;
        let costs   = dijkstra(db.storage(), db.adjacency(), start, &config)
            .map_err(graph_err)?;

        // Sort by cost for a deterministic response
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
        let config = DiffusionConfig { k_chebyshev, ..Default::default() };

        let db = self.db.lock().await;
        let engine = DiffusionEngine::new(config);
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
        let r = req.into_inner();
        let noise_val  = if r.noise > 0.0   { r.noise }   else { 0.02 };
        let lr_val     = if r.adam_lr > 0.0 { r.adam_lr } else { 5e-3 };
        validate_sleep_params(noise_val, lr_val)?;

        let config = SleepConfig {
            noise:               noise_val,
            adam_steps:          if r.adam_steps > 0 { r.adam_steps as usize } else { 10 },
            adam_lr:             lr_val,
            hausdorff_threshold: if r.hausdorff_threshold > 0.0 { r.hausdorff_threshold } else { 0.15 },
        };
        warn!(
            noise = config.noise,
            adam_steps = config.adam_steps,
            "triggering sleep cycle — DB will be locked"
        );

        let seed = if r.rng_seed == 0 {
            rand::random::<u64>()
        } else {
            r.rng_seed
        };
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(seed);

        let mut db = self.db.lock().await;
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

    // ── Admin ─────────────────────────────────────────────────────────────

    async fn get_stats(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::StatsResponse>, Status> {
        let db = self.db.lock().await;
        let node_count = db.node_count().map_err(graph_err)? as u64;
        let edge_count = db.edge_count().map_err(graph_err)? as u64;

        Ok(Response::new(nietzsche::StatsResponse {
            node_count,
            edge_count,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }))
    }

    async fn health_check(
        &self,
        _req: Request<nietzsche::Empty>,
    ) -> Result<Response<nietzsche::StatusResponse>, Status> {
        Ok(Response::new(ok_status()))
    }
}
