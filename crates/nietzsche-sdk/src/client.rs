//! Async Rust client for the NietzscheDB gRPC API.
//!
//! Wraps the tonic-generated `NietzscheDbClient` with ergonomic typed methods.
//!
//! ## Example
//!
//! ```rust,ignore
//! use nietzsche_sdk::NietzscheClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut client = NietzscheClient::connect("http://[::1]:50051").await?;
//!
//!     // Insert a node
//!     let node = client.insert_node(vec![0.1, 0.2, 0.3], None).await?;
//!     println!("inserted: {}", node.id);
//!
//!     // Query with NQL
//!     let results = client
//!         .query("MATCH (n:Memory) WHERE n.energy > 0.5 RETURN n LIMIT 10")
//!         .await?;
//!     println!("{} nodes returned", results.nodes.len());
//!
//!     // Trigger reconsolidation sleep
//!     let report = client.trigger_sleep(Default::default()).await?;
//!     println!("committed = {}, ΔH = {:.4}", report.committed, report.hausdorff_delta);
//!
//!     Ok(())
//! }
//! ```

use nietzsche_api::pb::{
    self,
    nietzsche_db_client::NietzscheDbClient,
    BatchInsertNodesRequest, BatchInsertNodesResponse,
    BatchInsertEdgesRequest, BatchInsertEdgesResponse,
    DiffusionRequest, DiffusionResponse, EdgeIdRequest, EdgeResponse,
    Empty, InsertEdgeRequest, InsertNodeRequest, KnnRequest, KnnResponse,
    NodeIdRequest, NodeResponse, QueryRequest, QueryResponse,
    SleepRequest, SleepResponse, StatsResponse, StatusResponse,
    TraversalRequest, TraversalResponse, UpdateEnergyRequest,
    // Algorithm RPCs
    PageRankRequest, AlgorithmScoreResponse, LouvainRequest, AlgorithmCommunityResponse,
    LabelPropRequest, BetweennessRequest, ClosenessRequest, DegreeCentralityRequest,
    WccRequest, SccRequest, AStarRequest, AStarResponse,
    TriangleCountRequest, TriangleCountResponse, JaccardRequest, SimilarityResponse,
};
use tonic::transport::{Channel, Uri};
use uuid::Uuid;

// ─────────────────────────────────────────────
// Client-side typed structs
// ─────────────────────────────────────────────

/// Parameters for `insert_node`.
#[derive(Debug, Clone)]
pub struct InsertNodeParams {
    /// Node UUID — auto-generated if `None`.
    pub id:        Option<Uuid>,
    /// Poincaré-ball coordinates.
    pub coords:    Vec<f64>,
    /// JSON content payload.
    pub content:   serde_json::Value,
    /// Node type string (e.g. `"Semantic"`, `"Episodic"`).
    pub node_type: String,
    /// Initial energy ∈ (0, 1]. `0.0` → server defaults to 1.0.
    pub energy:    f32,
}

impl Default for InsertNodeParams {
    fn default() -> Self {
        Self {
            id:        None,
            coords:    Vec::new(),
            content:   serde_json::Value::Null,
            node_type: "Semantic".into(),
            energy:    1.0,
        }
    }
}

/// Parameters for `insert_edge`.
#[derive(Debug, Clone)]
pub struct InsertEdgeParams {
    pub id:        Option<Uuid>,
    pub from:      Uuid,
    pub to:        Uuid,
    pub edge_type: String,
    pub weight:    f64,
}

impl Default for InsertEdgeParams {
    fn default() -> Self {
        Self {
            id:        None,
            from:      Uuid::nil(),
            to:        Uuid::nil(),
            edge_type: "Association".into(),
            weight:    1.0,
        }
    }
}

/// Parameters for `trigger_sleep`.
#[derive(Debug, Clone)]
pub struct SleepParams {
    pub noise:               f64,
    pub adam_steps:          u32,
    pub adam_lr:             f64,
    pub hausdorff_threshold: f32,
    /// `0` → non-deterministic.
    pub rng_seed:            u64,
}

impl Default for SleepParams {
    fn default() -> Self {
        Self {
            noise:               0.02,
            adam_steps:          10,
            adam_lr:             5e-3,
            hausdorff_threshold: 0.15,
            rng_seed:            0,
        }
    }
}

// ─────────────────────────────────────────────
// NietzscheClient
// ─────────────────────────────────────────────

/// Async gRPC client for NietzscheDB.
///
/// Internally holds a tonic [`Channel`]; use [`NietzscheClient::connect`] to
/// establish the connection before calling any RPC.
pub struct NietzscheClient {
    inner: NietzscheDbClient<Channel>,
}

impl NietzscheClient {
    // ── Construction ──────────────────────────────────

    /// Connect to a NietzscheDB gRPC endpoint.
    ///
    /// `uri` should be a string such as `"http://[::1]:50051"` or `"http://127.0.0.1:50051"`.
    pub async fn connect(uri: impl AsRef<str>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let parsed: Uri = uri.as_ref().parse()?;
        let channel = Channel::builder(parsed).connect().await?;
        Ok(Self { inner: NietzscheDbClient::new(channel) })
    }

    // ── Node CRUD ─────────────────────────────────────

    /// Insert a node into the graph.
    pub async fn insert_node(
        &mut self,
        params: InsertNodeParams,
    ) -> Result<NodeResponse, tonic::Status> {
        let content_bytes = serde_json::to_vec(&params.content).unwrap_or_default();
        let req = InsertNodeRequest {
            id:         params.id.map(|u| u.to_string()).unwrap_or_default(),
            embedding:  Some(pb::PoincareVector {
                coords: params.coords,
                dim:    0, // server infers from coords length
            }),
            content:    content_bytes,
            node_type:  params.node_type,
            energy:     params.energy,
            collection: String::new(), // empty → "default"
        };
        self.inner.insert_node(req).await.map(|r| r.into_inner())
    }

    /// Retrieve a node by UUID.
    pub async fn get_node(&mut self, id: Uuid) -> Result<NodeResponse, tonic::Status> {
        let req = NodeIdRequest { id: id.to_string(), collection: String::new() };
        self.inner.get_node(req).await.map(|r| r.into_inner())
    }

    /// Hard-delete a node (and all its edges).
    pub async fn delete_node(&mut self, id: Uuid) -> Result<StatusResponse, tonic::Status> {
        let req = NodeIdRequest { id: id.to_string(), collection: String::new() };
        self.inner.delete_node(req).await.map(|r| r.into_inner())
    }

    /// Update a node's energy level.
    pub async fn update_energy(
        &mut self,
        node_id: Uuid,
        energy:  f32,
    ) -> Result<StatusResponse, tonic::Status> {
        let req = UpdateEnergyRequest { node_id: node_id.to_string(), energy, collection: String::new() };
        self.inner.update_energy(req).await.map(|r| r.into_inner())
    }

    // ── Edge CRUD ─────────────────────────────────────

    /// Insert a directed edge between two nodes.
    pub async fn insert_edge(
        &mut self,
        params: InsertEdgeParams,
    ) -> Result<EdgeResponse, tonic::Status> {
        let req = InsertEdgeRequest {
            id:         params.id.map(|u| u.to_string()).unwrap_or_default(),
            from:       params.from.to_string(),
            to:         params.to.to_string(),
            edge_type:  params.edge_type,
            weight:     params.weight,
            collection: String::new(), // empty → "default"
        };
        self.inner.insert_edge(req).await.map(|r| r.into_inner())
    }

    /// Delete an edge by UUID.
    pub async fn delete_edge(&mut self, id: Uuid) -> Result<StatusResponse, tonic::Status> {
        let req = EdgeIdRequest { id: id.to_string(), collection: String::new() };
        self.inner.delete_edge(req).await.map(|r| r.into_inner())
    }

    // ── Batch operations ───────────────────────────────

    /// Bulk-insert multiple nodes in a single RPC call.
    pub async fn batch_insert_nodes(
        &mut self,
        nodes: Vec<InsertNodeParams>,
        collection: &str,
    ) -> Result<BatchInsertNodesResponse, tonic::Status> {
        let proto_nodes: Vec<InsertNodeRequest> = nodes.into_iter().map(|p| {
            InsertNodeRequest {
                id:         p.id.map(|u| u.to_string()).unwrap_or_default(),
                embedding:  Some(pb::PoincareVector { coords: p.coords, dim: 0 }),
                content:    serde_json::to_vec(&p.content).unwrap_or_default(),
                node_type:  p.node_type,
                energy:     p.energy,
                collection: String::new(),
            }
        }).collect();
        let req = BatchInsertNodesRequest { nodes: proto_nodes, collection: collection.into() };
        self.inner.batch_insert_nodes(req).await.map(|r| r.into_inner())
    }

    /// Bulk-insert multiple edges in a single RPC call.
    pub async fn batch_insert_edges(
        &mut self,
        edges: Vec<InsertEdgeParams>,
        collection: &str,
    ) -> Result<BatchInsertEdgesResponse, tonic::Status> {
        let proto_edges: Vec<InsertEdgeRequest> = edges.into_iter().map(|p| {
            InsertEdgeRequest {
                id:         p.id.map(|u| u.to_string()).unwrap_or_default(),
                from:       p.from.to_string(),
                to:         p.to.to_string(),
                edge_type:  p.edge_type,
                weight:     p.weight,
                collection: String::new(),
            }
        }).collect();
        let req = BatchInsertEdgesRequest { edges: proto_edges, collection: collection.into() };
        self.inner.batch_insert_edges(req).await.map(|r| r.into_inner())
    }

    // ── NQL query ─────────────────────────────────────

    /// Execute a Nietzsche Query Language (NQL) statement.
    pub async fn query(&mut self, nql: &str) -> Result<QueryResponse, tonic::Status> {
        let req = QueryRequest { nql: nql.to_string(), params: Default::default(), collection: String::new() };
        self.inner.query(req).await.map(|r| r.into_inner())
    }

    // ── Vector KNN ────────────────────────────────────

    /// Hyperbolic k-nearest-neighbour search.
    pub async fn knn_search(
        &mut self,
        coords: Vec<f64>,
        k:      u32,
    ) -> Result<KnnResponse, tonic::Status> {
        let req = KnnRequest { query_coords: coords, k, collection: String::new() };
        self.inner.knn_search(req).await.map(|r| r.into_inner())
    }

    // ── Traversal ─────────────────────────────────────

    /// Breadth-first traversal from `start`.
    pub async fn bfs(
        &mut self,
        start:     Uuid,
        max_depth: u32,
        max_nodes: u32,
    ) -> Result<TraversalResponse, tonic::Status> {
        let req = TraversalRequest {
            start_node_id: start.to_string(),
            max_depth,
            max_nodes,
            max_cost:      0.0,
            energy_min:    0.0,
            collection:    String::new(),
        };
        self.inner.bfs(req).await.map(|r| r.into_inner())
    }

    /// Dijkstra shortest-path from `start`.
    pub async fn dijkstra(
        &mut self,
        start:    Uuid,
        max_cost: f64,
    ) -> Result<TraversalResponse, tonic::Status> {
        let req = TraversalRequest {
            start_node_id: start.to_string(),
            max_depth:     0,
            max_nodes:     0,
            max_cost,
            energy_min:    0.0,
            collection:    String::new(),
        };
        self.inner.dijkstra(req).await.map(|r| r.into_inner())
    }

    // ── Diffusion ─────────────────────────────────────

    /// Run Chebyshev heat-kernel diffusion from `sources`.
    pub async fn diffuse(
        &mut self,
        sources:    Vec<Uuid>,
        t_values:   Vec<f64>,
        k_chebyshev: u32,
    ) -> Result<DiffusionResponse, tonic::Status> {
        let req = DiffusionRequest {
            source_ids:  sources.iter().map(|u| u.to_string()).collect(),
            t_values,
            k_chebyshev,
            collection:  String::new(),
        };
        self.inner.diffuse(req).await.map(|r| r.into_inner())
    }

    // ── Sleep cycle ───────────────────────────────────

    /// Trigger a Riemannian reconsolidation sleep cycle on the server.
    pub async fn trigger_sleep(
        &mut self,
        params: SleepParams,
    ) -> Result<SleepResponse, tonic::Status> {
        let req = SleepRequest {
            noise:               params.noise,
            adam_steps:          params.adam_steps,
            adam_lr:             params.adam_lr,
            hausdorff_threshold: params.hausdorff_threshold,
            rng_seed:            params.rng_seed,
            collection:          String::new(),
        };
        self.inner.trigger_sleep(req).await.map(|r| r.into_inner())
    }

    // ── Graph Algorithms ──────────────────────────────

    /// Run PageRank on the graph.
    pub async fn run_pagerank(
        &mut self,
        damping: f64,
        max_iterations: u32,
    ) -> Result<AlgorithmScoreResponse, tonic::Status> {
        let req = PageRankRequest {
            collection: String::new(),
            damping_factor: damping,
            max_iterations,
            convergence_threshold: 0.0,
        };
        self.inner.run_page_rank(req).await.map(|r| r.into_inner())
    }

    /// Run Louvain community detection.
    pub async fn run_louvain(
        &mut self,
        max_iterations: u32,
        resolution: f64,
    ) -> Result<AlgorithmCommunityResponse, tonic::Status> {
        let req = LouvainRequest {
            collection: String::new(),
            max_iterations,
            resolution,
        };
        self.inner.run_louvain(req).await.map(|r| r.into_inner())
    }

    /// Run Label Propagation community detection.
    pub async fn run_label_prop(
        &mut self,
        max_iterations: u32,
    ) -> Result<AlgorithmCommunityResponse, tonic::Status> {
        let req = LabelPropRequest {
            collection: String::new(),
            max_iterations,
        };
        self.inner.run_label_prop(req).await.map(|r| r.into_inner())
    }

    /// Run Betweenness centrality.
    pub async fn run_betweenness(
        &mut self,
        sample_size: u32,
    ) -> Result<AlgorithmScoreResponse, tonic::Status> {
        let req = BetweennessRequest {
            collection: String::new(),
            sample_size,
        };
        self.inner.run_betweenness(req).await.map(|r| r.into_inner())
    }

    /// Run Closeness centrality.
    pub async fn run_closeness(&mut self) -> Result<AlgorithmScoreResponse, tonic::Status> {
        let req = ClosenessRequest { collection: String::new() };
        self.inner.run_closeness(req).await.map(|r| r.into_inner())
    }

    /// Run Degree centrality.
    pub async fn run_degree_centrality(
        &mut self,
        direction: &str,
    ) -> Result<AlgorithmScoreResponse, tonic::Status> {
        let req = DegreeCentralityRequest {
            collection: String::new(),
            direction: direction.to_string(),
        };
        self.inner.run_degree_centrality(req).await.map(|r| r.into_inner())
    }

    /// Run Weakly Connected Components.
    pub async fn run_wcc(&mut self) -> Result<AlgorithmCommunityResponse, tonic::Status> {
        let req = WccRequest { collection: String::new() };
        self.inner.run_wcc(req).await.map(|r| r.into_inner())
    }

    /// Run Strongly Connected Components.
    pub async fn run_scc(&mut self) -> Result<AlgorithmCommunityResponse, tonic::Status> {
        let req = SccRequest { collection: String::new() };
        self.inner.run_scc(req).await.map(|r| r.into_inner())
    }

    /// A* pathfinding using Poincare distance heuristic.
    pub async fn run_astar(
        &mut self,
        start: Uuid,
        goal: Uuid,
    ) -> Result<AStarResponse, tonic::Status> {
        let req = AStarRequest {
            collection: String::new(),
            start_node_id: start.to_string(),
            goal_node_id: goal.to_string(),
        };
        self.inner.run_a_star(req).await.map(|r| r.into_inner())
    }

    /// Count triangles in the graph.
    pub async fn run_triangle_count(&mut self) -> Result<TriangleCountResponse, tonic::Status> {
        let req = TriangleCountRequest { collection: String::new() };
        self.inner.run_triangle_count(req).await.map(|r| r.into_inner())
    }

    /// Jaccard node similarity.
    pub async fn run_jaccard(
        &mut self,
        top_k: u32,
        threshold: f64,
    ) -> Result<SimilarityResponse, tonic::Status> {
        let req = JaccardRequest {
            collection: String::new(),
            top_k,
            threshold,
        };
        self.inner.run_jaccard_similarity(req).await.map(|r| r.into_inner())
    }

    // ── Admin ─────────────────────────────────────────

    /// Retrieve server-side node/edge counts and version.
    pub async fn get_stats(&mut self) -> Result<StatsResponse, tonic::Status> {
        self.inner.get_stats(Empty {}).await.map(|r| r.into_inner())
    }

    /// Check server liveness.
    pub async fn health_check(&mut self) -> Result<StatusResponse, tonic::Status> {
        self.inner.health_check(Empty {}).await.map(|r| r.into_inner())
    }
}
