//! # Phase XVIII — Reasoning Engine (Graph Cognitive Kernel)
//!
//! Orchestrates the full cognitive pipeline:
//!
//! ```text
//! query (embedding or node IDs)
//!     ↓
//! ego subgraph extraction (Phase XVII cache)
//!     ↓
//! multi-path discovery (greedy route, concept path, diffusion walk)
//!     ↓
//! trajectory validation (GCS per hop)
//!     ↓
//! inference classification (generalization / specialization / dialectical / rupture)
//!     ↓
//! synthesis (Fréchet mean on Riemann sphere, if dialectical)
//!     ↓
//! structured ReasoningResult
//! ```
//!
//! ## Design
//!
//! The Reasoning Engine is a **pure orchestrator** — it calls into existing modules
//! (traversal, concept_path, ego_cache, synthesis, inference_engine) and composes
//! their outputs into a unified result. No new geometric primitives are introduced.
//!
//! ## Usage
//!
//! ```ignore
//! let engine = ReasoningEngine::new(config);
//! let result = engine.reason(&db, query)?;
//! // result.paths — multiple discovered paths
//! // result.rationale — inference classification for best path
//! // result.synthesis — optional synthesis node (if dialectical)
//! // result.explanation — human-readable narrative
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use nietzsche_graph::{
    AdjacencyIndex, GraphStorage, PoincareVector, Node, NodeMeta, NodeType,
    EgoCacheEntry, CausalType,
};
use nietzsche_graph::traversal::{
    greedy_route, greedy_route_to_embedding,
    diffusion_walk, shortest_path,
    GreedyRouteConfig, DiffusionConfig, DijkstraConfig,
    RouteResult,
};
use nietzsche_graph::concept_path::{
    concept_path, concept_path_from_embedding, explain_path,
    ConceptPath, PathHop,
};
use nietzsche_graph::ego_cache::EgoNeighbor;

use crate::error::{AgiError, AgiResult};
use crate::inference_engine::{InferenceEngine, InferenceConfig, EdgeInfo};
use crate::rationale::{InferenceType, Rationale};
use crate::trajectory::{GeodesicTrajectory, GeodesicCoherenceScore, validate_trajectory, GcsConfig};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Reasoning Engine.
#[derive(Debug, Clone)]
pub struct ReasoningConfig {
    /// Maximum hops for greedy routing.
    pub max_hops: usize,
    /// Minimum energy for traversal filters.
    pub energy_min: f32,
    /// Enable A* fallback when greedy routing hits local minimum.
    pub astar_fallback: bool,
    /// Maximum diffusion walk steps.
    pub diffusion_steps: usize,
    /// Energy bias for diffusion walk.
    pub diffusion_energy_bias: f32,
    /// Number of alternative paths to discover.
    pub max_alternative_paths: usize,
    /// Ego-cache depth-1 limit.
    pub ego_max_depth1: usize,
    /// Ego-cache depth-2 limit.
    pub ego_max_depth2: usize,
    /// GCS threshold for trajectory validation.
    pub gcs_threshold: f64,
    /// Whether to attempt synthesis on dialectical results.
    pub auto_synthesize: bool,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            max_hops: 20,
            energy_min: 0.0,
            astar_fallback: true,
            diffusion_steps: 30,
            diffusion_energy_bias: 1.5,
            max_alternative_paths: 3,
            ego_max_depth1: 50,
            ego_max_depth2: 100,
            gcs_threshold: 0.5,
            auto_synthesize: true,
        }
    }
}

// ─────────────────────────────────────────────
// Query types
// ─────────────────────────────────────────────

/// Input to the reasoning engine.
#[derive(Debug, Clone)]
pub enum ReasoningQuery {
    /// Reason about the path between two known nodes.
    NodeToNode {
        source: Uuid,
        target: Uuid,
    },
    /// Reason about the neighborhood of a single node (ego reasoning).
    EgoReasoning {
        node_id: Uuid,
    },
    /// Reason toward a target embedding (e.g., from a text query).
    EmbeddingQuery {
        start_node: Uuid,
        target_coords: Vec<f32>,
    },
}

// ─────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────

/// A discovered reasoning path with its analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPath {
    /// Path node IDs in order.
    pub node_ids: Vec<Uuid>,
    /// Method used to discover this path.
    pub method: String,
    /// Total hyperbolic distance.
    pub total_distance: f64,
    /// Number of hops.
    pub hop_count: usize,
    /// Whether the path reached its target.
    pub reached_target: bool,
    /// Per-hop annotations (if concept path was used).
    pub hops: Vec<PathAnnotation>,
}

/// Lightweight annotation for each hop in a reasoning path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAnnotation {
    pub node_id: Uuid,
    pub node_type: String,
    pub energy: f32,
    pub depth: f64,
    pub summary: String,
    pub distance_from_prev: f64,
}

/// Ego subgraph summary included in reasoning results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoSummary {
    pub root_id: Uuid,
    pub depth1_count: usize,
    pub depth2_count: usize,
    pub edge_count: usize,
    /// High-energy nodes in the ego neighborhood (top 5).
    pub key_neighbors: Vec<KeyNeighbor>,
}

/// A notable neighbor in the ego subgraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyNeighbor {
    pub id: Uuid,
    pub energy: f32,
    pub depth: f32,
    pub node_type: String,
    pub edge_type: String,
}

/// Complete result from the reasoning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    /// The query that was analyzed.
    pub query_type: String,
    /// Ego subgraph summary for the source node.
    pub ego_summary: Option<EgoSummary>,
    /// All discovered paths (best first).
    pub paths: Vec<ReasoningPath>,
    /// Inference classification for the best path.
    pub inference_type: Option<String>,
    /// GCS (Geodesic Coherence Score) for the best path.
    pub gcs: Option<f64>,
    /// Radial gradient (positive = specialization, negative = generalization).
    pub radial_gradient: Option<f64>,
    /// Causal fraction (% of timelike edges on best path).
    pub causal_fraction: Option<f64>,
    /// Synthesis coordinates (if dialectical synthesis was triggered).
    pub synthesis_coords: Option<Vec<f64>>,
    /// Human-readable explanation of the reasoning.
    pub explanation: String,
    /// Computation time in microseconds.
    pub duration_us: u64,
}

// ─────────────────────────────────────────────
// Reasoning Engine
// ─────────────────────────────────────────────

/// The Graph Cognitive Kernel — orchestrates multi-path reasoning.
pub struct ReasoningEngine {
    config: ReasoningConfig,
    inference: InferenceEngine,
}

impl ReasoningEngine {
    pub fn new(config: ReasoningConfig) -> Self {
        let inference = InferenceEngine::new(InferenceConfig {
            gcs_threshold: config.gcs_threshold,
            ..Default::default()
        });
        Self { config, inference }
    }

    pub fn with_defaults() -> Self {
        Self::new(ReasoningConfig::default())
    }

    /// Execute the full reasoning pipeline.
    pub fn reason(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        query: ReasoningQuery,
    ) -> AgiResult<ReasoningResult> {
        let t0 = std::time::Instant::now();

        match query {
            ReasoningQuery::NodeToNode { source, target } => {
                self.reason_node_to_node(storage, adjacency, source, target, t0)
            }
            ReasoningQuery::EgoReasoning { node_id } => {
                self.reason_ego(storage, adjacency, node_id, t0)
            }
            ReasoningQuery::EmbeddingQuery { start_node, target_coords } => {
                self.reason_to_embedding(storage, adjacency, start_node, target_coords, t0)
            }
        }
    }

    // ── Node-to-Node reasoning ───────────────────────

    fn reason_node_to_node(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        source: Uuid,
        target: Uuid,
        t0: std::time::Instant,
    ) -> AgiResult<ReasoningResult> {
        // 1. Build ego summary for source
        let ego_summary = self.build_ego_summary(storage, adjacency, source);

        // 2. Discover multiple paths
        let mut paths = Vec::new();

        // Path 1: Concept path (annotated greedy route)
        if let Ok(cp) = concept_path(storage, adjacency, source, target, &self.greedy_config()) {
            paths.push(concept_path_to_reasoning_path(cp, "concept_path"));
        }

        // Path 2: Shortest path (Dijkstra with Poincaré distance)
        if let Ok(Some(sp)) = shortest_path(storage, adjacency, source, target, &self.dijkstra_config()) {
            let dist = self.compute_path_distance(storage, &sp);
            paths.push(ReasoningPath {
                hop_count: sp.len().saturating_sub(1),
                node_ids: sp,
                method: "dijkstra".into(),
                total_distance: dist,
                reached_target: true,
                hops: vec![],
            });
        }

        // Path 3: Diffusion walk (energy-biased random walk)
        if let Ok(dw) = diffusion_walk(storage, adjacency, source, &self.diffusion_config()) {
            // Check if the walk passed through or near the target
            let reached = dw.contains(&target);
            let dist = self.compute_path_distance(storage, &dw);
            paths.push(ReasoningPath {
                hop_count: dw.len().saturating_sub(1),
                node_ids: dw,
                method: "diffusion_walk".into(),
                total_distance: dist,
                reached_target: reached,
                hops: vec![],
            });
        }

        // Sort by: reached_target (true first), then shortest distance
        paths.sort_by(|a, b| {
            b.reached_target.cmp(&a.reached_target)
                .then(a.total_distance.partial_cmp(&b.total_distance).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Trim to max_alternative_paths
        paths.truncate(self.config.max_alternative_paths);

        // 3. Validate best path with GCS + inference engine
        let (inference_type, gcs, radial_gradient, causal_fraction, synthesis_coords) =
            if let Some(best) = paths.first() {
                self.analyze_path(storage, adjacency, &best.node_ids)
            } else {
                (None, None, None, None, None)
            };

        // 4. Build explanation
        let explanation = self.build_explanation(&paths, &inference_type, &ego_summary);

        Ok(ReasoningResult {
            query_type: "node_to_node".into(),
            ego_summary,
            paths,
            inference_type,
            gcs,
            radial_gradient,
            causal_fraction,
            synthesis_coords,
            explanation,
            duration_us: t0.elapsed().as_micros() as u64,
        })
    }

    // ── Ego reasoning ─────────────────────────────────

    fn reason_ego(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        node_id: Uuid,
        t0: std::time::Instant,
    ) -> AgiResult<ReasoningResult> {
        let ego_summary = self.build_ego_summary(storage, adjacency, node_id);

        // Find paths to top-energy neighbors
        let mut paths = Vec::new();
        if let Some(ref ego) = ego_summary {
            for neighbor in ego.key_neighbors.iter().take(3) {
                if let Ok(cp) = concept_path(storage, adjacency, node_id, neighbor.id, &self.greedy_config()) {
                    paths.push(concept_path_to_reasoning_path(cp, "ego_exploration"));
                }
            }
        }

        // Diffusion walk for stochastic exploration
        if let Ok(dw) = diffusion_walk(storage, adjacency, node_id, &self.diffusion_config()) {
            let dist = self.compute_path_distance(storage, &dw);
            paths.push(ReasoningPath {
                hop_count: dw.len().saturating_sub(1),
                node_ids: dw,
                method: "ego_diffusion".into(),
                total_distance: dist,
                reached_target: false,
                hops: vec![],
            });
        }

        paths.truncate(self.config.max_alternative_paths);

        let (inference_type, gcs, radial_gradient, causal_fraction, synthesis_coords) =
            if let Some(best) = paths.first() {
                self.analyze_path(storage, adjacency, &best.node_ids)
            } else {
                (None, None, None, None, None)
            };

        let explanation = self.build_explanation(&paths, &inference_type, &ego_summary);

        Ok(ReasoningResult {
            query_type: "ego_reasoning".into(),
            ego_summary,
            paths,
            inference_type,
            gcs,
            radial_gradient,
            causal_fraction,
            synthesis_coords,
            explanation,
            duration_us: t0.elapsed().as_micros() as u64,
        })
    }

    // ── Embedding query reasoning ────────────────────

    fn reason_to_embedding(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        start_node: Uuid,
        target_coords: Vec<f32>,
        t0: std::time::Instant,
    ) -> AgiResult<ReasoningResult> {
        let ego_summary = self.build_ego_summary(storage, adjacency, start_node);

        let mut paths = Vec::new();

        // Concept path toward embedding
        if let Ok(cp) = concept_path_from_embedding(
            storage, adjacency, start_node, &target_coords, &self.greedy_config(),
        ) {
            paths.push(concept_path_to_reasoning_path(cp, "embedding_concept_path"));
        }

        // Greedy route toward embedding
        if let Ok(route) = greedy_route_to_embedding(
            storage, adjacency, start_node, &target_coords, &self.greedy_config(),
        ) {
            paths.push(route_to_reasoning_path(route, "embedding_greedy"));
        }

        paths.truncate(self.config.max_alternative_paths);

        let (inference_type, gcs, radial_gradient, causal_fraction, synthesis_coords) =
            if let Some(best) = paths.first() {
                self.analyze_path(storage, adjacency, &best.node_ids)
            } else {
                (None, None, None, None, None)
            };

        let explanation = self.build_explanation(&paths, &inference_type, &ego_summary);

        Ok(ReasoningResult {
            query_type: "embedding_query".into(),
            ego_summary,
            paths,
            inference_type,
            gcs,
            radial_gradient,
            causal_fraction,
            synthesis_coords,
            explanation,
            duration_us: t0.elapsed().as_micros() as u64,
        })
    }

    // ── Internal helpers ─────────────────────────────

    fn greedy_config(&self) -> GreedyRouteConfig {
        GreedyRouteConfig {
            energy_min: self.config.energy_min,
            max_hops: self.config.max_hops,
            astar_fallback: self.config.astar_fallback,
            astar_max_nodes: 2000,
        }
    }

    fn dijkstra_config(&self) -> DijkstraConfig {
        DijkstraConfig {
            energy_min: self.config.energy_min,
            max_nodes: 2000,
            max_distance: f64::MAX,
        }
    }

    fn diffusion_config(&self) -> DiffusionConfig {
        DiffusionConfig {
            steps: self.config.diffusion_steps,
            energy_bias: self.config.diffusion_energy_bias,
            seed: None,
        }
    }

    fn build_ego_summary(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        node_id: Uuid,
    ) -> Option<EgoSummary> {
        // Try cache first, then build
        let entry = match storage.get_ego(&node_id) {
            Ok(Some(e)) => e,
            _ => {
                match nietzsche_graph::ego_cache::build_ego_entry(
                    node_id, storage, adjacency,
                    self.config.ego_max_depth1,
                    self.config.ego_max_depth2,
                ) {
                    Ok(Some(e)) => {
                        let _ = storage.put_ego(&e);
                        e
                    }
                    _ => return None,
                }
            }
        };

        // Extract top-5 highest-energy neighbors
        let mut all_neighbors: Vec<&EgoNeighbor> = entry.depth1.iter()
            .chain(entry.depth2.iter())
            .collect();
        all_neighbors.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));

        let key_neighbors = all_neighbors.iter().take(5).map(|n| KeyNeighbor {
            id: n.id,
            energy: n.energy,
            depth: n.depth,
            node_type: n.node_type.clone(),
            edge_type: n.edge_type.clone(),
        }).collect();

        Some(EgoSummary {
            root_id: entry.root_id,
            depth1_count: entry.depth1.len(),
            depth2_count: entry.depth2.len(),
            edge_count: entry.edge_count,
            key_neighbors,
        })
    }

    fn compute_path_distance(&self, storage: &GraphStorage, path: &[Uuid]) -> f64 {
        if path.len() < 2 { return 0.0; }

        let mut total = 0.0;
        let mut prev_emb: Option<Vec<f64>> = None;

        for id in path {
            if let Ok(Some(node)) = storage.get_node(id) {
                let coords: Vec<f64> = node.embedding.coords.iter().map(|&c| c as f64).collect();
                if let Some(ref prev) = prev_emb {
                    total += poincare_distance_f64(prev, &coords);
                }
                prev_emb = Some(coords);
            }
        }
        total
    }

    fn analyze_path(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        path: &[Uuid],
    ) -> (Option<String>, Option<f64>, Option<f64>, Option<f64>, Option<Vec<f64>>) {
        if path.len() < 2 {
            return (None, None, None, None, None);
        }

        // Load embeddings for GCS computation
        let mut embeddings: Vec<Vec<f64>> = Vec::new();
        for id in path {
            match storage.get_node(id) {
                Ok(Some(node)) => {
                    embeddings.push(node.embedding.coords.iter().map(|&c| c as f64).collect());
                }
                _ => {
                    // Use zero embedding as fallback
                    let dim = embeddings.first().map(|e| e.len()).unwrap_or(128);
                    embeddings.push(vec![0.0; dim]);
                }
            }
        }

        // Validate trajectory (compute GCS)
        let gcs_config = GcsConfig::default();
        let trajectory = match validate_trajectory(path, &embeddings, &gcs_config) {
            Ok(t) => t,
            Err(_) => return (None, None, None, None, None),
        };

        // Collect edge causal info
        let mut edge_infos = Vec::new();
        for i in 0..path.len() - 1 {
            let entries = adjacency.entries_out(&path[i]);
            let causal_type = entries.iter()
                .find(|e| e.neighbor_id == path[i + 1])
                .map(|_| {
                    // Compute basic causal type from timestamps
                    match (storage.get_node_meta(&path[i]), storage.get_node_meta(&path[i + 1])) {
                        (Ok(Some(a)), Ok(Some(b))) => {
                            if a.created_at < b.created_at {
                                CausalType::Timelike
                            } else if a.created_at == b.created_at {
                                CausalType::Lightlike
                            } else {
                                CausalType::Spacelike
                            }
                        }
                        _ => CausalType::Spacelike,
                    }
                })
                .unwrap_or(CausalType::Spacelike);
            edge_infos.push(EdgeInfo { causal_type });
        }

        // Run inference engine
        let rationale = match self.inference.analyze(&trajectory, &edge_infos, None) {
            Ok(r) => r,
            Err(_) => return (Some("error".into()), None, None, None, None),
        };

        // Attempt synthesis if dialectical
        let synthesis_coords = if rationale.inference_type == InferenceType::DialecticalSynthesis
            && self.config.auto_synthesize
        {
            self.attempt_synthesis(storage, path)
        } else {
            None
        };

        (
            Some(format!("{}", rationale.inference_type)),
            Some(rationale.gcs),
            Some(trajectory.radial_gradient),
            Some(if edge_infos.is_empty() { 0.0 } else {
                edge_infos.iter().filter(|e| e.causal_type == CausalType::Timelike).count() as f64
                    / edge_infos.len() as f64
            }),
            synthesis_coords,
        )
    }

    fn attempt_synthesis(&self, storage: &GraphStorage, path: &[Uuid]) -> Option<Vec<f64>> {
        // Collect embeddings for synthesis
        let mut embs: Vec<Vec<f64>> = Vec::new();
        for id in path {
            if let Ok(Some(node)) = storage.get_node(id) {
                embs.push(node.embedding.coords.iter().map(|&c| c as f64).collect());
            }
        }
        if embs.len() < 2 { return None; }

        // Compute Fréchet mean on Riemann sphere
        use nietzsche_hyp_ops::riemann;
        let refs: Vec<&[f64]> = embs.iter().map(|e| e.as_slice()).collect();
        match riemann::synthesis_multi(&refs) {
            Ok(coords) => {
                // Sanitize to Poincaré ball
                let sanitized = nietzsche_hyp_ops::manifold::sanitize_poincare_f64(&coords);
                Some(sanitized)
            }
            Err(_) => None,
        }
    }

    fn build_explanation(
        &self,
        paths: &[ReasoningPath],
        inference_type: &Option<String>,
        ego: &Option<EgoSummary>,
    ) -> String {
        let mut parts = Vec::new();

        if let Some(ref ego) = ego {
            parts.push(format!(
                "Ego subgraph: {} direct + {} depth-2 neighbors, {} edges",
                ego.depth1_count, ego.depth2_count, ego.edge_count
            ));
        }

        if paths.is_empty() {
            parts.push("No reasoning paths found.".into());
        } else {
            parts.push(format!("{} paths discovered:", paths.len()));
            for (i, p) in paths.iter().enumerate() {
                let status = if p.reached_target { "reached" } else { "exploratory" };
                parts.push(format!(
                    "  [{}] {} — {} hops, dist={:.4}, {}",
                    i + 1, p.method, p.hop_count, p.total_distance, status
                ));
            }
        }

        if let Some(ref itype) = inference_type {
            parts.push(format!("Inference: {}", itype));
        }

        parts.join("\n")
    }
}

// ─────────────────────────────────────────────
// Conversion helpers
// ─────────────────────────────────────────────

fn concept_path_to_reasoning_path(cp: ConceptPath, method: &str) -> ReasoningPath {
    let hops: Vec<PathAnnotation> = cp.hops.iter().map(|h| PathAnnotation {
        node_id: h.node_id,
        node_type: format!("{:?}", h.node_type),
        energy: h.energy,
        depth: h.depth,
        summary: h.summary.clone(),
        distance_from_prev: h.distance_from_prev,
    }).collect();

    ReasoningPath {
        node_ids: cp.hops.iter().map(|h| h.node_id).collect(),
        method: method.into(),
        total_distance: cp.total_distance,
        hop_count: cp.hop_count,
        reached_target: cp.reached_target,
        hops,
    }
}

fn route_to_reasoning_path(route: RouteResult, method: &str) -> ReasoningPath {
    ReasoningPath {
        hop_count: route.path.len().saturating_sub(1),
        node_ids: route.path,
        method: method.into(),
        total_distance: route.final_distance,
        reached_target: route.reached_target,
        hops: vec![],
    }
}

/// Poincaré ball distance in f64.
fn poincare_distance_f64(u: &[f64], v: &[f64]) -> f64 {
    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let norm_u_sq: f64 = u.iter().map(|x| x.powi(2)).sum();
    let norm_v_sq: f64 = v.iter().map(|x| x.powi(2)).sum();

    let denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq);
    if denom <= 0.0 { return f64::MAX; }

    let arg = 1.0 + 2.0 * diff_sq / denom;
    arg.max(1.0).acosh()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = ReasoningConfig::default();
        assert_eq!(cfg.max_hops, 20);
        assert_eq!(cfg.max_alternative_paths, 3);
        assert!(cfg.auto_synthesize);
        assert!(cfg.astar_fallback);
    }

    #[test]
    fn poincare_distance_origin() {
        let u = vec![0.0, 0.0, 0.0];
        let v = vec![0.1, 0.0, 0.0];
        let d = poincare_distance_f64(&u, &v);
        assert!(d > 0.0);
        assert!(d < 1.0); // small distance near origin
    }

    #[test]
    fn poincare_distance_self() {
        let u = vec![0.3, 0.2, 0.1];
        let d = poincare_distance_f64(&u, &u);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn concept_path_conversion() {
        let cp = ConceptPath {
            hops: vec![PathHop {
                node_id: Uuid::from_u128(1),
                node_type: NodeType::Semantic,
                energy: 0.8,
                depth: 0.3,
                generation: 0,
                summary: "test".into(),
                distance_from_prev: 0.0,
                distance_to_target: 0.5,
            }],
            total_distance: 0.5,
            hop_count: 0,
            reached_target: true,
            used_fallback: false,
            residual_distance: 0.0,
        };
        let rp = concept_path_to_reasoning_path(cp, "test");
        assert_eq!(rp.method, "test");
        assert!(rp.reached_target);
        assert_eq!(rp.hops.len(), 1);
    }

    #[test]
    fn reasoning_result_serializes() {
        let result = ReasoningResult {
            query_type: "node_to_node".into(),
            ego_summary: None,
            paths: vec![],
            inference_type: Some("Generalization".into()),
            gcs: Some(0.85),
            radial_gradient: Some(-0.15),
            causal_fraction: Some(0.6),
            synthesis_coords: None,
            explanation: "test".into(),
            duration_us: 1234,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Generalization"));
        assert!(json.contains("1234"));
    }
}
