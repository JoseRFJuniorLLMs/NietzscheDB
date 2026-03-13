// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XI.5 + XI.7 — Hyperbolic Space Export with Angular Clustering.
//!
//! Exports the graph's Poincaré ball embedding as JSON suitable for
//! rendering in a 2-D Poincaré disk visualizer (WebGL, D3, perspektive.js).
//!
//! ## Projection
//!
//! High-dimensional embeddings (3072-D) are projected to 2-D using the
//! first two principal components of the radial + angular decomposition:
//!
//! - **r** = ‖embedding‖ (radial distance, preserves depth hierarchy)
//! - **θ** = atan2(PC2, PC1) (angular position from first 2 components)
//!
//! This preserves the most important structural information: radial
//! stratification (rings) and angular clustering (semantic continents).
//!
//! ## Angular Clustering (Phase XI.7)
//!
//! Detects **semantic continents** — dense angular sectors in the Poincaré
//! disk. Each sector represents a knowledge domain. Algorithm:
//!
//! 1. Histogram of θ values → angular density
//! 2. Gaussian kernel smoothing (eliminates noise)
//! 3. Peak detection (density > mean × threshold)
//! 4. Sector expansion until density drops below mean
//! 5. Hub labeling: highest `energy × log(degree + 1)` node names the sector
//!
//! ## Output format
//!
//! ```json
//! {
//!   "metadata": { ... },
//!   "nodes": [ { ..., "sector_id": 0 } ],
//!   "edges": [ ... ],
//!   "rings": [ ... ],
//!   "sectors": [
//!     { "id": 0, "theta_min": 0.2, "theta_max": 0.7,
//!       "node_count": 134, "hub_node": "uuid", "label": "machine learning" }
//!   ]
//! }
//! ```

use serde::Serialize;
use uuid::Uuid;

use crate::adjacency::AdjacencyIndex;
use crate::error::GraphError;
use crate::storage::GraphStorage;

// ─────────────────────────────────────────────
// Export types
// ─────────────────────────────────────────────

/// Full export of the hyperbolic space for visualization.
#[derive(Debug, Clone, Serialize)]
pub struct HyperbolicExport {
    pub metadata: ExportMetadata,
    pub nodes: Vec<ExportNode>,
    pub edges: Vec<ExportEdge>,
    pub rings: Vec<RadialBin>,
    pub sectors: Vec<AngularSector>,
}

/// Summary metadata for the export.
#[derive(Debug, Clone, Serialize)]
pub struct ExportMetadata {
    pub node_count: usize,
    pub edge_count: usize,
    pub dimension: usize,
    pub mean_radius: f64,
    pub max_radius: f64,
    pub ring_count: usize,
    pub sector_count: usize,
}

/// A node projected to the 2-D Poincaré disk.
#[derive(Debug, Clone, Serialize)]
pub struct ExportNode {
    pub id: String,
    /// X coordinate in the Poincaré disk (2-D projection).
    pub x: f64,
    /// Y coordinate in the Poincaré disk (2-D projection).
    pub y: f64,
    /// Original radial distance ‖embedding‖.
    pub r: f64,
    /// Angular position (radians) from the 2-D projection.
    pub theta: f64,
    /// Node type.
    pub node_type: String,
    /// Energy level [0, 1].
    pub energy: f32,
    /// L-System generation.
    pub generation: u32,
    /// Hausdorff local dimension.
    pub hausdorff: f32,
    /// Content summary (truncated).
    pub label: String,
    /// Outgoing degree.
    pub degree_out: usize,
    /// Incoming degree.
    pub degree_in: usize,
    /// Angular sector ID (None if outside any sector).
    pub sector_id: Option<usize>,
}

/// A semantic continent — a dense angular sector in the Poincaré disk.
#[derive(Debug, Clone, Serialize)]
pub struct AngularSector {
    /// Sector index (0-based).
    pub id: usize,
    /// Start angle (radians, in [−π, π]).
    pub theta_min: f64,
    /// End angle (radians, in [−π, π]).
    pub theta_max: f64,
    /// Number of nodes in this sector.
    pub node_count: usize,
    /// Hub node ID (highest semantic mass in sector).
    pub hub_node: Option<String>,
    /// Auto-generated label from hub node's content.
    pub label: String,
    /// Mean energy of nodes in this sector.
    pub mean_energy: f32,
    /// Mean radial distance of nodes in this sector.
    pub mean_radius: f64,
}

/// An edge in the export.
#[derive(Debug, Clone, Serialize)]
pub struct ExportEdge {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub edge_type: String,
}

/// A radial density bin for ring visualization.
#[derive(Debug, Clone, Serialize)]
pub struct RadialBin {
    pub r_min: f64,
    pub r_max: f64,
    pub count: usize,
    pub avg_energy: f32,
    pub is_peak: bool,
}

// ─────────────────────────────────────────────
// Export configuration
// ─────────────────────────────────────────────

/// Configuration for the hyperbolic export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Maximum nodes to export (0 = all).
    pub max_nodes: usize,
    /// Minimum energy to include a node (filters noise).
    pub energy_min: f32,
    /// Include edges in the export.
    pub include_edges: bool,
    /// Number of radial bins for the ring histogram.
    pub radial_bins: usize,
    /// Maximum label length.
    pub max_label_len: usize,
    /// Number of angular bins for sector detection (default 360).
    pub angular_bins: usize,
    /// Sector detection threshold: density must exceed mean × threshold (default 1.5).
    pub sector_threshold: f64,
    /// Minimum nodes to consider an angular region a sector (default 5).
    pub min_sector_nodes: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            max_nodes: 0,
            energy_min: 0.0,
            include_edges: true,
            radial_bins: 50,
            max_label_len: 60,
            angular_bins: 360,
            sector_threshold: 1.5,
            min_sector_nodes: 5,
        }
    }
}

// ─────────────────────────────────────────────
// Export function
// ─────────────────────────────────────────────

/// Export the hyperbolic graph space for 2-D Poincaré disk visualization.
///
/// Projects high-dimensional embeddings to 2-D using the first two
/// coordinate components, preserving radial structure (depth hierarchy)
/// and angular clustering (semantic continents).
///
/// ## Performance
///
/// Streams nodes via iterator — does not load all embeddings into memory
/// at once. For graphs with >100K nodes, use `config.max_nodes` to cap
/// the export size.
pub fn export_hyperbolic_space(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &ExportConfig,
) -> Result<HyperbolicExport, GraphError> {
    let mut nodes = Vec::new();
    let mut radii = Vec::new();
    let mut dim = 0usize;

    // ── Scan nodes ──────────────────────────────────────────
    for result in storage.iter_nodes() {
        let node = result?;

        if node.energy < config.energy_min {
            continue;
        }

        let coords = &node.embedding.coords;
        if dim == 0 && !coords.is_empty() {
            dim = coords.len();
        }

        // Radial distance (original high-D norm)
        let r: f64 = coords.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();

        // 2-D projection: use first two components for angle
        let (c0, c1) = if coords.len() >= 2 {
            (coords[0] as f64, coords[1] as f64)
        } else if coords.len() == 1 {
            (coords[0] as f64, 0.0)
        } else {
            (0.0, 0.0)
        };

        // Angular position from the 2-D projection
        let theta = c1.atan2(c0);

        // Project to 2-D Poincaré disk: preserve r, use theta for direction
        let x = r * theta.cos();
        let y = r * theta.sin();

        let label = crate::concept_path::extract_summary_public(&node.meta.content, config.max_label_len);

        let degree_out = adjacency.degree_out(&node.id);
        let degree_in = adjacency.degree_in(&node.id);

        nodes.push(ExportNode {
            id: node.id.to_string(),
            x,
            y,
            r,
            theta,
            node_type: format!("{:?}", node.meta.node_type),
            energy: node.energy,
            generation: node.lsystem_generation,
            hausdorff: node.hausdorff_local,
            label,
            degree_out,
            degree_in,
            sector_id: None, // assigned after clustering
        });

        radii.push(r);

        if config.max_nodes > 0 && nodes.len() >= config.max_nodes {
            break;
        }
    }

    // ── Radial histogram (ring detection) ───────────────────
    let max_r = radii.iter().cloned().fold(0.0_f64, f64::max);
    let mean_r = if radii.is_empty() { 0.0 } else {
        radii.iter().sum::<f64>() / radii.len() as f64
    };

    let bin_width = if max_r > 0.0 { max_r / config.radial_bins as f64 } else { 1.0 };
    let mut bins: Vec<(usize, f32)> = vec![(0, 0.0); config.radial_bins];

    for (i, node) in nodes.iter().enumerate() {
        let bin_idx = ((node.r / bin_width) as usize).min(config.radial_bins - 1);
        bins[bin_idx].0 += 1;
        bins[bin_idx].1 += node.energy;
    }

    // Detect peaks (local maxima in the histogram)
    let counts: Vec<usize> = bins.iter().map(|(c, _)| *c).collect();
    let rings: Vec<RadialBin> = (0..config.radial_bins)
        .map(|i| {
            let r_min = i as f64 * bin_width;
            let r_max = r_min + bin_width;
            let (count, energy_sum) = bins[i];
            let avg_energy = if count > 0 { energy_sum / count as f32 } else { 0.0 };
            let is_peak = is_local_peak(&counts, i);
            RadialBin { r_min, r_max, count, avg_energy, is_peak }
        })
        .collect();

    let ring_count = rings.iter().filter(|r| r.is_peak).count();

    // ── Edges ───────────────────────────────────────────────
    let mut edges = Vec::new();
    if config.include_edges {
        let node_ids: std::collections::HashSet<String> =
            nodes.iter().map(|n| n.id.clone()).collect();

        for result in storage.iter_edges() {
            let edge = result?;
            let from = edge.from.to_string();
            let to = edge.to.to_string();
            // Only include edges where both endpoints are in the export
            if node_ids.contains(&from) && node_ids.contains(&to) {
                edges.push(ExportEdge {
                    from,
                    to,
                    weight: edge.weight,
                    edge_type: format!("{:?}", edge.edge_type),
                });
            }
        }
    }

    // ── Angular clustering (sector detection) ──────────────
    let sectors = detect_angular_sectors(&mut nodes, config);
    let sector_count = sectors.len();

    Ok(HyperbolicExport {
        metadata: ExportMetadata {
            node_count: nodes.len(),
            edge_count: edges.len(),
            dimension: dim,
            mean_radius: mean_r,
            max_radius: max_r,
            ring_count,
            sector_count,
        },
        nodes,
        edges,
        rings,
        sectors,
    })
}

// ─────────────────────────────────────────────
// Angular clustering (Phase XI.7)
// ─────────────────────────────────────────────

/// Detect semantic continents via angular density clustering.
///
/// Algorithm:
/// 1. Build angular histogram (θ ∈ [−π, π] mapped to bins)
/// 2. Smooth with a 3-tap Gaussian kernel [0.25, 0.5, 0.25]
/// 3. Find contiguous runs where smoothed density > mean × threshold
/// 4. Merge adjacent runs, filter by min_sector_nodes
/// 5. Label each sector by its highest-mass hub node
///
/// Mutates `nodes` to assign `sector_id` per node.
fn detect_angular_sectors(
    nodes: &mut [ExportNode],
    config: &ExportConfig,
) -> Vec<AngularSector> {
    if nodes.is_empty() || config.angular_bins == 0 {
        return Vec::new();
    }

    let n_bins = config.angular_bins;
    let bin_width = std::f64::consts::TAU / n_bins as f64;

    // Step 1: angular histogram
    let mut histogram = vec![0usize; n_bins];
    for node in nodes.iter() {
        // Map θ from [−π, π] to [0, 2π) then to bin index
        let theta_pos = if node.theta < 0.0 {
            node.theta + std::f64::consts::TAU
        } else {
            node.theta
        };
        let bin = ((theta_pos / bin_width) as usize).min(n_bins - 1);
        histogram[bin] += 1;
    }

    // Step 2: smooth with [0.25, 0.5, 0.25] kernel (circular)
    let mut smoothed = vec![0.0f64; n_bins];
    for i in 0..n_bins {
        let left = if i == 0 { n_bins - 1 } else { i - 1 };
        let right = if i == n_bins - 1 { 0 } else { i + 1 };
        smoothed[i] = histogram[left] as f64 * 0.25
            + histogram[i] as f64 * 0.5
            + histogram[right] as f64 * 0.25;
    }

    // Mean density
    let total: f64 = smoothed.iter().sum();
    let mean_density = total / n_bins as f64;

    if mean_density < 1e-9 {
        return Vec::new();
    }

    let threshold = mean_density * config.sector_threshold;

    // Step 3: find contiguous runs above threshold
    // We handle circularity by scanning twice the array length
    let mut runs: Vec<(usize, usize)> = Vec::new(); // (start_bin, end_bin) inclusive
    let mut in_run = false;
    let mut run_start = 0usize;

    for i in 0..n_bins {
        if smoothed[i] >= threshold {
            if !in_run {
                run_start = i;
                in_run = true;
            }
        } else if in_run {
            runs.push((run_start, i - 1));
            in_run = false;
        }
    }
    if in_run {
        runs.push((run_start, n_bins - 1));
    }

    // Handle circular wrap-around: if first and last runs touch the boundary, merge them
    if runs.len() >= 2 {
        let first = runs[0];
        let last = runs[runs.len() - 1];
        if first.0 == 0 && last.1 == n_bins - 1 {
            let merged_start = last.0;
            let merged_end = first.1;
            runs.remove(0);
            let last_idx = runs.len() - 1;
            runs[last_idx] = (merged_start, merged_end);
        }
    }

    // Step 4: convert runs to sectors, filter by min_sector_nodes
    let mut sectors = Vec::new();
    let mut sector_id = 0usize;

    for (start_bin, end_bin) in &runs {
        let (start_bin, end_bin) = (*start_bin, *end_bin);

        // Convert bins back to theta range
        let theta_min = bin_to_theta(start_bin, bin_width);
        let theta_max = bin_to_theta_end(end_bin, bin_width);

        // Count nodes and find hub
        let mut count = 0usize;
        let mut energy_sum = 0.0f32;
        let mut radius_sum = 0.0f64;
        let mut best_hub_score = -1.0f64;
        let mut best_hub_idx: Option<usize> = None;

        for (ni, node) in nodes.iter().enumerate() {
            if angle_in_sector(node.theta, theta_min, theta_max) {
                count += 1;
                energy_sum += node.energy;
                radius_sum += node.r;

                // Semantic mass: energy × log(degree + 1)
                let degree = node.degree_out + node.degree_in;
                let mass = node.energy as f64 * ((degree as f64) + 1.0).ln();
                if mass > best_hub_score {
                    best_hub_score = mass;
                    best_hub_idx = Some(ni);
                }
            }
        }

        if count < config.min_sector_nodes {
            continue;
        }

        let hub_node = best_hub_idx.map(|i| nodes[i].id.clone());
        let label = best_hub_idx
            .map(|i| nodes[i].label.clone())
            .unwrap_or_default();

        sectors.push(AngularSector {
            id: sector_id,
            theta_min,
            theta_max,
            node_count: count,
            hub_node,
            label,
            mean_energy: if count > 0 { energy_sum / count as f32 } else { 0.0 },
            mean_radius: if count > 0 { radius_sum / count as f64 } else { 0.0 },
        });

        // Assign sector_id to nodes
        for node in nodes.iter_mut() {
            if node.sector_id.is_none() && angle_in_sector(node.theta, theta_min, theta_max) {
                node.sector_id = Some(sector_id);
            }
        }

        sector_id += 1;
    }

    sectors
}

/// Map bin index to theta start (in [−π, π]).
fn bin_to_theta(bin: usize, bin_width: f64) -> f64 {
    let theta_pos = bin as f64 * bin_width; // [0, 2π)
    if theta_pos > std::f64::consts::PI {
        theta_pos - std::f64::consts::TAU
    } else {
        theta_pos
    }
}

/// Map bin index to theta end (in [−π, π]).
fn bin_to_theta_end(bin: usize, bin_width: f64) -> f64 {
    let theta_pos = (bin + 1) as f64 * bin_width; // (0, 2π]
    if theta_pos > std::f64::consts::PI {
        theta_pos - std::f64::consts::TAU
    } else {
        theta_pos
    }
}

/// Check if angle θ falls within sector [theta_min, theta_max].
/// Handles wrap-around (when theta_min > theta_max, sector crosses −π/π boundary).
fn angle_in_sector(theta: f64, theta_min: f64, theta_max: f64) -> bool {
    if theta_min <= theta_max {
        theta >= theta_min && theta <= theta_max
    } else {
        // Wraps around: sector covers [theta_min, π] ∪ [−π, theta_max]
        theta >= theta_min || theta <= theta_max
    }
}

/// Check if bin `i` is a local peak (more nodes than both neighbours).
fn is_local_peak(counts: &[usize], i: usize) -> bool {
    if counts[i] < 3 { return false; } // minimum significance

    let left = if i > 0 { counts[i - 1] } else { 0 };
    let right = if i + 1 < counts.len() { counts[i + 1] } else { 0 };

    counts[i] > left && counts[i] > right
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::AdjacencyIndex;
    use crate::model::{Edge, Node, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_storage(dir: &TempDir) -> GraphStorage {
        let p = dir.path().join("rocksdb");
        GraphStorage::open(p.to_str().unwrap()).unwrap()
    }

    fn node_at_2d(x: f64, y: f64, name: &str) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, y as f32]),
            serde_json::json!({ "name": name }),
        )
    }

    fn build_test_graph(dir: &TempDir) -> (GraphStorage, AdjacencyIndex, Vec<Node>) {
        let storage = open_storage(dir);
        let adjacency = AdjacencyIndex::new();

        // Create nodes at different radii (simulating ring structure)
        let nodes = vec![
            node_at_2d(0.1, 0.05, "Core Axiom"),       // r ≈ 0.11
            node_at_2d(0.08, -0.06, "Foundation"),       // r ≈ 0.10
            node_at_2d(0.35, 0.2, "Category A"),         // r ≈ 0.40
            node_at_2d(-0.3, 0.25, "Category B"),        // r ≈ 0.39
            node_at_2d(0.4, -0.1, "Category C"),         // r ≈ 0.41
            node_at_2d(0.7, 0.3, "Episodic 1"),          // r ≈ 0.76
            node_at_2d(-0.6, 0.5, "Episodic 2"),         // r ≈ 0.78
            node_at_2d(0.65, -0.4, "Episodic 3"),        // r ≈ 0.76
        ];

        for n in &nodes { storage.put_node(n).unwrap(); }

        // Chain: core → categories → episodics
        let edges = vec![
            Edge::association(nodes[0].id, nodes[2].id, 0.9),
            Edge::association(nodes[0].id, nodes[3].id, 0.8),
            Edge::association(nodes[1].id, nodes[4].id, 0.85),
            Edge::association(nodes[2].id, nodes[5].id, 0.7),
            Edge::association(nodes[3].id, nodes[6].id, 0.6),
            Edge::association(nodes[4].id, nodes[7].id, 0.75),
            // Cross-links between categories
            Edge::association(nodes[2].id, nodes[3].id, 0.5),
        ];

        for e in &edges {
            storage.put_edge(e).unwrap();
            adjacency.add_edge(e);
        }

        (storage, adjacency, nodes)
    }

    #[test]
    fn export_basic_graph() {
        let dir = tmp();
        let (storage, adjacency, nodes) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency, &ExportConfig::default(),
        ).unwrap();

        assert_eq!(export.metadata.node_count, 8);
        assert_eq!(export.metadata.edge_count, 7);
        assert_eq!(export.metadata.dimension, 2);
        assert!(export.metadata.mean_radius > 0.0);
        assert!(export.metadata.max_radius > 0.0);
    }

    #[test]
    fn export_nodes_have_valid_coordinates() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency, &ExportConfig::default(),
        ).unwrap();

        for node in &export.nodes {
            let norm = (node.x * node.x + node.y * node.y).sqrt();
            assert!(norm < 1.0, "node {} outside disk: norm={:.4}", node.label, norm);
            assert!((norm - node.r).abs() < 0.01,
                "norm={:.4} should match r={:.4}", norm, node.r);
            assert!(!node.id.is_empty());
            assert!(!node.label.is_empty());
        }
    }

    #[test]
    fn export_detects_rings() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency,
            &ExportConfig { radial_bins: 10, ..Default::default() },
        ).unwrap();

        // With nodes at r≈0.1, r≈0.4, r≈0.76, we should see peaks
        assert!(export.rings.len() == 10);
        let peak_count = export.rings.iter().filter(|r| r.is_peak).count();
        assert!(peak_count >= 2, "expected at least 2 radial peaks, got {peak_count}");
    }

    #[test]
    fn export_max_nodes_limits_output() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency,
            &ExportConfig { max_nodes: 3, ..Default::default() },
        ).unwrap();

        assert!(export.metadata.node_count <= 3);
    }

    #[test]
    fn export_energy_filter() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let mut alive = node_at_2d(0.3, 0.1, "Alive");
        alive.energy = 0.8;
        let mut dead = node_at_2d(0.5, 0.2, "Dead");
        dead.energy = 0.0;

        storage.put_node(&alive).unwrap();
        storage.put_node(&dead).unwrap();

        let export = export_hyperbolic_space(
            &storage, &adjacency,
            &ExportConfig { energy_min: 0.1, ..Default::default() },
        ).unwrap();

        assert_eq!(export.metadata.node_count, 1);
        assert_eq!(export.nodes[0].label, "Alive");
    }

    #[test]
    fn export_without_edges() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency,
            &ExportConfig { include_edges: false, ..Default::default() },
        ).unwrap();

        assert_eq!(export.metadata.edge_count, 0);
        assert!(export.edges.is_empty());
    }

    #[test]
    fn export_serializes_to_json() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency, &ExportConfig::default(),
        ).unwrap();

        let json = serde_json::to_string_pretty(&export).unwrap();
        assert!(json.contains("\"node_count\""));
        assert!(json.contains("\"Core Axiom\""));
        assert!(json.contains("\"r_min\""));
    }

    #[test]
    fn export_preserves_ring_structure() {
        let dir = tmp();
        let (storage, adjacency, _) = build_test_graph(&dir);

        let export = export_hyperbolic_space(
            &storage, &adjacency, &ExportConfig::default(),
        ).unwrap();

        // Core nodes (r < 0.2) should exist
        let cores: Vec<_> = export.nodes.iter().filter(|n| n.r < 0.2).collect();
        assert!(!cores.is_empty(), "should have core nodes near center");

        // Peripheral nodes (r > 0.6) should exist
        let periphery: Vec<_> = export.nodes.iter().filter(|n| n.r > 0.6).collect();
        assert!(!periphery.is_empty(), "should have peripheral nodes");

        // Core should have higher degree than periphery (hub structure)
        let core_avg_degree: f64 = cores.iter()
            .map(|n| n.degree_out as f64)
            .sum::<f64>() / cores.len() as f64;
        let periph_avg_degree: f64 = periphery.iter()
            .map(|n| n.degree_out as f64)
            .sum::<f64>() / periphery.len() as f64;

        assert!(core_avg_degree >= periph_avg_degree,
            "core avg degree ({core_avg_degree:.1}) should >= periphery ({periph_avg_degree:.1})");
    }

    // ── Angular clustering tests ─────────────────────────────

    /// Build a graph with two clear angular clusters for sector detection.
    fn build_clustered_graph(dir: &TempDir) -> (GraphStorage, AdjacencyIndex) {
        let storage = open_storage(dir);
        let adjacency = AdjacencyIndex::new();

        // Cluster A: θ ≈ 0.5 rad (upper-right quadrant), 10 nodes
        for i in 0..10 {
            let r = 0.3 + (i as f64) * 0.03;
            let theta = 0.4 + (i as f64) * 0.02; // θ ∈ [0.4, 0.58]
            let x = r * theta.cos();
            let y = r * theta.sin();
            let mut n = node_at_2d(x, y, &format!("ML-{i}"));
            n.energy = 0.7 + (i as f32) * 0.02;
            storage.put_node(&n).unwrap();
        }

        // Cluster B: θ ≈ −2.0 rad (lower-left quadrant), 8 nodes
        for i in 0..8 {
            let r = 0.4 + (i as f64) * 0.02;
            let theta = -2.1 + (i as f64) * 0.025; // θ ∈ [−2.1, −1.925]
            let x = r * theta.cos();
            let y = r * theta.sin();
            let mut n = node_at_2d(x, y, &format!("Physics-{i}"));
            n.energy = 0.6 + (i as f32) * 0.03;
            storage.put_node(&n).unwrap();
        }

        // Sparse noise: 3 random nodes spread out
        let mut n1 = node_at_2d(0.2, -0.1, "Noise-1");
        n1.energy = 0.3;
        let mut n2 = node_at_2d(-0.1, 0.05, "Noise-2");
        n2.energy = 0.2;
        let mut n3 = node_at_2d(0.05, -0.3, "Noise-3");
        n3.energy = 0.4;
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_node(&n3).unwrap();

        (storage, adjacency)
    }

    #[test]
    fn angular_clustering_detects_two_sectors() {
        let dir = tmp();
        let (storage, adjacency) = build_clustered_graph(&dir);

        let config = ExportConfig {
            angular_bins: 72, // 5° resolution
            sector_threshold: 1.5,
            min_sector_nodes: 3,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();

        assert!(
            export.sectors.len() >= 2,
            "expected at least 2 sectors, got {}",
            export.sectors.len()
        );
        assert!(export.metadata.sector_count >= 2);
    }

    #[test]
    fn sectors_have_valid_ranges() {
        let dir = tmp();
        let (storage, adjacency) = build_clustered_graph(&dir);

        let config = ExportConfig {
            angular_bins: 72,
            sector_threshold: 1.5,
            min_sector_nodes: 3,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();

        for sector in &export.sectors {
            assert!(sector.node_count >= 3, "sector {} too small: {}", sector.id, sector.node_count);
            assert!(sector.mean_energy > 0.0, "sector {} has zero energy", sector.id);
            assert!(sector.mean_radius > 0.0, "sector {} has zero radius", sector.id);
            assert!(!sector.label.is_empty(), "sector {} has no label", sector.id);
        }
    }

    #[test]
    fn nodes_assigned_to_sectors() {
        let dir = tmp();
        let (storage, adjacency) = build_clustered_graph(&dir);

        let config = ExportConfig {
            angular_bins: 72,
            sector_threshold: 1.5,
            min_sector_nodes: 3,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();

        // At least the clustered nodes should have sector_id assigned
        let assigned: Vec<_> = export.nodes.iter().filter(|n| n.sector_id.is_some()).collect();
        assert!(
            assigned.len() >= 10,
            "expected at least 10 nodes assigned to sectors, got {}",
            assigned.len()
        );
    }

    #[test]
    fn hub_label_from_highest_mass() {
        let dir = tmp();
        let (storage, adjacency) = build_clustered_graph(&dir);

        let config = ExportConfig {
            angular_bins: 72,
            sector_threshold: 1.5,
            min_sector_nodes: 3,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();

        // Each sector should have a hub node
        for sector in &export.sectors {
            assert!(sector.hub_node.is_some(), "sector {} has no hub", sector.id);
        }
    }

    #[test]
    fn no_sectors_when_uniform_distribution() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        // Place nodes evenly around the circle (uniform θ)
        for i in 0..36 {
            let theta = (i as f64) * std::f64::consts::TAU / 36.0 - std::f64::consts::PI;
            let r = 0.5;
            let x = r * theta.cos();
            let y = r * theta.sin();
            let n = node_at_2d(x, y, &format!("Uniform-{i}"));
            storage.put_node(&n).unwrap();
        }

        let config = ExportConfig {
            angular_bins: 36,
            sector_threshold: 2.0, // strict threshold
            min_sector_nodes: 5,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();

        // Uniform distribution should produce no sectors above 2.0× mean
        assert_eq!(
            export.sectors.len(), 0,
            "uniform distribution should have no sectors, got {}",
            export.sectors.len()
        );
    }

    #[test]
    fn sectors_in_json_output() {
        let dir = tmp();
        let (storage, adjacency) = build_clustered_graph(&dir);

        let config = ExportConfig {
            angular_bins: 72,
            sector_threshold: 1.5,
            min_sector_nodes: 3,
            ..Default::default()
        };

        let export = export_hyperbolic_space(&storage, &adjacency, &config).unwrap();
        let json = serde_json::to_string_pretty(&export).unwrap();

        assert!(json.contains("\"sectors\""), "JSON should contain sectors array");
        assert!(json.contains("\"theta_min\""), "JSON should contain theta_min");
        assert!(json.contains("\"hub_node\""), "JSON should contain hub_node");
        assert!(json.contains("\"sector_id\""), "JSON should contain sector_id on nodes");
        assert!(json.contains("\"sector_count\""), "JSON should contain sector_count in metadata");
    }

    #[test]
    fn angle_in_sector_handles_wraparound() {
        // Normal range: [0.5, 1.5]
        assert!(angle_in_sector(1.0, 0.5, 1.5));
        assert!(!angle_in_sector(2.0, 0.5, 1.5));

        // Wrap-around: [2.5, -2.5] = covers [2.5, π] ∪ [−π, -2.5]
        assert!(angle_in_sector(3.0, 2.5, -2.5));
        assert!(angle_in_sector(-3.0, 2.5, -2.5));
        assert!(!angle_in_sector(0.0, 2.5, -2.5));
    }
}
