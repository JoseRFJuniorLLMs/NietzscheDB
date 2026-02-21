//! Integration tests for nietzsche-cugraph.
//!
//! Tests the CPU-fallback path (feature `cuda` OFF):
//! - CSR construction from AdjacencyIndex
//! - BFS traversal
//! - Dijkstra shortest paths
//! - PageRank centrality
//! - Error handling

use nietzsche_cugraph::{CuGraphIndex, CuGraphError};
use nietzsche_graph::{AdjacencyIndex, Edge, EdgeType};
use uuid::Uuid;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a simple graph:  A → B → C, with given weights.
fn three_chain() -> (AdjacencyIndex, Uuid, Uuid, Uuid) {
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();

    adj.add_edge(&Edge::association(a, b, 1.0));
    adj.add_edge(&Edge::association(b, c, 2.0));

    (adj, a, b, c)
}

/// Build a diamond graph: A → B, A → C, B → D, C → D.
fn diamond_graph() -> (AdjacencyIndex, Uuid, Uuid, Uuid, Uuid) {
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();

    adj.add_edge(&Edge::association(a, b, 1.0));
    adj.add_edge(&Edge::association(a, c, 3.0));
    adj.add_edge(&Edge::association(b, d, 1.0));
    adj.add_edge(&Edge::association(c, d, 1.0));

    (adj, a, b, c, d)
}

/// Build a cycle:  A → B → C → A.
fn triangle_cycle() -> (AdjacencyIndex, Uuid, Uuid, Uuid) {
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();

    adj.add_edge(&Edge::association(a, b, 1.0));
    adj.add_edge(&Edge::association(b, c, 1.0));
    adj.add_edge(&Edge::association(c, a, 1.0));

    (adj, a, b, c)
}

// ══════════════════════════════════════════════════════════════════════════════
// CSR Construction
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn csr_empty_graph() {
    let adj = AdjacencyIndex::new();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    assert_eq!(index.vertex_count(), 0);
    assert_eq!(index.edge_count(), 0);
    assert!(index.node_ids().is_empty());
}

#[test]
fn csr_single_edge() {
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    adj.add_edge(&Edge::association(a, b, 0.5));

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    assert_eq!(index.vertex_count(), 2);
    assert_eq!(index.edge_count(), 1);
    assert!(index.node_ids().contains(&a));
    assert!(index.node_ids().contains(&b));
}

#[test]
fn csr_vertex_ordering_is_deterministic() {
    let adj = AdjacencyIndex::new();
    let mut ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

    for i in 0..9 {
        adj.add_edge(&Edge::association(ids[i], ids[i + 1], 1.0));
    }

    // Build twice and verify same node ordering
    let idx1 = CuGraphIndex::from_adjacency(&adj).unwrap();
    let idx2 = CuGraphIndex::from_adjacency(&adj).unwrap();

    assert_eq!(idx1.node_ids(), idx2.node_ids());
}

#[test]
fn csr_preserves_edge_count() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // A→B and B→C = 2 edges
    assert_eq!(index.vertex_count(), 3);
    assert_eq!(index.edge_count(), 2);
}

#[test]
fn csr_diamond_has_four_edges() {
    let (adj, _, _, _, _) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    assert_eq!(index.vertex_count(), 4);
    assert_eq!(index.edge_count(), 4);
}

#[test]
fn csr_node_ids_sorted() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let ids = index.node_ids();
    for w in ids.windows(2) {
        assert!(w[0] < w[1], "node_ids should be sorted by UUID");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// BFS
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn bfs_chain_visits_all() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.bfs(a, 10).unwrap();

    assert_eq!(result.visited.len(), 3);
    assert_eq!(result.visited[0], a);
    assert_eq!(result.distances[0], 0); // A is at distance 0
}

#[test]
fn bfs_chain_distances() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.bfs(a, 10).unwrap();

    // Find distances by UUID
    let dist_a = result.visited.iter().zip(&result.distances)
        .find(|(id, _)| **id == a).unwrap().1;
    let dist_b = result.visited.iter().zip(&result.distances)
        .find(|(id, _)| **id == b).unwrap().1;
    let dist_c = result.visited.iter().zip(&result.distances)
        .find(|(id, _)| **id == c).unwrap().1;

    assert_eq!(*dist_a, 0);
    assert_eq!(*dist_b, 1);
    assert_eq!(*dist_c, 2);
}

#[test]
fn bfs_max_depth_limits_traversal() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // max_depth=1: should visit A (depth 0) and B (depth 1), but NOT C (depth 2)
    let result = index.bfs(a, 1).unwrap();

    let visited_ids: Vec<Uuid> = result.visited.clone();
    assert!(visited_ids.contains(&a));
    assert!(visited_ids.contains(&b));
    assert!(!visited_ids.contains(&c));
}

#[test]
fn bfs_from_middle_node() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // BFS from B: should visit B and C (B→C exists), but not A (no A→B edge reversed)
    let result = index.bfs(b, 10).unwrap();

    assert!(result.visited.contains(&b));
    assert!(result.visited.contains(&c));
    // A is not reachable from B (directed graph: A→B→C)
    assert!(!result.visited.contains(&a));
}

#[test]
fn bfs_from_leaf_node() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // BFS from C: only C itself (no outgoing edges from C)
    let result = index.bfs(c, 10).unwrap();
    assert_eq!(result.visited.len(), 1);
    assert_eq!(result.visited[0], c);
}

#[test]
fn bfs_nonexistent_node_returns_error() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let ghost = Uuid::new_v4();
    let result = index.bfs(ghost, 10);
    assert!(result.is_err());
}

#[test]
fn bfs_cycle_does_not_loop() {
    let (adj, a, b, c) = triangle_cycle();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.bfs(a, 100).unwrap();

    // Should visit all 3 nodes exactly once
    assert_eq!(result.visited.len(), 3);
}

#[test]
fn bfs_diamond_from_top() {
    let (adj, a, b, c, d) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.bfs(a, 10).unwrap();
    assert_eq!(result.visited.len(), 4);

    // A at depth 0, B and C at depth 1, D at depth 2
    let dist_of = |id: Uuid| -> u32 {
        result.visited.iter().zip(&result.distances)
            .find(|(uid, _)| **uid == id).unwrap().1.clone()
    };
    assert_eq!(dist_of(a), 0);
    assert_eq!(dist_of(b), 1);
    assert_eq!(dist_of(c), 1);
    assert_eq!(dist_of(d), 2);
}

// ══════════════════════════════════════════════════════════════════════════════
// Dijkstra
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn dijkstra_chain() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(a).unwrap();

    assert!((result.distances[&a] - 0.0).abs() < 1e-6);
    assert!((result.distances[&b] - 1.0).abs() < 1e-6);
    assert!((result.distances[&c] - 3.0).abs() < 1e-6); // 1.0 + 2.0
}

#[test]
fn dijkstra_diamond_shortest_path() {
    let (adj, a, b, c, d) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(a).unwrap();

    // A→B→D = 1.0 + 1.0 = 2.0 (shorter than A→C→D = 3.0 + 1.0 = 4.0)
    assert!((result.distances[&a] - 0.0).abs() < 1e-6);
    assert!((result.distances[&b] - 1.0).abs() < 1e-6);
    assert!((result.distances[&c] - 3.0).abs() < 1e-6);
    assert!((result.distances[&d] - 2.0).abs() < 1e-6);
}

#[test]
fn dijkstra_nonexistent_start_returns_error() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(Uuid::new_v4());
    assert!(result.is_err());
}

#[test]
fn dijkstra_leaf_only_reaches_self() {
    let (adj, _a, _b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(c).unwrap();

    // C has no outgoing edges — only C itself is reachable
    assert_eq!(result.distances.len(), 1);
    assert!((result.distances[&c] - 0.0).abs() < 1e-6);
}

#[test]
fn dijkstra_self_distance_is_zero() {
    let (adj, a, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(a).unwrap();
    assert!((result.distances[&a] - 0.0).abs() < 1e-6);
}

#[test]
fn dijkstra_cycle() {
    let (adj, a, b, c) = triangle_cycle();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.dijkstra(a).unwrap();

    assert!((result.distances[&a] - 0.0).abs() < 1e-6);
    assert!((result.distances[&b] - 1.0).abs() < 1e-6);
    assert!((result.distances[&c] - 2.0).abs() < 1e-6);
}

// ══════════════════════════════════════════════════════════════════════════════
// PageRank
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn pagerank_empty_graph() {
    let adj = AdjacencyIndex::new();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 100, 1e-6).unwrap();
    assert!(result.scores.is_empty());
}

#[test]
fn pagerank_scores_sum_to_one() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 100, 1e-8).unwrap();

    let total: f64 = result.scores.values().sum();
    assert!((total - 1.0).abs() < 0.01, "PR scores should sum to ~1.0, got {total}");
}

#[test]
fn pagerank_all_nodes_have_scores() {
    let (adj, a, b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 100, 1e-8).unwrap();

    assert!(result.scores.contains_key(&a));
    assert!(result.scores.contains_key(&b));
    assert!(result.scores.contains_key(&c));
}

#[test]
fn pagerank_sink_node_gets_high_score() {
    // In A → B → C, C is a sink (no outgoing edges).
    // With damping, C should accumulate the most rank.
    let (adj, a, _b, c) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 100, 1e-8).unwrap();

    assert!(result.scores[&c] > result.scores[&a],
        "Sink node C should have higher rank than source A");
}

#[test]
fn pagerank_cycle_uniform() {
    // In a symmetric cycle A → B → C → A, all nodes should have equal rank.
    let (adj, a, b, c) = triangle_cycle();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 200, 1e-10).unwrap();

    let expected = 1.0 / 3.0;
    assert!((result.scores[&a] - expected).abs() < 0.01);
    assert!((result.scores[&b] - expected).abs() < 0.01);
    assert!((result.scores[&c] - expected).abs() < 0.01);
}

#[test]
fn pagerank_scores_are_positive() {
    let (adj, _, _, _, _) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let result = index.pagerank(0.85, 100, 1e-6).unwrap();

    for (_, &score) in &result.scores {
        assert!(score > 0.0, "all PR scores must be positive");
    }
}

#[test]
fn pagerank_convergence_with_few_iterations() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // Very few iterations, loose tolerance
    let result = index.pagerank(0.85, 3, 1.0).unwrap();
    let total: f64 = result.scores.values().sum();
    // Should still roughly sum to 1.0
    assert!((total - 1.0).abs() < 0.1);
}

// ══════════════════════════════════════════════════════════════════════════════
// Error Types
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn error_display_node_not_found() {
    let id = Uuid::new_v4();
    let err = CuGraphError::NodeNotFound(id);
    let msg = format!("{err}");
    assert!(msg.contains("node not found"));
    assert!(msg.contains(&id.to_string()));
}

#[test]
fn error_display_csr_build() {
    let err = CuGraphError::CsrBuild("test error".into());
    assert!(format!("{err}").contains("CSR build failed"));
}

#[test]
fn error_display_api() {
    let err = CuGraphError::Api { code: 42, message: "oops".into() };
    assert!(format!("{err}").contains("42"));
    assert!(format!("{err}").contains("oops"));
}

#[test]
fn error_display_cuda() {
    let err = CuGraphError::Cuda("out of memory".into());
    assert!(format!("{err}").contains("CUDA device error"));
}

#[test]
fn error_display_kernel_compile() {
    let err = CuGraphError::KernelCompile("syntax error".into());
    assert!(format!("{err}").contains("NVRTC compilation error"));
}

#[test]
fn error_display_result_extract() {
    let err = CuGraphError::ResultExtract("unexpected shape".into());
    assert!(format!("{err}").contains("result extraction failed"));
}

// ══════════════════════════════════════════════════════════════════════════════
// Larger Graphs
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn bfs_star_graph() {
    // Hub → spoke_0, spoke_1, ..., spoke_n
    let adj = AdjacencyIndex::new();
    let hub = Uuid::new_v4();
    let spokes: Vec<Uuid> = (0..20).map(|_| Uuid::new_v4()).collect();

    for &spoke in &spokes {
        adj.add_edge(&Edge::association(hub, spoke, 1.0));
    }

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();
    let result = index.bfs(hub, 1).unwrap();

    // Hub + all spokes = 21
    assert_eq!(result.visited.len(), 21);
}

#[test]
fn dijkstra_weighted_star() {
    let adj = AdjacencyIndex::new();
    let hub = Uuid::new_v4();
    let spokes: Vec<(Uuid, f32)> = (0..5)
        .map(|i| (Uuid::new_v4(), (i + 1) as f32 * 0.5))
        .collect();

    for &(spoke, weight) in &spokes {
        adj.add_edge(&Edge::new(hub, spoke, EdgeType::Association, weight));
    }

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();
    let result = index.dijkstra(hub).unwrap();

    for &(spoke, expected_dist) in &spokes {
        let actual = result.distances[&spoke];
        assert!((actual - expected_dist as f64).abs() < 1e-5,
            "spoke distance mismatch: expected {expected_dist}, got {actual}");
    }
}

#[test]
fn pagerank_disconnected_components() {
    // Two disconnected edges: A → B, C → D
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let d = Uuid::new_v4();

    adj.add_edge(&Edge::association(a, b, 1.0));
    adj.add_edge(&Edge::association(c, d, 1.0));

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();
    let result = index.pagerank(0.85, 100, 1e-8).unwrap();

    assert_eq!(result.scores.len(), 4);
    let total: f64 = result.scores.values().sum();
    assert!((total - 1.0).abs() < 0.01);
}

#[test]
fn bfs_disconnected_only_visits_component() {
    let adj = AdjacencyIndex::new();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4(); // isolated from A→B

    adj.add_edge(&Edge::association(a, b, 1.0));
    // Add C as a node with an edge to itself (or a separate edge)
    adj.add_edge(&Edge::association(c, c, 1.0));

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();
    let result = index.bfs(a, 10).unwrap();

    assert!(result.visited.contains(&a));
    assert!(result.visited.contains(&b));
    assert!(!result.visited.contains(&c));
}

// ══════════════════════════════════════════════════════════════════════════════
// CSR Structural Invariants (via public API)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn csr_vertex_and_edge_counts_consistent() {
    let (adj, _, _, _) = three_chain();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    // 3 nodes, 2 edges (A→B, B→C)
    assert_eq!(index.vertex_count(), 3);
    assert_eq!(index.edge_count(), 2);
    assert_eq!(index.node_ids().len(), 3);
}

#[test]
fn csr_diamond_counts() {
    let (adj, _, _, _, _) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    assert_eq!(index.vertex_count(), 4);
    assert_eq!(index.edge_count(), 4);
    assert_eq!(index.node_ids().len(), 4);
}

#[test]
fn csr_node_ids_unique() {
    let (adj, _, _, _, _) = diamond_graph();
    let index = CuGraphIndex::from_adjacency(&adj).unwrap();

    let ids = index.node_ids();
    let mut dedup = ids.to_vec();
    dedup.sort();
    dedup.dedup();
    assert_eq!(ids.len(), dedup.len(), "node_ids should be unique");
}

#[test]
fn csr_large_graph_counts() {
    let adj = AdjacencyIndex::new();
    let nodes: Vec<Uuid> = (0..50).map(|_| Uuid::new_v4()).collect();

    // Create a chain: 0→1→2→...→49
    for i in 0..49 {
        adj.add_edge(&Edge::association(nodes[i], nodes[i + 1], 1.0));
    }

    let index = CuGraphIndex::from_adjacency(&adj).unwrap();
    assert_eq!(index.vertex_count(), 50);
    assert_eq!(index.edge_count(), 49);
}
