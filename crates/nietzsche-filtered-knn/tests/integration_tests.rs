//! Integration tests for nietzsche-filtered-knn.
//!
//! These tests exercise the full pipeline: create a temp RocksDB, insert nodes,
//! build filter bitmaps, and run filtered KNN searches.

use std::collections::HashMap;

use roaring::RoaringBitmap;
use uuid::Uuid;

use nietzsche_graph::{GraphStorage, Node, NodeType, PoincareVector};
use nietzsche_filtered_knn::{
    build_filter_bitmap, build_id_to_idx, filtered_knn, json_path_exists,
    json_path_matches, NodeFilter,
};

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

fn open_temp_db() -> (GraphStorage, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
    (storage, dir)
}

/// Create a node with a 2D Poincare embedding and specific energy.
fn make_node(x: f32, y: f32, energy: f32, node_type: NodeType, content: serde_json::Value) -> Node {
    let mut node = Node::new(
        Uuid::new_v4(),
        PoincareVector::new(vec![x, y]),
        content,
    );
    node.meta.energy = energy;
    node.meta.node_type = node_type;
    node
}

/// Insert nodes, return their IDs and the id_to_idx map.
fn insert_nodes(
    storage: &GraphStorage,
    nodes: &[Node],
) -> (Vec<Uuid>, HashMap<Uuid, u32>) {
    for node in nodes {
        storage.put_node(node).unwrap();
    }
    let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
    let id_to_idx = build_id_to_idx(&ids);
    (ids, id_to_idx)
}

// ─────────────────────────────────────────────
// Test 1: Energy range filter
// ─────────────────────────────────────────────

#[test]
fn test_energy_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 0.2, NodeType::Semantic, serde_json::json!({})),
        make_node(0.2, 0.1, 0.5, NodeType::Semantic, serde_json::json!({})),
        make_node(0.1, 0.2, 0.8, NodeType::Semantic, serde_json::json!({})),
        make_node(0.2, 0.2, 1.0, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // Filter: energy in [0.4, 0.9]
    let filter = NodeFilter::EnergyRange { min: 0.4, max: 0.9 };
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    // Should match nodes with energy 0.5 and 0.8
    assert_eq!(bm.len(), 2, "Expected 2 nodes in bitmap, got {}", bm.len());

    // Verify the correct indices are set
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&1), "Node at index 1 (energy=0.5) should be in bitmap");
    assert!(set_indices.contains(&2), "Node at index 2 (energy=0.8) should be in bitmap");
}

// ─────────────────────────────────────────────
// Test 2: Node type filter
// ─────────────────────────────────────────────

#[test]
fn test_node_type_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Episodic, serde_json::json!({})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
        make_node(0.1, 0.2, 1.0, NodeType::Episodic, serde_json::json!({})),
        make_node(0.2, 0.2, 1.0, NodeType::Concept, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    let filter = NodeFilter::NodeType("Episodic".to_string());
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 2, "Expected 2 Episodic nodes");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&0));
    assert!(set_indices.contains(&2));
}

// ─────────────────────────────────────────────
// Test 3: Content field filter
// ─────────────────────────────────────────────

#[test]
fn test_content_field_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"category": "science"})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"category": "art"})),
        make_node(0.1, 0.2, 1.0, NodeType::Semantic, serde_json::json!({"category": "science"})),
        make_node(0.2, 0.2, 1.0, NodeType::Semantic, serde_json::json!({"other": "field"})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    let filter = NodeFilter::ContentField {
        path: "category".to_string(),
        value: serde_json::json!("science"),
    };
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 2, "Expected 2 nodes with category=science");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&0));
    assert!(set_indices.contains(&2));
}

// ─────────────────────────────────────────────
// Test 4: AND filter
// ─────────────────────────────────────────────

#[test]
fn test_and_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 0.3, NodeType::Episodic, serde_json::json!({})),
        make_node(0.2, 0.1, 0.7, NodeType::Episodic, serde_json::json!({})),
        make_node(0.1, 0.2, 0.7, NodeType::Semantic, serde_json::json!({})),
        make_node(0.2, 0.2, 0.9, NodeType::Episodic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // AND: energy >= 0.5 AND node_type == Episodic
    let filter = NodeFilter::And(vec![
        NodeFilter::EnergyRange { min: 0.5, max: 1.0 },
        NodeFilter::NodeType("Episodic".to_string()),
    ]);
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    // Node 1 (energy=0.7, Episodic) and Node 3 (energy=0.9, Episodic) match
    assert_eq!(bm.len(), 2, "Expected 2 nodes matching AND filter");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&1));
    assert!(set_indices.contains(&3));
}

// ─────────────────────────────────────────────
// Test 5: OR filter
// ─────────────────────────────────────────────

#[test]
fn test_or_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 0.3, NodeType::Episodic, serde_json::json!({})),
        make_node(0.2, 0.1, 0.7, NodeType::Semantic, serde_json::json!({})),
        make_node(0.1, 0.2, 0.2, NodeType::Concept, serde_json::json!({})),
        make_node(0.2, 0.2, 0.9, NodeType::Episodic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // OR: Concept OR energy >= 0.8
    let filter = NodeFilter::Or(vec![
        NodeFilter::NodeType("Concept".to_string()),
        NodeFilter::EnergyRange { min: 0.8, max: 1.0 },
    ]);
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    // Node 2 (Concept) and Node 3 (energy=0.9) match
    assert_eq!(bm.len(), 2, "Expected 2 nodes matching OR filter");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&2));
    assert!(set_indices.contains(&3));
}

// ─────────────────────────────────────────────
// Test 6: Filtered KNN basic
// ─────────────────────────────────────────────

#[test]
fn test_filtered_knn_basic() {
    let (storage, _dir) = open_temp_db();

    // Create nodes at different positions in the Poincare ball.
    // Query will be near (0.1, 0.1). We filter to only high-energy nodes.
    let nodes = vec![
        // Index 0: close to query, LOW energy (should be filtered out)
        make_node(0.11, 0.11, 0.1, NodeType::Semantic, serde_json::json!({})),
        // Index 1: close to query, HIGH energy (should be result #1)
        make_node(0.12, 0.12, 0.8, NodeType::Semantic, serde_json::json!({})),
        // Index 2: far from query, HIGH energy (should be result #2)
        make_node(0.5, 0.5, 0.9, NodeType::Semantic, serde_json::json!({})),
        // Index 3: medium distance, HIGH energy (should be between #1 and #2)
        make_node(0.3, 0.3, 0.7, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // Build energy filter: energy >= 0.5
    let filter = NodeFilter::EnergyRange { min: 0.5, max: 1.0 };
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    // Should have 3 nodes in bitmap (indices 1, 2, 3)
    assert_eq!(bm.len(), 3);

    // KNN search: query near origin, k=2
    let query = vec![0.1_f32, 0.1];
    let results = filtered_knn(&query, 2, &bm, &ids, &storage).unwrap();

    assert_eq!(results.len(), 2, "Expected 2 results from KNN");

    // Result #1 should be closest high-energy node (index 1, at 0.12, 0.12)
    assert_eq!(results[0].0, ids[1], "Nearest should be node at (0.12, 0.12)");

    // Results should be sorted by distance ascending
    assert!(
        results[0].1 <= results[1].1,
        "Results must be sorted ascending by distance: {} <= {}",
        results[0].1,
        results[1].1,
    );
}

// ─────────────────────────────────────────────
// Test 7: Filtered KNN with empty bitmap
// ─────────────────────────────────────────────

#[test]
fn test_filtered_knn_empty_bitmap() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, _id_to_idx) = insert_nodes(&storage, &nodes);

    // Empty bitmap
    let empty_bm = RoaringBitmap::new();
    let query = vec![0.1_f32, 0.1];
    let results = filtered_knn(&query, 5, &empty_bm, &ids, &storage).unwrap();

    assert!(results.is_empty(), "Empty bitmap should yield empty results");
}

// ─────────────────────────────────────────────
// Test 8: JSON path matches
// ─────────────────────────────────────────────

#[test]
fn test_json_path_matches() {
    let content = serde_json::json!({
        "a": {
            "b": {
                "c": 42
            }
        },
        "name": "test",
        "tags": ["x", "y"]
    });

    // Simple path
    assert!(json_path_matches(&content, "name", &serde_json::json!("test")));
    assert!(!json_path_matches(&content, "name", &serde_json::json!("other")));

    // Nested path
    assert!(json_path_matches(&content, "a.b.c", &serde_json::json!(42)));
    assert!(!json_path_matches(&content, "a.b.c", &serde_json::json!(99)));

    // Non-existent path
    assert!(!json_path_matches(&content, "x.y.z", &serde_json::json!(1)));

    // Array value
    assert!(json_path_matches(&content, "tags", &serde_json::json!(["x", "y"])));
}

// ─────────────────────────────────────────────
// Test 9: JSON path exists
// ─────────────────────────────────────────────

#[test]
fn test_json_path_exists() {
    let content = serde_json::json!({
        "a": {
            "b": 10
        },
        "c": null,
        "d": "hello"
    });

    assert!(json_path_exists(&content, "a.b"));
    assert!(json_path_exists(&content, "d"));
    assert!(!json_path_exists(&content, "c")); // null is treated as non-existent
    assert!(!json_path_exists(&content, "x"));
    assert!(!json_path_exists(&content, "a.z"));
}

// ─────────────────────────────────────────────
// Test 10: ContentFieldExists filter
// ─────────────────────────────────────────────

#[test]
fn test_content_field_exists_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"title": "Hello"})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"body": "World"})),
        make_node(0.1, 0.2, 1.0, NodeType::Semantic, serde_json::json!({"title": "Foo", "body": "Bar"})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    let filter = NodeFilter::ContentFieldExists("title".to_string());
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 2, "Expected 2 nodes with 'title' field");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&0));
    assert!(set_indices.contains(&2));
}

// ─────────────────────────────────────────────
// Test 11: Empty AND filter (universal set)
// ─────────────────────────────────────────────

#[test]
fn test_empty_and_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // Empty AND = all nodes pass
    let filter = NodeFilter::And(vec![]);
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 2, "Empty AND should match all nodes");
}

// ─────────────────────────────────────────────
// Test 12: Empty OR filter (empty set)
// ─────────────────────────────────────────────

#[test]
fn test_empty_or_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    // Empty OR = no nodes pass
    let filter = NodeFilter::Or(vec![]);
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 0, "Empty OR should match no nodes");
}

// ─────────────────────────────────────────────
// Test 13: KNN with k=0 returns error
// ─────────────────────────────────────────────

#[test]
fn test_filtered_knn_k_zero_is_error() {
    let (storage, _dir) = open_temp_db();
    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({})),
    ];
    let (ids, _) = insert_nodes(&storage, &nodes);

    let mut bm = RoaringBitmap::new();
    bm.insert(0);

    let query = vec![0.1_f32, 0.1];
    let result = filtered_knn(&query, 0, &bm, &ids, &storage);

    assert!(result.is_err(), "k=0 should return an error");
}

// ─────────────────────────────────────────────
// Test 14: Nested content field filter
// ─────────────────────────────────────────────

#[test]
fn test_nested_content_field_filter() {
    let (storage, _dir) = open_temp_db();

    let nodes = vec![
        make_node(0.1, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"meta": {"lang": "en"}})),
        make_node(0.2, 0.1, 1.0, NodeType::Semantic, serde_json::json!({"meta": {"lang": "pt"}})),
        make_node(0.1, 0.2, 1.0, NodeType::Semantic, serde_json::json!({"meta": {"lang": "en"}})),
    ];
    let (ids, id_to_idx) = insert_nodes(&storage, &nodes);

    let filter = NodeFilter::ContentField {
        path: "meta.lang".to_string(),
        value: serde_json::json!("en"),
    };
    let bm = build_filter_bitmap(&filter, &storage, &ids, &id_to_idx).unwrap();

    assert_eq!(bm.len(), 2, "Expected 2 nodes with meta.lang=en");
    let set_indices: Vec<u32> = bm.iter().collect();
    assert!(set_indices.contains(&0));
    assert!(set_indices.contains(&2));
}

// ─────────────────────────────────────────────
// Test 15: Build id_to_idx correctness
// ─────────────────────────────────────────────

#[test]
fn test_build_id_to_idx() {
    let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
    let map = build_id_to_idx(&ids);

    assert_eq!(map.len(), 5);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(map[id], i as u32);
    }
}
