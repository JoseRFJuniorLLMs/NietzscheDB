//! Unit tests for nietzsche-sdk parameter structs and serialization.
//!
//! These tests do NOT require a running NietzscheDB server — they validate
//! client-side logic: default values, parameter construction, serialization,
//! URI parsing, and proto conversions.

use nietzsche_sdk::{InsertNodeParams, InsertEdgeParams, SleepParams};
use uuid::Uuid;

// ══════════════════════════════════════════════════════════════════════════════
// InsertNodeParams
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn insert_node_params_default() {
    let p = InsertNodeParams::default();

    assert!(p.id.is_none());
    assert!(p.coords.is_empty());
    assert_eq!(p.content, serde_json::Value::Null);
    assert_eq!(p.node_type, "Semantic");
    assert!((p.energy - 1.0).abs() < f32::EPSILON);
}

#[test]
fn insert_node_params_custom() {
    let id = Uuid::new_v4();
    let p = InsertNodeParams {
        id: Some(id),
        coords: vec![0.1, 0.2, 0.3],
        content: serde_json::json!({"key": "value"}),
        node_type: "Episodic".into(),
        energy: 0.7,
    };

    assert_eq!(p.id, Some(id));
    assert_eq!(p.coords.len(), 3);
    assert_eq!(p.node_type, "Episodic");
    assert!((p.energy - 0.7).abs() < f32::EPSILON);
}

#[test]
fn insert_node_params_struct_update_syntax() {
    let p = InsertNodeParams {
        coords: vec![0.5, 0.5, 0.5],
        node_type: "Procedural".into(),
        ..Default::default()
    };

    assert!(p.id.is_none());
    assert_eq!(p.coords, vec![0.5, 0.5, 0.5]);
    assert_eq!(p.node_type, "Procedural");
    assert!((p.energy - 1.0).abs() < f32::EPSILON);
}

#[test]
fn insert_node_params_clone() {
    let original = InsertNodeParams {
        coords: vec![0.1, 0.2],
        ..Default::default()
    };
    let cloned = original.clone();
    assert_eq!(original.coords, cloned.coords);
    assert_eq!(original.node_type, cloned.node_type);
}

#[test]
fn insert_node_params_debug() {
    let p = InsertNodeParams::default();
    let dbg = format!("{p:?}");
    assert!(dbg.contains("InsertNodeParams"));
    assert!(dbg.contains("Semantic"));
}

#[test]
fn insert_node_params_empty_coords() {
    let p = InsertNodeParams::default();
    assert!(p.coords.is_empty());
}

#[test]
fn insert_node_params_high_dimensional_coords() {
    let p = InsertNodeParams {
        coords: vec![0.01; 3072],
        ..Default::default()
    };
    assert_eq!(p.coords.len(), 3072);
}

#[test]
fn insert_node_params_json_content_types() {
    // Null
    let p1 = InsertNodeParams { content: serde_json::Value::Null, ..Default::default() };
    assert!(p1.content.is_null());

    // String
    let p2 = InsertNodeParams {
        content: serde_json::json!("hello"),
        ..Default::default()
    };
    assert!(p2.content.is_string());

    // Object
    let p3 = InsertNodeParams {
        content: serde_json::json!({"nested": {"deep": true}}),
        ..Default::default()
    };
    assert!(p3.content.is_object());

    // Array
    let p4 = InsertNodeParams {
        content: serde_json::json!([1, 2, 3]),
        ..Default::default()
    };
    assert!(p4.content.is_array());
}

// ══════════════════════════════════════════════════════════════════════════════
// InsertEdgeParams
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn insert_edge_params_default() {
    let p = InsertEdgeParams::default();

    assert!(p.id.is_none());
    assert_eq!(p.from, Uuid::nil());
    assert_eq!(p.to, Uuid::nil());
    assert_eq!(p.edge_type, "Association");
    assert!((p.weight - 1.0).abs() < f64::EPSILON);
}

#[test]
fn insert_edge_params_custom() {
    let from = Uuid::new_v4();
    let to = Uuid::new_v4();
    let edge_id = Uuid::new_v4();

    let p = InsertEdgeParams {
        id: Some(edge_id),
        from,
        to,
        edge_type: "Hierarchical".into(),
        weight: 0.5,
    };

    assert_eq!(p.id, Some(edge_id));
    assert_eq!(p.from, from);
    assert_eq!(p.to, to);
    assert_eq!(p.edge_type, "Hierarchical");
    assert!((p.weight - 0.5).abs() < f64::EPSILON);
}

#[test]
fn insert_edge_params_clone() {
    let p = InsertEdgeParams {
        from: Uuid::new_v4(),
        to: Uuid::new_v4(),
        weight: 0.42,
        ..Default::default()
    };
    let c = p.clone();
    assert_eq!(p.from, c.from);
    assert_eq!(p.to, c.to);
    assert!((p.weight - c.weight).abs() < f64::EPSILON);
}

#[test]
fn insert_edge_params_debug() {
    let p = InsertEdgeParams::default();
    let dbg = format!("{p:?}");
    assert!(dbg.contains("InsertEdgeParams"));
}

// ══════════════════════════════════════════════════════════════════════════════
// SleepParams
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn sleep_params_default() {
    let p = SleepParams::default();

    assert!((p.noise - 0.02).abs() < f64::EPSILON);
    assert_eq!(p.adam_steps, 10);
    assert!((p.adam_lr - 5e-3).abs() < f64::EPSILON);
    assert!((p.hausdorff_threshold - 0.15).abs() < f32::EPSILON);
    assert_eq!(p.rng_seed, 0);
}

#[test]
fn sleep_params_custom() {
    let p = SleepParams {
        noise: 0.1,
        adam_steps: 50,
        adam_lr: 1e-4,
        hausdorff_threshold: 0.5,
        rng_seed: 42,
    };

    assert!((p.noise - 0.1).abs() < f64::EPSILON);
    assert_eq!(p.adam_steps, 50);
    assert!((p.adam_lr - 1e-4).abs() < f64::EPSILON);
    assert!((p.hausdorff_threshold - 0.5).abs() < f32::EPSILON);
    assert_eq!(p.rng_seed, 42);
}

#[test]
fn sleep_params_clone() {
    let p = SleepParams { rng_seed: 123, ..Default::default() };
    let c = p.clone();
    assert_eq!(p.rng_seed, c.rng_seed);
}

#[test]
fn sleep_params_debug() {
    let p = SleepParams::default();
    let dbg = format!("{p:?}");
    assert!(dbg.contains("SleepParams"));
}

// ══════════════════════════════════════════════════════════════════════════════
// Proto Re-exports
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn re_exported_types_are_accessible() {
    // Verify that re-exported proto types are usable from the SDK crate
    let _: nietzsche_sdk::StatusResponse = nietzsche_sdk::StatusResponse {
        status: "ok".into(),
        error: String::new(),
    };
    let _: nietzsche_sdk::KnnResponse = nietzsche_sdk::KnnResponse {
        results: vec![],
    };
}

// ══════════════════════════════════════════════════════════════════════════════
// Batch Parameter Lists
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn batch_node_params() {
    let nodes: Vec<InsertNodeParams> = (0..100)
        .map(|i| InsertNodeParams {
            coords: vec![i as f64 * 0.001; 128],
            node_type: "Semantic".into(),
            ..Default::default()
        })
        .collect();

    assert_eq!(nodes.len(), 100);
    assert_eq!(nodes[0].coords.len(), 128);
    assert_eq!(nodes[99].coords.len(), 128);
}

#[test]
fn batch_edge_params() {
    let a = Uuid::new_v4();
    let targets: Vec<Uuid> = (0..50).map(|_| Uuid::new_v4()).collect();

    let edges: Vec<InsertEdgeParams> = targets
        .iter()
        .map(|&t| InsertEdgeParams {
            from: a,
            to: t,
            weight: 0.5,
            ..Default::default()
        })
        .collect();

    assert_eq!(edges.len(), 50);
    assert!(edges.iter().all(|e| e.from == a));
}

// ══════════════════════════════════════════════════════════════════════════════
// NietzscheClient (connection error without a server)
// ══════════════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn connect_to_invalid_address_fails() {
    use nietzsche_sdk::NietzscheClient;

    // Connecting to a non-listening port should fail
    let result = NietzscheClient::connect("http://127.0.0.1:1").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn connect_parses_valid_uri() {
    use nietzsche_sdk::NietzscheClient;

    // This will fail at the TCP level (no server) but the URI should parse OK.
    // The error should be a connection error, not a parse error.
    let result = NietzscheClient::connect("http://[::1]:50051").await;
    // We just verify it doesn't panic — connection failure is expected
    let _ = result;
}

#[tokio::test]
async fn connect_rejects_garbage_uri() {
    use nietzsche_sdk::NietzscheClient;

    let result = NietzscheClient::connect("not a valid uri!!!").await;
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════════════════════
// Content Serialization
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn content_json_roundtrip() {
    let original = serde_json::json!({
        "text": "Thus spoke Zarathustra",
        "embedding_model": "text-embedding-3-large",
        "tokens": 42,
        "nested": { "deep": [1, 2, 3] }
    });

    let bytes = serde_json::to_vec(&original).unwrap();
    let restored: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    assert_eq!(original, restored);
}

#[test]
fn empty_content_serializes() {
    let null_content = serde_json::Value::Null;
    let bytes = serde_json::to_vec(&null_content).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn large_content_serializes() {
    let big = serde_json::json!({
        "text": "a".repeat(100_000),
    });
    let bytes = serde_json::to_vec(&big).unwrap();
    assert!(bytes.len() > 100_000);
}

// ══════════════════════════════════════════════════════════════════════════════
// UUID handling
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn uuid_to_string_roundtrip() {
    let id = Uuid::new_v4();
    let s = id.to_string();
    let parsed: Uuid = s.parse().unwrap();
    assert_eq!(id, parsed);
}

#[test]
fn nil_uuid_to_string() {
    let nil = Uuid::nil();
    let s = nil.to_string();
    assert_eq!(s, "00000000-0000-0000-0000-000000000000");
}

#[test]
fn params_with_nil_uuid() {
    let p = InsertEdgeParams::default();
    assert_eq!(p.from.to_string(), "00000000-0000-0000-0000-000000000000");
    assert_eq!(p.to.to_string(), "00000000-0000-0000-0000-000000000000");
}
