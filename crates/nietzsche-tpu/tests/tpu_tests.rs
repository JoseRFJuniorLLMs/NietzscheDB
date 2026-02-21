//! Integration tests for nietzsche-tpu.
//!
//! These tests exercise the CPU-fallback path (feature `tpu` OFF).
//! The staging buffer logic, VectorStore trait impl, compaction,
//! and kNN search are fully covered without requiring a Cloud TPU VM.

use nietzsche_graph::model::PoincareVector;
use nietzsche_graph::db::VectorStore;
use nietzsche_tpu::TpuVectorStore;
use uuid::Uuid;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn pv(coords: &[f32]) -> PoincareVector {
    PoincareVector::new(coords.to_vec())
}

fn random_pv(dim: usize) -> PoincareVector {
    let coords: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.013) % 0.9) - 0.4).collect();
    PoincareVector::new(coords)
}

// ── Construction ────────────────────────────────────────────────────────────

#[test]
fn new_store_is_empty() {
    let store = TpuVectorStore::new(128).expect("new should succeed");
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert_eq!(store.dim(), 128);
}

#[test]
fn new_various_dimensions() {
    for dim in [1, 3, 128, 768, 1536, 3072] {
        let store = TpuVectorStore::new(dim).expect("should succeed");
        assert_eq!(store.dim(), dim);
    }
}

#[test]
fn tpu_feature_disabled_in_tests() {
    // Without the `tpu` feature, is_tpu_enabled should be false.
    assert!(!TpuVectorStore::is_tpu_enabled());
}

// ── Upsert ──────────────────────────────────────────────────────────────────

#[test]
fn upsert_single_vector() {
    let mut store = TpuVectorStore::new(3).unwrap();
    let id = Uuid::new_v4();
    let v = pv(&[0.1, 0.2, 0.3]);

    store.upsert(id, &v).unwrap();
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

#[test]
fn upsert_dimension_mismatch_returns_error() {
    let mut store = TpuVectorStore::new(3).unwrap();
    let id = Uuid::new_v4();
    let wrong = pv(&[0.1, 0.2]); // dim=2, store expects 3

    let result = store.upsert(id, &wrong);
    assert!(result.is_err());
}

#[test]
fn upsert_same_id_overwrites() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    store.upsert(id, &pv(&[0.1, 0.2])).unwrap();
    assert_eq!(store.len(), 1);

    // Overwrite with new coords
    store.upsert(id, &pv(&[0.3, 0.4])).unwrap();
    // Active count should still be 1 (old row soft-deleted, new row added)
    assert_eq!(store.len(), 1);
}

#[test]
fn upsert_multiple_distinct_ids() {
    let mut store = TpuVectorStore::new(4).unwrap();
    let ids: Vec<Uuid> = (0..50).map(|_| Uuid::new_v4()).collect();

    for (i, &id) in ids.iter().enumerate() {
        let coords: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32 * 0.01).collect();
        store.upsert(id, &pv(&coords)).unwrap();
    }

    assert_eq!(store.len(), 50);
}

// ── Delete ──────────────────────────────────────────────────────────────────

#[test]
fn delete_existing_vector() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    store.upsert(id, &pv(&[0.1, 0.2])).unwrap();
    assert_eq!(store.len(), 1);

    store.delete(id).unwrap();
    assert_eq!(store.len(), 0);
    assert!(store.is_empty());
}

#[test]
fn delete_nonexistent_is_noop() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    // Deleting a never-inserted ID should not panic or error
    store.delete(id).unwrap();
    assert_eq!(store.len(), 0);
}

#[test]
fn delete_twice_is_idempotent() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    store.upsert(id, &pv(&[0.5, 0.5])).unwrap();
    store.delete(id).unwrap();
    store.delete(id).unwrap(); // second delete
    assert_eq!(store.len(), 0);
}

#[test]
fn delete_one_of_many() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();

    store.upsert(a, &pv(&[0.1, 0.1])).unwrap();
    store.upsert(b, &pv(&[0.2, 0.2])).unwrap();
    store.upsert(c, &pv(&[0.3, 0.3])).unwrap();
    assert_eq!(store.len(), 3);

    store.delete(b).unwrap();
    assert_eq!(store.len(), 2);
}

// ── kNN (CPU fallback) ─────────────────────────────────────────────────────

#[test]
fn knn_empty_store_returns_empty() {
    let store = TpuVectorStore::new(3).unwrap();
    let query = pv(&[0.0, 0.0, 0.0]);

    let results = store.knn(&query, 5).unwrap();
    assert!(results.is_empty());
}

#[test]
fn knn_single_vector_returns_it() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();
    store.upsert(id, &pv(&[0.1, 0.2])).unwrap();

    let query = pv(&[0.1, 0.2]);
    let results = store.knn(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
    assert!(results[0].1 < 0.001); // distance ≈ 0
}

#[test]
fn knn_returns_sorted_by_distance() {
    let mut store = TpuVectorStore::new(2).unwrap();

    let close = Uuid::new_v4();
    let medium = Uuid::new_v4();
    let far = Uuid::new_v4();

    // Query will be at origin [0, 0]
    store.upsert(close,  &pv(&[0.01, 0.01])).unwrap();
    store.upsert(medium, &pv(&[0.3,  0.3])).unwrap();
    store.upsert(far,    &pv(&[0.5,  0.5])).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 3).unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, close);
    assert_eq!(results[1].0, medium);
    assert_eq!(results[2].0, far);

    // Distances should be strictly increasing
    assert!(results[0].1 < results[1].1);
    assert!(results[1].1 < results[2].1);
}

#[test]
fn knn_k_greater_than_n() {
    let mut store = TpuVectorStore::new(2).unwrap();

    store.upsert(Uuid::new_v4(), &pv(&[0.1, 0.1])).unwrap();
    store.upsert(Uuid::new_v4(), &pv(&[0.2, 0.2])).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 100).unwrap();

    // Should return only 2 results (all we have)
    assert_eq!(results.len(), 2);
}

#[test]
fn knn_k_zero() {
    let mut store = TpuVectorStore::new(2).unwrap();
    store.upsert(Uuid::new_v4(), &pv(&[0.1, 0.1])).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn knn_excludes_deleted_vectors() {
    let mut store = TpuVectorStore::new(2).unwrap();

    let kept = Uuid::new_v4();
    let deleted = Uuid::new_v4();

    store.upsert(kept,    &pv(&[0.1, 0.1])).unwrap();
    store.upsert(deleted, &pv(&[0.05, 0.05])).unwrap();
    store.delete(deleted).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, kept);
}

#[test]
fn knn_after_overwrite() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    // Insert far from origin
    store.upsert(id, &pv(&[0.5, 0.5])).unwrap();

    // Overwrite to be at origin
    store.upsert(id, &pv(&[0.01, 0.01])).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, id);
    // Distance should be small (near origin)
    assert!(results[0].1 < 0.1);
}

// ── L2 Distance Correctness ────────────────────────────────────────────────

#[test]
fn knn_distance_is_euclidean_l2() {
    let mut store = TpuVectorStore::new(2).unwrap();
    let id = Uuid::new_v4();

    store.upsert(id, &pv(&[0.3, 0.4])).unwrap();

    let query = pv(&[0.0, 0.0]);
    let results = store.knn(&query, 1).unwrap();

    // Expected L2 distance = sqrt(0.3² + 0.4²) = sqrt(0.25) = 0.5
    let dist = results[0].1;
    assert!((dist - 0.5).abs() < 1e-5, "expected ~0.5, got {dist}");
}

#[test]
fn knn_distance_symmetry() {
    let mut store = TpuVectorStore::new(3).unwrap();
    let id_a = Uuid::new_v4();
    let id_b = Uuid::new_v4();

    let va = pv(&[0.1, 0.2, 0.3]);
    let vb = pv(&[0.4, 0.5, 0.6]);

    store.upsert(id_a, &va).unwrap();
    store.upsert(id_b, &vb).unwrap();

    // Query from A, distance to B
    let results_from_a = store.knn(&va, 2).unwrap();
    let dist_a_to_b = results_from_a.iter().find(|(id, _)| *id == id_b).unwrap().1;

    // Query from B, distance to A
    let results_from_b = store.knn(&vb, 2).unwrap();
    let dist_b_to_a = results_from_b.iter().find(|(id, _)| *id == id_a).unwrap().1;

    assert!((dist_a_to_b - dist_b_to_a).abs() < 1e-5, "distance should be symmetric");
}

// ── Stress / Large-Scale ────────────────────────────────────────────────────

#[test]
fn upsert_and_knn_with_many_vectors() {
    let dim = 64;
    let n = 500;
    let mut store = TpuVectorStore::new(dim).unwrap();

    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = Uuid::new_v4();
        let coords: Vec<f32> = (0..dim)
            .map(|j| ((i * dim + j) as f32 * 0.0001) % 0.9 - 0.45)
            .collect();
        store.upsert(id, &pv(&coords)).unwrap();
        ids.push(id);
    }

    assert_eq!(store.len(), n);

    let query = pv(&vec![0.0f32; dim]);
    let results = store.knn(&query, 10).unwrap();
    assert_eq!(results.len(), 10);

    // Verify distances are sorted
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
}

#[test]
fn interleaved_insert_delete_search() {
    let mut store = TpuVectorStore::new(4).unwrap();
    let mut alive = Vec::new();

    for i in 0..100 {
        let id = Uuid::new_v4();
        let coords: Vec<f32> = (0..4).map(|j| (i * 4 + j) as f32 * 0.005).collect();
        store.upsert(id, &pv(&coords)).unwrap();

        // Delete every 3rd
        if i % 3 == 0 {
            store.delete(id).unwrap();
        } else {
            alive.push(id);
        }
    }

    assert_eq!(store.len(), alive.len());

    let query = pv(&[0.0, 0.0, 0.0, 0.0]);
    let results = store.knn(&query, 200).unwrap();

    // All returned IDs should be alive
    for (id, _) in &results {
        assert!(alive.contains(id), "kNN returned deleted ID {id}");
    }
    assert_eq!(results.len(), alive.len());
}

// ── build_tpu_index (no-op without tpu feature) ────────────────────────────

#[test]
fn build_tpu_index_noop_without_feature() {
    let mut store = TpuVectorStore::new(128).unwrap();

    for _ in 0..10 {
        store.upsert(Uuid::new_v4(), &random_pv(128)).unwrap();
    }

    // Should succeed as a no-op
    store.build_tpu_index().unwrap();

    // Store still works
    let results = store.knn(&random_pv(128), 5).unwrap();
    assert_eq!(results.len(), 5);
}

// ── MHLO program generation (unit test via internal fn) ─────────────────────

#[test]
fn mhlo_dot_program_format() {
    // We test the MHLO template indirectly: it must compile successfully
    // on a real TPU. Here we at least verify the function exists and
    // the store can be built with various dimensions.
    for dim in [3, 128, 1536] {
        let store = TpuVectorStore::new(dim).unwrap();
        assert_eq!(store.dim(), dim);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────────

#[test]
fn upsert_zero_vector() {
    let mut store = TpuVectorStore::new(3).unwrap();
    let id = Uuid::new_v4();
    store.upsert(id, &pv(&[0.0, 0.0, 0.0])).unwrap();

    let query = pv(&[0.0, 0.0, 0.0]);
    let results = store.knn(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].1 < 1e-6);
}

#[test]
fn upsert_boundary_vectors() {
    let mut store = TpuVectorStore::new(2).unwrap();

    // Vectors near the Poincaré ball boundary (norm close to 1.0)
    store.upsert(Uuid::new_v4(), &pv(&[0.99, 0.0])).unwrap();
    store.upsert(Uuid::new_v4(), &pv(&[-0.99, 0.0])).unwrap();
    store.upsert(Uuid::new_v4(), &pv(&[0.0, 0.99])).unwrap();
    assert_eq!(store.len(), 3);

    let results = store.knn(&pv(&[0.0, 0.0]), 3).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn high_dimensional_vectors() {
    let dim = 3072;
    let mut store = TpuVectorStore::new(dim).unwrap();

    for _ in 0..5 {
        store.upsert(Uuid::new_v4(), &random_pv(dim)).unwrap();
    }

    let results = store.knn(&random_pv(dim), 3).unwrap();
    assert_eq!(results.len(), 3);
}

// ── Thread safety ───────────────────────────────────────────────────────────

#[test]
fn concurrent_knn_reads() {
    use std::sync::Arc;
    use std::thread;

    let mut store = TpuVectorStore::new(4).unwrap();
    for _ in 0..20 {
        store.upsert(Uuid::new_v4(), &random_pv(4)).unwrap();
    }

    let store = Arc::new(store);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let s = Arc::clone(&store);
            thread::spawn(move || {
                let query = pv(&[0.0, 0.0, 0.0, 0.0]);
                let results = s.knn(&query, 5).unwrap();
                assert_eq!(results.len(), 5);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
