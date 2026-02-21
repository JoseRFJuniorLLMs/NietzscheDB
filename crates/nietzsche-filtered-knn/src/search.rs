//! Filtered KNN search: brute-force search over bitmap-selected nodes.
//!
//! This module implements the core filtered vector search: given a query point
//! in the Poincare ball, a Roaring bitmap of candidate indices, and a mapping
//! from indices to node UUIDs, it returns the top-k nearest neighbors measured
//! by hyperbolic (Poincare) distance.
//!
//! ## Design
//!
//! The search is intentionally brute-force over the filtered subset. This is
//! the correct strategy when:
//! - The bitmap is small (e.g. < 10k nodes after WHERE filtering)
//! - Exact results are required (no ANN approximation)
//! - The HNSW index does not natively support the filter predicates
//!
//! For large unfiltered searches, use the HNSW index directly. The HNSW
//! integration (passing the bitmap into `search_layer0`) can be wired later.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use roaring::RoaringBitmap;
use uuid::Uuid;

use nietzsche_graph::{GraphStorage, PoincareVector};

use crate::error::FilteredKnnError;

// ─────────────────────────────────────────────
// Candidate for max-heap
// ─────────────────────────────────────────────

/// A search candidate with reverse ordering so `BinaryHeap` acts as a max-heap
/// on distance (we pop the farthest candidate when the heap exceeds k).
#[derive(Debug, Clone)]
struct KnnCandidate {
    idx: u32,
    id: Uuid,
    distance: f64,
}

impl PartialEq for KnnCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.idx == other.idx
    }
}

impl Eq for KnnCandidate {}

impl PartialOrd for KnnCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KnnCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority (gets popped first)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

// ─────────────────────────────────────────────
// Filtered KNN search
// ─────────────────────────────────────────────

/// Perform a filtered KNN search over the Poincare ball.
///
/// Iterates over every index set in `filter_bitmap`, loads the corresponding
/// embedding from storage, computes the Poincare distance to `query`, and
/// maintains a max-heap of size `k` to select the nearest neighbors.
///
/// # Arguments
///
/// - `query` — Query point coordinates (f32, same dimensionality as stored embeddings).
/// - `k` — Number of nearest neighbors to return.
/// - `filter_bitmap` — Roaring bitmap of allowed indices into `node_ids`.
/// - `node_ids` — Ordered list of node UUIDs (indices in this array correspond
///   to bits in the bitmap).
/// - `storage` — Graph storage from which to load embeddings.
///
/// # Returns
///
/// A `Vec<(Uuid, f64)>` of up to `k` results, sorted ascending by Poincare
/// distance. Each tuple is `(node_id, distance)`.
///
/// # Errors
///
/// Returns [`FilteredKnnError`] if:
/// - `k` is zero.
/// - A node referenced in the bitmap cannot be found in storage.
/// - A dimension mismatch is detected between query and stored embeddings.
pub fn filtered_knn(
    query: &[f32],
    k: usize,
    filter_bitmap: &RoaringBitmap,
    node_ids: &[Uuid],
    storage: &GraphStorage,
) -> Result<Vec<(Uuid, f64)>, FilteredKnnError> {
    if k == 0 {
        return Err(FilteredKnnError::InvalidK);
    }

    if filter_bitmap.is_empty() {
        return Ok(Vec::new());
    }

    let query_pv = PoincareVector::new(query.to_vec());

    // Max-heap: the root is the farthest candidate. When heap.len() > k, pop root.
    let mut heap: BinaryHeap<KnnCandidate> = BinaryHeap::with_capacity(k + 1);

    for idx in filter_bitmap.iter() {
        let i = idx as usize;
        if i >= node_ids.len() {
            continue;
        }

        let node_id = node_ids[i];
        let embedding = match storage.get_embedding(&node_id)? {
            Some(emb) => emb,
            None => continue, // Skip nodes whose embeddings are missing
        };

        // Dimension check (only on first mismatch to avoid repeated overhead)
        if embedding.dim != query.len() {
            return Err(FilteredKnnError::DimensionMismatch {
                query: query.len(),
                storage: embedding.dim,
            });
        }

        let distance = query_pv.distance(&embedding);

        // Early pruning: if heap is full and this node is farther than the worst, skip
        if heap.len() >= k {
            if let Some(worst) = heap.peek() {
                if distance >= worst.distance {
                    continue;
                }
            }
        }

        heap.push(KnnCandidate {
            idx,
            id: node_id,
            distance,
        });

        if heap.len() > k {
            heap.pop(); // Remove the farthest
        }
    }

    // Extract results and sort ascending by distance
    let mut results: Vec<(Uuid, f64)> = heap
        .into_iter()
        .map(|c| (c.id, c.distance))
        .collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    Ok(results)
}

/// Perform an unfiltered KNN search (brute-force over all provided node_ids).
///
/// Convenience wrapper that creates a full bitmap (all bits set) and delegates
/// to [`filtered_knn`].
pub fn brute_force_knn(
    query: &[f32],
    k: usize,
    node_ids: &[Uuid],
    storage: &GraphStorage,
) -> Result<Vec<(Uuid, f64)>, FilteredKnnError> {
    let mut full_bitmap = RoaringBitmap::new();
    for i in 0..node_ids.len() as u32 {
        full_bitmap.insert(i);
    }
    filtered_knn(query, k, &full_bitmap, node_ids, storage)
}
