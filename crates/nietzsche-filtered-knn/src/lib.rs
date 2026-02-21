//! # nietzsche-filtered-knn
//!
//! Filtered KNN search with Roaring Bitmap pre-filtering for NietzscheDB.
//!
//! This crate provides a clean, standalone filtered vector search that
//! works with any Roaring bitmap and falls back to brute-force when needed.
//!
//! ## Architecture
//!
//! The workflow is:
//!
//! 1. **Build filters**: Convert NQL `WHERE` clause predicates into
//!    [`NodeFilter`] instances.
//! 2. **Build bitmap**: Call [`build_filter_bitmap`] against [`GraphStorage`]
//!    to produce a [`RoaringBitmap`] of candidate node indices.
//! 3. **Search**: Call [`filtered_knn`] with the bitmap to perform brute-force
//!    Poincare-distance KNN over only the matching nodes.
//!
//! ## Relationship to HNSW
//!
//! The HNSW index in `hyperspace-index` already has internal Roaring Bitmap
//! support for metadata-based filtering. This crate provides an **orthogonal**
//! filtered KNN that operates at the NietzscheDB graph layer (content fields,
//! energy, node type) rather than the raw vector metadata layer. The two can
//! be composed: use this crate's bitmap as input to HNSW's `search_layer0`
//! allowed bitmap for best-of-both-worlds filtering.
//!
//! ## Example
//!
//! ```ignore
//! use nietzsche_filtered_knn::{NodeFilter, build_filter_bitmap, build_id_to_idx, filtered_knn};
//!
//! // 1. Collect candidate node IDs
//! let node_ids: Vec<Uuid> = storage.scan_nodes_meta()?.iter().map(|m| m.id).collect();
//! let id_to_idx = build_id_to_idx(&node_ids);
//!
//! // 2. Build filter bitmap
//! let filter = NodeFilter::EnergyRange { min: 0.5, max: 1.0 };
//! let bitmap = build_filter_bitmap(&filter, &storage, &node_ids, &id_to_idx)?;
//!
//! // 3. Search
//! let results = filtered_knn(&query_vec, 10, &bitmap, &node_ids, &storage)?;
//! ```

pub mod error;
pub mod filter;
pub mod search;

// Re-exports for convenience
pub use error::FilteredKnnError;
pub use filter::{build_filter_bitmap, build_id_to_idx, json_path_exists, json_path_matches, NodeFilter};
pub use search::{brute_force_knn, filtered_knn};
