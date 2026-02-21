//! Build Roaring Bitmaps from filter conditions against NietzscheDB graph storage.
//!
//! Each [`NodeFilter`] variant maps to a specific scan strategy over
//! [`GraphStorage`]. The resulting [`RoaringBitmap`] contains integer indices
//! into a caller-provided `node_ids` slice, making it composable with any
//! downstream search (brute-force, HNSW, etc.).

use std::collections::HashMap;

use roaring::RoaringBitmap;
use uuid::Uuid;

use nietzsche_graph::GraphStorage;

use crate::error::FilteredKnnError;

// ─────────────────────────────────────────────
// NodeFilter enum
// ─────────────────────────────────────────────

/// A declarative filter that produces a Roaring bitmap of matching node indices.
///
/// Filters are applied against [`GraphStorage`] and produce a bitmap whose
/// set bits correspond to positions in a caller-provided `node_ids` array.
#[derive(Debug, Clone)]
pub enum NodeFilter {
    /// Node energy is within the inclusive range `[min, max]`.
    ///
    /// Uses the `energy_idx` secondary index for O(log N + k) scans.
    EnergyRange { min: f32, max: f32 },

    /// Node type matches the given string representation.
    ///
    /// Compared against `format!("{:?}", node_meta.node_type)`.
    NodeType(String),

    /// A content JSON field at `path` (dot-separated) equals `value`.
    ///
    /// Full scan: no secondary index on content fields.
    ContentField {
        path: String,
        value: serde_json::Value,
    },

    /// A content JSON field at `path` (dot-separated) exists and is not null.
    ContentFieldExists(String),

    /// Intersection of multiple filters (logical AND).
    And(Vec<NodeFilter>),

    /// Union of multiple filters (logical OR).
    Or(Vec<NodeFilter>),
}

// ─────────────────────────────────────────────
// Build bitmap from filter
// ─────────────────────────────────────────────

/// Build a [`RoaringBitmap`] from a filter against storage.
///
/// The bitmap contains integer indices into the provided `node_ids` list.
/// `id_to_idx` must map every UUID in `node_ids` to its corresponding index.
///
/// # Errors
///
/// Returns [`FilteredKnnError`] if the underlying storage scan fails.
pub fn build_filter_bitmap(
    filter: &NodeFilter,
    storage: &GraphStorage,
    node_ids: &[Uuid],
    id_to_idx: &HashMap<Uuid, u32>,
) -> Result<RoaringBitmap, FilteredKnnError> {
    match filter {
        NodeFilter::EnergyRange { min, max } => {
            let mut bm = RoaringBitmap::new();
            let nodes = storage.scan_nodes_energy_range(*min, *max)?;
            for node in nodes {
                if let Some(&idx) = id_to_idx.get(&node.id) {
                    bm.insert(idx);
                }
            }
            Ok(bm)
        }

        NodeFilter::NodeType(nt) => {
            let mut bm = RoaringBitmap::new();
            for meta_result in storage.iter_nodes_meta() {
                let meta = meta_result?;
                if format!("{:?}", meta.node_type) == *nt {
                    if let Some(&idx) = id_to_idx.get(&meta.id) {
                        bm.insert(idx);
                    }
                }
            }
            Ok(bm)
        }

        NodeFilter::ContentField { path, value } => {
            let mut bm = RoaringBitmap::new();
            for meta_result in storage.iter_nodes_meta() {
                let meta = meta_result?;
                if json_path_matches(&meta.content, path, value) {
                    if let Some(&idx) = id_to_idx.get(&meta.id) {
                        bm.insert(idx);
                    }
                }
            }
            Ok(bm)
        }

        NodeFilter::ContentFieldExists(path) => {
            let mut bm = RoaringBitmap::new();
            for meta_result in storage.iter_nodes_meta() {
                let meta = meta_result?;
                if json_path_exists(&meta.content, path) {
                    if let Some(&idx) = id_to_idx.get(&meta.id) {
                        bm.insert(idx);
                    }
                }
            }
            Ok(bm)
        }

        NodeFilter::And(filters) => {
            if filters.is_empty() {
                // Empty AND = universal set (all nodes pass)
                let mut bm = RoaringBitmap::new();
                for idx in 0..node_ids.len() as u32 {
                    bm.insert(idx);
                }
                return Ok(bm);
            }
            let mut result = build_filter_bitmap(&filters[0], storage, node_ids, id_to_idx)?;
            for f in &filters[1..] {
                let sub = build_filter_bitmap(f, storage, node_ids, id_to_idx)?;
                result &= sub;
            }
            Ok(result)
        }

        NodeFilter::Or(filters) => {
            if filters.is_empty() {
                // Empty OR = empty set (no nodes pass)
                return Ok(RoaringBitmap::new());
            }
            let mut result = build_filter_bitmap(&filters[0], storage, node_ids, id_to_idx)?;
            for f in &filters[1..] {
                let sub = build_filter_bitmap(f, storage, node_ids, id_to_idx)?;
                result |= sub;
            }
            Ok(result)
        }
    }
}

// ─────────────────────────────────────────────
// JSON path helpers
// ─────────────────────────────────────────────

/// Navigate a JSON value by a dot-separated path and compare with `expected`.
///
/// # Examples
///
/// ```ignore
/// let json = serde_json::json!({"a": {"b": 42}});
/// assert!(json_path_matches(&json, "a.b", &serde_json::json!(42)));
/// ```
pub fn json_path_matches(
    content: &serde_json::Value,
    path: &str,
    expected: &serde_json::Value,
) -> bool {
    let mut current = content;
    for part in path.split('.') {
        match current.get(part) {
            Some(v) => current = v,
            None => return false,
        }
    }
    current == expected
}

/// Check whether a dot-separated JSON path exists and is not null.
pub fn json_path_exists(content: &serde_json::Value, path: &str) -> bool {
    let mut current = content;
    for part in path.split('.') {
        match current.get(part) {
            Some(v) => current = v,
            None => return false,
        }
    }
    !current.is_null()
}

// ─────────────────────────────────────────────
// Utility: build id-to-index mapping
// ─────────────────────────────────────────────

/// Build a `HashMap<Uuid, u32>` mapping node UUIDs to their index in `node_ids`.
///
/// This is a convenience function for callers that need to construct the
/// `id_to_idx` argument required by [`build_filter_bitmap`].
pub fn build_id_to_idx(node_ids: &[Uuid]) -> HashMap<Uuid, u32> {
    node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i as u32))
        .collect()
}
