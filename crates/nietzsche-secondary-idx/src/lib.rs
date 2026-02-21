//! # nietzsche-secondary-idx
//!
//! Secondary indexes on arbitrary JSON content fields for NietzscheDB.
//!
//! This crate allows creating indexes on dot-separated field paths within a
//! node's `content` JSON value (e.g. `"title"`, `"category"`, `"nested.score"`).
//!
//! ## Supported index types
//!
//! - **String** -- exact match and lexicographic range queries.
//! - **Float** -- IEEE 754 f64 values with sign-magnitude ordered encoding.
//! - **Int** -- signed 64-bit integers with big-endian sign-flip encoding.
//!
//! ## Storage layout
//!
//! All data is stored in the existing `CF_META` column family:
//!
//! - Index definitions: `"idx_def:{name}"` -> JSON-serialized [`IndexDef`]
//! - Index entries: `"sidx:{name}:{sortable_value}:{node_id}"` -> empty value
//!
//! ## Usage
//!
//! ```rust,no_run
//! use nietzsche_secondary_idx::{SecondaryIndexBuilder, IndexDef, IndexType};
//! use nietzsche_graph::GraphStorage;
//!
//! let storage = GraphStorage::open("/tmp/mydb").unwrap();
//!
//! // Create an index on content.title
//! let def = IndexDef {
//!     name: "title_idx".to_string(),
//!     field_path: "title".to_string(),
//!     index_type: IndexType::String,
//! };
//! let backfilled = SecondaryIndexBuilder::create_index(&storage, &def).unwrap();
//!
//! // Lookup
//! let ids = SecondaryIndexBuilder::lookup(
//!     &storage,
//!     "title_idx",
//!     &serde_json::json!("Hello World"),
//! ).unwrap();
//! ```

pub mod builder;
pub mod error;
pub mod model;
pub mod store;

pub use builder::SecondaryIndexBuilder;
pub use error::SecondaryIdxError;
pub use model::{IndexDef, IndexType};
pub use store::SecondaryIndexStore;

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    /// Open a temporary RocksDB-backed GraphStorage for testing.
    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    /// Create a test node with the given content JSON.
    fn make_node_with_content(content: serde_json::Value) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.1, 0.2]),
            content,
        )
    }

    // ── Test 1: create string index and verify lookup ──────────

    #[test]
    fn test_create_string_index() {
        let (storage, _dir) = open_temp_db();

        // Insert nodes with content.title
        let node1 = make_node_with_content(serde_json::json!({"title": "Alpha"}));
        let node2 = make_node_with_content(serde_json::json!({"title": "Beta"}));
        let node3 = make_node_with_content(serde_json::json!({"other": "no_title"}));
        storage.put_node(&node1).unwrap();
        storage.put_node(&node2).unwrap();
        storage.put_node(&node3).unwrap();

        // Create index on content.title
        let def = IndexDef {
            name: "title_idx".to_string(),
            field_path: "title".to_string(),
            index_type: IndexType::String,
        };
        let backfilled = SecondaryIndexBuilder::create_index(&storage, &def).unwrap();
        assert_eq!(backfilled, 2, "only 2 nodes have content.title");

        // Lookup "Alpha"
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "title_idx",
            &serde_json::json!("Alpha"),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node1.id);

        // Lookup "Beta"
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "title_idx",
            &serde_json::json!("Beta"),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node2.id);

        // Lookup missing value
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "title_idx",
            &serde_json::json!("Gamma"),
        )
        .unwrap();
        assert!(ids.is_empty());
    }

    // ── Test 2: create float index and verify range lookup ─────

    #[test]
    fn test_create_float_index() {
        let (storage, _dir) = open_temp_db();

        let node1 = make_node_with_content(serde_json::json!({"score": 1.5}));
        let node2 = make_node_with_content(serde_json::json!({"score": 3.0}));
        let node3 = make_node_with_content(serde_json::json!({"score": 5.5}));
        storage.put_node(&node1).unwrap();
        storage.put_node(&node2).unwrap();
        storage.put_node(&node3).unwrap();

        let def = IndexDef {
            name: "score_idx".to_string(),
            field_path: "score".to_string(),
            index_type: IndexType::Float,
        };
        let backfilled = SecondaryIndexBuilder::create_index(&storage, &def).unwrap();
        assert_eq!(backfilled, 3);

        // Range lookup [2.0, 4.0] should return node2 only
        let ids = SecondaryIndexBuilder::range_lookup(
            &storage,
            "score_idx",
            &serde_json::json!(2.0),
            &serde_json::json!(4.0),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node2.id);
    }

    // ── Test 3: drop index ────────────────────────────────────

    #[test]
    fn test_drop_index() {
        let (storage, _dir) = open_temp_db();

        let node = make_node_with_content(serde_json::json!({"category": "science"}));
        storage.put_node(&node).unwrap();

        let def = IndexDef {
            name: "cat_idx".to_string(),
            field_path: "category".to_string(),
            index_type: IndexType::String,
        };
        SecondaryIndexBuilder::create_index(&storage, &def).unwrap();

        // Verify the index works
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "cat_idx",
            &serde_json::json!("science"),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);

        // Drop the index
        let deleted = SecondaryIndexBuilder::drop_index(&storage, "cat_idx").unwrap();
        assert_eq!(deleted, 1);

        // Verify definition is gone
        let indexes = SecondaryIndexBuilder::list_indexes(&storage).unwrap();
        assert!(indexes.is_empty());

        // Verify lookup fails with IndexNotFound
        let result = SecondaryIndexBuilder::lookup(
            &storage,
            "cat_idx",
            &serde_json::json!("science"),
        );
        assert!(matches!(result, Err(SecondaryIdxError::IndexNotFound(_))));
    }

    // ── Test 4: manually insert entry ─────────────────────────

    #[test]
    fn test_insert_entry() {
        let (storage, _dir) = open_temp_db();

        // Create index def (no backfill since no nodes exist yet)
        let def = IndexDef {
            name: "tag_idx".to_string(),
            field_path: "tag".to_string(),
            index_type: IndexType::String,
        };
        SecondaryIndexBuilder::create_index(&storage, &def).unwrap();

        // Insert a node after index creation
        let node = make_node_with_content(serde_json::json!({"tag": "important"}));
        storage.put_node(&node).unwrap();

        // Manually insert the entry
        let indexed = SecondaryIndexBuilder::insert_entry(
            &storage,
            "tag_idx",
            &node.id,
            &node.content,
        )
        .unwrap();
        assert!(indexed);

        // Verify lookup
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "tag_idx",
            &serde_json::json!("important"),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node.id);
    }

    // ── Test 5: lookup multiple nodes with same value ─────────

    #[test]
    fn test_lookup_multiple() {
        let (storage, _dir) = open_temp_db();

        let node1 = make_node_with_content(serde_json::json!({"category": "physics"}));
        let node2 = make_node_with_content(serde_json::json!({"category": "physics"}));
        let node3 = make_node_with_content(serde_json::json!({"category": "math"}));
        storage.put_node(&node1).unwrap();
        storage.put_node(&node2).unwrap();
        storage.put_node(&node3).unwrap();

        let def = IndexDef {
            name: "cat_idx".to_string(),
            field_path: "category".to_string(),
            index_type: IndexType::String,
        };
        SecondaryIndexBuilder::create_index(&storage, &def).unwrap();

        // Lookup "physics" should return both nodes
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "cat_idx",
            &serde_json::json!("physics"),
        )
        .unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&node1.id));
        assert!(ids.contains(&node2.id));

        // Lookup "math" should return only node3
        let ids = SecondaryIndexBuilder::lookup(
            &storage,
            "cat_idx",
            &serde_json::json!("math"),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node3.id);
    }

    // ── Test 6: float range lookup ───────────────────────────

    #[test]
    fn test_range_lookup() {
        let (storage, _dir) = open_temp_db();

        let node_neg = make_node_with_content(serde_json::json!({"val": -5.0}));
        let node_zero = make_node_with_content(serde_json::json!({"val": 0.0}));
        let node_small = make_node_with_content(serde_json::json!({"val": 2.5}));
        let node_mid = make_node_with_content(serde_json::json!({"val": 7.0}));
        let node_large = make_node_with_content(serde_json::json!({"val": 15.0}));
        storage.put_node(&node_neg).unwrap();
        storage.put_node(&node_zero).unwrap();
        storage.put_node(&node_small).unwrap();
        storage.put_node(&node_mid).unwrap();
        storage.put_node(&node_large).unwrap();

        let def = IndexDef {
            name: "val_idx".to_string(),
            field_path: "val".to_string(),
            index_type: IndexType::Float,
        };
        SecondaryIndexBuilder::create_index(&storage, &def).unwrap();

        // Range [0.0, 10.0] should return zero, small, mid
        let ids = SecondaryIndexBuilder::range_lookup(
            &storage,
            "val_idx",
            &serde_json::json!(0.0),
            &serde_json::json!(10.0),
        )
        .unwrap();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&node_zero.id));
        assert!(ids.contains(&node_small.id));
        assert!(ids.contains(&node_mid.id));
        assert!(!ids.contains(&node_neg.id));
        assert!(!ids.contains(&node_large.id));

        // Range [-10.0, -1.0] should return neg
        let ids = SecondaryIndexBuilder::range_lookup(
            &storage,
            "val_idx",
            &serde_json::json!(-10.0),
            &serde_json::json!(-1.0),
        )
        .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], node_neg.id);
    }

    // ── Test 7: list indexes ─────────────────────────────────

    #[test]
    fn test_list_indexes() {
        let (storage, _dir) = open_temp_db();

        let def1 = IndexDef {
            name: "idx_a".to_string(),
            field_path: "field_a".to_string(),
            index_type: IndexType::String,
        };
        let def2 = IndexDef {
            name: "idx_b".to_string(),
            field_path: "field_b".to_string(),
            index_type: IndexType::Float,
        };

        SecondaryIndexBuilder::create_index(&storage, &def1).unwrap();
        SecondaryIndexBuilder::create_index(&storage, &def2).unwrap();

        let indexes = SecondaryIndexBuilder::list_indexes(&storage).unwrap();
        assert_eq!(indexes.len(), 2);

        let names: Vec<&str> = indexes.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"idx_a"));
        assert!(names.contains(&"idx_b"));
    }

    // ── Test 8: different indexes are independent ────────────

    #[test]
    fn test_different_indexes_independent() {
        let (storage, _dir) = open_temp_db();

        let node1 = make_node_with_content(serde_json::json!({"title": "Alpha", "category": "X"}));
        let node2 = make_node_with_content(serde_json::json!({"title": "Beta", "category": "Y"}));
        storage.put_node(&node1).unwrap();
        storage.put_node(&node2).unwrap();

        let title_def = IndexDef {
            name: "title_idx".to_string(),
            field_path: "title".to_string(),
            index_type: IndexType::String,
        };
        let cat_def = IndexDef {
            name: "category_idx".to_string(),
            field_path: "category".to_string(),
            index_type: IndexType::String,
        };
        SecondaryIndexBuilder::create_index(&storage, &title_def).unwrap();
        SecondaryIndexBuilder::create_index(&storage, &cat_def).unwrap();

        // Lookup in title_idx
        let title_ids = SecondaryIndexBuilder::lookup(
            &storage,
            "title_idx",
            &serde_json::json!("Alpha"),
        )
        .unwrap();
        assert_eq!(title_ids.len(), 1);
        assert_eq!(title_ids[0], node1.id);

        // "Alpha" should NOT appear in category_idx
        let cat_ids = SecondaryIndexBuilder::lookup(
            &storage,
            "category_idx",
            &serde_json::json!("Alpha"),
        )
        .unwrap();
        assert!(cat_ids.is_empty(), "Alpha should not be in category_idx");

        // "X" should appear in category_idx but NOT in title_idx
        let cat_ids = SecondaryIndexBuilder::lookup(
            &storage,
            "category_idx",
            &serde_json::json!("X"),
        )
        .unwrap();
        assert_eq!(cat_ids.len(), 1);
        assert_eq!(cat_ids[0], node1.id);

        let title_ids = SecondaryIndexBuilder::lookup(
            &storage,
            "title_idx",
            &serde_json::json!("X"),
        )
        .unwrap();
        assert!(title_ids.is_empty(), "X should not be in title_idx");
    }
}
