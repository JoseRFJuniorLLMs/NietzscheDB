use nietzsche_graph::GraphStorage;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::SecondaryIdxError;
use crate::model::IndexDef;
use crate::store::{resolve_field_path, SecondaryIndexStore};

/// Builder that creates, backfills, and drops secondary indexes.
///
/// All operations are stateless — the builder reads/writes through
/// [`SecondaryIndexStore`] which persists everything in `CF_META`.
pub struct SecondaryIndexBuilder;

impl SecondaryIndexBuilder {
    /// Create a new secondary index and backfill it from all existing nodes.
    ///
    /// 1. Checks that no index with the same name already exists.
    /// 2. Persists the [`IndexDef`] under `"idx_def:{name}"`.
    /// 3. Scans all node metadata and inserts an index entry for each node
    ///    whose `content` has a value at the specified `field_path`.
    ///
    /// Returns the number of nodes indexed during backfill.
    pub fn create_index(
        storage: &GraphStorage,
        def: &IndexDef,
    ) -> Result<u64, SecondaryIdxError> {
        // 1. Check for duplicates
        if SecondaryIndexStore::get_index_def(storage, &def.name)?.is_some() {
            return Err(SecondaryIdxError::IndexAlreadyExists(def.name.clone()));
        }

        // 2. Persist definition
        SecondaryIndexStore::put_index_def(storage, def)?;
        info!(index_name = %def.name, field_path = %def.field_path, "created secondary index definition");

        // 3. Backfill from existing nodes
        let mut count = 0u64;
        for result in storage.iter_nodes_meta() {
            let meta = result?;
            if let Some(field_value) = resolve_field_path(&meta.content, &def.field_path) {
                match SecondaryIndexStore::insert_entry(
                    storage,
                    &def.name,
                    &def.index_type,
                    field_value,
                    &meta.id,
                ) {
                    Ok(()) => count += 1,
                    Err(SecondaryIdxError::TypeMismatch { expected, got }) => {
                        warn!(
                            node_id = %meta.id,
                            field_path = %def.field_path,
                            expected = %expected,
                            got = %got,
                            "skipping node: type mismatch during backfill"
                        );
                    }
                    Err(e) => return Err(e),
                }
            } else {
                debug!(
                    node_id = %meta.id,
                    field_path = %def.field_path,
                    "skipping node: field not found"
                );
            }
        }

        info!(index_name = %def.name, backfilled = count, "secondary index backfill complete");
        Ok(count)
    }

    /// Drop a secondary index: remove all entries and the definition.
    ///
    /// Returns the number of index entries deleted.
    pub fn drop_index(
        storage: &GraphStorage,
        name: &str,
    ) -> Result<u64, SecondaryIdxError> {
        // Verify the index exists
        if SecondaryIndexStore::get_index_def(storage, name)?.is_none() {
            return Err(SecondaryIdxError::IndexNotFound(name.to_string()));
        }

        // Delete all entries
        let count = SecondaryIndexStore::delete_all_entries(storage, name)?;

        // Delete the definition
        SecondaryIndexStore::delete_index_def(storage, name)?;

        info!(index_name = %name, entries_deleted = count, "secondary index dropped");
        Ok(count)
    }

    /// Insert a single node into an existing secondary index.
    ///
    /// Call this when a new node is inserted into the graph to keep the
    /// secondary index up to date.
    ///
    /// Returns `Ok(true)` if the node was indexed, `Ok(false)` if the field
    /// was not present in the node's content.
    pub fn insert_entry(
        storage: &GraphStorage,
        index_name: &str,
        node_id: &Uuid,
        content: &serde_json::Value,
    ) -> Result<bool, SecondaryIdxError> {
        let def = SecondaryIndexStore::get_index_def(storage, index_name)?
            .ok_or_else(|| SecondaryIdxError::IndexNotFound(index_name.to_string()))?;

        if let Some(field_value) = resolve_field_path(content, &def.field_path) {
            SecondaryIndexStore::insert_entry(
                storage,
                &def.name,
                &def.index_type,
                field_value,
                node_id,
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Delete a single node from an existing secondary index.
    ///
    /// Call this when a node is removed or updated to keep the index consistent.
    pub fn delete_entry(
        storage: &GraphStorage,
        index_name: &str,
        node_id: &Uuid,
        old_value: &serde_json::Value,
    ) -> Result<(), SecondaryIdxError> {
        let def = SecondaryIndexStore::get_index_def(storage, index_name)?
            .ok_or_else(|| SecondaryIdxError::IndexNotFound(index_name.to_string()))?;

        SecondaryIndexStore::delete_entry(
            storage,
            &def.name,
            &def.index_type,
            old_value,
            node_id,
        )?;
        Ok(())
    }

    /// Look up all node IDs matching an exact value in a secondary index.
    pub fn lookup(
        storage: &GraphStorage,
        index_name: &str,
        value: &serde_json::Value,
    ) -> Result<Vec<Uuid>, SecondaryIdxError> {
        let def = SecondaryIndexStore::get_index_def(storage, index_name)?
            .ok_or_else(|| SecondaryIdxError::IndexNotFound(index_name.to_string()))?;

        SecondaryIndexStore::lookup(storage, &def.name, &def.index_type, value)
    }

    /// Range lookup: find all node IDs where the indexed value is in `[min, max]`.
    ///
    /// Only meaningful for `Float` and `Int` index types.
    pub fn range_lookup(
        storage: &GraphStorage,
        index_name: &str,
        min_value: &serde_json::Value,
        max_value: &serde_json::Value,
    ) -> Result<Vec<Uuid>, SecondaryIdxError> {
        let def = SecondaryIndexStore::get_index_def(storage, index_name)?
            .ok_or_else(|| SecondaryIdxError::IndexNotFound(index_name.to_string()))?;

        SecondaryIndexStore::range_lookup(
            storage,
            &def.name,
            &def.index_type,
            min_value,
            max_value,
        )
    }

    /// List all secondary index definitions.
    pub fn list_indexes(
        storage: &GraphStorage,
    ) -> Result<Vec<IndexDef>, SecondaryIdxError> {
        SecondaryIndexStore::list_indexes(storage)
    }
}
