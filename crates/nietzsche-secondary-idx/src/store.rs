use nietzsche_graph::GraphStorage;
use uuid::Uuid;

use crate::error::SecondaryIdxError;
use crate::model::{IndexDef, IndexType};

// ─────────────────────────────────────────────
// Key prefixes stored in CF_META
// ─────────────────────────────────────────────

/// Prefix for index definition entries: `"idx_def:{name}"`.
const IDX_DEF_PREFIX: &str = "idx_def:";

/// Prefix for index data entries: `"sidx:{name}:{sortable_value}:{node_id}"`.
const SIDX_PREFIX: &str = "sidx:";

// ─────────────────────────────────────────────
// SecondaryIndexStore
// ─────────────────────────────────────────────

/// Manages secondary index entries stored in the `CF_META` column family
/// of the underlying [`GraphStorage`].
///
/// Index definitions are stored under `"idx_def:{name}"`.
/// Index entries are stored under `"sidx:{name}:{sortable_value}:{node_id}"`.
pub struct SecondaryIndexStore;

impl SecondaryIndexStore {
    // ── Index definition CRUD ──────────────────────

    /// Persist an index definition.
    pub fn put_index_def(
        storage: &GraphStorage,
        def: &IndexDef,
    ) -> Result<(), SecondaryIdxError> {
        let key = format!("{}{}", IDX_DEF_PREFIX, def.name);
        let value = serde_json::to_vec(def)?;
        storage.put_meta(&key, &value)?;
        Ok(())
    }

    /// Read an index definition by name.
    pub fn get_index_def(
        storage: &GraphStorage,
        name: &str,
    ) -> Result<Option<IndexDef>, SecondaryIdxError> {
        let key = format!("{}{}", IDX_DEF_PREFIX, name);
        match storage.get_meta(&key)? {
            Some(bytes) => {
                let def: IndexDef = serde_json::from_slice(&bytes)?;
                Ok(Some(def))
            }
            None => Ok(None),
        }
    }

    /// Delete an index definition.
    pub fn delete_index_def(
        storage: &GraphStorage,
        name: &str,
    ) -> Result<(), SecondaryIdxError> {
        let key = format!("{}{}", IDX_DEF_PREFIX, name);
        storage.delete_meta(&key)?;
        Ok(())
    }

    /// List all index definitions.
    pub fn list_indexes(
        storage: &GraphStorage,
    ) -> Result<Vec<IndexDef>, SecondaryIdxError> {
        let entries = storage.scan_meta_prefix(IDX_DEF_PREFIX.as_bytes())?;
        let mut defs = Vec::with_capacity(entries.len());
        for (_key, value) in entries {
            let def: IndexDef = serde_json::from_slice(&value)?;
            defs.push(def);
        }
        Ok(defs)
    }

    // ── Index entry operations ────────────────────

    /// Insert a single index entry for a node.
    ///
    /// Key format: `"sidx:{index_name}:{sortable_value}:{node_id}"`
    pub fn insert_entry(
        storage: &GraphStorage,
        index_name: &str,
        index_type: &IndexType,
        value: &serde_json::Value,
        node_id: &Uuid,
    ) -> Result<(), SecondaryIdxError> {
        let sortable = encode_sortable_value(index_type, value)?;
        let key = format!(
            "{}{}:{}:{}",
            SIDX_PREFIX, index_name, sortable, node_id
        );
        storage.put_meta(&key, &[])?;
        Ok(())
    }

    /// Delete a single index entry for a node.
    pub fn delete_entry(
        storage: &GraphStorage,
        index_name: &str,
        index_type: &IndexType,
        old_value: &serde_json::Value,
        node_id: &Uuid,
    ) -> Result<(), SecondaryIdxError> {
        let sortable = encode_sortable_value(index_type, old_value)?;
        let key = format!(
            "{}{}:{}:{}",
            SIDX_PREFIX, index_name, sortable, node_id
        );
        storage.delete_meta(&key)?;
        Ok(())
    }

    /// Look up all node IDs matching an exact value in the given index.
    pub fn lookup(
        storage: &GraphStorage,
        index_name: &str,
        index_type: &IndexType,
        value: &serde_json::Value,
    ) -> Result<Vec<Uuid>, SecondaryIdxError> {
        let sortable = encode_sortable_value(index_type, value)?;
        let prefix = format!("{}{}:{}:", SIDX_PREFIX, index_name, sortable);
        let entries = storage.scan_meta_prefix(prefix.as_bytes())?;
        let mut ids = Vec::with_capacity(entries.len());
        for (key_bytes, _) in entries {
            let key_str = std::str::from_utf8(&key_bytes)
                .map_err(|e| SecondaryIdxError::Storage(e.to_string()))?;
            if let Some(uuid_str) = key_str.rsplit(':').next() {
                let id = Uuid::parse_str(uuid_str)
                    .map_err(|e| SecondaryIdxError::Storage(e.to_string()))?;
                ids.push(id);
            }
        }
        Ok(ids)
    }

    /// Range lookup: find all node IDs where the indexed value is in `[min, max]` (inclusive).
    ///
    /// This works by scanning the prefix `"sidx:{index_name}:"` and filtering entries
    /// whose sortable value falls within the range.
    pub fn range_lookup(
        storage: &GraphStorage,
        index_name: &str,
        index_type: &IndexType,
        min_value: &serde_json::Value,
        max_value: &serde_json::Value,
    ) -> Result<Vec<Uuid>, SecondaryIdxError> {
        let min_sortable = encode_sortable_value(index_type, min_value)?;
        let max_sortable = encode_sortable_value(index_type, max_value)?;

        // Scan all entries for this index
        let prefix = format!("{}{}:", SIDX_PREFIX, index_name);
        let entries = storage.scan_meta_prefix(prefix.as_bytes())?;

        let mut ids = Vec::new();
        for (key_bytes, _) in entries {
            let key_str = std::str::from_utf8(&key_bytes)
                .map_err(|e| SecondaryIdxError::Storage(e.to_string()))?;

            // Parse: "sidx:{name}:{sortable_value}:{node_id}"
            // After removing the prefix "sidx:{name}:", we have "{sortable_value}:{node_id}"
            let after_prefix = &key_str[prefix.len()..];

            // The node_id is a UUID (36 chars with hyphens), always at the end after the last ':'
            // The sortable_value is everything before the last ':'
            if let Some(last_colon) = after_prefix.rfind(':') {
                let entry_sortable = &after_prefix[..last_colon];
                let uuid_str = &after_prefix[last_colon + 1..];

                if entry_sortable >= min_sortable.as_str()
                    && entry_sortable <= max_sortable.as_str()
                {
                    let id = Uuid::parse_str(uuid_str)
                        .map_err(|e| SecondaryIdxError::Storage(e.to_string()))?;
                    ids.push(id);
                }
            }
        }
        Ok(ids)
    }

    /// Delete all index entries for a given index name.
    pub fn delete_all_entries(
        storage: &GraphStorage,
        index_name: &str,
    ) -> Result<u64, SecondaryIdxError> {
        let prefix = format!("{}{}:", SIDX_PREFIX, index_name);
        let entries = storage.scan_meta_prefix(prefix.as_bytes())?;
        let mut count = 0u64;
        for (key_bytes, _) in entries {
            let key_str = std::str::from_utf8(&key_bytes)
                .map_err(|e| SecondaryIdxError::Storage(e.to_string()))?;
            storage.delete_meta(key_str)?;
            count += 1;
        }
        Ok(count)
    }
}

// ─────────────────────────────────────────────
// Sortable value encoding
// ─────────────────────────────────────────────

/// Encode a JSON value into a sortable string representation suitable for
/// lexicographic ordering in RocksDB prefix scans.
///
/// - **String**: The raw UTF-8 string value.
/// - **Float**: IEEE 754 sign-magnitude to ordered encoding, rendered as 16 hex chars.
/// - **Int**: Big-endian encoding with sign flip, rendered as 16 hex chars.
fn encode_sortable_value(
    index_type: &IndexType,
    value: &serde_json::Value,
) -> Result<String, SecondaryIdxError> {
    match index_type {
        IndexType::String => {
            match value.as_str() {
                Some(s) => Ok(s.to_string()),
                None => Err(SecondaryIdxError::TypeMismatch {
                    expected: "String".to_string(),
                    got: value_type_name(value),
                }),
            }
        }
        IndexType::Float => {
            let f = value
                .as_f64()
                .ok_or_else(|| SecondaryIdxError::TypeMismatch {
                    expected: "Float".to_string(),
                    got: value_type_name(value),
                })?;
            Ok(encode_f64_sortable(f))
        }
        IndexType::Int => {
            let i = value
                .as_i64()
                .ok_or_else(|| SecondaryIdxError::TypeMismatch {
                    expected: "Int".to_string(),
                    got: value_type_name(value),
                })?;
            Ok(encode_i64_sortable(i))
        }
    }
}

/// IEEE 754 sign-magnitude to ordered encoding for f64.
///
/// Positive floats: XOR with 0x8000_0000_0000_0000 (flip sign bit).
/// Negative floats: bitwise NOT (flip all bits).
///
/// This produces a u64 whose big-endian byte order matches the natural
/// ordering of f64 values.
fn encode_f64_sortable(f: f64) -> String {
    let raw = f.to_bits();
    let sortable = if raw >> 63 == 0 {
        raw ^ 0x8000_0000_0000_0000
    } else {
        !raw
    };
    format!("{:016x}", sortable)
}

/// Big-endian encoding with sign flip for i64.
///
/// XOR with 0x8000_0000_0000_0000 flips the sign bit so that negative values
/// sort before positive values in lexicographic (big-endian) order.
fn encode_i64_sortable(i: i64) -> String {
    let sortable = (i as u64) ^ 0x8000_0000_0000_0000;
    format!("{:016x}", sortable)
}

/// Returns a human-readable type name for a JSON value.
fn value_type_name(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "Null".to_string(),
        serde_json::Value::Bool(_) => "Bool".to_string(),
        serde_json::Value::Number(_) => "Number".to_string(),
        serde_json::Value::String(_) => "String".to_string(),
        serde_json::Value::Array(_) => "Array".to_string(),
        serde_json::Value::Object(_) => "Object".to_string(),
    }
}

/// Resolve a dot-separated field path against a JSON value.
///
/// For example, `resolve_field_path(content, "title")` returns `content["title"]`.
/// `resolve_field_path(content, "nested.field")` returns `content["nested"]["field"]`.
pub fn resolve_field_path<'a>(
    content: &'a serde_json::Value,
    field_path: &str,
) -> Option<&'a serde_json::Value> {
    let mut current = content;
    for segment in field_path.split('.') {
        current = current.get(segment)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_f64_sortable_ordering() {
        let neg = encode_f64_sortable(-10.0);
        let zero = encode_f64_sortable(0.0);
        let pos_small = encode_f64_sortable(1.5);
        let pos_large = encode_f64_sortable(100.0);

        assert!(neg < zero, "neg={neg} should < zero={zero}");
        assert!(zero < pos_small, "zero={zero} should < pos_small={pos_small}");
        assert!(
            pos_small < pos_large,
            "pos_small={pos_small} should < pos_large={pos_large}"
        );
    }

    #[test]
    fn test_encode_i64_sortable_ordering() {
        let neg = encode_i64_sortable(-100);
        let zero = encode_i64_sortable(0);
        let pos = encode_i64_sortable(100);

        assert!(neg < zero, "neg={neg} should < zero={zero}");
        assert!(zero < pos, "zero={zero} should < pos={pos}");
    }

    #[test]
    fn test_resolve_field_path_simple() {
        let content = serde_json::json!({"title": "hello", "score": 42});
        assert_eq!(
            resolve_field_path(&content, "title"),
            Some(&serde_json::json!("hello"))
        );
        assert_eq!(
            resolve_field_path(&content, "score"),
            Some(&serde_json::json!(42))
        );
    }

    #[test]
    fn test_resolve_field_path_nested() {
        let content = serde_json::json!({"nested": {"field": "value"}});
        assert_eq!(
            resolve_field_path(&content, "nested.field"),
            Some(&serde_json::json!("value"))
        );
    }

    #[test]
    fn test_resolve_field_path_missing() {
        let content = serde_json::json!({"title": "hello"});
        assert_eq!(resolve_field_path(&content, "missing"), None);
    }
}
