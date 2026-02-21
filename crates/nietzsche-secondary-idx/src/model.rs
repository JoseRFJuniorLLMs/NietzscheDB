use serde::{Deserialize, Serialize};

/// The type of values stored in a secondary index.
///
/// Determines encoding and comparison semantics for the sortable key bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// UTF-8 string values. Sorted lexicographically.
    String,
    /// IEEE 754 f64 values. Sorted using sign-magnitude to ordered encoding.
    Float,
    /// Signed 64-bit integer values. Sorted using big-endian with sign flip.
    Int,
}

impl std::fmt::Display for IndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexType::String => write!(f, "String"),
            IndexType::Float => write!(f, "Float"),
            IndexType::Int => write!(f, "Int"),
        }
    }
}

/// Definition of a secondary index on a JSON field path.
///
/// # Example
/// ```
/// use nietzsche_secondary_idx::model::{IndexDef, IndexType};
///
/// let def = IndexDef {
///     name: "title_idx".to_string(),
///     field_path: "content.title".to_string(),
///     index_type: IndexType::String,
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDef {
    /// Unique name for this index (e.g. "title_idx").
    pub name: String,

    /// Dot-separated JSON field path (e.g. "content.title", "content.score").
    /// The path is resolved against the node's `content` JSON value.
    /// For a path like "title", it resolves to `content["title"]`.
    /// For a path like "nested.field", it resolves to `content["nested"]["field"]`.
    pub field_path: String,

    /// The expected type of the indexed values.
    pub index_type: IndexType,
}
