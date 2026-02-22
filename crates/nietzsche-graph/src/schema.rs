//! Schema validation for NietzscheDB nodes.
//!
//! Allows defining per-`NodeType` constraints: required fields, field types,
//! and optional field-level constraints.  Schemas are persisted in `CF_META`
//! with key prefix `schema:`.
//!
//! ## Usage
//!
//! ```text
//! SetSchema { node_type: "Episodic",
//!             required_fields: ["title", "source"],
//!             field_types: { "title": "string", "score": "number" } }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::GraphError;
use crate::model::NodeMeta;
use crate::storage::GraphStorage;

// ─────────────────────────────────────────────
// Constraint types
// ─────────────────────────────────────────────

/// Expected JSON type for a metadata field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    String,
    Number,
    Bool,
    Array,
    Object,
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldType::String => write!(f, "string"),
            FieldType::Number => write!(f, "number"),
            FieldType::Bool   => write!(f, "bool"),
            FieldType::Array  => write!(f, "array"),
            FieldType::Object => write!(f, "object"),
        }
    }
}

/// Schema constraint for a single `NodeType`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConstraint {
    /// The node type this constraint applies to (e.g. "Episodic", "Semantic").
    pub node_type: String,
    /// Fields that must be present in `metadata`.
    pub required_fields: Vec<String>,
    /// Expected types for specific fields.  Fields not listed here accept any type.
    pub field_types: HashMap<String, FieldType>,
}

// ─────────────────────────────────────────────
// Validator
// ─────────────────────────────────────────────

/// Validates node metadata against registered schema constraints.
#[derive(Debug, Clone, Default)]
pub struct SchemaValidator {
    constraints: HashMap<String, SchemaConstraint>,
}

impl SchemaValidator {
    pub fn new() -> Self {
        Self { constraints: HashMap::new() }
    }

    /// Register or replace a constraint for a node type.
    pub fn set_constraint(&mut self, constraint: SchemaConstraint) {
        self.constraints.insert(constraint.node_type.clone(), constraint);
    }

    /// Remove the constraint for a node type.
    pub fn remove_constraint(&mut self, node_type: &str) {
        self.constraints.remove(node_type);
    }

    /// Get the constraint for a node type (if any).
    pub fn get_constraint(&self, node_type: &str) -> Option<&SchemaConstraint> {
        self.constraints.get(node_type)
    }

    /// List all registered constraints.
    pub fn list_constraints(&self) -> Vec<&SchemaConstraint> {
        self.constraints.values().collect()
    }

    /// Validate a node's metadata against the schema for its type.
    ///
    /// Returns `Ok(())` if valid or no schema exists for this type.
    /// Returns `Err` with a list of violations if invalid.
    pub fn validate_node(&self, meta: &NodeMeta) -> Result<(), Vec<String>> {
        let type_name = format!("{:?}", meta.node_type);
        let constraint = match self.constraints.get(&type_name) {
            Some(c) => c,
            None => return Ok(()), // No schema → always valid
        };

        let mut violations = Vec::new();

        // Check required fields
        for field in &constraint.required_fields {
            if !meta.metadata.contains_key(field) {
                violations.push(format!(
                    "missing required field '{}' for node type '{}'",
                    field, type_name,
                ));
            }
        }

        // Check field types
        for (field, expected_type) in &constraint.field_types {
            if let Some(value) = meta.metadata.get(field) {
                let type_ok = match expected_type {
                    FieldType::String => value.is_string(),
                    FieldType::Number => value.is_number(),
                    FieldType::Bool   => value.is_boolean(),
                    FieldType::Array  => value.is_array(),
                    FieldType::Object => value.is_object(),
                };
                if !type_ok {
                    violations.push(format!(
                        "field '{}' expected type '{}' but got '{}'",
                        field, expected_type, json_type_name(value),
                    ));
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }

    // ── Persistence ──────────────────────────────────────

    /// Persist a constraint to RocksDB CF_META.
    pub fn save_constraint(
        &self,
        storage: &GraphStorage,
        constraint: &SchemaConstraint,
    ) -> Result<(), GraphError> {
        let key = format!("schema:{}", constraint.node_type);
        let bytes = serde_json::to_vec(constraint)
            .map_err(|e| GraphError::Storage(format!("schema serialize: {e}")))?;
        storage.put_meta(&key, &bytes)
    }

    /// Delete a constraint from RocksDB CF_META.
    pub fn delete_constraint(
        &self,
        storage: &GraphStorage,
        node_type: &str,
    ) -> Result<(), GraphError> {
        let key = format!("schema:{}", node_type);
        storage.delete_meta(&key)
    }

    /// Load all persisted constraints from RocksDB CF_META.
    pub fn load_all(storage: &GraphStorage) -> Result<Self, GraphError> {
        let mut validator = Self::new();
        // Scan all keys with prefix "schema:"
        let prefix = b"schema:";
        let iter = storage.scan_meta_prefix(prefix)?;
        for (_, value) in iter {
            let constraint: SchemaConstraint = serde_json::from_slice(&value)
                .map_err(|e| GraphError::Storage(format!("schema deserialize: {e}")))?;
            validator.set_constraint(constraint);
        }
        Ok(validator)
    }
}

/// Human-readable JSON type name.
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null      => "null",
        serde_json::Value::Bool(_)   => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_)  => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{NodeMeta, NodeType};
    use uuid::Uuid;

    fn make_meta(node_type: NodeType, metadata: HashMap<String, serde_json::Value>) -> NodeMeta {
        NodeMeta {
            id: Uuid::new_v4(),
            depth: 0.5,
            content: serde_json::Value::Null,
            node_type,
            energy: 0.8,
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: 0,
            expires_at: None,
            metadata,
            valence: 0.0,
            arousal: 0.0,
            is_phantom: false,
        }
    }

    #[test]
    fn no_schema_always_valid() {
        let validator = SchemaValidator::new();
        let meta = make_meta(NodeType::Episodic, HashMap::new());
        assert!(validator.validate_node(&meta).is_ok());
    }

    #[test]
    fn required_fields_enforced() {
        let mut validator = SchemaValidator::new();
        validator.set_constraint(SchemaConstraint {
            node_type: "Episodic".into(),
            required_fields: vec!["title".into(), "source".into()],
            field_types: HashMap::new(),
        });

        // Missing both fields
        let meta = make_meta(NodeType::Episodic, HashMap::new());
        let errors = validator.validate_node(&meta).unwrap_err();
        assert_eq!(errors.len(), 2);

        // Has title but not source
        let mut md = HashMap::new();
        md.insert("title".into(), serde_json::json!("hello"));
        let meta = make_meta(NodeType::Episodic, md);
        let errors = validator.validate_node(&meta).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("source"));
    }

    #[test]
    fn field_types_enforced() {
        let mut validator = SchemaValidator::new();
        let mut field_types = HashMap::new();
        field_types.insert("score".into(), FieldType::Number);
        field_types.insert("tags".into(), FieldType::Array);
        validator.set_constraint(SchemaConstraint {
            node_type: "Semantic".into(),
            required_fields: vec![],
            field_types,
        });

        // Wrong type for score
        let mut md = HashMap::new();
        md.insert("score".into(), serde_json::json!("not a number"));
        let meta = make_meta(NodeType::Semantic, md);
        let errors = validator.validate_node(&meta).unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("score"));
        assert!(errors[0].contains("number"));

        // Correct types
        let mut md = HashMap::new();
        md.insert("score".into(), serde_json::json!(0.95));
        md.insert("tags".into(), serde_json::json!(["a", "b"]));
        let meta = make_meta(NodeType::Semantic, md);
        assert!(validator.validate_node(&meta).is_ok());
    }

    #[test]
    fn unregistered_type_passes() {
        let mut validator = SchemaValidator::new();
        validator.set_constraint(SchemaConstraint {
            node_type: "Episodic".into(),
            required_fields: vec!["title".into()],
            field_types: HashMap::new(),
        });

        // Semantic has no constraint → passes
        let meta = make_meta(NodeType::Semantic, HashMap::new());
        assert!(validator.validate_node(&meta).is_ok());
    }
}
