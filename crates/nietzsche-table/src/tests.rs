use serde_json::json;
use uuid::Uuid;

use crate::{ColumnDef, ColumnType, TableSchema, TableStore};

/// Helper to build a simple schema with the given columns.
fn make_schema(name: &str, columns: Vec<ColumnDef>) -> TableSchema {
    TableSchema {
        name: name.to_string(),
        columns,
    }
}

/// Helper to build a common "people" schema.
fn people_schema() -> TableSchema {
    make_schema(
        "people",
        vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Integer,
                nullable: false,
                primary_key: true,
            },
            ColumnDef {
                name: "name".into(),
                col_type: ColumnType::Text,
                nullable: false,
                primary_key: false,
            },
            ColumnDef {
                name: "age".into(),
                col_type: ColumnType::Integer,
                nullable: true,
                primary_key: false,
            },
        ],
    )
}

// -----------------------------------------------------------------------
// 1. test_create_table
// -----------------------------------------------------------------------
#[test]
fn test_create_table() {
    let store = TableStore::open_memory().unwrap();
    let schema = people_schema();
    store.create_table(&schema).unwrap();

    let tables = store.list_tables().unwrap();
    assert!(tables.contains(&"people".to_string()));
}

// -----------------------------------------------------------------------
// 2. test_drop_table
// -----------------------------------------------------------------------
#[test]
fn test_drop_table() {
    let store = TableStore::open_memory().unwrap();
    let schema = people_schema();
    store.create_table(&schema).unwrap();

    assert!(store.list_tables().unwrap().contains(&"people".to_string()));

    store.drop_table("people").unwrap();

    assert!(!store.list_tables().unwrap().contains(&"people".to_string()));
}

// -----------------------------------------------------------------------
// 3. test_insert_and_query
// -----------------------------------------------------------------------
#[test]
fn test_insert_and_query() {
    let store = TableStore::open_memory().unwrap();
    store.create_table(&people_schema()).unwrap();

    let rowid = store
        .insert_row("people", &json!({"id": 1, "name": "Nietzsche", "age": 55}))
        .unwrap();
    assert!(rowid > 0);

    let rows = store.query_rows("people", None, &[]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Nietzsche");
    assert_eq!(rows[0]["age"], 55);
}

// -----------------------------------------------------------------------
// 4. test_insert_multiple_rows
// -----------------------------------------------------------------------
#[test]
fn test_insert_multiple_rows() {
    let store = TableStore::open_memory().unwrap();
    store.create_table(&people_schema()).unwrap();

    for i in 1..=5 {
        store
            .insert_row(
                "people",
                &json!({"id": i, "name": format!("Person_{}", i), "age": 20 + i}),
            )
            .unwrap();
    }

    let rows = store.query_rows("people", None, &[]).unwrap();
    assert_eq!(rows.len(), 5);
}

// -----------------------------------------------------------------------
// 5. test_query_with_filter
// -----------------------------------------------------------------------
#[test]
fn test_query_with_filter() {
    let store = TableStore::open_memory().unwrap();
    store.create_table(&people_schema()).unwrap();

    store
        .insert_row("people", &json!({"id": 1, "name": "Alice", "age": 30}))
        .unwrap();
    store
        .insert_row("people", &json!({"id": 2, "name": "Bob", "age": 25}))
        .unwrap();
    store
        .insert_row("people", &json!({"id": 3, "name": "Charlie", "age": 35}))
        .unwrap();

    let params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(28i64)];
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|b| b.as_ref()).collect();
    let rows = store
        .query_rows("people", Some("age > ?1"), param_refs.as_slice())
        .unwrap();

    assert_eq!(rows.len(), 2);
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"Alice"));
    assert!(names.contains(&"Charlie"));
}

// -----------------------------------------------------------------------
// 6. test_delete_rows
// -----------------------------------------------------------------------
#[test]
fn test_delete_rows() {
    let store = TableStore::open_memory().unwrap();
    store.create_table(&people_schema()).unwrap();

    store
        .insert_row("people", &json!({"id": 1, "name": "Alice", "age": 30}))
        .unwrap();
    store
        .insert_row("people", &json!({"id": 2, "name": "Bob", "age": 25}))
        .unwrap();
    store
        .insert_row("people", &json!({"id": 3, "name": "Charlie", "age": 35}))
        .unwrap();

    let params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new("Bob".to_string())];
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|b| b.as_ref()).collect();
    let deleted = store
        .delete_rows("people", "name = ?1", param_refs.as_slice())
        .unwrap();
    assert_eq!(deleted, 1);

    let rows = store.query_rows("people", None, &[]).unwrap();
    assert_eq!(rows.len(), 2);
    let names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    assert!(!names.contains(&"Bob"));
}

// -----------------------------------------------------------------------
// 7. test_node_ref_column
// -----------------------------------------------------------------------
#[test]
fn test_node_ref_column() {
    let store = TableStore::open_memory().unwrap();

    let schema = make_schema(
        "edges",
        vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Integer,
                nullable: false,
                primary_key: true,
            },
            ColumnDef {
                name: "source_node".into(),
                col_type: ColumnType::NodeRef,
                nullable: false,
                primary_key: false,
            },
            ColumnDef {
                name: "target_node".into(),
                col_type: ColumnType::NodeRef,
                nullable: false,
                primary_key: false,
            },
        ],
    );
    store.create_table(&schema).unwrap();

    let src = Uuid::new_v4().to_string();
    let tgt = Uuid::new_v4().to_string();

    // Valid UUIDs should succeed.
    store
        .insert_row(
            "edges",
            &json!({"id": 1, "source_node": src, "target_node": tgt}),
        )
        .unwrap();

    // Invalid UUID should fail.
    let result = store.insert_row(
        "edges",
        &json!({"id": 2, "source_node": "not-a-uuid", "target_node": tgt}),
    );
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("UUID"));

    // Verify the valid row is stored correctly.
    let rows = store.query_rows("edges", None, &[]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["source_node"], src);
    assert_eq!(rows[0]["target_node"], tgt);
}

// -----------------------------------------------------------------------
// 8. test_json_column
// -----------------------------------------------------------------------
#[test]
fn test_json_column() {
    let store = TableStore::open_memory().unwrap();

    let schema = make_schema(
        "documents",
        vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Integer,
                nullable: false,
                primary_key: true,
            },
            ColumnDef {
                name: "metadata".into(),
                col_type: ColumnType::Json,
                nullable: true,
                primary_key: false,
            },
        ],
    );
    store.create_table(&schema).unwrap();

    let meta = json!({"tags": ["philosophy", "existentialism"], "score": 0.95});
    store
        .insert_row("documents", &json!({"id": 1, "metadata": meta}))
        .unwrap();

    let rows = store.query_rows("documents", None, &[]).unwrap();
    assert_eq!(rows.len(), 1);

    // The metadata column is stored as a JSON string in SQLite and returned as a
    // string value. We can parse it back.
    let stored = rows[0]["metadata"].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(stored).unwrap();
    assert_eq!(parsed["score"], 0.95);
    assert_eq!(parsed["tags"][0], "philosophy");
}

// -----------------------------------------------------------------------
// 9. test_table_schema
// -----------------------------------------------------------------------
#[test]
fn test_table_schema() {
    let store = TableStore::open_memory().unwrap();
    let original = people_schema();
    store.create_table(&original).unwrap();

    let retrieved = store.table_schema("people").unwrap();
    assert_eq!(retrieved.name, "people");
    assert_eq!(retrieved.columns.len(), 3);

    // Verify column details.
    let id_col = &retrieved.columns[0];
    assert_eq!(id_col.name, "id");
    assert_eq!(id_col.col_type, ColumnType::Integer);
    assert!(id_col.primary_key);

    let name_col = &retrieved.columns[1];
    assert_eq!(name_col.name, "name");
    assert_eq!(name_col.col_type, ColumnType::Text);
    assert!(!name_col.nullable);

    let age_col = &retrieved.columns[2];
    assert_eq!(age_col.name, "age");
    assert_eq!(age_col.col_type, ColumnType::Integer);
    assert!(age_col.nullable);
}

// -----------------------------------------------------------------------
// 10. test_open_memory
// -----------------------------------------------------------------------
#[test]
fn test_open_memory() {
    let store = TableStore::open_memory().unwrap();

    // Should start with no tables.
    let tables = store.list_tables().unwrap();
    assert!(tables.is_empty());

    // Should be able to do basic operations.
    store.create_table(&people_schema()).unwrap();
    store
        .insert_row("people", &json!({"id": 1, "name": "Test", "age": 42}))
        .unwrap();
    let rows = store.query_rows("people", None, &[]).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Test");
}

// -----------------------------------------------------------------------
// 11. test_open_file (bonus test using tempfile)
// -----------------------------------------------------------------------
#[test]
fn test_open_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    let path_str = path.to_str().unwrap();

    {
        let store = TableStore::open(path_str).unwrap();
        store.create_table(&people_schema()).unwrap();
        store
            .insert_row("people", &json!({"id": 1, "name": "Persistent", "age": 99}))
            .unwrap();
    }

    // Re-open and verify data persists.
    {
        let store = TableStore::open(path_str).unwrap();
        let rows = store.query_rows("people", None, &[]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["name"], "Persistent");
    }
}

// -----------------------------------------------------------------------
// 12. test_drop_nonexistent_table
// -----------------------------------------------------------------------
#[test]
fn test_drop_nonexistent_table() {
    let store = TableStore::open_memory().unwrap();
    let result = store.drop_table("nonexistent");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Table not found"));
}

// -----------------------------------------------------------------------
// 13. test_invalid_schema_no_columns
// -----------------------------------------------------------------------
#[test]
fn test_invalid_schema_no_columns() {
    let store = TableStore::open_memory().unwrap();
    let schema = make_schema("empty", vec![]);
    let result = store.create_table(&schema);
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Invalid schema"));
}

// -----------------------------------------------------------------------
// 14. test_bool_column
// -----------------------------------------------------------------------
#[test]
fn test_bool_column() {
    let store = TableStore::open_memory().unwrap();

    let schema = make_schema(
        "flags",
        vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Integer,
                nullable: false,
                primary_key: true,
            },
            ColumnDef {
                name: "active".into(),
                col_type: ColumnType::Bool,
                nullable: false,
                primary_key: false,
            },
        ],
    );
    store.create_table(&schema).unwrap();

    store
        .insert_row("flags", &json!({"id": 1, "active": true}))
        .unwrap();
    store
        .insert_row("flags", &json!({"id": 2, "active": false}))
        .unwrap();

    let rows = store.query_rows("flags", None, &[]).unwrap();
    assert_eq!(rows.len(), 2);
    // Booleans are stored as INTEGER 0/1.
    assert_eq!(rows[0]["active"], 1);
    assert_eq!(rows[1]["active"], 0);
}

// -----------------------------------------------------------------------
// 15. test_float_column
// -----------------------------------------------------------------------
#[test]
fn test_float_column() {
    let store = TableStore::open_memory().unwrap();

    let schema = make_schema(
        "measurements",
        vec![
            ColumnDef {
                name: "id".into(),
                col_type: ColumnType::Integer,
                nullable: false,
                primary_key: true,
            },
            ColumnDef {
                name: "value".into(),
                col_type: ColumnType::Float,
                nullable: false,
                primary_key: false,
            },
        ],
    );
    store.create_table(&schema).unwrap();

    store
        .insert_row("measurements", &json!({"id": 1, "value": 3.14159}))
        .unwrap();

    let rows = store.query_rows("measurements", None, &[]).unwrap();
    assert_eq!(rows.len(), 1);
    let val = rows[0]["value"].as_f64().unwrap();
    assert!((val - 3.14159).abs() < 1e-5);
}
