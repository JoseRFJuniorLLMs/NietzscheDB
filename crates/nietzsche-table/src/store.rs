use rusqlite::{params_from_iter, Connection};
use serde_json::{Map, Value};
use tracing::{debug, instrument};

use crate::error::TableError;
use crate::schema::{ColumnType, TableSchema};

/// A relational table store backed by SQLite.
///
/// `TableStore` allows creating, dropping, and querying relational tables that
/// can coexist alongside NietzscheDB's hyperbolic graph. Columns of type
/// [`ColumnType::NodeRef`] hold UUID references to graph nodes, bridging the
/// relational and graph paradigms.
pub struct TableStore {
    conn: Connection,
}

impl TableStore {
    /// Opens (or creates) a SQLite database at the given file path.
    #[instrument(skip_all, fields(path = %path))]
    pub fn open(path: &str) -> Result<Self, TableError> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        debug!("Opened TableStore at {}", path);
        Ok(Self { conn })
    }

    /// Opens an in-memory SQLite database (useful for testing).
    pub fn open_memory() -> Result<Self, TableError> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        debug!("Opened in-memory TableStore");
        Ok(Self { conn })
    }

    /// Creates a new table according to the given schema.
    ///
    /// Maps each [`ColumnType`] to its corresponding SQLite type affinity.
    /// Returns [`TableError::InvalidSchema`] if the schema has no columns or
    /// contains duplicate column names.
    #[instrument(skip_all, fields(table = %schema.name))]
    pub fn create_table(&self, schema: &TableSchema) -> Result<(), TableError> {
        if schema.columns.is_empty() {
            return Err(TableError::InvalidSchema(
                "Table must have at least one column".into(),
            ));
        }

        // Check for duplicate column names.
        let mut seen = std::collections::HashSet::new();
        for col in &schema.columns {
            if !seen.insert(&col.name) {
                return Err(TableError::InvalidSchema(format!(
                    "Duplicate column name: {}",
                    col.name
                )));
            }
        }

        let col_defs: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let mut def = format!(
                    "\"{}\" {}",
                    col.name,
                    col.col_type.to_sqlite_type()
                );
                if col.primary_key {
                    def.push_str(" PRIMARY KEY");
                }
                if !col.nullable && !col.primary_key {
                    def.push_str(" NOT NULL");
                }
                def
            })
            .collect();

        let sql = format!(
            "CREATE TABLE IF NOT EXISTS \"{}\" ({})",
            schema.name,
            col_defs.join(", ")
        );

        debug!("CREATE TABLE SQL: {}", sql);
        self.conn.execute(&sql, [])?;

        // Store schema metadata in a special table so we can reconstruct it later.
        self.ensure_meta_table()?;
        let schema_json = serde_json::to_string(schema).map_err(|e| {
            TableError::SerializationError(format!("Failed to serialize schema: {}", e))
        })?;
        self.conn.execute(
            "INSERT OR REPLACE INTO _nietzsche_table_meta (table_name, schema_json) VALUES (?1, ?2)",
            rusqlite::params![schema.name, schema_json],
        )?;

        Ok(())
    }

    /// Drops a table by name. Returns [`TableError::TableNotFound`] if it does
    /// not exist.
    #[instrument(skip_all, fields(table = %name))]
    pub fn drop_table(&self, name: &str) -> Result<(), TableError> {
        if !self.table_exists(name)? {
            return Err(TableError::TableNotFound(name.to_string()));
        }

        let sql = format!("DROP TABLE \"{}\"", name);
        self.conn.execute(&sql, [])?;

        // Remove metadata.
        self.conn.execute(
            "DELETE FROM _nietzsche_table_meta WHERE table_name = ?1",
            rusqlite::params![name],
        )?;

        debug!("Dropped table {}", name);
        Ok(())
    }

    /// Inserts a row into the named table. The `row` must be a JSON object
    /// whose keys correspond to column names.
    ///
    /// Returns the SQLite `rowid` of the newly inserted row.
    #[instrument(skip_all, fields(table = %table))]
    pub fn insert_row(&self, table: &str, row: &Value) -> Result<i64, TableError> {
        let obj = row.as_object().ok_or_else(|| {
            TableError::SerializationError("Row must be a JSON object".into())
        })?;

        if obj.is_empty() {
            return Err(TableError::SerializationError(
                "Row must have at least one field".into(),
            ));
        }

        // Validate UUID and NodeRef columns, and Json columns.
        let schema = self.table_schema(table)?;
        for col_def in &schema.columns {
            if let Some(val) = obj.get(&col_def.name) {
                match col_def.col_type {
                    ColumnType::Uuid | ColumnType::NodeRef => {
                        if let Some(s) = val.as_str() {
                            uuid::Uuid::parse_str(s).map_err(|e| {
                                TableError::SerializationError(format!(
                                    "Column '{}' requires a valid UUID, got '{}': {}",
                                    col_def.name, s, e
                                ))
                            })?;
                        } else if !val.is_null() {
                            return Err(TableError::SerializationError(format!(
                                "Column '{}' requires a UUID string, got {}",
                                col_def.name, val
                            )));
                        }
                    }
                    ColumnType::Json => {
                        // Serialize nested JSON to a string for storage.
                        // Validation: the value must be valid JSON (it already is since it
                        // came from serde_json::Value, so this is always satisfied).
                    }
                    _ => {}
                }
            }
        }

        let columns: Vec<&String> = obj.keys().collect();
        let placeholders: Vec<String> = (1..=columns.len()).map(|i| format!("?{}", i)).collect();

        let sql = format!(
            "INSERT INTO \"{}\" ({}) VALUES ({})",
            table,
            columns
                .iter()
                .map(|c| format!("\"{}\"", c))
                .collect::<Vec<_>>()
                .join(", "),
            placeholders.join(", ")
        );

        let values: Vec<Box<dyn rusqlite::types::ToSql>> = columns
            .iter()
            .map(|col_name| {
                let val = &obj[col_name.as_str()];
                Self::json_value_to_sql(val, &schema, col_name)
            })
            .collect();

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            values.iter().map(|b| b.as_ref()).collect();

        self.conn.execute(&sql, param_refs.as_slice())?;
        let rowid = self.conn.last_insert_rowid();
        debug!("Inserted row into {} with rowid {}", table, rowid);
        Ok(rowid)
    }

    /// Queries rows from a table with an optional WHERE filter.
    ///
    /// The `filter` parameter, if provided, is appended after `WHERE`.
    /// Use `params` to pass bind parameters for the filter.
    ///
    /// Returns a `Vec` of JSON objects, one per row.
    #[instrument(skip_all, fields(table = %table))]
    pub fn query_rows(
        &self,
        table: &str,
        filter: Option<&str>,
        params: &[&dyn rusqlite::types::ToSql],
    ) -> Result<Vec<Value>, TableError> {
        if !self.table_exists(table)? {
            return Err(TableError::TableNotFound(table.to_string()));
        }

        let sql = match filter {
            Some(f) => format!("SELECT * FROM \"{}\" WHERE {}", table, f),
            None => format!("SELECT * FROM \"{}\"", table),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let col_count = stmt.column_count();
        let col_names: Vec<String> = (0..col_count)
            .map(|i| stmt.column_name(i).unwrap().to_string())
            .collect();

        let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
            let mut map = Map::new();
            for (i, col_name) in col_names.iter().enumerate() {
                let val = row.get_ref(i)?;
                let json_val = match val {
                    rusqlite::types::ValueRef::Null => Value::Null,
                    rusqlite::types::ValueRef::Integer(n) => Value::Number(n.into()),
                    rusqlite::types::ValueRef::Real(f) => {
                        Value::Number(serde_json::Number::from_f64(f).unwrap_or(0.into()))
                    }
                    rusqlite::types::ValueRef::Text(t) => {
                        let s = std::str::from_utf8(t).unwrap_or("");
                        Value::String(s.to_string())
                    }
                    rusqlite::types::ValueRef::Blob(b) => {
                        Value::String(format!("<blob {} bytes>", b.len()))
                    }
                };
                map.insert(col_name.clone(), json_val);
            }
            Ok(Value::Object(map))
        })?;

        let mut result = Vec::new();
        for row_result in rows {
            result.push(row_result?);
        }

        debug!("Queried {} rows from {}", result.len(), table);
        Ok(result)
    }

    /// Deletes rows from a table matching the given WHERE clause.
    ///
    /// Returns the number of rows deleted.
    #[instrument(skip_all, fields(table = %table))]
    pub fn delete_rows(
        &self,
        table: &str,
        filter: &str,
        params: &[&dyn rusqlite::types::ToSql],
    ) -> Result<usize, TableError> {
        if !self.table_exists(table)? {
            return Err(TableError::TableNotFound(table.to_string()));
        }

        let sql = format!("DELETE FROM \"{}\" WHERE {}", table, filter);
        let deleted = self.conn.execute(&sql, params_from_iter(params.iter()))?;
        debug!("Deleted {} rows from {}", deleted, table);
        Ok(deleted)
    }

    /// Lists all user-created table names (excludes internal metadata tables
    /// and SQLite system tables).
    pub fn list_tables(&self) -> Result<Vec<String>, TableError> {
        let mut stmt = self.conn.prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY name",
        )?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(names)
    }

    /// Retrieves the schema of a table by name.
    ///
    /// Returns [`TableError::TableNotFound`] if the table does not exist or
    /// has no stored schema metadata.
    pub fn table_schema(&self, name: &str) -> Result<TableSchema, TableError> {
        self.ensure_meta_table()?;
        let result: Result<String, _> = self.conn.query_row(
            "SELECT schema_json FROM _nietzsche_table_meta WHERE table_name = ?1",
            rusqlite::params![name],
            |row| row.get(0),
        );

        match result {
            Ok(json_str) => {
                let schema: TableSchema = serde_json::from_str(&json_str).map_err(|e| {
                    TableError::SerializationError(format!("Failed to parse schema: {}", e))
                })?;
                Ok(schema)
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(TableError::TableNotFound(name.to_string()))
            }
            Err(e) => Err(TableError::Sqlite(e)),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Ensures the internal metadata table exists.
    fn ensure_meta_table(&self) -> Result<(), TableError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS _nietzsche_table_meta (
                table_name TEXT PRIMARY KEY,
                schema_json TEXT NOT NULL
            )",
        )?;
        Ok(())
    }

    /// Checks whether a user table exists.
    fn table_exists(&self, name: &str) -> Result<bool, TableError> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
            rusqlite::params![name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Converts a `serde_json::Value` into a boxed `ToSql` for binding to
    /// a prepared statement. For `Json` columns, nested values are serialized
    /// to a JSON string.
    fn json_value_to_sql(
        val: &Value,
        schema: &TableSchema,
        col_name: &str,
    ) -> Box<dyn rusqlite::types::ToSql> {
        // Check if this column is a Json column — if so, store the value as a
        // serialized JSON string regardless of its structure.
        let is_json_col = schema
            .columns
            .iter()
            .any(|c| c.name == *col_name && c.col_type == ColumnType::Json);

        if is_json_col && !val.is_null() {
            return Box::new(serde_json::to_string(val).unwrap_or_default());
        }

        match val {
            Value::Null => Box::new(rusqlite::types::Null),
            Value::Bool(b) => Box::new(*b as i64),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Box::new(i)
                } else if let Some(f) = n.as_f64() {
                    Box::new(f)
                } else {
                    Box::new(n.to_string())
                }
            }
            Value::String(s) => Box::new(s.clone()),
            Value::Array(_) | Value::Object(_) => {
                Box::new(serde_json::to_string(val).unwrap_or_default())
            }
        }
    }
}
