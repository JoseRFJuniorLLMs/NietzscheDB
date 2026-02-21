use serde::{Deserialize, Serialize};

/// Defines the schema of a relational table, including its name and column definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSchema {
    /// The name of the table.
    pub name: String,
    /// Ordered list of column definitions.
    pub columns: Vec<ColumnDef>,
}

/// Defines a single column within a table schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnDef {
    /// The column name.
    pub name: String,
    /// The logical type of the column.
    pub col_type: ColumnType,
    /// Whether the column accepts NULL values.
    pub nullable: bool,
    /// Whether this column is (part of) the primary key.
    pub primary_key: bool,
}

/// Logical column types supported by NietzscheDB tables.
///
/// Each type maps to an underlying SQLite storage type:
/// - `Text` -> TEXT
/// - `Integer` -> INTEGER
/// - `Float` -> REAL
/// - `Bool` -> INTEGER (0/1)
/// - `Uuid` -> TEXT (validated as UUID)
/// - `Json` -> TEXT (parsed as JSON)
/// - `NodeRef` -> TEXT (UUID reference to a graph node)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ColumnType {
    /// A UTF-8 text string.
    Text,
    /// A 64-bit signed integer.
    Integer,
    /// A 64-bit floating-point number.
    Float,
    /// A boolean stored as INTEGER (0 or 1).
    Bool,
    /// A UUID stored as TEXT but validated on insert.
    Uuid,
    /// Arbitrary JSON stored as TEXT.
    Json,
    /// A UUID reference to a graph node, stored as TEXT.
    NodeRef,
}

impl ColumnType {
    /// Returns the SQLite type affinity string for this column type.
    pub fn to_sqlite_type(self) -> &'static str {
        match self {
            ColumnType::Text => "TEXT",
            ColumnType::Integer => "INTEGER",
            ColumnType::Float => "REAL",
            ColumnType::Bool => "INTEGER",
            ColumnType::Uuid => "TEXT",
            ColumnType::Json => "TEXT",
            ColumnType::NodeRef => "TEXT",
        }
    }
}
