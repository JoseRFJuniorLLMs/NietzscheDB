// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # NAQ Builder API
//!
//! Construct NietzscheDB AST nodes directly in Rust — zero text parsing.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use nietzsche_naq::builder::Naq;
//!
//! // Match with filter + sort + limit
//! let q = Naq::match_nodes()
//!     .label("Semantic")
//!     .where_gt("energy", 0.5)
//!     .order_desc("energy")
//!     .limit(10)
//!     .build();
//!
//! // Match edges
//! let q = Naq::match_edges("Association")
//!     .from_label("Semantic")
//!     .where_gt("a.energy", 0.5)
//!     .build();
//!
//! // Create node
//! let q = Naq::create()
//!     .label("Semantic")
//!     .prop("name", "quark")
//!     .prop_f64("energy", 0.8)
//!     .build();
//!
//! // Update
//! let q = Naq::match_set()
//!     .where_gt("energy", 0.0)
//!     .set("energy", 0.0)
//!     .build();
//!
//! // Delete
//! let q = Naq::match_delete()
//!     .where_lt("energy", 0.1)
//!     .detach()
//!     .build();
//! ```

mod match_builder;
mod mutation_builder;

pub use match_builder::{MatchBuilder, EdgeMatchBuilder};
pub use mutation_builder::{CreateBuilder, MatchSetBuilder, MatchDeleteBuilder};

use nietzsche_query::Query;

/// Entry point for the NAQ Builder API.
///
/// All methods return builder structs that construct `Query` AST nodes
/// without any text parsing.
pub struct Naq;

impl Naq {
    // ── Match ────────────────────────────────────────────────

    /// Start building a `MATCH (n) ...` query.
    pub fn match_nodes() -> MatchBuilder {
        MatchBuilder::new()
    }

    /// Start building a `MATCH (a)-[:edge_type]->(b) ...` query.
    pub fn match_edges(edge_type: &str) -> EdgeMatchBuilder {
        EdgeMatchBuilder::new(edge_type)
    }

    /// Build `MATCH (n) RETURN n LIMIT {limit}` — the most common dashboard query.
    pub fn match_all(limit: usize) -> Query {
        MatchBuilder::new().limit(limit).build()
    }

    /// Build `MATCH (a)-[]->(b) RETURN a, b LIMIT {limit}` — edge listing.
    pub fn match_all_edges(limit: usize) -> Query {
        EdgeMatchBuilder::anonymous().limit(limit).build()
    }

    /// Build `MATCH (a)-[]->(b) RETURN COUNT(a)` — edge count.
    pub fn count_edges() -> Query {
        EdgeMatchBuilder::anonymous().return_count().build()
    }

    // ── Create ───────────────────────────────────────────────

    /// Start building a `CREATE (n:Label {...})` query.
    pub fn create() -> CreateBuilder {
        CreateBuilder::new()
    }

    // ── Update ───────────────────────────────────────────────

    /// Start building a `MATCH (n) WHERE ... SET ...` query.
    pub fn match_set() -> MatchSetBuilder {
        MatchSetBuilder::new()
    }

    // ── Delete ───────────────────────────────────────────────

    /// Start building a `MATCH (n) WHERE ... DELETE n` query.
    pub fn match_delete() -> MatchDeleteBuilder {
        MatchDeleteBuilder::new()
    }
}
