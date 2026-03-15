// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # nietzsche-naq
//!
//! **NAQ — NietzscheDB Agent Query**: Builder API, AST cache, and compact
//! agent protocol for NietzscheDB.
//!
//! ## Three layers of optimization
//!
//! | Layer | Target | Gain |
//! |-------|--------|------|
//! | Builder API | Rust internal (agency, reactors) | Eliminates 100% text parsing |
//! | AST Cache | code_as_data actions that repeat | ~100x less parsing per tick |
//! | NAQ Protocol | External agents (MCP, EVA, LLMs) | ~70% fewer tokens |
//!
//! ## Builder API (Layer 1)
//!
//! ```rust,no_run
//! use nietzsche_naq::prelude::*;
//!
//! // Zero parsing — builds AST directly
//! let query = Naq::match_nodes()
//!     .label("Semantic")
//!     .where_gt("energy", 0.5)
//!     .where_contains("content", "physics")
//!     .order_desc("energy")
//!     .limit(10)
//!     .build();
//! ```
//!
//! ## AST Cache (Layer 2)
//!
//! ```rust,no_run
//! use nietzsche_naq::cache::NaqCache;
//!
//! let mut cache = NaqCache::new(256);
//! let ast = cache.get_or_parse("MATCH (n) WHERE n.energy > 0.5 RETURN n").unwrap();
//! // Second call: instant (no parsing)
//! let ast2 = cache.get_or_parse("MATCH (n) WHERE n.energy > 0.5 RETURN n").unwrap();
//! ```

pub mod builder;
pub mod cache;

/// Convenience re-exports for `use nietzsche_naq::prelude::*`
pub mod prelude {
    pub use crate::builder::Naq;
    pub use crate::cache::NaqCache;
}

// Re-export the AST types for downstream users
pub use nietzsche_query::ast;
pub use nietzsche_query::{Query, QueryError};
