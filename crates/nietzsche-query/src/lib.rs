//! # nietzsche-query
//!
//! **NQL — Nietzsche Query Language** parser, planner, and executor.
//!
//! ## Supported query forms
//!
//! ```text
//! -- Node scan with energy + depth filters
//! MATCH (n:Memory)
//! WHERE n.energy > 0.3 AND n.depth < 0.8
//! RETURN n ORDER BY n.energy DESC LIMIT 10
//!
//! -- Hyperbolic nearest-neighbour search
//! MATCH (n)
//! WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5
//! RETURN n ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC LIMIT 5
//!
//! -- Typed edge traversal
//! MATCH (a)-[:Association]->(b) WHERE a.energy > 0.5 RETURN b
//!
//! -- Energy-biased diffusion walk
//! DIFFUSE FROM $node WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5 RETURN path
//!
//! -- Phase 11: Sensory nearest-neighbour search
//! MATCH (n)
//! WHERE SENSORY_DIST(n.latent, $q) < 0.3
//! RETURN n ORDER BY SENSORY_DIST(n.latent, $q) ASC LIMIT 5
//!
//! -- Phase 11: Reconstruct sensory data from latent
//! RECONSTRUCT $node_id MODALITY audio QUALITY high
//! ```

pub mod ast;
pub mod cost;
pub mod error;
pub mod executor;
pub mod parser;

pub use ast::{
    Query, MatchQuery, DiffuseQuery, ReconstructQuery, ReconstructTarget,
    InvokeZaratustraQuery,
    CreateQuery, MatchSetQuery, MatchDeleteQuery,
    CreateDaemonQuery, DropDaemonQuery, DaemonAction,
    DreamFromQuery, ApplyDreamQuery, RejectDreamQuery,
    TranslateQuery, CounterfactualQuery,
    ShareArchetypeQuery, NarrateQuery,
    Pattern, NodePattern, PathPattern, Direction,
    Condition, CompOp, StringCompOp,
    Expr, HDistArg, MathFunc, MathFuncArg, ArithOp,
    SetAssignment,
    ReturnClause, ReturnItem, ReturnExpr,
    AggFunc, AggArg, GroupByItem,
    OrderBy, OrderExpr, OrderDir,
    DiffuseFrom,
};
pub use error::QueryError;
pub use executor::{execute, execute_with_indexes, ParamValue, Params, QueryResult, ScalarValue};
pub use parser::parse;
