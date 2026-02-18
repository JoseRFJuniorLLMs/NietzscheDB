//! # nietzsche-graph
//!
//! Hyperbolic graph engine for NietzscheDB.
//!
//! Provides the core data model and in-memory index for the knowledge graph:
//! - [`model::PoincareVector`] — points in the Poincaré ball (‖x‖ < 1.0)
//! - [`model::Node`]          — hyperbolic graph node with energy + Hausdorff fields
//! - [`model::Edge`]          — directed typed edge
//! - [`adjacency::AdjacencyIndex`] — lock-free bidirectional adjacency index

pub mod adjacency;
pub mod error;
pub mod model;

pub use adjacency::AdjacencyIndex;
pub use error::GraphError;
pub use model::{Edge, EdgeType, Node, NodeType, PoincareVector};
