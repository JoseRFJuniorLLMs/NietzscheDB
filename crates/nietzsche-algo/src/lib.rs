//! Graph algorithm library for NietzscheDB.
//!
//! Provides production-grade implementations of common graph algorithms,
//! matching the capabilities of Neo4j GDS and TigerGraph:
//!
//! - **Centrality**: PageRank, Betweenness (Brandes), Closeness, Degree
//! - **Community**: Louvain modularity, Label Propagation
//! - **Components**: Weakly Connected (Union-Find), Strongly Connected (Tarjan)
//! - **Pathfinding**: A* (with Poincaré heuristic), Triangle Count
//! - **Similarity**: Jaccard, Overlap

pub mod pagerank;
pub mod community;
pub mod centrality;
pub mod components;
pub mod pathfinding;
pub mod similarity;

pub use pagerank::{PageRankConfig, PageRankResult, pagerank};
pub use community::{LouvainConfig, LouvainResult, louvain, LabelPropResult, label_propagation};
pub use centrality::{
    BetweennessResult, betweenness_centrality,
    closeness_centrality, degree_centrality, Direction,
};
pub use components::{ComponentResult, weakly_connected_components, strongly_connected_components};
pub use pathfinding::{astar, triangle_count};
pub use similarity::{SimilarityPair, jaccard_similarity};
