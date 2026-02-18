//! # nietzsche-sdk
//!
//! Rust async client SDK for NietzscheDB.
//!
//! Wraps the tonic-generated gRPC client from `nietzsche-api` with ergonomic
//! typed methods and builder-friendly parameter structs.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_sdk::{NietzscheClient, InsertNodeParams, SleepParams};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut client = NietzscheClient::connect("http://[::1]:50051").await?;
//!
//!     // Health check
//!     let health = client.health_check().await?;
//!     assert_eq!(health.status, "ok");
//!
//!     // Insert a 3-D Poincaré-ball node
//!     let node = client.insert_node(InsertNodeParams {
//!         coords:    vec![0.1, 0.2, 0.3],
//!         content:   serde_json::json!({ "text": "Thus spoke Zarathustra" }),
//!         node_type: "Episodic".into(),
//!         ..Default::default()
//!     }).await?;
//!
//!     // NQL query
//!     let result = client
//!         .query("MATCH (n) WHERE n.energy > 0.5 RETURN n LIMIT 5")
//!         .await?;
//!     println!("found {} nodes", result.nodes.len());
//!
//!     // Reconsolidation sleep
//!     let report = client.trigger_sleep(SleepParams::default()).await?;
//!     println!("Hausdorff delta = {:.4}, committed = {}", report.hausdorff_delta, report.committed);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Multi-language SDKs
//!
//! - `python/nietzsche_db.py`    — Python gRPC client stub
//! - `typescript/nietzsche_db.ts` — TypeScript/Node.js gRPC client stub

pub mod client;

pub use client::{
    InsertEdgeParams, InsertNodeParams, NietzscheClient, SleepParams,
};

// Re-export common proto response types so callers do not need to depend on
// nietzsche-api directly.
pub use nietzsche_api::pb::{
    DiffusionResponse, DiffusionScale, EdgeResponse, KnnResponse, KnnResult,
    NodePair, NodeResponse, QueryResponse, SleepResponse, StatsResponse,
    StatusResponse, TraversalResponse,
};
