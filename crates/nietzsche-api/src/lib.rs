//! # nietzsche-api
//!
//! Unified gRPC API exposing all NietzscheDB subsystems over a single
//! `NietzscheDB` service defined in `proto/nietzsche.proto`.
//!
//! ## Subsystems exposed
//!
//! | RPC              | Phase | Description                              |
//! |------------------|-------|------------------------------------------|
//! | InsertNode/GetNode/DeleteNode/UpdateEnergy | 1 | Graph CRUD |
//! | InsertEdge/DeleteEdge | 1 | Edge CRUD |
//! | Query            | 4     | Nietzsche Query Language (NQL)            |
//! | KnnSearch        | 1     | Hyperbolic k-nearest-neighbour search     |
//! | Bfs / Dijkstra   | 3     | Graph traversal                           |
//! | Diffuse          | 6     | Chebyshev heat-kernel diffusion           |
//! | TriggerSleep     | 8     | Riemannian reconsolidation cycle          |
//! | GetStats         | —     | Node / edge counts + version              |
//! | HealthCheck      | —     | Liveness probe                            |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_api::server::NietzscheServer;
//! use nietzsche_graph::{MockVectorStore, NietzscheDB};
//! use std::{net::SocketAddr, path::Path, sync::Arc};
//! use tokio::sync::Mutex;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let db = NietzscheDB::open(Path::new("/data/nietzsche"), MockVectorStore::default())?;
//!     let shared = Arc::new(Mutex::new(db));
//!     let addr: SocketAddr = "[::1]:50051".parse()?;
//!     NietzscheServer::new(shared).serve(addr).await?;
//!     Ok(())
//! }
//! ```

// Generated protobuf / tonic code (compiled by build.rs)
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
pub mod proto {
    pub mod nietzsche {
        tonic::include_proto!("nietzsche");
    }
}

pub mod server;
pub mod validation;

pub use server::NietzscheServer;
pub use proto::nietzsche as pb;
pub use validation::{
    validate_embedding, validate_energy, validate_k, validate_nql,
    validate_sleep_params, validate_source_count, validate_t_values,
    parse_uuid as validate_uuid,
};
