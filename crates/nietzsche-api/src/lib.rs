//! # nietzsche-api
//!
//! Unified gRPC API exposing all NietzscheDB subsystems over a single
//! `NietzscheDB` service defined in `proto/nietzsche.proto`.
//!
//! ## Multi-collection (Fase B)
//! The server is backed by a [`CollectionManager`] which routes each RPC to
//! the collection named in the request's `collection` field (empty → `"default"`).
//!
//! ## Subsystems exposed
//!
//! | RPC group         | Description                                      |
//! |-------------------|--------------------------------------------------|
//! | CreateCollection / DropCollection / ListCollections | Collection management |
//! | InsertNode / GetNode / DeleteNode / UpdateEnergy | Graph CRUD          |
//! | InsertEdge / DeleteEdge    | Edge CRUD                                |
//! | Query             | Nietzsche Query Language (NQL)                   |
//! | KnnSearch         | Hyperbolic k-nearest-neighbour search            |
//! | Bfs / Dijkstra    | Graph traversal                                  |
//! | Diffuse           | Chebyshev heat-kernel diffusion                  |
//! | TriggerSleep      | Riemannian reconsolidation cycle                 |
//! | GetStats          | Aggregate node / edge counts + version           |
//! | HealthCheck       | Liveness probe                                   |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_api::server::NietzscheServer;
//! use nietzsche_graph::CollectionManager;
//! use std::{net::SocketAddr, path::Path};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let cm   = CollectionManager::open(Path::new("/data/nietzsche"))?;
//!     let addr: SocketAddr = "[::1]:50051".parse()?;
//!     NietzscheServer::new(cm).serve(addr).await?;
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

/// Encoded protobuf file descriptor set — used to register gRPC reflection.
pub const NIETZSCHE_DESCRIPTOR: &[u8] =
    tonic::include_file_descriptor_set!("nietzsche_descriptor");

pub mod cdc;
pub mod server;
pub mod validation;

pub use cdc::CdcBroadcaster;
pub use server::NietzscheServer;
pub use proto::nietzsche as pb;
pub use validation::{
    validate_embedding, validate_energy, validate_k, validate_nql,
    validate_sleep_params, validate_source_count, validate_t_values,
    parse_uuid as validate_uuid,
};
