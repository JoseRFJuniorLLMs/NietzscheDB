//! Named Vectors — multi-vector-per-node for NietzscheDB.
//!
//! Each node can have multiple named embedding spaces:
//! - `"default"` — the standard Poincare ball embedding
//! - `"text"` — text-derived embedding
//! - `"image"` — image-derived embedding
//! - `"audio"` — audio-derived embedding
//!
//! Stored in CF_META under key prefix `"nvec:{node_id}:{name}"`.

pub mod model;
pub mod store;
pub mod error;

pub use model::NamedVector;
pub use store::NamedVectorStore;
pub use error::NamedVectorError;
