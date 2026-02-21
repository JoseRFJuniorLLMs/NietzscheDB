//! # nietzsche-dream
//!
//! **Dream Queries** — speculative graph exploration via hyperbolic diffusion.
//!
//! Dreams are transient explorations that start from a seed node,
//! diffuse energy with stochastic noise, and detect events like
//! cluster collisions, energy spikes, and curvature anomalies.
//!
//! ## NQL examples
//!
//! ```text
//! DREAM FROM $node DEPTH 3 NOISE 0.1
//! APPLY DREAM $dream_id
//! REJECT DREAM $dream_id
//! SHOW DREAMS
//! ```

pub mod engine;
pub mod error;
pub mod model;
pub mod store;

pub use engine::{DreamConfig, DreamEngine};
pub use error::DreamError;
pub use model::{DreamEvent, DreamEventType, DreamNodeDelta, DreamSession, DreamStatus};
pub use store::{put_dream, get_dream, delete_dream, list_dreams, list_pending_dreams};
