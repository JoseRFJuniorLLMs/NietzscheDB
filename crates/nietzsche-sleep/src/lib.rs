//! # nietzsche-sleep
//!
//! Reconsolidation sleep cycle for NietzscheDB.
//!
//! Implements a biologically-inspired memory consolidation loop that
//! periodically perturbs and re-optimises node embeddings in the Poincaré
//! ball, preserving the fractal geometry of the knowledge graph.
//!
//! ## Modules
//!
//! - [`riemannian`] — Riemannian geometry primitives (exp map, grad, Adam)
//! - [`snapshot`]   — Embedding checkpoint / rollback
//! - [`cycle`]      — 7-step reconsolidation protocol
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_sleep::{SleepCycle, SleepConfig};
//! use rand::SeedableRng;
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let config  = SleepConfig::default();
//! let report  = SleepCycle::run(&config, &mut db, &mut rng)?;
//!
//! println!("Hausdorff Δ = {:.4}", report.hausdorff_delta);
//! println!("committed   = {}", report.committed);
//! ```

pub mod cycle;
pub mod riemannian;
pub mod snapshot;

pub use cycle::{SleepConfig, SleepCycle, SleepError, SleepReport};
pub use riemannian::{
    conformal_factor, exp_map, project_into_ball, random_tangent, riemannian_grad, AdamState,
    RiemannianAdam,
};
pub use snapshot::Snapshot;
