//! `nietzsche-pregel` — Hyperbolic heat kernel diffusion engine.
//!
//! Implements **multi-scale activation propagation** across the Poincaré-ball
//! knowledge graph using a Chebyshev polynomial approximation of `e^{−tL̃}`.
//!
//! ## Crate structure
//!
//! | Module          | Responsibility                                              |
//! |-----------------|-------------------------------------------------------------|
//! | [`laplacian`]   | [`HyperbolicLaplacian`] — sparse, Poincaré-weighted         |
//! | [`chebyshev`]   | Modified Bessel coefficients + Chebyshev recurrence         |
//! | [`diffusion`]   | [`DiffusionEngine`] multi-scale tick + overlap metrics       |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_pregel::{DiffusionEngine, DiffusionConfig};
//!
//! let engine = DiffusionEngine::new(DiffusionConfig::default());
//!
//! let results = engine.diffuse(
//!     db.storage(),
//!     db.adjacency(),
//!     &[source_node_id],
//!     &[0.1, 1.0, 10.0],
//! )?;
//!
//! let overlap = DiffusionEngine::scale_overlap(&results, 0.1, 10.0);
//! assert!(overlap < 0.30);
//! ```
//!
//! ## Scale semantics
//!
//! | `t`   | Activated region              | Cognitive analogue    |
//! |-------|-------------------------------|------------------------|
//! | 0.1   | Immediate neighbors           | Focused recall         |
//! | 1.0   | 2–3 hop neighborhood          | Associative thinking   |
//! | 10.0  | Distant structural relatives  | Free association       |

pub mod chebyshev;
pub mod diffusion;
pub mod laplacian;

// ── Chebyshev ─────────────────────────────────────────────────────────────────
pub use chebyshev::{
    apply_heat_kernel, chebyshev_coefficients, modified_bessel_i, K_DEFAULT, LAMBDA_MAX,
};

// ── Laplacian ─────────────────────────────────────────────────────────────────
pub use laplacian::{HyperbolicLaplacian, NodeIndex};

// ── Diffusion engine ──────────────────────────────────────────────────────────
pub use diffusion::{DiffusionConfig, DiffusionEngine, DiffusionError, DiffusionResult};
