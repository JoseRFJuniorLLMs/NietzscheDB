//! `nietzsche-lsystem` — L-System fractal growth engine for the Poincaré-ball graph.
//!
//! ## Crate structure
//!
//! | Module        | Responsibility                                          |
//! |---------------|---------------------------------------------------------|
//! | [`engine`]    | [`LSystemEngine`] tick loop + pending-action execution  |
//! | [`hausdorff`] | Box-counting Hausdorff dimension (global + local)       |
//! | [`mobius`]    | Möbius addition, child/sibling placement in ℍⁿ          |
//! | [`rules`]     | [`ProductionRule`], [`RuleCondition`], [`RuleAction`]   |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use nietzsche_lsystem::{LSystemEngine, ProductionRule};
//!
//! let engine = LSystemEngine::new(vec![
//!     ProductionRule::growth_child("grow", 3),
//!     ProductionRule::lateral_association("assoc", 3),
//!     ProductionRule::prune_fading("prune", 0.1),
//! ]);
//!
//! let report = engine.tick(&mut db)?;
//! println!("spawned={} pruned={} D={:.3}", report.nodes_spawned, report.nodes_pruned, report.global_hausdorff);
//! ```

pub mod engine;
pub mod hausdorff;
pub mod mobius;
pub mod rules;

// ── Engine ────────────────────────────────────────────────────────────────────
pub use engine::{
    LSystemEngine, LSystemError, LSystemReport, SensoryDegrader,
    DEFAULT_HAUSDORFF_HI, DEFAULT_HAUSDORFF_LO,
};

// ── Hausdorff ─────────────────────────────────────────────────────────────────
pub use hausdorff::{box_counting, global_hausdorff, local_hausdorff, LOCAL_K};

// ── Möbius / geometry ─────────────────────────────────────────────────────────
pub use mobius::{mobius_add, project_into_ball, spawn_child, spawn_sibling};

// ── Rules ─────────────────────────────────────────────────────────────────────
pub use rules::{check_condition, ProductionRule, RuleAction, RuleCondition};
