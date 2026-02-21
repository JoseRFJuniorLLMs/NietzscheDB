//! # nietzsche-wiederkehr
//!
//! **Wiederkehr** — Autonomous DAEMON agents that patrol the NietzscheDB graph.
//!
//! Daemons are persistent, energy-bounded agents that periodically scan nodes
//! matching an ON pattern, evaluate a WHEN condition, and execute a THEN action.
//!
//! ## NQL examples
//!
//! ```text
//! CREATE DAEMON guardian ON (n:Memory)
//!   WHEN n.energy > 0.8
//!   THEN DIFFUSE FROM n WITH t=[0.1, 1.0] MAX_HOPS 5
//!   EVERY INTERVAL("1h")
//!   ENERGY 0.8
//!
//! DROP DAEMON guardian
//! SHOW DAEMONS
//! ```

pub mod config;
pub mod engine;
pub mod error;
pub mod evaluator;
pub mod model;
pub mod priority;
pub mod store;

pub use config::DaemonEngineConfig;
pub use engine::{DaemonEngine, DaemonIntent, DaemonTickResult};
pub use error::DaemonError;
pub use evaluator::{evaluate_condition, parse_interval_str};
pub use model::DaemonDef;
pub use priority::{WillToPowerConfig, PriorityEntry, prioritize_daemons, calculate_priority};
pub use store::{put_daemon, get_daemon, delete_daemon, list_daemons};
