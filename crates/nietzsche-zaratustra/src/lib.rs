//! # nietzsche-zaratustra
//!
//! **Zaratustra** is the autonomous evolution engine for NietzscheDB.
//! Inspired by Nietzsche's *Also sprach Zarathustra* (1883), it implements
//! three interlinked concepts as concrete database algorithms:
//!
//! | Concept | Algorithm | Effect |
//! |---|---|---|
//! | **Will to Power** (`Wille zur Macht`) | Energy propagation (graph heat diffusion) | Hubs grow; isolated nodes decay |
//! | **Eternal Recurrence** (`Ewige Wiederkehr`) | Temporal echo ring-buffer | High-energy states are snapshotted; the graph remembers its own past |
//! | **Übermensch** | Energy-rank elite-tier selection | Top-N% identified for hot-cache promotion |
//!
//! ## Quick start
//! ```rust,ignore
//! use nietzsche_zaratustra::{ZaratustraEngine, ZaratustraConfig};
//!
//! let engine = ZaratustraEngine::from_env();
//! let report = engine.run_cycle(db.storage_mut(), db.adjacency())?;
//!
//! println!("Will to Power: {:.1}% nodes updated, energy Δ={:.4}",
//!     report.will_to_power.nodes_updated as f64 / total as f64 * 100.0,
//!     report.will_to_power.total_energy_delta,
//! );
//! println!("Eternal Recurrence: {} echoes created", report.eternal_recurrence.echoes_created);
//! println!("Übermensch elite: {} nodes (threshold={:.3})",
//!     report.ubermensch.elite_count,
//!     report.ubermensch.energy_threshold,
//! );
//! ```
//!
//! ## Environment variables
//! | Variable | Default | Description |
//! |---|---|---|
//! | `ZARATUSTRA_ALPHA` | `0.10` | Energy propagation coefficient |
//! | `ZARATUSTRA_DECAY` | `0.02` | Temporal energy decay rate |
//! | `ZARATUSTRA_ECHO_THRESHOLD` | `0.70` | Min energy to create an echo |
//! | `ZARATUSTRA_MAX_ECHOES` | `5` | Max echoes per node (ring buffer) |
//! | `ZARATUSTRA_UBERMENSCH_FRACTION` | `0.10` | Top fraction for elite tier |
//! | `ZARATUSTRA_PROPAGATION_STEPS` | `3` | Propagation steps per cycle |
//!
//! ## NQL integration
//! After running a cycle, echoes are visible via standard NQL:
//! ```text
//! -- Find nodes with at least one Zaratustra echo
//! MATCH (n)
//! WHERE n.energy > 0.5
//! RETURN n ORDER BY n.energy DESC LIMIT 20
//!
//! -- The _zaratustra_echoes field is stored inside n.content (JSON)
//! -- and can be inspected from the application layer.
//! ```

pub mod config;
pub mod engine;
pub mod error;
pub mod eternal_recurrence;
pub mod ubermensch;
pub mod will_to_power;

pub use config::ZaratustraConfig;
pub use engine::{ZaratustraEngine, ZaratustraReport};
pub use error::ZaratustraError;
pub use eternal_recurrence::EternalRecurrenceReport;
pub use ubermensch::UbermenschReport;
pub use will_to_power::WillToPowerReport;
