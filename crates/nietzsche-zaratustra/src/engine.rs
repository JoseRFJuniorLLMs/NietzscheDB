//! **ZaratustraEngine** — orchestrates the three-phase Zaratustra cycle.
//!
//! ```text
//!                        ┌──────────────────────────────┐
//!                        │       ZARATUSTRA CYCLE       │
//!                        └──────────────┬───────────────┘
//!                                       │
//!             ┌─────────────────────────▼──────────────────────────┐
//!             │  Phase 1 · WILL TO POWER (energy propagation)      │
//!             │  Nodes draw strength from energetic neighbours.    │
//!             │  Isolated nodes decay; hubs amplify.               │
//!             └─────────────────────────┬──────────────────────────┘
//!                                       │
//!             ┌─────────────────────────▼──────────────────────────┐
//!             │  Phase 2 · ETERNAL RECURRENCE (temporal echoes)    │
//!             │  High-energy nodes leave a snapshot "echo" stored  │
//!             │  inside their JSON content.  Echoes form a ring    │
//!             │  buffer — the database remembers its own past.     │
//!             └─────────────────────────┬──────────────────────────┘
//!                                       │
//!             ┌─────────────────────────▼──────────────────────────┐
//!             │  Phase 3 · ÜBERMENSCH (elite tier analytics)       │
//!             │  Identifies the top-N% nodes — the Übermensch.    │
//!             │  Returns their IDs for hot-cache/priority use.     │
//!             └──────────────────────────────────────────────────────┘
//! ```

use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::config::ZaratustraConfig;
use crate::error::ZaratustraError;
use crate::eternal_recurrence::{run_eternal_recurrence, EternalRecurrenceReport};
use crate::ubermensch::{run_ubermensch, UbermenschReport};
use crate::will_to_power::{run_will_to_power, WillToPowerReport};

/// Full report produced by a single Zaratustra cycle.
#[derive(Debug, Clone, Default)]
pub struct ZaratustraReport {
    /// Phase 1 results.
    pub will_to_power: WillToPowerReport,
    /// Phase 2 results.
    pub eternal_recurrence: EternalRecurrenceReport,
    /// Phase 3 results.
    pub ubermensch: UbermenschReport,
    /// Wall-clock duration of the full cycle in milliseconds.
    pub duration_ms: u64,
}

/// The Zaratustra engine.
///
/// Create once and call [`run_cycle`] periodically (e.g. from a background
/// Tokio task in `nietzsche-server`).  The engine is cheap to clone — it
/// holds only configuration.
#[derive(Debug, Clone)]
pub struct ZaratustraEngine {
    pub config: ZaratustraConfig,
}

impl ZaratustraEngine {
    /// Create with explicit config.
    pub fn new(config: ZaratustraConfig) -> Self {
        Self { config }
    }

    /// Create from environment variables (see [`ZaratustraConfig::from_env`]).
    pub fn from_env() -> Self {
        Self::new(ZaratustraConfig::from_env())
    }

    /// Run one full Zaratustra cycle (all three phases) against the given
    /// `storage` and `adjacency` index.
    ///
    /// `GraphStorage::put_node` takes `&self` (RocksDB handles its own
    /// locking), so a shared reference is sufficient.  The caller is
    /// responsible for holding the `NietzscheDB` mutex to prevent concurrent
    /// writes from racing with Zaratustra's propagation.
    pub fn run_cycle(
        &self,
        storage:   &GraphStorage,
        adjacency: &AdjacencyIndex,
    ) -> Result<ZaratustraReport, ZaratustraError> {
        let t0 = std::time::Instant::now();

        let will_to_power = run_will_to_power(storage, adjacency, &self.config)?;

        // Eternal Recurrence reads the freshly-updated energies from storage.
        let eternal_recurrence = run_eternal_recurrence(storage, &self.config)?;

        // Übermensch reads final energies for ranking.
        let ubermensch = run_ubermensch(storage, &self.config)?;

        let duration_ms = t0.elapsed().as_millis() as u64;

        Ok(ZaratustraReport {
            will_to_power,
            eternal_recurrence,
            ubermensch,
            duration_ms,
        })
    }
}
