pub mod coherence;
pub mod entropy;
pub mod gap;

use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::AgencyEventBus;

/// Report produced by a single daemon tick.
#[derive(Debug, Clone, Default)]
pub struct DaemonReport {
    pub daemon_name: String,
    pub events_emitted: usize,
    pub nodes_scanned: usize,
    pub duration_us: u64,
    pub details: Vec<String>,
}

/// All agency daemons implement this trait.
///
/// Daemons are **read-only** — they scan the graph, analyze it, and emit
/// events via the bus. They never mutate the graph.
pub trait AgencyDaemon: Send + Sync {
    fn name(&self) -> &str;

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError>;
}

pub use coherence::CoherenceDaemon;
pub use entropy::EntropyDaemon;
pub use gap::GapDaemon;
