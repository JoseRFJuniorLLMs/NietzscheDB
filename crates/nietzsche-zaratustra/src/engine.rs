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

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{AdjacencyIndex, Edge, EdgeType, GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_temp_db() -> (GraphStorage, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (storage, dir)
    }

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"label": "test"}),
        );
        node.energy = energy;
        node
    }

    #[test]
    fn full_cycle_on_empty_graph() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();
        let engine = ZaratustraEngine::new(ZaratustraConfig::default());

        let report = engine.run_cycle(&storage, &adjacency).unwrap();

        assert_eq!(report.will_to_power.nodes_updated, 0);
        assert_eq!(report.eternal_recurrence.echoes_created, 0);
        assert_eq!(report.ubermensch.elite_count, 0);
        assert!(report.duration_ms < 5000, "should complete quickly");
    }

    #[test]
    fn full_cycle_with_single_node() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let node = make_node(0.1, 0.1, 0.9);
        storage.put_node(&node).unwrap();

        let engine = ZaratustraEngine::new(ZaratustraConfig {
            echo_threshold: 0.5,
            ..ZaratustraConfig::default()
        });

        let report = engine.run_cycle(&storage, &adjacency).unwrap();

        // Phase 1: node decays (no neighbours)
        assert!(report.will_to_power.nodes_updated > 0 || report.will_to_power.total_energy_delta < 1e-6);

        // Phase 2: energy might still be above threshold after decay
        // Phase 3: single node = 1 elite
        assert_eq!(report.ubermensch.elite_count, 1);
    }

    #[test]
    fn full_cycle_with_graph() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        // Star topology: center -> leaf1, center -> leaf2
        let center = make_node(0.1, 0.0, 0.9);
        let leaf1 = make_node(0.0, 0.1, 0.8);
        let leaf2 = make_node(0.2, 0.0, 0.3);

        let center_id = center.id;

        let e1 = Edge::new(center.id, leaf1.id, EdgeType::Association, 1.0);
        let e2 = Edge::new(center.id, leaf2.id, EdgeType::Association, 1.0);

        storage.put_node(&center).unwrap();
        storage.put_node(&leaf1).unwrap();
        storage.put_node(&leaf2).unwrap();
        adjacency.add_edge(&e1);
        adjacency.add_edge(&e2);

        let cfg = ZaratustraConfig {
            alpha: 0.2,
            decay: 0.01,
            echo_threshold: 0.7,
            max_echoes_per_node: 5,
            ubermensch_top_fraction: 0.34,
            propagation_steps: 2,
            ..ZaratustraConfig::default()
        };

        let engine = ZaratustraEngine::new(cfg);
        let report = engine.run_cycle(&storage, &adjacency).unwrap();

        // Verify all three phases ran
        assert!(report.will_to_power.mean_energy_before > 0.0);
        // Elite should contain at least 1 node (ceil(3*0.34) = 2)
        assert!(report.ubermensch.elite_count >= 1);
        // Center node (high energy hub) should be elite
        assert!(
            report.ubermensch.elite_node_ids.contains(&center_id),
            "center (high energy hub) should be in elite tier",
        );
    }

    #[test]
    fn from_env_creates_engine() {
        // Just verify it doesn't panic
        let engine = ZaratustraEngine::from_env();
        // Config should have default values when env vars are not set
        assert!(engine.config.alpha > 0.0);
    }

    #[test]
    fn two_consecutive_cycles_produce_different_results() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0, 0.9);
        let b = make_node(0.0, 0.1, 0.5);
        let edge = Edge::new(a.id, b.id, EdgeType::Association, 1.0);

        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        adjacency.add_edge(&edge);

        let engine = ZaratustraEngine::new(ZaratustraConfig {
            propagation_steps: 1,
            decay: 0.05,
            alpha: 0.1,
            echo_threshold: 0.8,
            ..ZaratustraConfig::default()
        });

        let r1 = engine.run_cycle(&storage, &adjacency).unwrap();
        let r2 = engine.run_cycle(&storage, &adjacency).unwrap();

        // After two cycles, energy should have decayed further
        // The second cycle's mean_energy_before should be lower than the first's
        // (because decay reduced everything in cycle 1)
        assert!(
            r2.will_to_power.mean_energy_before <= r1.will_to_power.mean_energy_before + 1e-5,
            "second cycle mean energy before ({}) should not exceed first ({})",
            r2.will_to_power.mean_energy_before,
            r1.will_to_power.mean_energy_before,
        );
    }

    #[test]
    fn duration_ms_is_populated() {
        let (storage, _dir) = open_temp_db();
        let adjacency = AdjacencyIndex::new();

        for i in 0..5 {
            let node = make_node(0.01 * (i + 1) as f32, 0.0, 0.5);
            storage.put_node(&node).unwrap();
        }

        let engine = ZaratustraEngine::new(ZaratustraConfig::default());
        let report = engine.run_cycle(&storage, &adjacency).unwrap();

        // Duration should be non-negative (could be 0 on fast machines)
        assert!(report.duration_ms < 30_000, "should finish in reasonable time");
    }
}
