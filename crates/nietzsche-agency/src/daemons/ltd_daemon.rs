//! Long-Term Depression (LTD) Daemon
//!
//! Implements the biological LTD mechanism: synaptic connections are
//! *weakened* when recurring caregiver corrections mark them as incorrect
//! or harmful. This is the counterpart to Hebbian strengthening (Will to Power).
//!
//! ## Mechanism
//!
//! 1. The caregiver correction pipeline sets a `correction_count` field in
//!    edge metadata whenever it marks an association as wrong.
//! 2. This daemon scans all edges for `correction_count > 0`.
//! 3. For each such edge, it emits a `CorrectionAccumulated` event carrying
//!    the edge endpoints and correction count.
//! 4. The `AgencyReactor` converts these events into `ApplyLTD` intents.
//! 5. The server applies `weight -= LTD_RATE * correction_count` under write lock.
//!
//! ## LTD Rate
//!
//! The default LTD rate is `0.05` per correction (5% weight reduction per
//! correction event). This is configurable via `AgencyConfig::ltd_rate`.
//!
//! Edges that reach `weight <= 0` are candidates for pruning by the Niilista GC.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

use super::{AgencyDaemon, DaemonReport};

/// The metadata field written by the caregiver correction pipeline.
const CORRECTION_COUNT_KEY: &str = "correction_count";

/// Minimum correction count to trigger LTD (avoids processing noise).
const MIN_CORRECTION_THRESHOLD: u64 = 1;

/// LTD daemon: weakens edges that have accumulated caregiver corrections.
pub struct LTDDaemon;

impl AgencyDaemon for LTDDaemon {
    fn name(&self) -> &str { "ltd" }

    fn tick(
        &self,
        storage:   &GraphStorage,
        _adjacency: &AdjacencyIndex,
        bus:       &AgencyEventBus,
        config:    &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        let mut edges_scanned = 0usize;
        let mut events_emitted = 0usize;
        let mut details = Vec::new();

        let min_threshold = config.ltd_correction_threshold
            .unwrap_or(MIN_CORRECTION_THRESHOLD);

        // Scan all edges — edges are small (~64 bytes), this is O(E) but cheap.
        for result in storage.iter_edges() {
            let edge = match result {
                Ok(e) => e,
                Err(_) => continue,
            };
            edges_scanned += 1;

            // Check for correction_count in edge metadata
            let correction_count = match edge.metadata.get(CORRECTION_COUNT_KEY) {
                Some(serde_json::Value::Number(n)) => {
                    n.as_u64().unwrap_or(0)
                }
                Some(serde_json::Value::String(s)) => {
                    s.parse::<u64>().unwrap_or(0)
                }
                _ => continue, // No correction field → skip
            };

            if correction_count < min_threshold {
                continue;
            }

            // Compute the LTD delta: rate * correction_count, capped at current weight
            let ltd_rate = config.ltd_rate.unwrap_or(0.05);
            let weight_delta = (ltd_rate * correction_count as f64)
                .min(edge.weight as f64); // cannot go negative

            bus.publish(AgencyEvent::CorrectionAccumulated {
                from_id:          edge.from,
                to_id:            edge.to,
                correction_count,
                weight_delta:     weight_delta as f32,
            });

            events_emitted += 1;
            details.push(format!(
                "edge {}->{}: corrections={}, weight_delta={:.3}",
                edge.from, edge.to, correction_count, weight_delta
            ));
        }

        Ok(DaemonReport {
            daemon_name: "ltd".into(),
            events_emitted,
            nodes_scanned: edges_scanned, // reusing the field for edge count
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Edge, EdgeType, Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node() -> Node {
        Node::new(Uuid::new_v4(), PoincareVector::new(vec![0.1, 0.1]), serde_json::json!({}))
    }

    fn make_edge_with_correction(from: Uuid, to: Uuid, weight: f32, count: u64) -> Edge {
        let mut edge = Edge::new(from, to, EdgeType::Association, weight);
        edge.metadata.insert("correction_count".into(), serde_json::json!(count));
        edge
    }

    fn make_edge_clean(from: Uuid, to: Uuid) -> Edge {
        Edge::new(from, to, EdgeType::Association, 0.8)
    }

    #[test]
    fn no_corrections_yields_no_events() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        let n1 = make_node();
        let n2 = make_node();
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_edge(&make_edge_clean(n1.id, n2.id)).unwrap();

        let report = LTDDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0);
        assert_eq!(report.nodes_scanned, 1); // edge count
    }

    #[test]
    fn edge_with_corrections_emits_event() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let mut rx = bus.subscribe();
        let config = AgencyConfig::default();

        let n1 = make_node();
        let n2 = make_node();
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        storage.put_edge(&make_edge_with_correction(n1.id, n2.id, 0.7, 3)).unwrap();

        let report = LTDDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 1);

        let event = rx.try_recv().unwrap();
        match event {
            AgencyEvent::CorrectionAccumulated { correction_count, weight_delta, .. } => {
                assert_eq!(correction_count, 3);
                // Default rate 0.05 × 3 = 0.15
                assert!((weight_delta - 0.15).abs() < 0.001);
            }
            _ => panic!("expected CorrectionAccumulated"),
        }
    }

    #[test]
    fn weight_delta_capped_at_current_weight() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let mut rx = bus.subscribe();
        let config = AgencyConfig::default();

        let n1 = make_node();
        let n2 = make_node();
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        // Tiny weight, large correction count → delta capped at weight
        storage.put_edge(&make_edge_with_correction(n1.id, n2.id, 0.1, 100)).unwrap();

        let _ = LTDDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        let event = rx.try_recv().unwrap();
        match event {
            AgencyEvent::CorrectionAccumulated { weight_delta, .. } => {
                // 0.05 × 100 = 5.0, capped at weight 0.1
                assert!(weight_delta <= 0.1 + 1e-4, "delta must not exceed weight");
            }
            _ => panic!("expected CorrectionAccumulated"),
        }
    }

    #[test]
    fn below_threshold_no_event() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig {
            ltd_correction_threshold: Some(5), // require ≥5 corrections
            ..AgencyConfig::default()
        };

        let n1 = make_node();
        let n2 = make_node();
        storage.put_node(&n1).unwrap();
        storage.put_node(&n2).unwrap();
        // Only 2 corrections → below threshold of 5
        storage.put_edge(&make_edge_with_correction(n1.id, n2.id, 0.8, 2)).unwrap();

        let report = LTDDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0);
    }
}
