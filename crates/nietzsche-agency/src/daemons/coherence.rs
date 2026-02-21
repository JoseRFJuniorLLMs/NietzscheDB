use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_pregel::{DiffusionConfig, DiffusionEngine};

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

use super::{AgencyDaemon, DaemonReport};

/// Detects when the graph has become too uniform — when focused recall (t=0.1)
/// and free association (t=10.0) activate the same nodes, meaning the
/// hyperbolic hierarchy has lost discriminative power.
///
/// ## Algorithm
///
/// 1. Select up to 10 high-energy probe nodes spread across depth quantiles.
/// 2. Run multi-scale diffusion at `[0.1, 1.0, 10.0]` for each probe.
/// 3. Compute `jaccard_overlap(t=0.1, t=10.0)` per probe, then average.
/// 4. If `mean_overlap > threshold` (default 0.70), emit `CoherenceDrop`.
///
/// This is the most expensive daemon (Laplacian build + Chebyshev), but
/// runs only once per agency tick.
pub struct CoherenceDaemon;

const MAX_PROBES: usize = 10;
const T_SCALES: &[f64] = &[0.1, 1.0, 10.0];
const MIN_ENERGY_FOR_PROBE: f32 = 0.5;

impl AgencyDaemon for CoherenceDaemon {
    fn name(&self) -> &str { "coherence" }

    fn tick(
        &self,
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        // Collect candidate probes: high-energy nodes
        let mut candidates: Vec<(uuid::Uuid, f32)> = Vec::new(); // (id, depth)
        let mut nodes_scanned = 0usize;

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };
            nodes_scanned += 1;

            if meta.energy >= MIN_ENERGY_FOR_PROBE {
                candidates.push((meta.id, meta.depth));
            }
        }

        if candidates.len() < 3 || nodes_scanned < 5 {
            return Ok(DaemonReport {
                daemon_name: "coherence".into(),
                nodes_scanned,
                details: vec!["too few nodes for coherence analysis".into()],
                ..Default::default()
            });
        }

        // Spread probes across depth quantiles
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let step = candidates.len().max(1) / MAX_PROBES.min(candidates.len()).max(1);
        let probes: Vec<uuid::Uuid> = candidates
            .iter()
            .step_by(step.max(1))
            .take(MAX_PROBES)
            .map(|(id, _)| *id)
            .collect();

        if probes.is_empty() {
            return Ok(DaemonReport {
                daemon_name: "coherence".into(),
                nodes_scanned,
                ..Default::default()
            });
        }

        let engine = DiffusionEngine::new(DiffusionConfig::default());

        let mut sum_overlap_01_10 = 0.0f64;
        let mut sum_overlap_01_1 = 0.0f64;
        let mut probe_count = 0usize;

        for probe_id in &probes {
            let results = engine.diffuse(storage, adjacency, &[*probe_id], T_SCALES)?;
            if results.len() < 3 { continue; }

            let o_01_10 = DiffusionEngine::scale_overlap(&results, 0.1, 10.0);
            let o_01_1 = DiffusionEngine::scale_overlap(&results, 0.1, 1.0);

            sum_overlap_01_10 += o_01_10;
            sum_overlap_01_1 += o_01_1;
            probe_count += 1;
        }

        let mut events_emitted = 0;
        let mut details = Vec::new();

        if probe_count > 0 {
            let mean_overlap_01_10 = sum_overlap_01_10 / probe_count as f64;
            let mean_overlap_01_1 = sum_overlap_01_1 / probe_count as f64;

            details.push(format!(
                "coherence: probes={probe_count}, J(0.1,10.0)={mean_overlap_01_10:.3}, J(0.1,1.0)={mean_overlap_01_1:.3}"
            ));

            if mean_overlap_01_10 > config.coherence_overlap_threshold {
                bus.publish(AgencyEvent::CoherenceDrop {
                    overlap_01_10: mean_overlap_01_10,
                    overlap_01_1: mean_overlap_01_1,
                });
                events_emitted = 1;
            }
        }

        Ok(DaemonReport {
            daemon_name: "coherence".into(),
            events_emitted,
            nodes_scanned,
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

    fn make_node(x: f32, y: f32, energy: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.meta.energy = energy;
        node
    }

    #[test]
    fn chain_graph_runs_without_panic() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        // Build a chain: n0 -> n1 -> n2 -> ... -> n19
        let mut prev_id = None;
        for i in 0..20 {
            let depth = 0.05 + (i as f32 / 20.0) * 0.8;
            let node = make_node(depth, 0.01, 0.8);
            let nid = node.id;
            storage.put_node(&node).unwrap();

            if let Some(pid) = prev_id {
                let edge = Edge::new(pid, nid, EdgeType::Association, 0.9);
                storage.put_edge(&edge).unwrap();
                adjacency.add_edge(&edge);
            }
            prev_id = Some(nid);
        }

        let report = CoherenceDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 20);
    }

    #[test]
    fn too_few_nodes_skips() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        // Only 2 nodes — below threshold
        storage.put_node(&make_node(0.1, 0.0, 0.8)).unwrap();
        storage.put_node(&make_node(0.2, 0.0, 0.8)).unwrap();

        let report = CoherenceDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0);
        assert!(report.details.iter().any(|d| d.contains("too few")));
    }
}
