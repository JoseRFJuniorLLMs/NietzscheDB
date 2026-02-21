use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

use super::{AgencyDaemon, DaemonReport};

/// Detects regions in the Poincare ball where local Hausdorff dimensions are
/// wildly inconsistent, indicating chaotic or poorly structured sub-graphs.
///
/// ## Algorithm
///
/// 1. Scan all `NodeMeta` (~100 bytes each, no embeddings).
/// 2. Partition into regions using `(depth_bin, angular_bin)`:
///    - `depth_bin = min(floor(depth * 4), 3)` — 4 radial bands.
///    - `angular_bin` derived from hash of `node.id` — deterministic, avoids
///      loading 12 KB embeddings just for an angle.
/// 3. Compute variance of `hausdorff_local` per region.
/// 4. If variance exceeds threshold, emit `EntropySpike`.
pub struct EntropyDaemon;

impl AgencyDaemon for EntropyDaemon {
    fn name(&self) -> &str { "entropy" }

    fn tick(
        &self,
        storage: &GraphStorage,
        _adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        let depth_bands: usize = 4;
        let angular_slices = config.entropy_region_count.max(1);
        let total_regions = depth_bands * angular_slices;

        // Per-region accumulators: (sum, sum_sq, count, sample_ids)
        let mut regions: Vec<(f64, f64, usize, Vec<uuid::Uuid>)> =
            vec![(0.0, 0.0, 0, Vec::new()); total_regions];

        let mut nodes_scanned = 0usize;

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };
            nodes_scanned += 1;

            let depth_bin = ((meta.depth * depth_bands as f32).floor() as usize)
                .min(depth_bands - 1);
            let angular_bin = angular_bin_from_id(&meta.id, angular_slices);
            let region_idx = depth_bin * angular_slices + angular_bin;

            let r = &mut regions[region_idx];
            let h = meta.hausdorff_local as f64;
            r.0 += h;
            r.1 += h * h;
            r.2 += 1;
            if r.3.len() < 10 {
                r.3.push(meta.id);
            }
        }

        let mut events_emitted = 0usize;
        let mut details = Vec::new();

        for (idx, (sum, sum_sq, count, ref sample_ids)) in regions.iter().enumerate() {
            if *count < 2 { continue; }
            let n = *count as f64;
            let mean = sum / n;
            let variance = (sum_sq / n - mean * mean).max(0.0) as f32;

            if variance > config.entropy_variance_threshold {
                bus.publish(AgencyEvent::EntropySpike {
                    region_id: idx,
                    variance,
                    sample_node_ids: sample_ids.clone(),
                });
                events_emitted += 1;
                details.push(format!("region {idx}: var={variance:.4}"));
            }
        }

        Ok(DaemonReport {
            daemon_name: "entropy".into(),
            events_emitted,
            nodes_scanned,
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

/// Derive an angular bin from a UUID deterministically.
///
/// Uses the first 2 bytes of the UUID as a pseudo-random angle proxy.
/// This avoids loading the full embedding (~12 KB) just to compute
/// `atan2(coords[1], coords[0])`.
fn angular_bin_from_id(id: &uuid::Uuid, slices: usize) -> usize {
    let bytes = id.as_bytes();
    let hash = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    hash % slices
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

    fn make_node(x: f32, y: f32, energy: f32, hausdorff: f32) -> Node {
        let mut node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        );
        node.meta.energy = energy;
        node.meta.hausdorff_local = hausdorff;
        node
    }

    #[test]
    fn uniform_hausdorff_no_spike() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        // All nodes with the same hausdorff → zero variance → no spike
        for i in 0..20 {
            let node = make_node(0.01 * i as f32, 0.01, 0.5, 1.0);
            storage.put_node(&node).unwrap();
        }

        let report = EntropyDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0);
        assert_eq!(report.nodes_scanned, 20);
    }

    #[test]
    fn mixed_hausdorff_detects_spike() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let mut rx = bus.subscribe();
        let config = AgencyConfig {
            entropy_region_count: 1, // force all into one region
            entropy_variance_threshold: 0.01,
            ..AgencyConfig::default()
        };

        // Mix hausdorff: half 0.5, half 1.8 → high variance
        for i in 0..20 {
            let h = if i < 10 { 0.5 } else { 1.8 };
            let node = make_node(0.01 * (i + 1) as f32, 0.01, 0.5, h);
            storage.put_node(&node).unwrap();
        }

        let report = EntropyDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert!(report.events_emitted > 0, "should detect spike");

        let event = rx.try_recv().unwrap();
        match event {
            AgencyEvent::EntropySpike { variance, .. } => {
                assert!(variance > 0.01);
            }
            _ => panic!("expected EntropySpike"),
        }
    }

    #[test]
    fn empty_graph_no_crash() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        let report = EntropyDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.events_emitted, 0);
    }
}
