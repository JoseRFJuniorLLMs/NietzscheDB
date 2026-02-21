use nietzsche_graph::{AdjacencyIndex, GraphStorage};

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus, SectorId};

use super::{AgencyDaemon, DaemonReport};

/// Detects knowledge gaps — sparse sectors in the Poincare ball where few
/// or no nodes exist.
///
/// ## Algorithm
///
/// 1. Partition the ball into a `(depth, angle)` grid:
///    - `D` radial bands (default 5): `[0, 0.2), [0.2, 0.4), ..., [0.8, 1.0)`
///    - `A` angular slices (default 16)
///    - Total sectors: `D * A`
///
/// 2. Scan all `NodeMeta` (~100 bytes each). Assign each node to a sector
///    using `node.depth` for the radial band and a UUID hash for the angle
///    (avoids loading embeddings).
///
/// 3. Sectors with count below `max(1, total_nodes / (sectors * 10))` are
///    knowledge gaps.
pub struct GapDaemon;

impl AgencyDaemon for GapDaemon {
    fn name(&self) -> &str { "gap" }

    fn tick(
        &self,
        storage: &GraphStorage,
        _adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        let d = config.gap_depth_bins.max(1);
        let a = config.gap_sector_count.max(1);
        let total_sectors = d * a;

        // count[depth_bin * a + angular_bin]
        let mut counts = vec![0usize; total_sectors];
        let mut total_nodes = 0usize;

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };
            total_nodes += 1;

            let depth_bin = ((meta.depth * d as f32).floor() as usize).min(d - 1);
            let angular_bin = angular_bin_from_id(&meta.id, a);
            counts[depth_bin * a + angular_bin] += 1;
        }

        if total_nodes == 0 {
            return Ok(DaemonReport {
                daemon_name: "gap".into(),
                nodes_scanned: 0,
                ..Default::default()
            });
        }

        // Gap threshold: sectors with fewer nodes than expected * min_density
        let expected_per_sector = total_nodes as f64 / total_sectors as f64;
        let threshold = (expected_per_sector * config.gap_min_density).ceil().max(1.0) as usize;

        let mut events_emitted = 0usize;
        let mut details = Vec::new();

        for (idx, &count) in counts.iter().enumerate() {
            if count < threshold {
                let depth_bin = idx / a;
                let angular_bin = idx % a;
                let depth_lo = depth_bin as f32 / d as f32;
                let depth_hi = (depth_bin + 1) as f32 / d as f32;
                let density = count as f64 / total_nodes as f64;

                bus.publish(AgencyEvent::KnowledgeGap {
                    sector: SectorId { depth_bin, angular_bin },
                    density,
                    suggested_depth_range: (depth_lo, depth_hi),
                });
                events_emitted += 1;
                details.push(format!(
                    "gap at d={depth_bin} a={angular_bin}: {count}/{threshold} nodes"
                ));
            }
        }

        Ok(DaemonReport {
            daemon_name: "gap".into(),
            events_emitted,
            nodes_scanned: total_nodes,
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

fn angular_bin_from_id(id: &uuid::Uuid, slices: usize) -> usize {
    let bytes = id.as_bytes();
    let hash = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
    hash % slices
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, PoincareVector};
    use tempfile::TempDir;
    use uuid::Uuid;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node_at_depth(depth_target: f32) -> Node {
        // Place node at (depth_target, 0) — inside ball since depth_target < 1
        let x = depth_target.min(0.99);
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, 0.0]),
            serde_json::json!({}),
        )
    }

    #[test]
    fn many_nodes_few_gaps() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(256);
        let config = AgencyConfig {
            gap_depth_bins: 2,
            gap_sector_count: 2,
            gap_min_density: 0.01, // very low threshold
            ..AgencyConfig::default()
        };

        // Insert 200 nodes — with random UUIDs they'll spread across sectors
        for i in 0..200 {
            let depth = (i as f32 / 200.0) * 0.98;
            let node = make_node_at_depth(depth);
            storage.put_node(&node).unwrap();
        }

        let report = GapDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 200);
        // With 200 nodes across 4 sectors, gaps should be minimal
    }

    #[test]
    fn single_node_many_gaps() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(256);
        let config = AgencyConfig {
            gap_depth_bins: 5,
            gap_sector_count: 4,
            gap_min_density: 0.1,
            ..AgencyConfig::default()
        };

        // Single node → most sectors are empty → many gaps
        storage.put_node(&make_node_at_depth(0.1)).unwrap();

        let report = GapDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 1);
        // With 20 total sectors and 1 node, at least 18 should be gaps
        assert!(report.events_emitted >= 18, "expected many gaps, got {}", report.events_emitted);
    }

    #[test]
    fn empty_graph_no_crash() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = nietzsche_graph::AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        let report = GapDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.events_emitted, 0);
    }
}
