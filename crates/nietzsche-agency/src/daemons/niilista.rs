//! Daemon Niilista — Semantic Garbage Collector.
//!
//! The "immune system" of NietzscheDB.  Inspired by Nietzsche's *Amor Fati*
//! (acceptance of destruction as counterpart to the Übermensch's will to
//! power), the Niilista hunts for **semantically redundant** nodes — those
//! whose embeddings are >99% similar — and emits events so the reactor can
//! fuse them into a single **Archetype** node.
//!
//! Like all agency daemons this is **read-only**: it scans, analyses, and
//! emits events.  The actual mutation (merge + phantomize) is performed by
//! the server in response to `AgencyIntent::TriggerSemanticGc`.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use uuid::Uuid;

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

use super::{AgencyDaemon, DaemonReport};

/// Semantic GC daemon — detects near-duplicate embeddings.
///
/// ## Algorithm
///
/// 1. Scan `NodeMeta` for non-phantom nodes with `energy > 0`.
/// 2. Load embeddings for those nodes (up to `max_scan`).
/// 3. Pairwise sq-Euclidean pre-filter → full Poincaré distance if close.
/// 4. Union-Find clustering of near-duplicates (distance < threshold).
/// 5. Emit `SemanticRedundancy` for each group ≥ `min_group_size`.
pub struct NiilistaGcDaemon;

impl AgencyDaemon for NiilistaGcDaemon {
    fn name(&self) -> &str { "niilista" }

    fn tick(
        &self,
        storage: &GraphStorage,
        _adjacency: &AdjacencyIndex,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<DaemonReport, AgencyError> {
        let t0 = std::time::Instant::now();

        let max_scan = config.niilista_max_scan;
        let distance_threshold = config.niilista_distance_threshold as f64;
        let min_group = config.niilista_min_group_size;
        // Cheap pre-filter: sq-euclidean upper bound for Poincaré distance
        // For small distances, d_poincaré ≈ 2·sqrt(sq_euclidean), so
        // sq_euclidean < (threshold/2)² is a safe pre-filter.
        let sq_prefilter = (distance_threshold / 2.0).powi(2);

        // ── 1. Collect candidate node IDs ─────────────────────────
        let mut candidates: Vec<(Uuid, nietzsche_graph::PoincareVector)> = Vec::new();
        let mut nodes_scanned = 0usize;

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };
            nodes_scanned += 1;

            if meta.is_phantom || meta.energy <= 0.0 {
                continue;
            }

            if candidates.len() >= max_scan {
                break;
            }

            // Load embedding
            if let Ok(Some(emb)) = storage.get_embedding(&meta.id) {
                candidates.push((meta.id, emb));
            }
        }

        if candidates.len() < min_group {
            return Ok(DaemonReport {
                daemon_name: "niilista".into(),
                events_emitted: 0,
                nodes_scanned,
                duration_us: t0.elapsed().as_micros() as u64,
                details: vec![],
            });
        }

        // ── 2. Cluster near-duplicates ────────────────────────────
        let n = candidates.len();
        let mut uf = UnionFind::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                // Cheap pre-filter
                let sq = candidates[i].1.sq_euclidean(&candidates[j].1);
                if sq > sq_prefilter {
                    continue;
                }
                // Full Poincaré distance
                let dist = candidates[i].1.distance(&candidates[j].1);
                if dist < distance_threshold {
                    uf.union(i, j);
                }
            }
        }

        // Collect groups
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            groups.entry(uf.find(i)).or_default().push(i);
        }

        let mut events_emitted = 0usize;
        let mut details = Vec::new();

        for indices in groups.into_values() {
            if indices.len() < min_group {
                continue;
            }

            let node_ids: Vec<Uuid> = indices.iter().map(|&i| candidates[i].0).collect();

            // Pick archetype: highest energy node
            let archetype_id = node_ids[
                indices.iter()
                    .enumerate()
                    .max_by(|(_, &a), (_, &b)| {
                        // Compare by energy — need to scan meta again (cheap)
                        let ea = storage.get_node_meta(&candidates[a].0)
                            .ok().flatten().map(|m| m.energy).unwrap_or(0.0);
                        let eb = storage.get_node_meta(&candidates[b].0)
                            .ok().flatten().map(|m| m.energy).unwrap_or(0.0);
                        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            ];

            bus.publish(AgencyEvent::SemanticRedundancy {
                group_size: node_ids.len(),
                archetype_id,
                redundant_ids: node_ids.iter().filter(|&&id| id != archetype_id).cloned().collect(),
            });

            events_emitted += 1;
            details.push(format!(
                "group: {} nodes, archetype={}",
                node_ids.len(),
                archetype_id,
            ));
        }

        Ok(DaemonReport {
            daemon_name: "niilista".into(),
            events_emitted,
            nodes_scanned,
            duration_us: t0.elapsed().as_micros() as u64,
            details,
        })
    }
}

// ── Union-Find ──────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank:   Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb { return; }
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
        } else {
            self.parent[rb] = ra;
            self.rank[ra] += 1;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, PoincareVector};
    use tempfile::TempDir;

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
        node.meta.is_phantom = false;
        node
    }

    #[test]
    fn detects_near_duplicate_cluster() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);
        let mut rx = bus.subscribe();

        // 5 nodes at nearly identical positions
        for i in 0..5 {
            let tiny = i as f32 * 0.0001; // << threshold
            let node = make_node(0.3 + tiny, 0.2 + tiny, 0.7);
            storage.put_node(&node).unwrap();
        }

        let config = AgencyConfig {
            niilista_distance_threshold: 0.05,
            niilista_min_group_size: 3,
            niilista_max_scan: 100,
            ..AgencyConfig::default()
        };

        let report = NiilistaGcDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 1, "should detect one redundancy group");
        assert_eq!(report.nodes_scanned, 5);

        let event = rx.try_recv().unwrap();
        match event {
            AgencyEvent::SemanticRedundancy { group_size, archetype_id, redundant_ids } => {
                assert_eq!(group_size, 5);
                assert_eq!(redundant_ids.len(), 4);
                assert!(!redundant_ids.contains(&archetype_id));
            }
            _ => panic!("expected SemanticRedundancy event"),
        }
    }

    #[test]
    fn skips_distant_nodes() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);

        // 5 nodes spread far apart
        for i in 0..5 {
            let node = make_node(0.1 * (i + 1) as f32, 0.0, 0.7);
            storage.put_node(&node).unwrap();
        }

        let config = AgencyConfig {
            niilista_distance_threshold: 0.01,
            niilista_min_group_size: 2,
            niilista_max_scan: 100,
            ..AgencyConfig::default()
        };

        let report = NiilistaGcDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0, "distant nodes → no redundancy");
    }

    #[test]
    fn skips_phantom_nodes() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(64);

        // 5 phantom nodes at identical positions — should be ignored
        for _ in 0..5 {
            let mut node = make_node(0.3, 0.2, 0.7);
            node.meta.is_phantom = true;
            storage.put_node(&node).unwrap();
        }

        let config = AgencyConfig {
            niilista_distance_threshold: 0.05,
            niilista_min_group_size: 2,
            niilista_max_scan: 100,
            ..AgencyConfig::default()
        };

        let report = NiilistaGcDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.events_emitted, 0, "phantom nodes should be skipped");
    }

    #[test]
    fn empty_graph_no_crash() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        let bus = AgencyEventBus::new(16);
        let config = AgencyConfig::default();

        let report = NiilistaGcDaemon.tick(&storage, &adjacency, &bus, &config).unwrap();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.events_emitted, 0);
    }
}
