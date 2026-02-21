//! Memory Fusion: Episodic → Semantic consolidation.
//!
//! During sleep, groups of similar episodic nodes are fused into a single
//! semantic node near the centre of the Poincaré ball. The original episodic
//! nodes are phantomized (topology preserved, geometry intact).
//!
//! This mirrors the brain's consolidation of episodic memory traces into
//! abstract semantic knowledge.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::{
    Edge, EdgeType, Node, NodeType, PoincareVector, NietzscheDB, VectorStore,
};

use crate::SleepError;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for episodic → semantic memory fusion.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Minimum episodic nodes in a group to trigger fusion (default: 5).
    pub min_group_size: usize,
    /// Maximum Poincaré distance between nodes for them to belong to the
    /// same cluster (default: 0.5).
    pub distance_threshold: f64,
    /// Maximum number of episodic nodes to consider per cycle (default: 500).
    pub max_scan: usize,
    /// Minimum energy for a node to be eligible for fusion (default: 0.3).
    pub min_energy: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            min_group_size: 5,
            distance_threshold: 0.5,
            max_scan: 500,
            min_energy: 0.3,
        }
    }
}

// ── Report ──────────────────────────────────────────────────────────────────

/// Outcome of a single fusion cycle.
#[derive(Debug, Clone, Default)]
pub struct FusionReport {
    /// Number of episodic clusters identified.
    pub groups_found: usize,
    /// Total episodic nodes that were fused (phantomized).
    pub nodes_fused: usize,
    /// New semantic nodes created from merged clusters.
    pub semantic_nodes_created: usize,
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

// ── Core ────────────────────────────────────────────────────────────────────

/// Run one episodic → semantic fusion cycle.
///
/// 1. Scan episodic nodes with energy ≥ `min_energy`
/// 2. Group by embedding proximity (Poincaré distance < `distance_threshold`)
/// 3. For each group ≥ `min_group_size`: create a semantic centroid node
/// 4. Phantomize originals (topology preserved via Phantom Nodes)
pub fn fuse_episodic_memories<V: VectorStore>(
    config: &FusionConfig,
    db: &mut NietzscheDB<V>,
) -> Result<FusionReport, SleepError> {
    let mut report = FusionReport::default();

    // ── 1. Collect eligible episodic nodes ────────────────────
    let all_nodes = db.storage().scan_nodes()
        .map_err(|e| SleepError::Graph(e))?;

    let episodics: Vec<Node> = all_nodes
        .into_iter()
        .filter(|n| {
            n.meta.node_type == NodeType::Episodic
                && n.meta.energy >= config.min_energy
                && !n.meta.is_phantom
        })
        .take(config.max_scan)
        .collect();

    if episodics.len() < config.min_group_size {
        return Ok(report);
    }

    // ── 2. Cluster by Poincaré distance ──────────────────────
    let n = episodics.len();
    let mut uf = UnionFind::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = episodics[i].embedding.distance(&episodics[j].embedding);
            if dist < config.distance_threshold {
                uf.union(i, j);
            }
        }
    }

    // Collect groups by root representative
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        groups.entry(uf.find(i)).or_default().push(i);
    }

    // Filter to groups large enough for fusion
    let fusable: Vec<Vec<usize>> = groups
        .into_values()
        .filter(|g| g.len() >= config.min_group_size)
        .collect();

    report.groups_found = fusable.len();

    // ── 3. Merge each group into a semantic node ─────────────
    for group_indices in &fusable {
        let dim = episodics[group_indices[0]].embedding.dim;
        let group_len = group_indices.len() as f32;

        // Compute centroid embedding (average of coordinates)
        let mut centroid = vec![0.0f32; dim];
        for &idx in group_indices {
            for (c, coord) in centroid.iter_mut().zip(episodics[idx].embedding.coords.iter()) {
                *c += coord;
            }
        }
        for c in centroid.iter_mut() {
            *c /= group_len;
        }

        // Project into Poincaré ball
        let centroid_vec = PoincareVector::new(centroid).project_into_ball();

        // Energy = max of the group (strongest memory survives)
        let max_energy = group_indices.iter()
            .map(|&i| episodics[i].meta.energy)
            .fold(0.0f32, f32::max);

        // Content = array of original contents
        let merged_content = serde_json::json!({
            "fusion": "episodic_to_semantic",
            "source_count": group_indices.len(),
            "sources": group_indices.iter()
                .map(|&i| serde_json::json!({
                    "id": episodics[i].id.to_string(),
                    "content": episodics[i].meta.content.clone(),
                }))
                .collect::<Vec<_>>(),
        });

        // Create the new semantic node
        let mut semantic_node = Node::new(Uuid::new_v4(), centroid_vec, merged_content);
        semantic_node.meta.node_type = NodeType::Semantic;
        semantic_node.meta.energy = max_energy;

        let semantic_id = semantic_node.id;
        db.insert_node(semantic_node)
            .map_err(|e| SleepError::Graph(e))?;

        // Connect the new semantic node to all neighbours of the fused nodes
        // (preserves reachability from the rest of the graph)
        let mut connected: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        let fused_ids: std::collections::HashSet<Uuid> = group_indices.iter()
            .map(|&i| episodics[i].id)
            .collect();

        for &idx in group_indices {
            let node_id = episodics[idx].id;
            let neighbours = db.adjacency().neighbors_both(&node_id);
            for nb in neighbours {
                // Don't connect to other fused nodes in the same group
                if !fused_ids.contains(&nb) && connected.insert(nb) {
                    db.insert_edge(Edge::new(semantic_id, nb, EdgeType::Association, 1.0))
                        .map_err(|e| SleepError::Graph(e))?;
                }
            }
        }

        // ── 4. Phantomize originals ──────────────────────────
        for &idx in group_indices {
            db.phantomize_node(episodics[idx].id)
                .map_err(|e| SleepError::Graph(e))?;
        }

        report.nodes_fused += group_indices.len();
        report.semantic_nodes_created += 1;
    }

    Ok(report)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::MockVectorStore;
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_db(dir: &TempDir) -> NietzscheDB<MockVectorStore> {
        NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap()
    }

    fn episodic_node(x: f32, y: f32) -> Node {
        let mut n = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"memory": format!("event at ({x},{y})")}),
        );
        n.meta.node_type = NodeType::Episodic;
        n.meta.energy = 0.8;
        n
    }

    #[test]
    fn fusion_merges_close_episodic_nodes() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Insert 6 episodic nodes clustered near (0.1, 0.1)
        let mut ids = Vec::new();
        for i in 0..6 {
            let offset = i as f32 * 0.01;
            let node = episodic_node(0.1 + offset, 0.1 + offset);
            ids.push(node.id);
            db.insert_node(node).unwrap();
        }

        let config = FusionConfig {
            min_group_size: 5,
            distance_threshold: 0.5,
            ..Default::default()
        };

        let report = fuse_episodic_memories(&config, &mut db).unwrap();
        assert_eq!(report.groups_found, 1);
        assert_eq!(report.nodes_fused, 6);
        assert_eq!(report.semantic_nodes_created, 1);

        // Original nodes should be phantom
        for id in &ids {
            let meta = db.storage().get_node_meta(id).unwrap().unwrap();
            assert!(meta.is_phantom, "original episodic node should be phantom");
        }

        // The new semantic node should exist
        let all = db.storage().scan_nodes_meta()
            .unwrap()
            .into_iter()
            .filter(|m| m.node_type == NodeType::Semantic && !m.is_phantom)
            .collect::<Vec<_>>();
        assert_eq!(all.len(), 1, "one semantic node should be created");
        assert!(all[0].energy >= 0.8 - 1e-6);
    }

    #[test]
    fn fusion_skips_small_groups() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Insert only 3 episodic nodes (below min_group_size=5)
        for i in 0..3 {
            let node = episodic_node(0.1 + i as f32 * 0.01, 0.1);
            db.insert_node(node).unwrap();
        }

        let config = FusionConfig::default();
        let report = fuse_episodic_memories(&config, &mut db).unwrap();
        assert_eq!(report.groups_found, 0);
        assert_eq!(report.nodes_fused, 0);
    }

    #[test]
    fn fusion_separates_distant_clusters() {
        let dir = tmp();
        let mut db = open_db(&dir);

        // Cluster A: 5 nodes near (0.1, 0.0)
        for i in 0..5 {
            let node = episodic_node(0.1 + i as f32 * 0.01, 0.0);
            db.insert_node(node).unwrap();
        }

        // Cluster B: 5 nodes near (0.8, 0.0) — far from A
        for i in 0..5 {
            let node = episodic_node(0.8 + i as f32 * 0.005, 0.0);
            db.insert_node(node).unwrap();
        }

        let config = FusionConfig {
            min_group_size: 5,
            distance_threshold: 0.5,
            ..Default::default()
        };

        let report = fuse_episodic_memories(&config, &mut db).unwrap();
        assert_eq!(report.groups_found, 2, "two distant clusters");
        assert_eq!(report.semantic_nodes_created, 2);
        assert_eq!(report.nodes_fused, 10);
    }
}
