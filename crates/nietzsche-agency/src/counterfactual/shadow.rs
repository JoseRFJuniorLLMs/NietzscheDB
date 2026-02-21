use std::collections::HashMap;

use uuid::Uuid;
use nietzsche_graph::{AdjEntry, AdjacencyIndex, GraphStorage, NodeMeta};

use crate::error::AgencyError;

/// Lightweight shadow copy of the graph for what-if simulation.
///
/// Contains **only** metadata (~100 bytes/node) and adjacency lists —
/// no embeddings. This keeps the memory footprint at ~130 bytes/node
/// instead of ~12 KB/node.
pub struct ShadowGraph {
    pub nodes: HashMap<Uuid, NodeMeta>,
    pub adj_out: HashMap<Uuid, Vec<AdjEntry>>,
    pub adj_in: HashMap<Uuid, Vec<AdjEntry>>,
}

impl ShadowGraph {
    /// Build a shadow copy from the real graph.
    ///
    /// Copies all NodeMeta and adjacency entries. O(N + E).
    pub fn snapshot(
        storage: &GraphStorage,
        adjacency: &AdjacencyIndex,
    ) -> Result<Self, AgencyError> {
        let mut nodes = HashMap::new();
        for result in storage.iter_nodes_meta() {
            let meta = result?;
            nodes.insert(meta.id, meta);
        }

        let mut adj_out = HashMap::new();
        let mut adj_in: HashMap<Uuid, Vec<AdjEntry>> = HashMap::new();

        for (node_id, entries) in adjacency.snapshot_outgoing() {
            for entry in &entries {
                adj_in
                    .entry(entry.neighbor_id)
                    .or_default()
                    .push(AdjEntry {
                        edge_id: entry.edge_id,
                        neighbor_id: node_id,
                        weight: entry.weight,
                        edge_type: entry.edge_type.clone(),
                    });
            }
            adj_out.insert(node_id, entries);
        }

        Ok(Self { nodes, adj_out, adj_in })
    }

    /// Simulate adding a node (shadow only — no real mutation).
    pub fn simulate_add_node(&mut self, meta: NodeMeta) {
        self.nodes.insert(meta.id, meta);
    }

    /// Simulate adding an edge (shadow only — updates adjacency).
    pub fn simulate_add_edge(&mut self, from: Uuid, to: Uuid, weight: f32) {
        let edge_id = Uuid::new_v4();
        self.adj_out
            .entry(from)
            .or_default()
            .push(AdjEntry {
                edge_id,
                neighbor_id: to,
                weight,
                edge_type: nietzsche_graph::EdgeType::Association,
            });
        self.adj_in
            .entry(to)
            .or_default()
            .push(AdjEntry {
                edge_id,
                neighbor_id: from,
                weight,
                edge_type: nietzsche_graph::EdgeType::Association,
            });
    }

    /// Simulate removing a node (shadow only).
    ///
    /// Removes the node and cleans up adjacency in both directions.
    pub fn simulate_remove_node(&mut self, id: Uuid) {
        self.nodes.remove(&id);

        // Clean outgoing edges and their reverse pointers
        if let Some(out_entries) = self.adj_out.remove(&id) {
            for entry in &out_entries {
                if let Some(in_list) = self.adj_in.get_mut(&entry.neighbor_id) {
                    in_list.retain(|e| e.neighbor_id != id);
                }
            }
        }

        // Clean incoming edges and their forward pointers
        if let Some(in_entries) = self.adj_in.remove(&id) {
            for entry in &in_entries {
                if let Some(out_list) = self.adj_out.get_mut(&entry.neighbor_id) {
                    out_list.retain(|e| e.neighbor_id != id);
                }
            }
        }
    }

    /// Get all neighbor IDs (both directions, deduplicated).
    pub fn neighbors(&self, id: &Uuid) -> Vec<Uuid> {
        let mut seen = std::collections::HashSet::new();
        if let Some(out) = self.adj_out.get(id) {
            for e in out { seen.insert(e.neighbor_id); }
        }
        if let Some(inc) = self.adj_in.get(id) {
            for e in inc { seen.insert(e.neighbor_id); }
        }
        seen.into_iter().collect()
    }

    /// Get NodeMeta by ID.
    pub fn get_meta(&self, id: &Uuid) -> Option<&NodeMeta> {
        self.nodes.get(id)
    }

    /// Outgoing degree.
    pub fn degree_out(&self, id: &Uuid) -> usize {
        self.adj_out.get(id).map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Edge, EdgeType, Node, PoincareVector};
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn make_node(x: f32, y: f32) -> Node {
        Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({}),
        )
    }

    #[test]
    fn snapshot_matches_real_graph() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0);
        let b = make_node(0.2, 0.0);
        let aid = a.id;
        let bid = b.id;
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();

        let edge = Edge::new(aid, bid, EdgeType::Association, 0.8);
        storage.put_edge(&edge).unwrap();
        adjacency.add_edge(&edge);

        let shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        assert_eq!(shadow.nodes.len(), 2);
        assert!(shadow.get_meta(&aid).is_some());
        assert!(shadow.get_meta(&bid).is_some());
        assert_eq!(shadow.degree_out(&aid), 1);
    }

    #[test]
    fn simulate_remove_does_not_affect_real() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0);
        let aid = a.id;
        storage.put_node(&a).unwrap();

        let mut shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();
        shadow.simulate_remove_node(aid);

        // Shadow has no node
        assert!(shadow.get_meta(&aid).is_none());
        // Real storage still has the node
        assert!(storage.get_node_meta(&aid).unwrap().is_some());
    }

    #[test]
    fn simulate_add_node_and_edge() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let a = make_node(0.1, 0.0);
        let aid = a.id;
        storage.put_node(&a).unwrap();

        let mut shadow = ShadowGraph::snapshot(&storage, &adjacency).unwrap();

        // Simulate adding a new node
        let new_id = Uuid::new_v4();
        let new_meta = nietzsche_graph::NodeMeta {
            id: new_id,
            depth: 0.3,
            content: serde_json::json!({}),
            node_type: nietzsche_graph::NodeType::Semantic,
            energy: 0.8,
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: 0,
            expires_at: None,
            metadata: std::collections::HashMap::new(),
        };
        shadow.simulate_add_node(new_meta);
        shadow.simulate_add_edge(aid, new_id, 0.9);

        assert!(shadow.get_meta(&new_id).is_some());
        assert_eq!(shadow.degree_out(&aid), 1);
        assert_eq!(shadow.neighbors(&aid), vec![new_id]);

        // Real storage doesn't have the new node
        assert!(storage.get_node_meta(&new_id).unwrap().is_none());
    }
}
