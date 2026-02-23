use uuid::Uuid;
use std::collections::{HashSet, VecDeque};
use nietzsche_graph::{AdjacencyIndex, GraphStorage, Node};
use crate::Result;

#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    pub nodes: Vec<Node>,
    /// (source_idx, target_idx, edge_weight)
    pub edges: Vec<(usize, usize, f32)>,
}

pub struct NeighborSampler<'a> {
    storage: &'a GraphStorage,
    adjacency: &'a AdjacencyIndex,
}

impl<'a> NeighborSampler<'a> {
    pub fn new(storage: &'a GraphStorage, adjacency: &'a AdjacencyIndex) -> Self {
        Self { storage, adjacency }
    }

    /// Sample a K-hop neighborhood around a seed node.
    pub fn sample_k_hop(&self, seed_id: Uuid, k: usize) -> Result<SampledSubgraph> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((seed_id, 0));
        visited.insert(seed_id);

        let mut node_ids = Vec::new();
        
        // BFS to find nodes
        while let Some((node_id, depth)) = queue.pop_front() {
            node_ids.push(node_id);
            if depth < k {
                for neighbor in self.adjacency.neighbors_both(&node_id) {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        // Fetch nodes and build edge list
        let mut nodes = Vec::with_capacity(node_ids.len());
        let mut id_to_idx = std::collections::HashMap::with_capacity(node_ids.len());
        
        for (idx, &id) in node_ids.iter().enumerate() {
            if let Some(node) = self.storage.get_node(&id)? {
                nodes.push(node);
                id_to_idx.insert(id, idx);
            }
        }

        let mut edges = Vec::new();
        for (idx, &id) in node_ids.iter().enumerate() {
            for entry in self.adjacency.entries_out(&id) {
                if let Some(&target_idx) = id_to_idx.get(&entry.neighbor_id) {
                    edges.push((idx, target_idx, entry.weight));
                }
            }
        }

        Ok(SampledSubgraph { nodes, edges })
    }
}
