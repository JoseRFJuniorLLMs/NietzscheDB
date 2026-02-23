use uuid::Uuid;
use crate::{Result, MctsTree, MctsConfig};
use nietzsche_gnn::{GnnEngine, NeighborSampler};
use nietzsche_graph::{NietzscheDB, VectorStore};

pub struct AdvisorIntent {
    pub target_node: Uuid,
    pub confidence: f32,
    pub suggested_action: String,
}

pub struct MctsAdvisor<'a, V: VectorStore> {
    db: &'a NietzscheDB<V>,
    gnn: GnnEngine,
    config: MctsConfig,
}

impl<'a, V: VectorStore> MctsAdvisor<'a, V> {
    pub fn new(db: &'a NietzscheDB<V>, model_name: &str, config: MctsConfig) -> Self {
        Self {
            db,
            gnn: GnnEngine::new(model_name),
            config,
        }
    }

    pub async fn advise(&self, seed_node: Uuid) -> Result<Vec<AdvisorIntent>> {
        let mut tree = MctsTree::new(seed_node, MctsConfig {
            iterations: self.config.iterations,
            exploration_constant: self.config.exploration_constant,
        });
        
        for _ in 0..self.config.iterations {
            let leaf_id = tree.select_best_leaf();
            
            // Simulation/Expansion using GNN
            let sampler = NeighborSampler::new(self.db.storage(), self.db.adjacency());
            let subgraph = sampler.sample_k_hop(leaf_id, 1)?;
            let predictions = self.gnn.predict(&subgraph).await?;
            
            if !predictions.is_empty() {
                let children: Vec<Uuid> = predictions.iter().map(|p| p.node_id).collect();
                tree.expand(leaf_id, children);
                
                // Use the best prediction score as reward
                let best_reward = predictions.iter().map(|p| p.score).fold(0.0, f32::max);
                tree.backpropagate(leaf_id, best_reward);
            }
        }
        
        // Final selection: return intents for children of root
        let root = &tree.nodes[&tree.root];
        let mut intents = Vec::new();
        for &child_id in &root.children {
            let child = &tree.nodes[&child_id];
            intents.push(AdvisorIntent {
                target_node: child_id,
                confidence: child.value / child.visits.max(1) as f32,
                suggested_action: "explore".to_string(),
            });
        }
        
        Ok(intents)
    }
}

impl MctsConfig {
    fn default_config(&self) -> Self {
        Self {
            iterations: self.iterations,
            exploration_constant: self.exploration_constant,
        }
    }
}
