use std::collections::HashMap;
use uuid::Uuid;

pub struct MctsConfig {
    pub iterations: usize,
    pub exploration_constant: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            exploration_constant: 1.41,
        }
    }
}

pub struct MctsNode {
    pub id: Uuid,
    pub visits: u32,
    pub value: f32,
    pub children: Vec<Uuid>,
    pub parent: Option<Uuid>,
}

pub struct MctsTree {
    pub nodes: HashMap<Uuid, MctsNode>,
    pub root: Uuid,
    pub config: MctsConfig,
}

impl MctsTree {
    pub fn new(root_id: Uuid, config: MctsConfig) -> Self {
        let mut nodes = HashMap::new();
        nodes.insert(root_id, MctsNode {
            id: root_id,
            visits: 0,
            value: 0.0,
            children: Vec::new(),
            parent: None,
        });
        Self { nodes, root: root_id, config }
    }

    pub fn select_best_leaf(&self) -> Uuid {
        let mut current = self.root;
        while let Some(node) = self.nodes.get(&current) {
            if node.children.is_empty() {
                break;
            }
            // UCB1 Selection
            current = *node.children.iter()
                .max_by(|&a, &b| {
                    let ucb_a = self.calculate_ucb(a);
                    let ucb_b = self.calculate_ucb(b);
                    ucb_a.partial_cmp(&ucb_b).unwrap_or(std::cmp::Ordering::Equal)
                }).unwrap();
        }
        current
    }

    fn calculate_ucb(&self, node_id: &Uuid) -> f32 {
        let node = &self.nodes[node_id];
        if node.visits == 0 {
            return f32::INFINITY;
        }
        let parent_visits = self.nodes[&node.parent.unwrap()].visits as f32;
        (node.value / node.visits as f32) + self.config.exploration_constant * (parent_visits.ln() / node.visits as f32).sqrt()
    }

    pub fn expand(&mut self, parent_id: Uuid, children_ids: Vec<Uuid>) {
        // Collect existing children to avoid duplicate expansion if needed
        let mut added_children = Vec::new();
        if let Some(parent) = self.nodes.get(&parent_id) {
            for child_id in children_ids {
                if !parent.children.contains(&child_id) {
                    added_children.push(child_id);
                }
            }
        }

        // Now perform the mutations
        if !added_children.is_empty() {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.extend(added_children.iter().cloned());
            }

            for child_id in added_children {
                self.nodes.insert(child_id, MctsNode {
                    id: child_id,
                    visits: 0,
                    value: 0.0,
                    children: Vec::new(),
                    parent: Some(parent_id),
                });
            }
        }
    }

    pub fn backpropagate(&mut self, node_id: Uuid, reward: f32) {
        let mut current = Some(node_id);
        while let Some(id) = current {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.visits += 1;
                node.value += reward;
                current = node.parent;
            } else {
                break;
            }
        }
    }
}
