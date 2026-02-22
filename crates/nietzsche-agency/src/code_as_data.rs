//! **Code-as-Data** — NQL queries stored as activatable graph nodes (AGI-4).
//!
//! # Concept
//!
//! Nodes of type `Action` or `Skill` contain NQL queries in their content.
//! When the node's energy exceeds an activation threshold (via heat diffusion
//! / Will-to-Power), the stored query is extracted and can be executed,
//! creating energy-triggered chain reactions in the graph.
//!
//! This transforms NietzscheDB into a **Lisp-like interpreter in hyperbolic
//! geometry** — the database becomes Turing-complete and can "think" through
//! reactive rule chains.
//!
//! # Node content format
//!
//! ```json
//! {
//!   "action": {
//!     "nql": "MATCH (n) WHERE n.energy < 0.1 SET n.energy = 0.0",
//!     "activation_threshold": 0.8,
//!     "cooldown_ticks": 5,
//!     "max_firings": 100,
//!     "description": "Drain dying nodes"
//!   }
//! }
//! ```

use uuid::Uuid;

use nietzsche_graph::{GraphStorage, Node, NodeType};

// ─────────────────────────────────────────────
// Action Node
// ─────────────────────────────────────────────

/// An NQL query stored as a node in the graph.
#[derive(Debug, Clone)]
pub struct ActionNode {
    /// Node ID of the action node.
    pub node_id: Uuid,
    /// The NQL query string to execute when activated.
    pub nql: String,
    /// Energy threshold above which this action fires.
    pub activation_threshold: f32,
    /// Minimum ticks between firings (0 = no cooldown).
    pub cooldown_ticks: u32,
    /// Maximum total firings before the action is exhausted (0 = unlimited).
    pub max_firings: u32,
    /// Current number of times this action has fired.
    pub firings: u32,
    /// Current cooldown counter (decremented each tick).
    pub cooldown_remaining: u32,
    /// Human-readable description.
    pub description: String,
}

/// Result of scanning for activatable action nodes.
#[derive(Debug, Clone, Default)]
pub struct ActionScanReport {
    /// Total action/skill nodes found in the graph.
    pub total_action_nodes: usize,
    /// Action nodes whose energy exceeds their activation threshold.
    pub activated: Vec<ActionNode>,
    /// Action nodes on cooldown.
    pub on_cooldown: usize,
    /// Action nodes that are exhausted (max_firings reached).
    pub exhausted: usize,
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Extract an `ActionNode` from a graph node, if it's an action/skill node.
pub fn parse_action_node(node: &Node) -> Option<ActionNode> {
    // Check node type
    if node.meta.node_type != NodeType::Concept {
        return None;
    }

    // Check for action content
    let action = node.meta.content.get("action")?;
    let nql = action.get("nql")?.as_str()?.to_string();

    let activation_threshold = action.get("activation_threshold")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.8) as f32;
    let cooldown_ticks = action.get("cooldown_ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let max_firings = action.get("max_firings")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let firings = action.get("firings")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cooldown_remaining = action.get("cooldown_remaining")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let description = action.get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Some(ActionNode {
        node_id: node.id,
        nql,
        activation_threshold,
        cooldown_ticks,
        max_firings,
        firings,
        cooldown_remaining,
        description,
    })
}

/// Scan the graph for action nodes and return those that are ready to fire.
pub fn scan_activatable_actions(
    storage: &GraphStorage,
) -> Result<ActionScanReport, String> {
    let mut report = ActionScanReport::default();

    let nodes = storage.scan_nodes().map_err(|e| e.to_string())?;

    for node in &nodes {
        if node.meta.is_phantom || node.meta.energy <= 0.0 {
            continue;
        }

        if let Some(action) = parse_action_node(node) {
            report.total_action_nodes += 1;

            // Check exhaustion
            if action.max_firings > 0 && action.firings >= action.max_firings {
                report.exhausted += 1;
                continue;
            }

            // Check cooldown
            if action.cooldown_remaining > 0 {
                report.on_cooldown += 1;
                continue;
            }

            // Check activation threshold
            if node.meta.energy >= action.activation_threshold {
                report.activated.push(action);
            }
        }
    }

    Ok(report)
}

/// Record a firing event on an action node (increment firings, set cooldown).
///
/// Returns the updated JSON to write back to the node's content.
pub fn record_firing(
    storage: &GraphStorage,
    node_id: Uuid,
) -> Result<(), String> {
    let mut node = storage.get_node(&node_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("action node {} not found", node_id))?;

    if let Some(action) = node.meta.content.get("action").cloned() {
        let mut action = action;
        let firings = action.get("firings")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cooldown_ticks = action.get("cooldown_ticks")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        action.as_object_mut().map(|obj| {
            obj.insert("firings".into(), serde_json::json!(firings + 1));
            obj.insert("cooldown_remaining".into(), serde_json::json!(cooldown_ticks));
        });

        if let Some(obj) = node.meta.content.as_object_mut() {
            obj.insert("action".into(), action);
        }

        storage.put_node(&node).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Tick all action nodes' cooldowns (decrement cooldown_remaining by 1).
pub fn tick_cooldowns(
    storage: &GraphStorage,
) -> Result<usize, String> {
    let nodes = storage.scan_nodes().map_err(|e| e.to_string())?;
    let mut updated = 0;

    for mut node in nodes {
        if node.meta.is_phantom { continue; }
        if let Some(action) = node.meta.content.get("action").cloned() {
            let cooldown = action.get("cooldown_remaining")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);

            if cooldown > 0 {
                let mut action = action;
                action.as_object_mut().map(|obj| {
                    obj.insert("cooldown_remaining".into(), serde_json::json!(cooldown - 1));
                });
                if let Some(obj) = node.meta.content.as_object_mut() {
                    obj.insert("action".into(), action);
                }
                storage.put_node(&node).map_err(|e| e.to_string())?;
                updated += 1;
            }
        }
    }

    Ok(updated)
}

/// Create an Action node with the given NQL query.
pub fn create_action_node(
    nql: &str,
    description: &str,
    activation_threshold: f32,
    cooldown_ticks: u32,
    max_firings: u32,
    embedding: nietzsche_graph::PoincareVector,
) -> Node {
    let content = serde_json::json!({
        "action": {
            "nql": nql,
            "activation_threshold": activation_threshold,
            "cooldown_ticks": cooldown_ticks,
            "max_firings": max_firings,
            "firings": 0,
            "cooldown_remaining": 0,
            "description": description,
        }
    });

    let mut node = Node::new(Uuid::new_v4(), embedding, content);
    node.meta.node_type = NodeType::Concept;
    node
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::PoincareVector;
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_and_parse_action_node() {
        let node = create_action_node(
            "MATCH (n) WHERE n.energy < 0.1 SET n.energy = 0.0",
            "drain dying nodes",
            0.8,
            5,
            100,
            PoincareVector::new(vec![0.1, 0.0]),
        );

        let action = parse_action_node(&node).unwrap();
        assert_eq!(action.nql, "MATCH (n) WHERE n.energy < 0.1 SET n.energy = 0.0");
        assert_eq!(action.activation_threshold, 0.8);
        assert_eq!(action.cooldown_ticks, 5);
        assert_eq!(action.max_firings, 100);
        assert_eq!(action.firings, 0);
        assert_eq!(action.description, "drain dying nodes");
    }

    #[test]
    fn scan_finds_activated_actions() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Action node at high energy (above threshold)
        let mut node = create_action_node(
            "MATCH (n) SET n.energy = n.energy + 0.1",
            "boost",
            0.7,
            0,
            0,
            PoincareVector::new(vec![0.1, 0.0]),
        );
        node.meta.energy = 0.9; // > 0.7 threshold
        storage.put_node(&node).unwrap();

        // Action node at low energy (below threshold)
        let mut low = create_action_node(
            "MATCH (n) SET n.energy = 0",
            "wipe",
            0.8,
            0,
            0,
            PoincareVector::new(vec![0.2, 0.0]),
        );
        low.meta.energy = 0.3; // < 0.8 threshold
        storage.put_node(&low).unwrap();

        let report = scan_activatable_actions(&storage).unwrap();
        assert_eq!(report.total_action_nodes, 2);
        assert_eq!(report.activated.len(), 1);
        assert_eq!(report.activated[0].nql, "MATCH (n) SET n.energy = n.energy + 0.1");
    }

    #[test]
    fn firing_increments_counter_and_sets_cooldown() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let node = create_action_node(
            "MATCH (n) SET n.energy = 1.0",
            "test",
            0.5,
            3,
            10,
            PoincareVector::new(vec![0.1, 0.0]),
        );
        let id = node.id;
        storage.put_node(&node).unwrap();

        record_firing(&storage, id).unwrap();

        let updated = storage.get_node(&id).unwrap().unwrap();
        let action = parse_action_node(&updated).unwrap();
        assert_eq!(action.firings, 1);
        assert_eq!(action.cooldown_remaining, 3);
    }

    #[test]
    fn cooldown_tick_decrements() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let node = create_action_node(
            "MATCH (n) SET n.energy = 1.0",
            "test",
            0.5,
            5,
            0,
            PoincareVector::new(vec![0.1, 0.0]),
        );
        let id = node.id;
        storage.put_node(&node).unwrap();

        // Fire it
        record_firing(&storage, id).unwrap();
        let a = parse_action_node(&storage.get_node(&id).unwrap().unwrap()).unwrap();
        assert_eq!(a.cooldown_remaining, 5);

        // Tick cooldown
        tick_cooldowns(&storage).unwrap();
        let a = parse_action_node(&storage.get_node(&id).unwrap().unwrap()).unwrap();
        assert_eq!(a.cooldown_remaining, 4);

        // Tick 4 more times
        for _ in 0..4 {
            tick_cooldowns(&storage).unwrap();
        }
        let a = parse_action_node(&storage.get_node(&id).unwrap().unwrap()).unwrap();
        assert_eq!(a.cooldown_remaining, 0);
    }

    #[test]
    fn exhausted_action_not_activated() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let mut node = create_action_node(
            "MATCH (n) SET n.energy = 1.0",
            "test",
            0.5,
            0,
            2, // max 2 firings
            PoincareVector::new(vec![0.1, 0.0]),
        );
        node.meta.energy = 0.9;
        let id = node.id;
        storage.put_node(&node).unwrap();

        // Fire twice
        record_firing(&storage, id).unwrap();
        record_firing(&storage, id).unwrap();

        let report = scan_activatable_actions(&storage).unwrap();
        assert_eq!(report.exhausted, 1);
        assert!(report.activated.is_empty(), "exhausted action should not activate");
    }

    #[test]
    fn non_action_node_ignored() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Regular semantic node (no action content)
        let node = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![0.1, 0.0]),
            serde_json::json!({"title": "hello"}),
        );
        storage.put_node(&node).unwrap();

        let report = scan_activatable_actions(&storage).unwrap();
        assert_eq!(report.total_action_nodes, 0);
    }
}
