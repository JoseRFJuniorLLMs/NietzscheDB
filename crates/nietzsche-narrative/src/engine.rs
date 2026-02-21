//! Narrative Engine — compiles graph state into human-readable stories.
//!
//! Analyzes node energy distributions, edge patterns, and temporal evolution
//! to detect narrative arcs: emergence, conflict, resolution, decay.

use std::collections::HashMap;

use uuid::Uuid;
use nietzsche_graph::GraphStorage;

use crate::error::NarrativeError;
use crate::model::*;

/// Configuration for narrative generation.
#[derive(Debug, Clone)]
pub struct NarrativeConfig {
    /// Default time window in hours.
    pub default_window_hours: u64,
    /// Energy threshold for "elite" node detection.
    pub elite_threshold: f64,
    /// Energy threshold for "decayed" node detection.
    pub decay_threshold: f64,
    /// Minimum betweenness for bridge node detection.
    pub bridge_threshold: f64,
}

impl Default for NarrativeConfig {
    fn default() -> Self {
        Self {
            default_window_hours: 24,
            elite_threshold: 0.85,
            decay_threshold: 0.1,
            bridge_threshold: 0.5,
        }
    }
}

/// The Narrative Engine.
pub struct NarrativeEngine {
    pub config: NarrativeConfig,
}

impl NarrativeEngine {
    pub fn new(config: NarrativeConfig) -> Self {
        Self { config }
    }

    /// Generate a narrative from the current graph state.
    pub fn narrate(
        &self,
        storage: &GraphStorage,
        collection_name: &str,
        window_hours: Option<u64>,
    ) -> Result<NarrativeReport, NarrativeError> {
        let window = window_hours.unwrap_or(self.config.default_window_hours);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let window_start = now - (window as i64 * 3600);

        // Collect all node metadata
        let nodes: Vec<_> = storage.iter_nodes_meta()
            .filter_map(|r| r.ok())
            .filter(|n| n.created_at >= window_start || window == 0)
            .collect();

        if nodes.is_empty() {
            // Return empty narrative
            return Ok(NarrativeReport {
                collection: collection_name.to_string(),
                window_hours: window,
                generated_at: now,
                total_nodes: 0,
                total_edges: 0,
                events: vec![],
                statistics: NarrativeStats {
                    mean_energy: 0.0,
                    max_energy: 0.0,
                    min_energy: 0.0,
                    mean_depth: 0.0,
                    node_type_counts: vec![],
                },
                summary: "No nodes found in the specified time window.".to_string(),
            });
        }

        let total_edges = storage.edge_count().unwrap_or(0) as usize;

        // Compute statistics
        let energies: Vec<f64> = nodes.iter().map(|n| n.energy as f64).collect();
        let depths: Vec<f64> = nodes.iter().map(|n| n.depth as f64).collect();

        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let max_energy = energies.iter().cloned().fold(f64::MIN, f64::max);
        let min_energy = energies.iter().cloned().fold(f64::MAX, f64::min);
        let mean_depth = depths.iter().sum::<f64>() / depths.len() as f64;

        // Count node types
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for n in &nodes {
            *type_counts.entry(format!("{:?}", n.node_type)).or_default() += 1;
        }
        let node_type_counts: Vec<(String, usize)> = type_counts.into_iter().collect();

        // Detect events
        let mut events = Vec::new();

        // Elite emergence
        let elite_nodes: Vec<Uuid> = nodes.iter()
            .filter(|n| n.energy as f64 > self.config.elite_threshold)
            .map(|n| n.id)
            .collect();

        if !elite_nodes.is_empty() {
            events.push(NarrativeEvent {
                event_type: NarrativeEventType::EliteEmergence,
                description: format!(
                    "{} elite node(s) with energy > {:.2} emerged",
                    elite_nodes.len(), self.config.elite_threshold
                ),
                node_ids: elite_nodes.clone(),
                significance: max_energy,
            });
        }

        // Decay detection
        let decayed_nodes: Vec<Uuid> = nodes.iter()
            .filter(|n| (n.energy as f64) < self.config.decay_threshold)
            .map(|n| n.id)
            .collect();

        if !decayed_nodes.is_empty() {
            events.push(NarrativeEvent {
                event_type: NarrativeEventType::Decay,
                description: format!(
                    "{} node(s) with energy < {:.2} are decaying",
                    decayed_nodes.len(), self.config.decay_threshold
                ),
                node_ids: decayed_nodes,
                significance: 1.0 - min_energy,
            });
        }

        // Sort events by significance (descending)
        events.sort_by(|a, b| b.significance.partial_cmp(&a.significance).unwrap_or(std::cmp::Ordering::Equal));

        // Generate summary
        let summary = generate_summary(
            collection_name, &nodes, &events, mean_energy, max_energy, total_edges,
        );

        let statistics = NarrativeStats {
            mean_energy,
            max_energy,
            min_energy,
            mean_depth,
            node_type_counts,
        };

        Ok(NarrativeReport {
            collection: collection_name.to_string(),
            window_hours: window,
            generated_at: now,
            total_nodes: nodes.len(),
            total_edges,
            events,
            statistics,
            summary,
        })
    }
}

fn generate_summary(
    collection: &str,
    nodes: &[nietzsche_graph::NodeMeta],
    events: &[NarrativeEvent],
    mean_energy: f64,
    max_energy: f64,
    total_edges: usize,
) -> String {
    let mut parts = Vec::new();

    parts.push(format!(
        "Collection '{}' contains {} nodes and {} edges.",
        collection, nodes.len(), total_edges,
    ));

    parts.push(format!(
        "Energy landscape: mean={:.3}, peak={:.3}.",
        mean_energy, max_energy,
    ));

    if events.is_empty() {
        parts.push("The graph is in a quiet state with no significant events.".to_string());
    } else {
        for event in events.iter().take(3) {
            parts.push(event.description.clone());
        }
    }

    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{GraphStorage, Node, NodeMeta, NodeType, PoincareVector};

    fn temp_storage() -> (tempfile::TempDir, GraphStorage) {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (dir, storage)
    }

    fn insert_node(storage: &GraphStorage, energy: f32) -> Uuid {
        let id = Uuid::new_v4();
        let node = Node {
            meta: NodeMeta {
                id,
                node_type: NodeType::Semantic,
                energy,
                depth: 0.5,
                created_at: 1000,
                content: serde_json::json!({}),
                lsystem_generation: 0,
                hausdorff_local: 1.0,
                expires_at: None,
                metadata: HashMap::new(),
                is_phantom: false,
            },
            embedding: PoincareVector::origin(8),
        };
        storage.put_node(&node).unwrap();
        id
    }

    #[test]
    fn narrate_empty_graph() {
        let (_dir, storage) = temp_storage();
        let engine = NarrativeEngine::new(NarrativeConfig::default());
        let report = engine.narrate(&storage, "test", Some(0)).unwrap();
        assert!(report.summary.contains("No nodes found") || report.total_nodes == 0);
    }

    #[test]
    fn narrate_detects_elite() {
        let (_dir, storage) = temp_storage();
        insert_node(&storage, 0.95);
        insert_node(&storage, 0.5);

        let engine = NarrativeEngine::new(NarrativeConfig::default());
        let report = engine.narrate(&storage, "test", Some(0)).unwrap();
        assert!(report.events.iter().any(|e| e.event_type == NarrativeEventType::EliteEmergence));
        assert_eq!(report.total_nodes, 2);
    }

    #[test]
    fn narrate_detects_decay() {
        let (_dir, storage) = temp_storage();
        insert_node(&storage, 0.05);
        insert_node(&storage, 0.03);

        let engine = NarrativeEngine::new(NarrativeConfig::default());
        let report = engine.narrate(&storage, "test", Some(0)).unwrap();
        assert!(report.events.iter().any(|e| e.event_type == NarrativeEventType::Decay));
    }

    #[test]
    fn narrate_json_serializable() {
        let (_dir, storage) = temp_storage();
        insert_node(&storage, 0.7);

        let engine = NarrativeEngine::new(NarrativeConfig::default());
        let report = engine.narrate(&storage, "test", Some(0)).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        assert!(!json.is_empty());
    }
}
