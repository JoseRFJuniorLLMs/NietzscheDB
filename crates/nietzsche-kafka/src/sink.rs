use std::sync::Arc;

use tracing::{debug, warn};
use uuid::Uuid;

use nietzsche_graph::{
    AdjacencyIndex, Edge, EdgeType, GraphStorage, Node, PoincareVector,
};

use crate::batch::BatchResult;
use crate::error::KafkaError;
use crate::message::GraphMutation;

// ─────────────────────────────────────────────
// SinkConfig
// ─────────────────────────────────────────────

/// Configuration for the Kafka sink processor.
#[derive(Debug, Clone)]
pub struct SinkConfig {
    /// Maximum number of messages to buffer before flushing a batch.
    pub batch_size: usize,
    /// Maximum time (ms) between flushes, even if the batch is not full.
    pub flush_interval_ms: u64,
    /// If true, messages that fail to apply are forwarded to a dead-letter topic
    /// instead of being silently dropped.
    pub dead_letter_enabled: bool,
}

impl Default for SinkConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            flush_interval_ms: 1000,
            dead_letter_enabled: false,
        }
    }
}

// ─────────────────────────────────────────────
// SinkMetrics
// ─────────────────────────────────────────────

/// Runtime metrics for observability.
#[derive(Debug, Clone, Default)]
pub struct SinkMetrics {
    /// Total number of messages processed successfully.
    pub messages_processed: u64,
    /// Total number of messages that failed to apply.
    pub messages_failed: u64,
    /// Total number of batches flushed.
    pub batches_flushed: u64,
}

// ─────────────────────────────────────────────
// KafkaSink
// ─────────────────────────────────────────────

/// The core sink processor that applies [`GraphMutation`] messages to
/// a NietzscheDB [`GraphStorage`] instance.
///
/// # Architecture
///
/// ```text
/// Kafka topic
///   |
///   v
/// [Consumer] --> raw bytes
///   |
///   v
/// [Deserialize] --> GraphMutation
///   |
///   v
/// [KafkaSink.process_message()]
///   |
///   +-- InsertNode   --> storage.put_node()
///   +-- DeleteNode   --> storage.delete_node() + adjacency.remove_node()
///   +-- InsertEdge   --> storage.put_edge()    + adjacency.add_edge()
///   +-- DeleteEdge   --> storage.delete_edge() + adjacency.remove_edge()
///   +-- SetEnergy    --> storage.put_node_meta_update_energy()
///   +-- SetContent   --> storage.put_node_meta()
/// ```
///
/// The sink keeps both the persistent [`GraphStorage`] and the in-memory
/// [`AdjacencyIndex`] in sync.  Batch processing collects per-message
/// errors so that a single bad record does not block the entire batch.
pub struct KafkaSink {
    storage: Arc<GraphStorage>,
    adjacency: Arc<AdjacencyIndex>,
    config: SinkConfig,
    metrics: SinkMetrics,
}

impl KafkaSink {
    /// Create a new sink processor.
    pub fn new(
        storage: Arc<GraphStorage>,
        adjacency: Arc<AdjacencyIndex>,
        config: SinkConfig,
    ) -> Self {
        Self {
            storage,
            adjacency,
            config,
            metrics: SinkMetrics::default(),
        }
    }

    /// Apply a single mutation to the graph.
    ///
    /// On success, increments `messages_processed`.
    /// On failure, increments `messages_failed` and returns the error.
    pub fn process_message(&mut self, msg: &GraphMutation) -> Result<(), KafkaError> {
        let result = self.apply_mutation(msg);
        match &result {
            Ok(()) => {
                self.metrics.messages_processed += 1;
                debug!(?msg, "mutation applied successfully");
            }
            Err(e) => {
                self.metrics.messages_failed += 1;
                warn!(?msg, error = %e, "mutation failed");
            }
        }
        result
    }

    /// Process a batch of mutations, collecting per-message errors.
    ///
    /// Every message is attempted regardless of earlier failures in the batch.
    /// Returns a [`BatchResult`] summarising successes and failures.
    pub fn process_batch(&mut self, msgs: &[GraphMutation]) -> BatchResult {
        let mut result = BatchResult::empty();

        for (idx, msg) in msgs.iter().enumerate() {
            match self.apply_mutation(msg) {
                Ok(()) => {
                    self.metrics.messages_processed += 1;
                    result.record_success();
                }
                Err(e) => {
                    self.metrics.messages_failed += 1;
                    result.record_failure(idx, e);
                }
            }
        }

        self.metrics.batches_flushed += 1;
        result
    }

    /// Return a snapshot of the current metrics.
    pub fn metrics(&self) -> &SinkMetrics {
        &self.metrics
    }

    /// Return a reference to the current configuration.
    pub fn config(&self) -> &SinkConfig {
        &self.config
    }

    // ── Internal dispatch ─────────────────────────────

    fn apply_mutation(&self, msg: &GraphMutation) -> Result<(), KafkaError> {
        match msg {
            GraphMutation::InsertNode {
                id,
                content,
                energy,
                embedding,
            } => self.apply_insert_node(id, content, energy, embedding),

            GraphMutation::DeleteNode { id } => self.apply_delete_node(id),

            GraphMutation::InsertEdge {
                from,
                to,
                weight,
                edge_type,
            } => self.apply_insert_edge(from, to, weight, edge_type),

            GraphMutation::DeleteEdge { id } => self.apply_delete_edge(id),

            GraphMutation::SetEnergy { id, energy } => self.apply_set_energy(id, *energy),

            GraphMutation::SetContent { id, fields } => self.apply_set_content(id, fields),
        }
    }

    fn apply_insert_node(
        &self,
        id: &Option<Uuid>,
        content: &serde_json::Value,
        energy: &Option<f32>,
        embedding: &Option<Vec<f32>>,
    ) -> Result<(), KafkaError> {
        let node_id = id.unwrap_or_else(Uuid::new_v4);

        // Build embedding: use provided coordinates or default to origin.
        let poincare = match embedding {
            Some(coords) if !coords.is_empty() => {
                let pv = PoincareVector::new(coords.clone());
                if !pv.is_valid() {
                    // Project back into the ball rather than rejecting outright.
                    pv.project_into_ball()
                } else {
                    pv
                }
            }
            _ => PoincareVector::origin(2), // default 2-d origin
        };

        let mut node = Node::new(node_id, poincare, content.clone());

        // Override energy if provided (Node::new defaults to 1.0).
        if let Some(e) = energy {
            node.meta.energy = *e;
        }

        self.storage.put_node(&node)?;
        Ok(())
    }

    fn apply_delete_node(&self, id: &Uuid) -> Result<(), KafkaError> {
        // Clean up adjacency first (removes all connected edges).
        self.adjacency.remove_node(id);

        // Delete from persistent storage.
        self.storage.delete_node(id)?;
        Ok(())
    }

    fn apply_insert_edge(
        &self,
        from: &Uuid,
        to: &Uuid,
        weight: &Option<f32>,
        edge_type: &Option<String>,
    ) -> Result<(), KafkaError> {
        // Validate that both endpoints exist.
        if !self.storage.node_exists(from)? {
            return Err(KafkaError::InvalidMessage(format!(
                "source node {from} does not exist"
            )));
        }
        if !self.storage.node_exists(to)? {
            return Err(KafkaError::InvalidMessage(format!(
                "target node {to} does not exist"
            )));
        }

        let et = match edge_type.as_deref() {
            Some("Hierarchical") => EdgeType::Hierarchical,
            Some("LSystemGenerated") => EdgeType::LSystemGenerated,
            Some("Pruned") => EdgeType::Pruned,
            _ => EdgeType::Association,
        };

        let edge = Edge::new(*from, *to, et, weight.unwrap_or(1.0));

        // Persist and update RocksDB adjacency CFs.
        self.storage.put_edge(&edge)?;

        // Keep in-memory adjacency index in sync.
        self.adjacency.add_edge(&edge);

        Ok(())
    }

    fn apply_delete_edge(&self, id: &Uuid) -> Result<(), KafkaError> {
        // Need the full Edge record to update adjacency lists.
        let edge = self
            .storage
            .get_edge(id)?
            .ok_or_else(|| KafkaError::InvalidMessage(format!("edge {id} not found")))?;

        // Remove from adjacency (in-memory).
        self.adjacency.remove_edge(&edge);

        // Remove from persistent storage.
        self.storage.delete_edge(&edge)?;

        Ok(())
    }

    fn apply_set_energy(&self, id: &Uuid, energy: f32) -> Result<(), KafkaError> {
        let mut meta = self
            .storage
            .get_node_meta(id)?
            .ok_or_else(|| KafkaError::InvalidMessage(format!("node {id} not found")))?;

        let old_energy = meta.energy;
        meta.energy = energy;

        self.storage
            .put_node_meta_update_energy(&meta, old_energy)?;

        Ok(())
    }

    fn apply_set_content(&self, id: &Uuid, fields: &serde_json::Value) -> Result<(), KafkaError> {
        let mut meta = self
            .storage
            .get_node_meta(id)?
            .ok_or_else(|| KafkaError::InvalidMessage(format!("node {id} not found")))?;

        // Merge the provided fields into the existing content.
        if let (Some(existing), Some(new_fields)) = (meta.content.as_object_mut(), fields.as_object()) {
            for (k, v) in new_fields {
                existing.insert(k.clone(), v.clone());
            }
        } else {
            // If the existing content is not an object, replace it entirely.
            meta.content = fields.clone();
        }

        self.storage.put_node_meta(&meta)?;

        Ok(())
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Open an ephemeral RocksDB in a temp directory and return the
    /// storage, adjacency index, and _dir guard (keeps the dir alive).
    fn setup() -> (Arc<GraphStorage>, Arc<AdjacencyIndex>, TempDir) {
        let dir = TempDir::new().unwrap();
        let storage =
            Arc::new(GraphStorage::open(dir.path().to_str().unwrap()).unwrap());
        let adjacency = Arc::new(AdjacencyIndex::new());
        (storage, adjacency, dir)
    }

    fn make_sink(
        storage: Arc<GraphStorage>,
        adjacency: Arc<AdjacencyIndex>,
    ) -> KafkaSink {
        KafkaSink::new(storage, adjacency, SinkConfig::default())
    }

    // ── 1. test_process_insert_node ────────────────

    #[test]
    fn test_process_insert_node() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        let node_id = Uuid::new_v4();
        let msg = GraphMutation::InsertNode {
            id: Some(node_id),
            content: serde_json::json!({"label": "hello"}),
            energy: Some(0.8),
            embedding: Some(vec![0.1, 0.2]),
        };

        sink.process_message(&msg).unwrap();

        let node = storage.get_node(&node_id).unwrap().unwrap();
        assert_eq!(node.id, node_id);
        assert!((node.energy - 0.8).abs() < 1e-6);
        assert_eq!(node.embedding.coords, vec![0.1, 0.2]);
    }

    // ── 2. test_process_delete_node ────────────────

    #[test]
    fn test_process_delete_node() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        // Insert first
        let node_id = Uuid::new_v4();
        let insert = GraphMutation::InsertNode {
            id: Some(node_id),
            content: serde_json::json!({}),
            energy: None,
            embedding: None,
        };
        sink.process_message(&insert).unwrap();
        assert!(storage.node_exists(&node_id).unwrap());

        // Delete
        let delete = GraphMutation::DeleteNode { id: node_id };
        sink.process_message(&delete).unwrap();
        assert!(!storage.node_exists(&node_id).unwrap());
    }

    // ── 3. test_process_insert_edge ────────────────

    #[test]
    fn test_process_insert_edge() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency.clone());

        let a = Uuid::new_v4();
        let b = Uuid::new_v4();

        // Insert two nodes
        for id in [a, b] {
            sink.process_message(&GraphMutation::InsertNode {
                id: Some(id),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            })
            .unwrap();
        }

        // Insert edge
        let edge_msg = GraphMutation::InsertEdge {
            from: a,
            to: b,
            weight: Some(0.75),
            edge_type: Some("Association".to_string()),
        };
        sink.process_message(&edge_msg).unwrap();

        // Verify via adjacency index
        let neighbors = adjacency.neighbors_out(&a);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], b);

        // Verify via storage
        let edge_ids = storage.edge_ids_from(&a).unwrap();
        assert_eq!(edge_ids.len(), 1);
    }

    // ── 4. test_process_set_energy ─────────────────

    #[test]
    fn test_process_set_energy() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        let node_id = Uuid::new_v4();
        sink.process_message(&GraphMutation::InsertNode {
            id: Some(node_id),
            content: serde_json::json!({}),
            energy: Some(1.0),
            embedding: None,
        })
        .unwrap();

        // Set energy to 0.42
        sink.process_message(&GraphMutation::SetEnergy {
            id: node_id,
            energy: 0.42,
        })
        .unwrap();

        let meta = storage.get_node_meta(&node_id).unwrap().unwrap();
        assert!((meta.energy - 0.42).abs() < 1e-6);
    }

    // ── 5. test_process_set_content ────────────────

    #[test]
    fn test_process_set_content() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        let node_id = Uuid::new_v4();
        sink.process_message(&GraphMutation::InsertNode {
            id: Some(node_id),
            content: serde_json::json!({"name": "Alice", "age": 30}),
            energy: None,
            embedding: None,
        })
        .unwrap();

        // Update content fields (merge)
        sink.process_message(&GraphMutation::SetContent {
            id: node_id,
            fields: serde_json::json!({"age": 31, "city": "Berlin"}),
        })
        .unwrap();

        let meta = storage.get_node_meta(&node_id).unwrap().unwrap();
        let content = meta.content.as_object().unwrap();
        assert_eq!(content.get("name").unwrap(), "Alice");
        assert_eq!(content.get("age").unwrap(), 31);
        assert_eq!(content.get("city").unwrap(), "Berlin");
    }

    // ── 6. test_process_batch ──────────────────────

    #[test]
    fn test_process_batch() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        let ids: Vec<Uuid> = (0..5).map(|_| Uuid::new_v4()).collect();
        let msgs: Vec<GraphMutation> = ids
            .iter()
            .map(|id| GraphMutation::InsertNode {
                id: Some(*id),
                content: serde_json::json!({"batch": true}),
                energy: None,
                embedding: None,
            })
            .collect();

        let result = sink.process_batch(&msgs);
        assert_eq!(result.succeeded, 5);
        assert_eq!(result.failed, 0);
        assert!(result.is_all_ok());

        // Verify all nodes exist
        for id in &ids {
            assert!(storage.node_exists(id).unwrap());
        }
    }

    // ── 7. test_batch_partial_failure ──────────────

    #[test]
    fn test_batch_partial_failure() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage.clone(), adjacency);

        let valid_id = Uuid::new_v4();
        let nonexistent_id = Uuid::new_v4();

        let msgs = vec![
            // Valid: insert a node
            GraphMutation::InsertNode {
                id: Some(valid_id),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            },
            // Invalid: delete a nonexistent edge
            GraphMutation::DeleteEdge {
                id: nonexistent_id,
            },
            // Valid: insert another node
            GraphMutation::InsertNode {
                id: Some(Uuid::new_v4()),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            },
        ];

        let result = sink.process_batch(&msgs);
        assert_eq!(result.succeeded, 2);
        assert_eq!(result.failed, 1);
        assert!(!result.is_all_ok());

        // The failure should be at index 1
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, 1);

        // The valid node should still exist
        assert!(storage.node_exists(&valid_id).unwrap());
    }

    // ── 8. test_deserialize_mutations ──────────────

    #[test]
    fn test_deserialize_mutations() {
        // InsertNode
        let json = r#"{"type":"InsertNode","content":{"x":1},"energy":0.5,"embedding":[0.1,0.2]}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::InsertNode { .. }));

        // DeleteNode
        let json = r#"{"type":"DeleteNode","id":"550e8400-e29b-41d4-a716-446655440000"}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::DeleteNode { .. }));

        // InsertEdge
        let json = r#"{"type":"InsertEdge","from":"550e8400-e29b-41d4-a716-446655440000","to":"550e8400-e29b-41d4-a716-446655440001","weight":0.9}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::InsertEdge { .. }));

        // DeleteEdge
        let json = r#"{"type":"DeleteEdge","id":"550e8400-e29b-41d4-a716-446655440000"}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::DeleteEdge { .. }));

        // SetEnergy
        let json = r#"{"type":"SetEnergy","id":"550e8400-e29b-41d4-a716-446655440000","energy":0.75}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::SetEnergy { .. }));

        // SetContent
        let json = r#"{"type":"SetContent","id":"550e8400-e29b-41d4-a716-446655440000","fields":{"k":"v"}}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, GraphMutation::SetContent { .. }));

        // InsertNode with id = null (auto-generate)
        let json = r#"{"type":"InsertNode","id":null,"content":{}}"#;
        let msg: GraphMutation = serde_json::from_str(json).unwrap();
        if let GraphMutation::InsertNode { id, .. } = msg {
            assert!(id.is_none());
        } else {
            panic!("expected InsertNode");
        }
    }

    // ── 9. test_metrics_tracking ───────────────────

    #[test]
    fn test_metrics_tracking() {
        let (storage, adjacency, _dir) = setup();
        let mut sink = make_sink(storage, adjacency);

        // Process 3 successful inserts
        for _ in 0..3 {
            sink.process_message(&GraphMutation::InsertNode {
                id: Some(Uuid::new_v4()),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            })
            .unwrap();
        }

        // Process 1 failing delete (nonexistent edge)
        let _ = sink.process_message(&GraphMutation::DeleteEdge {
            id: Uuid::new_v4(),
        });

        let m = sink.metrics();
        assert_eq!(m.messages_processed, 3);
        assert_eq!(m.messages_failed, 1);

        // Process a batch of 2
        let batch = vec![
            GraphMutation::InsertNode {
                id: Some(Uuid::new_v4()),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            },
            GraphMutation::InsertNode {
                id: Some(Uuid::new_v4()),
                content: serde_json::json!({}),
                energy: None,
                embedding: None,
            },
        ];
        sink.process_batch(&batch);

        let m = sink.metrics();
        assert_eq!(m.messages_processed, 5);
        assert_eq!(m.messages_failed, 1);
        assert_eq!(m.batches_flushed, 1);
    }
}
