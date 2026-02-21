//! Observer Identity — the agency's self-representation in the graph.
//!
//! The Observer Identity is a special meta-node in the knowledge graph
//! that represents the agency system itself. This gives the graph a
//! form of "self-awareness": a node that stores the latest health
//! snapshot, whose energy reflects overall graph vitality, and whose
//! connections to other nodes encode what regions the agency is
//! actively monitoring.
//!
//! ## Philosophy
//!
//! In Nietzsche's concept of *Selbstüberwindung* (self-overcoming),
//! awareness of one's state is the prerequisite for growth. The
//! Observer Identity is the graph's mirror — it sees itself, and by
//! seeing, it can *will* to change.
//!
//! ## Design Decisions
//!
//! - Uses `NodeType::Concept` with a metadata tag `{"nietzsche_agency": "observer"}`
//!   to avoid modifying the core `NodeType` enum (which would ripple through
//!   all match statements in 27 crates).
//! - The observer node's UUID is persisted in CF_META at key
//!   `agency:observer_id` for O(1) lookup.
//! - The node's `content` field stores the latest HealthReport as JSON.
//! - The node's `energy` reflects normalized graph health (mean of
//!   energy, coherence, and fractal indicators).
//! - The node sits at the **center** of the Poincaré ball (depth ≈ 0.01)
//!   because it's the most abstract, high-level representation.

use uuid::Uuid;

use nietzsche_graph::{GraphStorage, NodeMeta, NodeType};

use crate::error::AgencyError;
use crate::observer::HealthReport;

const OBSERVER_META_KEY: &str = "agency:observer_id";
const OBSERVER_TAG_KEY: &str = "nietzsche_agency";
const OBSERVER_TAG_VALUE: &str = "observer";

/// Manage the Observer's identity node in the graph.
pub struct ObserverIdentity;

impl ObserverIdentity {
    /// Get or create the Observer meta-node.
    ///
    /// On first call, creates a new node at the center of the Poincaré ball
    /// and persists its UUID to CF_META. On subsequent calls, returns the
    /// existing node's UUID.
    pub fn ensure_exists(
        storage: &GraphStorage,
    ) -> Result<Uuid, AgencyError> {
        // Check if we already have an observer node
        if let Some(id) = Self::get_id(storage)? {
            // Verify the node still exists
            if storage.get_node_meta(&id)?.is_some() {
                return Ok(id);
            }
            // Node was deleted — recreate
        }

        // Create the observer node
        let id = Uuid::new_v4();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(OBSERVER_TAG_KEY.to_string(), serde_json::Value::String(OBSERVER_TAG_VALUE.to_string()));

        let meta = NodeMeta {
            id,
            depth: 0.01, // Near center — most abstract
            content: serde_json::json!({
                "role": "observer",
                "description": "Agency observer meta-node — the graph's self-awareness",
            }),
            node_type: NodeType::Concept,
            energy: 1.0, // Starts at full energy
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as i64,
            expires_at: None, // Never expires
            metadata,
            is_phantom: false,
        };

        storage.put_node_meta(&meta)?;

        // Persist the ID for future lookups
        storage.put_meta(OBSERVER_META_KEY, id.as_bytes())?;

        tracing::info!(observer_id = %id, "Observer identity node created");
        Ok(id)
    }

    /// Retrieve the Observer node's UUID, if it exists.
    pub fn get_id(
        storage: &GraphStorage,
    ) -> Result<Option<Uuid>, AgencyError> {
        match storage.get_meta(OBSERVER_META_KEY)? {
            Some(bytes) if bytes.len() == 16 => {
                Ok(Some(Uuid::from_bytes(bytes.try_into().unwrap())))
            }
            _ => Ok(None),
        }
    }

    /// Update the Observer node with the latest HealthReport.
    ///
    /// This encodes the graph's current state into the observer node:
    /// - `content`: full HealthReport as JSON
    /// - `energy`: normalized health score (0.0–1.0)
    /// - `hausdorff_local`: copied from global_hausdorff
    pub fn update_from_health(
        storage: &GraphStorage,
        report: &HealthReport,
    ) -> Result<(), AgencyError> {
        let id = match Self::get_id(storage)? {
            Some(id) => id,
            None => Self::ensure_exists(storage)?,
        };

        let mut meta = match storage.get_node_meta(&id)? {
            Some(m) => m,
            None => return Self::ensure_exists(storage).map(|_| ()),
        };

        // Encode health into content
        meta.content = serde_json::to_value(report)
            .unwrap_or(serde_json::json!({"error": "serialization failed"}));

        // Compute normalized health score
        // Components: energy health (0-1), coherence (0-1), fractal health (0-1)
        let energy_score = report.mean_energy.clamp(0.0, 1.0);
        let coherence_score = report.coherence_score.clamp(0.0, 1.0) as f32;
        let fractal_score = if report.is_fractal { 1.0 } else { 0.5 };
        let gap_penalty = (report.gap_count as f32 / 40.0).min(0.5); // max 0.5 penalty

        meta.energy = ((energy_score + coherence_score + fractal_score) / 3.0 - gap_penalty)
            .clamp(0.0, 1.0);

        meta.hausdorff_local = report.global_hausdorff;

        storage.put_node_meta(&meta)?;

        tracing::debug!(
            observer_id    = %id,
            observer_energy = meta.energy,
            "Observer identity updated"
        );

        Ok(())
    }

    /// Check whether a node is the Observer meta-node.
    pub fn is_observer(meta: &NodeMeta) -> bool {
        meta.metadata
            .get(OBSERVER_TAG_KEY)
            .and_then(|v| v.as_str())
            .map(|s| s == OBSERVER_TAG_VALUE)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn create_and_retrieve_observer() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let id = ObserverIdentity::ensure_exists(&storage).unwrap();

        // Should return the same ID on second call
        let id2 = ObserverIdentity::ensure_exists(&storage).unwrap();
        assert_eq!(id, id2);

        // Should be retrievable via get_id
        let stored_id = ObserverIdentity::get_id(&storage).unwrap();
        assert_eq!(stored_id, Some(id));
    }

    #[test]
    fn update_from_health_report() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        ObserverIdentity::ensure_exists(&storage).unwrap();

        let report = HealthReport {
            tick_number: 10,
            total_nodes: 500,
            mean_energy: 0.7,
            global_hausdorff: 1.5,
            is_fractal: true,
            coherence_score: 0.8,
            gap_count: 2,
            ..HealthReport::default()
        };

        ObserverIdentity::update_from_health(&storage, &report).unwrap();

        let id = ObserverIdentity::get_id(&storage).unwrap().unwrap();
        let meta = storage.get_node_meta(&id).unwrap().unwrap();

        // Energy should reflect health
        assert!(meta.energy > 0.5, "healthy graph should give high energy, got {}", meta.energy);
        assert_eq!(meta.hausdorff_local, 1.5);
        assert!(ObserverIdentity::is_observer(&meta));
    }

    #[test]
    fn is_observer_check() {
        let mut meta = NodeMeta {
            id: Uuid::new_v4(),
            depth: 0.01,
            content: serde_json::json!({}),
            node_type: NodeType::Concept,
            energy: 1.0,
            lsystem_generation: 0,
            hausdorff_local: 1.0,
            created_at: 0,
            expires_at: None,
            metadata: std::collections::HashMap::new(),
            is_phantom: false,
        };

        // Without tag: not observer
        assert!(!ObserverIdentity::is_observer(&meta));

        // With tag: is observer
        meta.metadata.insert(OBSERVER_TAG_KEY.to_string(), serde_json::Value::String(OBSERVER_TAG_VALUE.to_string()));
        assert!(ObserverIdentity::is_observer(&meta));
    }

    #[test]
    fn low_health_gives_low_energy() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        ObserverIdentity::ensure_exists(&storage).unwrap();

        let report = HealthReport {
            tick_number: 1,
            total_nodes: 10,
            mean_energy: 0.1,     // Very low
            global_hausdorff: 0.3, // Out of fractal range
            is_fractal: false,
            coherence_score: 0.2,  // Poor coherence
            gap_count: 30,         // Many gaps
            ..HealthReport::default()
        };

        ObserverIdentity::update_from_health(&storage, &report).unwrap();

        let id = ObserverIdentity::get_id(&storage).unwrap().unwrap();
        let meta = storage.get_node_meta(&id).unwrap().unwrap();

        assert!(meta.energy < 0.3, "unhealthy graph should give low observer energy, got {}", meta.energy);
    }
}
