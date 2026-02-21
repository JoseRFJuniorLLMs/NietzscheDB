//! Dream Engine — speculative graph exploration via hyperbolic diffusion.
//!
//! A dream simulation diffuses energy from a seed node, applying noise
//! to create stochastic perturbations. Events are detected when certain
//! thresholds are crossed (energy spikes, curvature anomalies, etc.).

use uuid::Uuid;
use nietzsche_graph::GraphStorage;

use crate::error::DreamError;
use crate::model::*;
use crate::store;

/// Configuration for dream simulation.
#[derive(Debug, Clone)]
pub struct DreamConfig {
    /// Default exploration depth (hops from seed).
    pub default_depth: usize,
    /// Default noise amplitude for perturbation.
    pub default_noise: f64,
    /// Energy threshold above which an energy spike event fires.
    pub energy_spike_threshold: f64,
    /// Hausdorff change threshold for curvature anomaly.
    pub curvature_anomaly_threshold: f64,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            default_depth: 5,
            default_noise: 0.05,
            energy_spike_threshold: 0.9,
            curvature_anomaly_threshold: 0.5,
        }
    }
}

impl DreamConfig {
    pub fn from_env() -> Self {
        Self {
            default_depth: std::env::var("DREAM_DEFAULT_DEPTH")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(5),
            default_noise: std::env::var("DREAM_DEFAULT_NOISE")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.05),
            energy_spike_threshold: std::env::var("DREAM_ENERGY_SPIKE_THRESHOLD")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.9),
            curvature_anomaly_threshold: std::env::var("DREAM_CURVATURE_THRESHOLD")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.5),
        }
    }
}

/// The Dream Engine runs speculative explorations.
pub struct DreamEngine {
    pub config: DreamConfig,
}

impl DreamEngine {
    pub fn new(config: DreamConfig) -> Self {
        Self { config }
    }

    /// Initiate a dream from a seed node. Returns a pending DreamSession.
    pub fn dream_from(
        &self,
        storage: &GraphStorage,
        seed_id: Uuid,
        depth: Option<usize>,
        noise: Option<f64>,
    ) -> Result<DreamSession, DreamError> {
        let depth = depth.unwrap_or(self.config.default_depth);
        let noise = noise.unwrap_or(self.config.default_noise);

        // Verify seed exists
        let _seed_meta = storage.get_node_meta(&seed_id)?
            .ok_or_else(|| DreamError::NotFound(format!("seed node {} not found", seed_id)))?;

        let mut events = Vec::new();
        let mut dream_nodes = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![seed_id];
        visited.insert(seed_id);

        // BFS exploration with noise-perturbed energy detection
        for _hop in 0..depth {
            let mut next_frontier = Vec::new();

            for nid in &frontier {
                if let Ok(Some(meta)) = storage.get_node_meta(nid) {
                    let energy = meta.energy as f64;
                    let hausdorff = meta.hausdorff_local as f64;

                    // Detect energy spike
                    if energy + noise > self.config.energy_spike_threshold {
                        events.push(DreamEvent {
                            event_type:  DreamEventType::EnergySpike,
                            node_id:     *nid,
                            energy,
                            depth:       meta.depth as f64,
                            description: format!(
                                "energy spike: {:.4} + noise {:.4} > {:.2}",
                                energy, noise, self.config.energy_spike_threshold
                            ),
                        });
                    }

                    // Detect curvature anomaly
                    if hausdorff > self.config.curvature_anomaly_threshold {
                        events.push(DreamEvent {
                            event_type:  DreamEventType::CurvatureAnomaly,
                            node_id:     *nid,
                            energy,
                            depth:       meta.depth as f64,
                            description: format!(
                                "curvature anomaly: hausdorff_local {:.4} > {:.2}",
                                hausdorff, self.config.curvature_anomaly_threshold
                            ),
                        });
                    }

                    // Record delta (dream proposes boosted energy)
                    let new_energy = (energy + noise).min(1.0);
                    if (new_energy - energy).abs() > 1e-6 {
                        dream_nodes.push(DreamNodeDelta {
                            node_id:    *nid,
                            old_energy: energy,
                            new_energy,
                            event_type: if energy + noise > self.config.energy_spike_threshold {
                                Some(DreamEventType::EnergySpike)
                            } else {
                                None
                            },
                        });
                    }

                    // Expand to neighbours via edge index
                    if let Ok(edge_ids) = storage.edge_ids_from(nid) {
                        for eid in edge_ids {
                            if let Ok(Some(edge)) = storage.get_edge(&eid) {
                                if visited.insert(edge.to) {
                                    next_frontier.push(edge.to);
                                }
                            }
                        }
                    }
                }
            }

            frontier = next_frontier;
            if frontier.is_empty() { break; }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let session = DreamSession {
            id:          format!("dream_{}", Uuid::new_v4().as_simple()),
            seed_node:   seed_id,
            depth,
            noise,
            events,
            created_at:  now,
            status:      DreamStatus::Pending,
            dream_nodes,
        };

        // Persist the dream session
        store::put_dream(storage, &session)?;

        Ok(session)
    }

    /// Apply a pending dream — persist energy changes to the graph.
    pub fn apply_dream(
        &self,
        storage: &GraphStorage,
        dream_id: &str,
    ) -> Result<DreamSession, DreamError> {
        let mut session = store::get_dream(storage, dream_id)?
            .ok_or_else(|| DreamError::NotFound(dream_id.to_string()))?;

        if session.status != DreamStatus::Pending {
            return Err(DreamError::AlreadyApplied(dream_id.to_string()));
        }

        // Apply energy deltas
        for delta in &session.dream_nodes {
            if let Ok(Some(mut meta)) = storage.get_node_meta(&delta.node_id) {
                meta.energy = delta.new_energy as f32;
                let _ = storage.put_node_meta(&meta);
            }
        }

        session.status = DreamStatus::Applied;
        store::put_dream(storage, &session)?;
        Ok(session)
    }

    /// Reject a pending dream — mark as rejected without modifying the graph.
    pub fn reject_dream(
        &self,
        storage: &GraphStorage,
        dream_id: &str,
    ) -> Result<DreamSession, DreamError> {
        let mut session = store::get_dream(storage, dream_id)?
            .ok_or_else(|| DreamError::NotFound(dream_id.to_string()))?;

        if session.status != DreamStatus::Pending {
            return Err(DreamError::AlreadyApplied(dream_id.to_string()));
        }

        session.status = DreamStatus::Rejected;
        store::put_dream(storage, &session)?;
        Ok(session)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{GraphStorage, Node, NodeMeta, NodeType, PoincareVector};
    use std::collections::HashMap;

    fn temp_storage() -> (tempfile::TempDir, GraphStorage) {
        let dir = tempfile::tempdir().unwrap();
        let storage = GraphStorage::open(dir.path().to_str().unwrap()).unwrap();
        (dir, storage)
    }

    fn insert_node(storage: &GraphStorage, energy: f32, hausdorff: f32) -> Uuid {
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
                hausdorff_local: hausdorff,
                expires_at: None,
                metadata: HashMap::new(),
            },
            embedding: PoincareVector::origin(8),
        };
        storage.put_node(&node).unwrap();
        id
    }

    #[test]
    fn dream_from_seed() {
        let (_dir, storage) = temp_storage();
        let seed_id = insert_node(&storage, 0.85, 0.1);

        let engine = DreamEngine::new(DreamConfig {
            energy_spike_threshold: 0.9,
            default_noise: 0.1,
            ..DreamConfig::default()
        });

        let session = engine.dream_from(&storage, seed_id, Some(1), Some(0.1)).unwrap();
        assert_eq!(session.status, DreamStatus::Pending);
        assert_eq!(session.seed_node, seed_id);
        // 0.85 + 0.1 = 0.95 > 0.9 → energy spike
        assert!(session.events.iter().any(|e| e.event_type == DreamEventType::EnergySpike));
    }

    #[test]
    fn apply_and_reject_dream() {
        let (_dir, storage) = temp_storage();
        let seed_id = insert_node(&storage, 0.5, 0.1);

        let engine = DreamEngine::new(DreamConfig::default());

        let session = engine.dream_from(&storage, seed_id, Some(1), Some(0.1)).unwrap();
        let dream_id = session.id.clone();

        // Apply dream
        let applied = engine.apply_dream(&storage, &dream_id).unwrap();
        assert_eq!(applied.status, DreamStatus::Applied);

        // Cannot apply again
        assert!(engine.apply_dream(&storage, &dream_id).is_err());
    }

    #[test]
    fn dream_not_found() {
        let (_dir, storage) = temp_storage();
        let engine = DreamEngine::new(DreamConfig::default());
        assert!(engine.apply_dream(&storage, "nonexistent").is_err());
    }

    #[test]
    fn curvature_anomaly_detected() {
        let (_dir, storage) = temp_storage();
        let seed_id = insert_node(&storage, 0.5, 0.8);

        let engine = DreamEngine::new(DreamConfig {
            curvature_anomaly_threshold: 0.5,
            ..DreamConfig::default()
        });

        let session = engine.dream_from(&storage, seed_id, Some(1), None).unwrap();
        assert!(session.events.iter().any(|e| e.event_type == DreamEventType::CurvatureAnomaly));
    }
}
