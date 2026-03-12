//! Dream Engine — speculative graph exploration via hyperbolic diffusion.
//!
//! A dream simulation diffuses energy from a seed node, applying noise
//! to create stochastic perturbations. Events are detected when certain
//! thresholds are crossed (energy spikes, curvature anomalies, etc.).
//!
//! When `neural_enabled = true` (env: `AGENCY_DREAM_NEURAL_ENABLED=true`),
//! the engine also generates new DreamSnapshot nodes using the
//! `dream_generator` ONNX model: it takes a seed node's embedding,
//! concatenates a noise vector, runs inference, and projects the output
//! into the Poincaré ball via `exp_map_zero`.

use uuid::Uuid;
use nietzsche_graph::{GraphStorage, Node, NodeMeta, NodeType, PoincareVector};
use nietzsche_hyp_ops::exp_map_zero;

use crate::error::DreamError;
use crate::generator::DreamGeneratorNet;
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
    /// Whether to use the neural dream generator (ONNX model) to create
    /// DreamSnapshot nodes with generated embeddings. Default: false.
    /// Env var: `AGENCY_DREAM_NEURAL_ENABLED`
    pub neural_enabled: bool,
    /// Creativity level for neural dream generation [0.0, 1.0].
    /// 0.0 = deterministic (same as seed), 1.0 = wild exploration.
    /// Default: 0.3
    pub neural_creativity: f32,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            default_depth: 5,
            default_noise: 0.05,
            energy_spike_threshold: 0.9,
            curvature_anomaly_threshold: 0.5,
            neural_enabled: false,
            neural_creativity: 0.3,
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
            neural_enabled: std::env::var("AGENCY_DREAM_NEURAL_ENABLED")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            neural_creativity: std::env::var("AGENCY_DREAM_NEURAL_CREATIVITY")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(0.3),
        }
    }
}

/// The Dream Engine runs speculative explorations.
///
/// When `config.neural_enabled` is true and the `dream_generator` ONNX model
/// is loaded in the neural registry, `dream_from` will also generate new
/// DreamSnapshot nodes with novel embeddings.
pub struct DreamEngine {
    pub config: DreamConfig,
    /// Optional neural dream generator. Loaded lazily; `None` if model file
    /// is missing or `neural_enabled` is false.
    generator: Option<DreamGeneratorNet>,
}

impl DreamEngine {
    pub fn new(config: DreamConfig) -> Self {
        // Only attempt to load the generator if neural dreams are enabled.
        // DreamGeneratorNet::new will silently skip if the .onnx file doesn't exist.
        let generator = if config.neural_enabled {
            let models_dir = std::env::var("NIETZSCHE_MODELS_DIR")
                .unwrap_or_else(|_| "models".to_string());
            let gen = DreamGeneratorNet::new(&models_dir);
            // Verify the model is actually loaded in the registry
            if nietzsche_neural::REGISTRY.get_session("dream_generator").is_ok() {
                tracing::info!("dream engine: neural dream generator loaded (creativity={:.2})", config.neural_creativity);
                Some(gen)
            } else {
                tracing::warn!("dream engine: neural_enabled=true but dream_generator model not found, falling back to diffusion-only");
                None
            }
        } else {
            None
        };
        Self { config, generator }
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

        // ── Neural Dream Generation ────────────────────────────────────
        // When the neural generator is available, create a new DreamSnapshot
        // node with a generated embedding. The model takes the seed node's
        // 128D embedding + 64D noise and produces a novel 128D embedding
        // that is then projected into the Poincaré ball via exp_map_zero.
        let mut generated_snapshot_id: Option<Uuid> = None;
        if let Some(ref generator) = self.generator {
            match self.generate_dream_snapshot(storage, seed_id, generator) {
                Ok(snapshot_id) => {
                    events.push(DreamEvent {
                        event_type:  DreamEventType::NeuralGeneration,
                        node_id:     snapshot_id,
                        energy:      0.5,
                        depth:       0.0, // will be set from embedding norm
                        description: format!(
                            "neural dream: generated DreamSnapshot {} from seed {} (creativity={:.2})",
                            snapshot_id, seed_id, self.config.neural_creativity
                        ),
                    });
                    generated_snapshot_id = Some(snapshot_id);
                    tracing::info!(
                        seed = %seed_id,
                        snapshot = %snapshot_id,
                        creativity = self.config.neural_creativity,
                        "dream engine: neural DreamSnapshot created"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        seed = %seed_id,
                        "dream engine: neural generation failed, continuing with diffusion-only dream"
                    );
                }
            }
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
            generated_node: generated_snapshot_id,
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

    /// Generate a DreamSnapshot node using the neural dream generator.
    ///
    /// Steps:
    /// 1. Load seed node's 128D embedding from storage
    /// 2. Run dream_generator ONNX: [1, 192] (seed ++ noise) → [1, 128]
    /// 3. Project output to Poincaré ball via exp_map_zero (guarantees ‖x‖ < 1)
    /// 4. Create a DreamSnapshot node with the generated embedding
    /// 5. Create a DREAM_GENERATED edge from seed → snapshot
    fn generate_dream_snapshot(
        &self,
        storage: &GraphStorage,
        seed_id: Uuid,
        generator: &DreamGeneratorNet,
    ) -> Result<Uuid, DreamError> {
        // 1. Get seed embedding (128D)
        let seed_embedding = storage.get_embedding(&seed_id)
            .map_err(|e| DreamError::Internal(format!("failed to get seed embedding: {e}")))?
            .ok_or_else(|| DreamError::Internal(format!("seed {} has no embedding", seed_id)))?;

        let seed_coords = &seed_embedding.coords;
        if seed_coords.len() != 128 {
            return Err(DreamError::Internal(format!(
                "seed embedding must be 128D, got {}D",
                seed_coords.len()
            )));
        }

        // 2. Run neural dream generator
        let generated = generator
            .dream(seed_coords, self.config.neural_creativity)
            .map_err(|e| DreamError::Internal(format!("dream_generator inference failed: {e}")))?;

        // 3. Project to Poincaré ball via exp_map_zero
        //    The model outputs Euclidean vectors; exp_map_zero maps them into
        //    the Poincaré ball with ‖x‖ < 1.0 guaranteed (tanh normalization).
        let generated_f64: Vec<f64> = generated.iter().map(|&x| x as f64).collect();
        let poincare_coords = exp_map_zero(&generated_f64);

        // Sanity check: verify we are inside the ball
        let norm_sq: f64 = poincare_coords.iter().map(|x| x * x).sum();
        if norm_sq >= 1.0 {
            // Defensive: clamp to 0.95 norm (should never happen with exp_map_zero)
            tracing::warn!(
                norm = norm_sq.sqrt(),
                "dream neural: generated embedding outside Poincaré ball, clamping"
            );
        }

        let embedding = PoincareVector::from_f64(poincare_coords.clone());
        let depth = embedding.depth() as f32;

        // 4. Create DreamSnapshot node
        let snapshot_id = Uuid::new_v4();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let content = serde_json::json!({
            "source": "dream_generator",
            "seed_node": seed_id.to_string(),
            "creativity": self.config.neural_creativity,
            "generated_at": now,
        });

        let node = Node {
            meta: NodeMeta {
                id: snapshot_id,
                node_type: NodeType::DreamSnapshot,
                energy: 0.5,  // moderate initial energy
                depth,
                created_at: now,
                content,
                lsystem_generation: 0,
                hausdorff_local: 0.0,
                expires_at: None,
                metadata: std::collections::HashMap::new(),
                is_phantom: false,
                valence: 0.0,
                arousal: 0.0,
            },
            embedding,
        };

        storage.put_node(&node)
            .map_err(|e| DreamError::Internal(format!("failed to persist DreamSnapshot: {e}")))?;

        // 5. Create DREAM_GENERATED edge from seed → snapshot
        let mut edge = nietzsche_graph::Edge::new(
            seed_id,
            snapshot_id,
            nietzsche_graph::EdgeType::Association,
            1.0,
        );
        edge.metadata.insert(
            "relation".to_string(),
            serde_json::json!("DREAM_GENERATED"),
        );
        storage.put_edge(&edge)
            .map_err(|e| DreamError::Internal(format!("failed to persist DREAM_GENERATED edge: {e}")))?;

        Ok(snapshot_id)
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
                is_phantom: false,
                valence: 0.0,
                arousal: 0.0,
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

    #[test]
    fn neural_disabled_no_generated_node() {
        // When neural_enabled = false, generated_node should be None
        let (_dir, storage) = temp_storage();
        let seed_id = insert_node(&storage, 0.5, 0.1);

        let engine = DreamEngine::new(DreamConfig {
            neural_enabled: false,
            ..DreamConfig::default()
        });

        let session = engine.dream_from(&storage, seed_id, Some(1), None).unwrap();
        assert!(session.generated_node.is_none(), "no neural generation when disabled");
    }

    #[test]
    fn neural_enabled_but_model_missing_falls_back() {
        // When neural_enabled = true but model file doesn't exist,
        // generator should be None and dream still works via diffusion.
        let (_dir, storage) = temp_storage();
        let seed_id = insert_node(&storage, 0.5, 0.1);

        let engine = DreamEngine::new(DreamConfig {
            neural_enabled: true,
            ..DreamConfig::default()
        });

        // Generator should be None because model file doesn't exist
        assert!(engine.generator.is_none(), "generator should be None when model is missing");

        let session = engine.dream_from(&storage, seed_id, Some(1), None).unwrap();
        assert!(session.generated_node.is_none(), "no neural generation without model");
        assert_eq!(session.status, DreamStatus::Pending);
    }
}
