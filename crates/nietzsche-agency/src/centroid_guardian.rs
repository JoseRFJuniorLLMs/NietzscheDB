//! # CentroidGuardian — The Geometric Self of the EVA
//!
//! Maintains the "civilizational centroid" $C_t$ — the weighted Fréchet mean
//! of all active node embeddings in the Poincaré ball. This centroid defines
//! the *identity* of the knowledge graph at any given moment.
//!
//! ## Topological Dependency
//!
//! ```text
//! CentroidGuardian → MaturityEvaluator → AxiomRegistry
//!       C_t                C_i(C_t)          Sagração
//! ```
//!
//! ## Mathematical Foundation
//!
//! ### Fréchet Mean (via tangent space at origin)
//! ```text
//! C_new = gyromidpoint_weighted({x_i}, {w_i})
//!       = exp₀(∑ w_i · log₀(x_i) / ∑ w_i)
//! ```
//!
//! ### Temporal Damping (Hysteresis of Identity)
//! ```text
//! C_t = exp₀(α · log₀(C_new) + (1−α) · log₀(C_prev))
//! ```
//! With α ≈ 0.05, the centroid moves slowly toward the new mean — preventing
//! cognitive shock from a single batch of outlier insertions.
//!
//! ### Civilizational Drift Veto
//! ```text
//! if d_𝔻(C_t, C_{t-1}) > ε  →  frozen = true
//! ```
//! When frozen, `MaturityEvaluator` rejects all promotions until
//! `nietzsche-sleep` reconsolidates and calls `unfreeze()`.
//!
//! ## Persistence
//!
//! The centroid is persisted to CF_META at key `agency:centroid` as
//! bincode-encoded `Vec<f64>`. On restart, the guardian resumes from
//! the last known position instead of recomputing from scratch.
//!
//! ## Scalability
//!
//! | Step | Cost |
//! |------|------|
//! | log_map per node | O(D) |
//! | tangent sum | O(N·D) |
//! | exp_map | O(D) |
//! | damping | O(D) |
//! | distance check | O(D) |
//!
//! For N=100k, D=3072: ~300M FLOPs per tick — well within budget.

use nietzsche_graph::GraphStorage;
use nietzsche_hyp_ops::{
    exp_map_zero, gyromidpoint_weighted, log_map_zero, poincare_distance, project_to_ball,
    error::HypError,
};
use uuid::Uuid;

use crate::config::AgencyConfig;
use crate::error::AgencyError;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

// ─────────────────────────────────────────────
// CF_META persistence keys
// ─────────────────────────────────────────────

const CENTROID_META_KEY: &str = "agency:centroid";
const CENTROID_EPOCH_KEY: &str = "agency:centroid_epoch";

// ─────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────

/// Errors produced by the CentroidGuardian.
#[derive(Debug, thiserror::Error)]
pub enum GuardianError {
    #[error("hyperbolic geometry error: {0}")]
    Hyp(#[from] HypError),
    #[error("empty node set: cannot compute centroid")]
    EmptyNodes,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

// ─────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────

/// Result of a centroid update cycle.
#[derive(Debug, Clone)]
pub struct CentroidUpdate {
    /// The new centroid position C_t.
    pub centroid: Vec<f64>,
    /// Poincaré distance C_t moved from the previous centroid.
    pub drift: f64,
    /// Whether the drift veto was triggered (drift > ε).
    pub froze: bool,
    /// Number of nodes used to compute this update.
    pub node_count: usize,
    /// Monotonically increasing epoch counter.
    pub epoch: u64,
}

// ─────────────────────────────────────────────
// CentroidGuardian
// ─────────────────────────────────────────────

/// Manages the civilizational centroid $C_t$ of the NietzscheDB knowledge graph.
///
/// # Thread Safety
///
/// `CentroidGuardian` is **not** `Send + Sync`. Wrap in `Arc<Mutex<>>` for
/// multi-threaded access (same pattern as `AgencyEngine`).
///
/// # Lifecycle
///
/// 1. At startup: `CentroidGuardian::from_storage(storage, dim)` — restores or starts at origin.
/// 2. Each agency tick: `guardian.tick(storage, bus, config)` — full update cycle.
/// 3. During `MaturityEvaluator`: call `guardian.is_frozen()` to gate promotions.
/// 4. After `nietzsche-sleep`: call `guardian.unfreeze()`.
/// 5. Centrality queries: `guardian.centrality(point)`.
pub struct CentroidGuardian {
    /// Current centroid C_t (Poincaré ball coords).
    current: Vec<f64>,
    /// Previous centroid for drift detection.
    previous: Option<Vec<f64>>,
    /// Frozen flag: true when drift > epsilon.
    frozen: bool,
    /// Monotonically increasing epoch counter.
    epoch: u64,
}

impl CentroidGuardian {
    /// Create a new CentroidGuardian with centroid at the origin.
    pub fn new(dim: usize) -> Self {
        Self {
            current: vec![0.0; dim],
            previous: None,
            frozen: false,
            epoch: 0,
        }
    }

    /// Create from storage, restoring persisted centroid + epoch if available.
    pub fn from_storage(storage: &GraphStorage, dim: usize) -> Self {
        let (centroid, epoch) = Self::load_persisted(storage);
        match centroid {
            Some(c) if c.len() == dim => {
                tracing::info!(epoch, dim, "CentroidGuardian: restored from CF_META");
                Self {
                    previous: Some(c.clone()),
                    current: c,
                    frozen: false,
                    epoch,
                }
            }
            _ => {
                tracing::info!(dim, "CentroidGuardian: starting at origin");
                Self::new(dim)
            }
        }
    }

    // ─────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────

    /// Current centroid C_t as a Poincaré ball point.
    pub fn current_centroid(&self) -> &[f64] {
        &self.current
    }

    /// Whether the civilizational drift veto is active.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Dimensionality of managed embeddings.
    pub fn dim(&self) -> usize {
        self.current.len()
    }

    /// How many update epochs have completed.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Release the drift veto after a successful `nietzsche-sleep` cycle.
    pub fn unfreeze(&mut self) {
        self.frozen = false;
        tracing::info!(epoch = self.epoch, "CentroidGuardian: drift veto released");
    }

    /// Hyperbolic centrality of a point relative to the centroid.
    ///
    /// ```text
    /// C_i = 1 / (1 + d_H(x_i, C_t))
    /// ```
    ///
    /// Returns 1.0 at the centroid, decays toward 0 at the boundary.
    pub fn centrality(&self, point: &[f64]) -> f64 {
        let d = poincare_distance(point, &self.current);
        1.0 / (1.0 + d)
    }

    // ─────────────────────────────────────────
    // Full tick (integrated with storage + bus)
    // ─────────────────────────────────────────

    /// Run one centroid update cycle, reading embeddings from storage.
    ///
    /// This is the primary entry point called by `AgencyEngine::tick()`.
    ///
    /// 1. Scans active, non-phantom nodes via `iter_nodes_meta`
    /// 2. Loads embeddings via `get_embedding` (f32 → f64 promotion)
    /// 3. Computes weighted Fréchet mean (weight = energy)
    /// 4. Applies temporal damping
    /// 5. Checks drift veto
    /// 6. Persists to CF_META
    /// 7. Emits `CentroidDrift` event if threshold exceeded
    pub fn tick(
        &mut self,
        storage: &GraphStorage,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Result<Option<CentroidUpdate>, AgencyError> {
        // 1. Collect active embeddings with energy weights
        let (embeddings, weights) = self.collect_active_embeddings(storage, config)?;

        if embeddings.is_empty() {
            tracing::debug!("CentroidGuardian: no active embeddings, skipping tick");
            return Ok(None);
        }

        // 2. Compute via core update logic
        let report = self.update_core(&embeddings, Some(&weights), config)?;

        // 3. Persist to CF_META
        self.persist(storage)?;

        // 4. Emit event if drift detected
        if report.froze {
            bus.publish(AgencyEvent::CentroidDrift {
                drift: report.drift,
                threshold: config.centroid_drift_threshold,
                epoch: report.epoch,
            });
        }

        Ok(Some(report))
    }

    /// Update the centroid from pre-collected embeddings (no storage access).
    ///
    /// Useful for external callers that already have embeddings in memory.
    pub fn update(
        &mut self,
        embeddings: &[(Uuid, Vec<f64>)],
        weights: Option<&[f64]>,
        config: &AgencyConfig,
    ) -> Result<CentroidUpdate, GuardianError> {
        if embeddings.is_empty() {
            return Err(GuardianError::EmptyNodes);
        }
        self.update_core(embeddings, weights, config)
    }

    /// Force-set the centroid (e.g. after loading persisted state).
    pub fn set_centroid(&mut self, centroid: Vec<f64>) -> Result<(), GuardianError> {
        if centroid.len() != self.current.len() {
            return Err(GuardianError::DimensionMismatch {
                expected: self.current.len(),
                got: centroid.len(),
            });
        }
        self.current = centroid;
        Ok(())
    }

    // ─────────────────────────────────────────
    // Core math (no I/O)
    // ─────────────────────────────────────────

    /// Core update: Fréchet mean → damping → drift check.
    fn update_core(
        &mut self,
        embeddings: &[(Uuid, Vec<f64>)],
        weights: Option<&[f64]>,
        config: &AgencyConfig,
    ) -> Result<CentroidUpdate, GuardianError> {
        let dim = self.current.len();

        // Validate dimensions
        if let Some((_, first)) = embeddings.first() {
            if first.len() != dim {
                return Err(GuardianError::DimensionMismatch {
                    expected: dim,
                    got: first.len(),
                });
            }
        }

        // Build slice refs for gyromidpoint_weighted
        let pts: Vec<&[f64]> = embeddings.iter().map(|(_, e)| e.as_slice()).collect();

        // Compute C_new via weighted Fréchet mean
        let c_new = match weights {
            Some(w) => gyromidpoint_weighted(&pts, w)?,
            None => {
                let uniform = vec![1.0; pts.len()];
                gyromidpoint_weighted(&pts, &uniform)?
            }
        };

        // Temporal damping: interpolate in tangent space at origin
        let c_damped = self.temporal_damping(&c_new, config.centroid_alpha);

        // Compute drift
        let drift = poincare_distance(&c_damped, &self.current);

        // Check veto
        let froze = drift > config.centroid_drift_threshold;
        if froze && !self.frozen {
            self.frozen = true;
            tracing::warn!(
                drift,
                epsilon = config.centroid_drift_threshold,
                "CentroidGuardian: civilizational drift exceeded ε — freezing maturity promotions"
            );
        }

        let node_count = embeddings.len();

        // Commit new centroid
        self.previous = Some(self.current.clone());
        self.current = c_damped;
        self.epoch += 1;

        Ok(CentroidUpdate {
            centroid: self.current.clone(),
            drift,
            froze,
            node_count,
            epoch: self.epoch,
        })
    }

    /// Temporal damping via tangent-space interpolation.
    ///
    /// ```text
    /// T_prev = log₀(C_prev)
    /// T_new  = log₀(C_new)
    /// T_damped = α · T_new + (1−α) · T_prev
    /// C_damped = exp₀(T_damped)
    /// ```
    fn temporal_damping(&self, new_centroid: &[f64], alpha: f64) -> Vec<f64> {
        let t_new = match log_map_zero(new_centroid) {
            Ok(v) => v,
            Err(_) => return new_centroid.to_vec(),
        };
        let t_prev = match log_map_zero(&self.current) {
            Ok(v) => v,
            Err(_) => return new_centroid.to_vec(),
        };

        let t_damped: Vec<f64> = t_new.iter()
            .zip(t_prev.iter())
            .map(|(&tn, &tp)| alpha * tn + (1.0 - alpha) * tp)
            .collect();

        project_to_ball(&exp_map_zero(&t_damped), 0.999)
    }

    // ─────────────────────────────────────────
    // Storage integration
    // ─────────────────────────────────────────

    /// Collect embeddings from active, non-phantom nodes.
    /// Returns (embeddings_f64, energy_weights).
    fn collect_active_embeddings(
        &self,
        storage: &GraphStorage,
        config: &AgencyConfig,
    ) -> Result<(Vec<(Uuid, Vec<f64>)>, Vec<f64>), AgencyError> {
        let dim = self.current.len();
        let mut embeddings = Vec::new();
        let mut weights = Vec::new();
        let mut scanned = 0usize;

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };

            // Skip phantom, expired, and dead nodes
            if meta.is_phantom || meta.energy <= 0.0 {
                continue;
            }

            // Respect scan limit to cap tick cost
            scanned += 1;
            if scanned > config.centroid_max_scan {
                break;
            }

            // Load embedding
            let emb = match storage.get_embedding(&meta.id) {
                Ok(Some(e)) if e.dim == dim => e,
                _ => continue,
            };

            // Promote f32 → f64 for geometric precision (ITEM C)
            let coords_f64: Vec<f64> = emb.coords.iter().map(|&x| x as f64).collect();

            // Validate: must be strictly inside the ball and non-zero
            let norm_sq: f64 = coords_f64.iter().map(|x| x * x).sum();
            if norm_sq >= 1.0 || norm_sq < 1e-30 {
                continue;
            }

            embeddings.push((meta.id, coords_f64));
            weights.push(meta.energy.max(0.001) as f64); // floor to avoid zero-weight
        }

        Ok((embeddings, weights))
    }

    /// Load centroid + epoch from CF_META.
    fn load_persisted(storage: &GraphStorage) -> (Option<Vec<f64>>, u64) {
        let centroid = storage.get_meta(CENTROID_META_KEY)
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize::<Vec<f64>>(&bytes).ok());

        let epoch = storage.get_meta(CENTROID_EPOCH_KEY)
            .ok()
            .flatten()
            .and_then(|bytes| {
                if bytes.len() == 8 {
                    Some(u64::from_le_bytes(bytes.try_into().unwrap()))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        (centroid, epoch)
    }

    /// Persist centroid + epoch to CF_META.
    fn persist(&self, storage: &GraphStorage) -> Result<(), AgencyError> {
        let bytes = bincode::serialize(&self.current)
            .map_err(|e| AgencyError::Internal(format!("centroid serialize: {e}")))?;
        storage.put_meta(CENTROID_META_KEY, &bytes)
            .map_err(|e| AgencyError::Internal(format!("centroid persist: {e}")))?;

        storage.put_meta(CENTROID_EPOCH_KEY, &self.epoch.to_le_bytes())
            .map_err(|e| AgencyError::Internal(format!("epoch persist: {e}")))?;

        Ok(())
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id() -> Uuid { Uuid::new_v4() }

    fn test_config() -> AgencyConfig {
        let mut c = AgencyConfig::default();
        c.centroid_alpha = 0.05;
        c.centroid_drift_threshold = 0.15;
        c.centroid_max_scan = 10_000;
        c
    }

    // ── basic update ──────────────────────────

    #[test]
    fn centroid_starts_at_origin() {
        let g = CentroidGuardian::new(2);
        assert!(g.current_centroid().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn single_node_update_moves_centroid() {
        let mut g = CentroidGuardian::new(2);
        let config = test_config();
        let embeddings = vec![(make_id(), vec![0.4, 0.0])];
        let report = g.update(&embeddings, None, &config).unwrap();
        let cx = g.current_centroid()[0];
        assert!(cx > 0.0 && cx < 0.4, "centroid x should be in (0, 0.4), got {cx}");
        assert_eq!(report.node_count, 1);
    }

    #[test]
    fn centroid_stays_inside_ball() {
        let mut g = CentroidGuardian::new(2);
        let config = test_config();
        let embeddings: Vec<(Uuid, Vec<f64>)> = (0..20)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::PI / 10.0;
                (make_id(), vec![angle.cos() * 0.6, angle.sin() * 0.6])
            })
            .collect();

        for _ in 0..50 {
            g.update(&embeddings, None, &config).unwrap();
        }

        let norm: f64 = g.current_centroid().iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "centroid escaped Poincaré ball: ‖C_t‖ = {norm}");
    }

    // ── temporal damping ──────────────────────

    #[test]
    fn damping_prevents_abrupt_jump() {
        let mut g = CentroidGuardian::new(2);
        let config = test_config();
        let embeddings = vec![(make_id(), vec![0.8, 0.0])];
        let _report = g.update(&embeddings, None, &config).unwrap();
        let cx = g.current_centroid()[0];
        assert!(cx < 0.2, "damping should keep C_t far from target on first step, got {cx}");
    }

    #[test]
    fn centroid_converges_to_target_over_iterations() {
        let mut g = CentroidGuardian::new(2);
        let mut config = test_config();
        config.centroid_alpha = 0.3;
        config.centroid_drift_threshold = 10.0;
        let target = vec![0.4, 0.0];
        let embeddings = vec![(make_id(), target.clone())];

        for _ in 0..100 {
            g.update(&embeddings, None, &config).unwrap();
        }

        let d = poincare_distance(g.current_centroid(), &target);
        assert!(d < 0.01, "centroid should converge after 100 steps, d={d}");
    }

    // ── drift veto ────────────────────────────

    #[test]
    fn small_drift_does_not_freeze() {
        let mut g = CentroidGuardian::new(2);
        let mut config = test_config();
        config.centroid_drift_threshold = 0.5;
        let embeddings = vec![(make_id(), vec![0.1, 0.0])];
        let report = g.update(&embeddings, None, &config).unwrap();
        assert!(!report.froze, "small drift should not trigger freeze");
        assert!(!g.is_frozen());
    }

    #[test]
    fn large_alpha_can_trigger_freeze() {
        let mut g = CentroidGuardian::new(2);
        let mut config = test_config();
        config.centroid_alpha = 1.0;
        config.centroid_drift_threshold = 0.001;
        let embeddings = vec![(make_id(), vec![0.6, 0.0])];
        let report = g.update(&embeddings, None, &config).unwrap();
        assert!(report.froze || report.drift > 0.001, "large step should cause drift");
    }

    #[test]
    fn unfreeze_clears_flag() {
        let mut g = CentroidGuardian::new(2);
        let mut config = test_config();
        config.centroid_alpha = 1.0;
        config.centroid_drift_threshold = 0.001;
        let embeddings = vec![(make_id(), vec![0.6, 0.0])];
        let _ = g.update(&embeddings, None, &config);
        if g.is_frozen() {
            g.unfreeze();
            assert!(!g.is_frozen());
        }
    }

    // ── centrality ────────────────────────────

    #[test]
    fn centrality_decreases_with_distance() {
        let g = CentroidGuardian::new(3);
        let near = vec![0.1, 0.0, 0.0];
        let far = vec![0.8, 0.0, 0.0];
        assert!(g.centrality(&near) > g.centrality(&far));
    }

    #[test]
    fn centrality_at_centroid_is_one() {
        let mut g = CentroidGuardian::new(3);
        g.current = vec![0.1, 0.2, 0.0];
        let c = g.centrality(&[0.1, 0.2, 0.0]);
        assert!((c - 1.0).abs() < 1e-10);
    }

    // ── errors ────────────────────────────────

    #[test]
    fn empty_update_errors() {
        let mut g = CentroidGuardian::new(2);
        assert!(g.update(&[], None, &test_config()).is_err());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut g = CentroidGuardian::new(2);
        let embeddings = vec![(make_id(), vec![0.1, 0.2, 0.3])];
        assert!(g.update(&embeddings, None, &test_config()).is_err());
    }

    // ── weighted update ───────────────────────

    #[test]
    fn weighted_update_biases_toward_high_weight_node() {
        let mut g = CentroidGuardian::new(2);
        let mut config = test_config();
        config.centroid_alpha = 1.0;
        config.centroid_drift_threshold = 10.0;

        let embeddings = vec![
            (make_id(), vec![0.5, 0.0]),   // high weight
            (make_id(), vec![-0.5, 0.0]),  // low weight
        ];
        let weights = vec![100.0, 0.001];
        g.update(&embeddings, Some(&weights), &config).unwrap();

        let cx = g.current_centroid()[0];
        assert!(cx > 0.3, "centroid should bias toward high-weight node, got {cx}");
    }

    #[test]
    fn epoch_increments() {
        let mut g = CentroidGuardian::new(2);
        let config = test_config();
        let embeddings = vec![(make_id(), vec![0.1, 0.0])];
        g.update(&embeddings, None, &config).unwrap();
        g.update(&embeddings, None, &config).unwrap();
        assert_eq!(g.epoch(), 2);
    }
}
