//! Phase XVIII — Hyperbolic Contrastive Training Loss.
//!
//! Trains Poincaré ball embeddings to reflect the true hierarchical structure
//! of the graph via contrastive loss + Riemannian SGD.
//!
//! ## Loss Function
//!
//! For a positive edge (u, v) with K negative samples n₁..nₖ:
//!
//! ```text
//! L(u,v) = -log σ(-d_H(u,v)) - Σᵢ log σ(d_H(u,nᵢ) - margin)
//! ```
//!
//! ## Integration
//!
//! Called from `AgencyEngine::tick()` every `hyp_training_interval` ticks.
//! Produces `AgencyIntent::UpdateEmbeddingBatch` for the server to write back.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use rand::seq::SliceRandom;
use tracing::info;
use uuid::Uuid;

// ──────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────

/// Configuration for hyperbolic contrastive training.
#[derive(Debug, Clone)]
pub struct HyperbolicTrainingConfig {
    /// Learning rate for Riemannian SGD (default: 0.01).
    pub learning_rate: f64,
    /// Number of negative samples per positive edge (default: 10).
    pub num_negatives: usize,
    /// Margin for negative samples in loss function (default: 0.1).
    pub margin: f64,
    /// Maximum radius in Poincaré ball — clips to prevent numerical instability (default: 0.95).
    pub max_norm: f64,
    /// Number of training epochs per tick (default: 5, lighter than experiment's 50).
    pub epochs: usize,
    /// Minimum improvement in loss to continue training (default: 1e-4).
    pub convergence_threshold: f64,
    /// Burn-in epochs with reduced learning rate (lr/10) for stabilization (default: 2).
    pub burn_in_epochs: usize,
    /// Maximum edges to sample per epoch for O(E) cap (default: 5000).
    pub max_edges_per_epoch: usize,
}

impl Default for HyperbolicTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            num_negatives: 10,
            margin: 0.1,
            max_norm: 0.95,
            epochs: 5,
            convergence_threshold: 1e-4,
            burn_in_epochs: 2,
            max_edges_per_epoch: 5_000,
        }
    }
}

/// Build training config from AgencyConfig.
pub fn build_training_config(cfg: &crate::config::AgencyConfig) -> HyperbolicTrainingConfig {
    HyperbolicTrainingConfig {
        learning_rate: cfg.hyp_training_lr,
        num_negatives: cfg.hyp_training_num_negatives,
        margin: cfg.hyp_training_margin,
        max_norm: cfg.hyp_training_max_norm,
        epochs: cfg.hyp_training_epochs,
        convergence_threshold: cfg.hyp_training_convergence,
        burn_in_epochs: cfg.hyp_training_burn_in,
        max_edges_per_epoch: cfg.hyp_training_max_edges,
    }
}

// ──────────────────────────────────────────────────────────────────
// Training result
// ──────────────────────────────────────────────────────────────────

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final mean loss over the last epoch.
    pub final_loss: f64,
    /// Loss per epoch for monitoring convergence.
    pub loss_history: Vec<f64>,
    /// Number of epochs actually run (may be < config.epochs if converged).
    pub epochs_run: usize,
    /// Number of embedding updates applied.
    pub updates_applied: usize,
    /// Node IDs whose embeddings were modified (for dirty-set marking).
    pub modified_nodes: Vec<Uuid>,
    /// Updated embeddings: (node_id, new_coords).
    pub updated_embeddings: Vec<(Uuid, Vec<f64>)>,
}

// ──────────────────────────────────────────────────────────────────
// Core math: Poincaré ball operations
// ──────────────────────────────────────────────────────────────────

/// Squared norm of a vector.
#[inline]
pub fn sq_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

/// Poincaré distance: d_H(u, v) = acosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))
#[inline]
pub fn poincare_distance(u: &[f64], v: &[f64]) -> f64 {
    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let norm_u_sq = sq_norm(u);
    let norm_v_sq = sq_norm(v);

    let denom = (1.0 - norm_u_sq).max(1e-15) * (1.0 - norm_v_sq).max(1e-15);
    let arg = 1.0 + 2.0 * diff_sq / denom;

    arg.max(1.0).acosh()
}

/// Sigmoid function: σ(x) = 1 / (1 + e^(-x))
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Euclidean gradient of Poincaré distance w.r.t. u.
pub fn grad_poincare_wrt_u(u: &[f64], v: &[f64]) -> Vec<f64> {
    let dim = u.len();
    let norm_u_sq = sq_norm(u);
    let norm_v_sq = sq_norm(v);
    let diff_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();

    let gamma_u = (1.0 - norm_u_sq).max(1e-15);
    let gamma_v = (1.0 - norm_v_sq).max(1e-15);

    let alpha = 1.0 + 2.0 * diff_sq / (gamma_u * gamma_v);
    let sqrt_alpha2_minus1 = ((alpha * alpha - 1.0).max(1e-15)).sqrt();

    let scale = 4.0 / (gamma_u * gamma_u * gamma_v * sqrt_alpha2_minus1 + 1e-15);

    let dot_uv: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

    let mut grad = vec![0.0; dim];
    let coeff_u = norm_v_sq - 2.0 * dot_uv + 1.0;
    for i in 0..dim {
        grad[i] = scale * (coeff_u * u[i] - gamma_u * v[i]);
    }

    grad
}

/// Riemannian rescaling: ∇_R = ((1 - ‖x‖²)² / 4) · ∇_E
#[inline]
pub fn riemannian_rescale(coords: &[f64], euclidean_grad: &mut [f64]) {
    let norm_sq = sq_norm(coords);
    let conformal = (1.0 - norm_sq).max(1e-15).powi(2) / 4.0;
    for g in euclidean_grad.iter_mut() {
        *g *= conformal;
    }
}

/// Project point back into Poincaré ball with max_norm radius.
#[inline]
pub fn project_to_ball(coords: &mut [f64], max_norm: f64) {
    let norm = sq_norm(coords).sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for c in coords.iter_mut() {
            *c *= scale;
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Contrastive loss computation
// ──────────────────────────────────────────────────────────────────

/// Contrastive loss for a positive pair (u, v) with negative samples.
///
/// L(u,v) = -log σ(-d_H(u,v)) - Σᵢ log σ(d_H(u,nᵢ) - margin)
pub fn contrastive_loss(
    u: &[f64],
    v: &[f64],
    negatives: &[&[f64]],
    margin: f64,
) -> f64 {
    let d_pos = poincare_distance(u, v);
    let pos_loss = -sigmoid(-d_pos).max(1e-15).ln();

    let mut neg_loss = 0.0;
    for neg in negatives {
        let d_neg = poincare_distance(u, neg);
        neg_loss -= sigmoid(d_neg - margin).max(1e-15).ln();
    }

    pos_loss + neg_loss
}

// ──────────────────────────────────────────────────────────────────
// Training loop
// ──────────────────────────────────────────────────────────────────

/// Train embeddings using hyperbolic contrastive loss with Riemannian SGD.
///
/// Pure computation over `GraphStorage` + `AdjacencyIndex`.
/// Returns updated embeddings as `TrainingResult` — the caller (AgencyEngine)
/// converts these into `AgencyIntent::UpdateEmbeddingBatch`.
pub fn train_hyperbolic_embeddings(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &HyperbolicTrainingConfig,
) -> TrainingResult {
    let mut rng = rand::thread_rng();

    // 1. Load all node embeddings into trainable format
    let all_node_ids = adjacency.all_nodes();
    if all_node_ids.is_empty() {
        return TrainingResult {
            final_loss: 0.0,
            loss_history: vec![],
            epochs_run: 0,
            updates_applied: 0,
            modified_nodes: vec![],
            updated_embeddings: vec![],
        };
    }

    // Build id→index map
    let id_to_idx: std::collections::HashMap<Uuid, usize> = all_node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Load embeddings (only nodes that have them)
    let mut embeddings: Vec<Option<Vec<f64>>> = Vec::with_capacity(all_node_ids.len());
    for id in &all_node_ids {
        let emb = storage
            .get_embedding(id)
            .ok()
            .flatten()
            .map(|e| e.coords_f64());
        embeddings.push(emb);
    }

    // Collect all edges for training
    let mut all_edges: Vec<(usize, usize)> = Vec::new();
    for (idx_u, id_u) in all_node_ids.iter().enumerate() {
        if embeddings[idx_u].is_none() {
            continue;
        }
        for neighbor_id in adjacency.neighbors_out(id_u) {
            if let Some(&idx_v) = id_to_idx.get(&neighbor_id) {
                if embeddings[idx_v].is_some() {
                    all_edges.push((idx_u, idx_v));
                }
            }
        }
    }

    if all_edges.is_empty() {
        return TrainingResult {
            final_loss: 0.0,
            loss_history: vec![],
            epochs_run: 0,
            updates_applied: 0,
            modified_nodes: vec![],
            updated_embeddings: vec![],
        };
    }

    // Nodes with embeddings (for negative sampling)
    let nodes_with_emb: Vec<usize> = (0..all_node_ids.len())
        .filter(|&i| embeddings[i].is_some())
        .collect();

    let mut loss_history = Vec::with_capacity(config.epochs);
    let mut total_updates = 0usize;
    let mut modified_set = std::collections::HashSet::new();

    // 2. Training loop
    for epoch in 0..config.epochs {
        let lr = if epoch < config.burn_in_epochs {
            config.learning_rate / 10.0
        } else {
            config.learning_rate
        };

        let mut epoch_loss = 0.0;
        let mut epoch_samples = 0usize;

        // Shuffle edges and cap per epoch
        let mut epoch_edges = all_edges.clone();
        epoch_edges.shuffle(&mut rng);
        if epoch_edges.len() > config.max_edges_per_epoch {
            epoch_edges.truncate(config.max_edges_per_epoch);
        }

        for &(idx_u, idx_v) in &epoch_edges {
            let u_coords = match &embeddings[idx_u] {
                Some(e) => e.clone(),
                None => continue,
            };
            let v_coords = match &embeddings[idx_v] {
                Some(e) => e.clone(),
                None => continue,
            };

            // Sample negatives for u
            let u_neighbors = adjacency.neighbors_both(&all_node_ids[idx_u]);
            let mut neg_indices = Vec::with_capacity(config.num_negatives);
            let mut attempts = 0;
            while neg_indices.len() < config.num_negatives && attempts < config.num_negatives * 5 {
                let &candidate_idx = nodes_with_emb.choose(&mut rng).unwrap();
                let candidate_id = &all_node_ids[candidate_idx];
                if candidate_idx != idx_u
                    && candidate_idx != idx_v
                    && !u_neighbors.contains(candidate_id)
                {
                    neg_indices.push(candidate_idx);
                }
                attempts += 1;
            }

            // Collect negative embeddings
            let neg_coords: Vec<Vec<f64>> = neg_indices
                .iter()
                .filter_map(|&i| embeddings[i].clone())
                .collect();
            let neg_refs: Vec<&[f64]> = neg_coords.iter().map(|v| v.as_slice()).collect();

            // Compute loss
            let loss = contrastive_loss(&u_coords, &v_coords, &neg_refs, config.margin);
            epoch_loss += loss;
            epoch_samples += 1;

            // ── Gradient for u ──────────────────────────────────────
            let d_pos = poincare_distance(&u_coords, &v_coords);
            let sig_neg_d = sigmoid(-d_pos);
            let mut grad_u = grad_poincare_wrt_u(&u_coords, &v_coords);
            let pos_scale = 1.0 - sig_neg_d;
            for g in grad_u.iter_mut() {
                *g *= pos_scale;
            }

            for neg in &neg_coords {
                let d_neg = poincare_distance(&u_coords, neg);
                let sig_d_neg_m = sigmoid(d_neg - config.margin);
                let neg_scale = -(1.0 - sig_d_neg_m);
                let grad_neg = grad_poincare_wrt_u(&u_coords, neg);
                for (g, gn) in grad_u.iter_mut().zip(grad_neg.iter()) {
                    *g += neg_scale * gn;
                }
            }

            riemannian_rescale(&u_coords, &mut grad_u);

            // SGD update for u
            if let Some(ref mut u_emb) = embeddings[idx_u] {
                for (c, g) in u_emb.iter_mut().zip(grad_u.iter()) {
                    *c -= lr * g;
                }
                project_to_ball(u_emb, config.max_norm);
                modified_set.insert(idx_u);
                total_updates += 1;
            }

            // ── Gradient for v (symmetric: push v toward u) ──────
            let mut grad_v = grad_poincare_wrt_u(&v_coords, &u_coords);
            let pos_scale_v = 1.0 - sigmoid(-d_pos);
            for g in grad_v.iter_mut() {
                *g *= pos_scale_v;
            }
            riemannian_rescale(&v_coords, &mut grad_v);

            if let Some(ref mut v_emb) = embeddings[idx_v] {
                for (c, g) in v_emb.iter_mut().zip(grad_v.iter()) {
                    *c -= lr * g;
                }
                project_to_ball(v_emb, config.max_norm);
                modified_set.insert(idx_v);
                total_updates += 1;
            }
        }

        let mean_loss = if epoch_samples > 0 {
            epoch_loss / epoch_samples as f64
        } else {
            0.0
        };
        loss_history.push(mean_loss);

        // Convergence check (after burn-in)
        if epoch >= config.burn_in_epochs && loss_history.len() >= 2 {
            let prev = loss_history[loss_history.len() - 2];
            let improvement = (prev - mean_loss).abs();
            if improvement < config.convergence_threshold {
                info!(
                    epoch,
                    mean_loss,
                    improvement,
                    "Hyperbolic training converged"
                );
                break;
            }
        }

        if epoch % 10 == 0 || epoch == config.epochs - 1 {
            info!(epoch, mean_loss, updates = total_updates, "Hyperbolic training epoch");
        }
    }

    // 3. Collect results
    let modified_nodes: Vec<Uuid> = modified_set
        .iter()
        .map(|&idx| all_node_ids[idx])
        .collect();

    let updated_embeddings: Vec<(Uuid, Vec<f64>)> = modified_set
        .into_iter()
        .filter_map(|idx| {
            embeddings[idx]
                .as_ref()
                .map(|e| (all_node_ids[idx], e.clone()))
        })
        .collect();

    let final_loss = loss_history.last().copied().unwrap_or(0.0);
    let epochs_run = loss_history.len();

    TrainingResult {
        final_loss,
        loss_history,
        epochs_run,
        updates_applied: total_updates,
        modified_nodes,
        updated_embeddings,
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_distance_same_point() {
        let u = vec![0.0, 0.0, 0.0];
        let d = poincare_distance(&u, &u);
        assert!(d.abs() < 1e-10, "Distance to self should be 0, got {d}");
    }

    #[test]
    fn test_poincare_distance_symmetry() {
        let u = vec![0.1, 0.2, 0.3];
        let v = vec![0.4, 0.1, -0.2];
        let d1 = poincare_distance(&u, &v);
        let d2 = poincare_distance(&v, &u);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Poincaré distance should be symmetric: {d1} vs {d2}"
        );
    }

    #[test]
    fn test_poincare_distance_origin_to_point() {
        let origin = vec![0.0, 0.0];
        let point = vec![0.5, 0.0];
        let d = poincare_distance(&origin, &point);
        let expected = (5.0_f64 / 3.0).acosh();
        assert!(
            (d - expected).abs() < 1e-10,
            "Expected {expected}, got {d}"
        );
    }

    #[test]
    fn test_poincare_boundary_large_distance() {
        let u = vec![0.9, 0.0];
        let v = vec![-0.9, 0.0];
        let d = poincare_distance(&u, &v);
        assert!(d > 5.0, "Boundary-to-boundary should be large: {d}");
    }

    #[test]
    fn test_sigmoid_basic() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_contrastive_loss_positive_closer() {
        let u = vec![0.1, 0.0];
        let v = vec![0.15, 0.0];
        let neg = vec![0.8, 0.0];
        let loss1 = contrastive_loss(&u, &v, &[&neg], 0.1);

        let v_far = vec![0.8, 0.0];
        let neg_close = vec![0.15, 0.0];
        let loss2 = contrastive_loss(&u, &v_far, &[&neg_close], 0.1);

        assert!(
            loss2 > loss1,
            "Misaligned pair should have higher loss: {loss1} vs {loss2}"
        );
    }

    #[test]
    fn test_gradient_finite_difference() {
        let u = vec![0.3, -0.2, 0.1];
        let v = vec![0.1, 0.4, -0.3];
        let eps = 1e-7;

        let grad = grad_poincare_wrt_u(&u, &v);

        for dim in 0..u.len() {
            let mut u_plus = u.clone();
            let mut u_minus = u.clone();
            u_plus[dim] += eps;
            u_minus[dim] -= eps;

            let d_plus = poincare_distance(&u_plus, &v);
            let d_minus = poincare_distance(&u_minus, &v);
            let numerical = (d_plus - d_minus) / (2.0 * eps);

            assert!(
                (grad[dim] - numerical).abs() < 1e-4,
                "Gradient mismatch in dim {dim}: analytical={}, numerical={}",
                grad[dim],
                numerical
            );
        }
    }

    #[test]
    fn test_project_to_ball() {
        let mut coords = vec![0.8, 0.6];
        project_to_ball(&mut coords, 0.95);
        let norm = sq_norm(&coords).sqrt();
        assert!(
            norm <= 0.95 + 1e-10,
            "Should be inside ball: norm={norm}"
        );
    }

    #[test]
    fn test_riemannian_rescale_at_origin() {
        let coords = vec![0.0, 0.0];
        let mut grad = vec![1.0, 1.0];
        riemannian_rescale(&coords, &mut grad);
        assert!(
            (grad[0] - 0.25).abs() < 1e-10,
            "At origin, rescale = 0.25: got {}",
            grad[0]
        );
    }

    #[test]
    fn test_riemannian_rescale_near_boundary() {
        let coords = vec![0.9, 0.0];
        let mut grad = vec![1.0, 1.0];
        riemannian_rescale(&coords, &mut grad);
        let expected = (1.0 - 0.81_f64).powi(2) / 4.0;
        assert!(
            (grad[0] - expected).abs() < 1e-10,
            "Near boundary rescale: expected {expected}, got {}",
            grad[0]
        );
    }
}
