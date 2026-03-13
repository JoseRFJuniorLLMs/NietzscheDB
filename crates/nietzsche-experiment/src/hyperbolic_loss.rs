// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase XVIII — Hyperbolic Contrastive Training Loss.
//!
//! Re-exports the training implementation from `nietzsche-agency::hyperbolic_training`.
//! This module exists for backward compatibility and experiment-specific extensions.

pub use nietzsche_agency::hyperbolic_training::{
    HyperbolicTrainingConfig, TrainingResult,
    train_hyperbolic_embeddings, build_training_config,
    poincare_distance, sigmoid, contrastive_loss,
    grad_poincare_wrt_u, riemannian_rescale, project_to_ball,
    sq_norm,
};

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
}
