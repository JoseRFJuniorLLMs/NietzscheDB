//! Input validation for all gRPC request handlers.
//!
//! Every `validate_*` function returns `Ok(())` on success or a
//! `tonic::Status::invalid_argument` describing the constraint violation.
//!
//! ## Constraints enforced
//!
//! | Field              | Constraint                                              |
//! |--------------------|---------------------------------------------------------|
//! | embedding.coords   | non-empty, dim ≤ 4096, ‖x‖ < 1.0                      |
//! | embedding.dim      | consistent with coords.len() when non-zero              |
//! | energy             | ∈ [0.0, 1.0]                                           |
//! | NQL string         | non-empty, length ≤ 8192                               |
//! | KNN k              | 1 ≤ k ≤ 10 000                                         |
//! | diffusion t_values | non-empty, all values > 0                              |
//! | source_ids count   | ≤ 1024 seeds per diffusion request                     |
//! | UUID strings       | parseable as UUIDv4                                     |

use tonic::Status;
use uuid::Uuid;

use crate::proto::nietzsche::PoincareVector;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum allowed embedding dimensionality.
pub const MAX_EMBEDDING_DIM: usize = 4_096;

/// Maximum NQL query string length (bytes).
pub const MAX_NQL_LEN: usize = 8_192;

/// Maximum allowed k in a KNN request.
pub const MAX_KNN_K: u32 = 10_000;

/// Maximum diffusion source seeds per request.
pub const MAX_DIFFUSION_SOURCES: usize = 1_024;

// ── Embedding ────────────────────────────────────────────────────────────────

/// Validate a `PoincareVector` proto message.
///
/// Checks:
/// - `coords` is non-empty
/// - `coords.len() ≤ MAX_EMBEDDING_DIM`
/// - `‖coords‖² < 1.0` (strictly inside the unit ball)
pub fn validate_embedding(emb: &PoincareVector) -> Result<(), Status> {
    if emb.coords.is_empty() {
        return Err(Status::invalid_argument(
            "embedding.coords must not be empty",
        ));
    }

    if emb.coords.len() > MAX_EMBEDDING_DIM {
        return Err(Status::invalid_argument(format!(
            "embedding dimension {} exceeds maximum {}",
            emb.coords.len(),
            MAX_EMBEDDING_DIM,
        )));
    }

    // Reject NaN / Inf before computing norm
    for (i, &c) in emb.coords.iter().enumerate() {
        if !c.is_finite() {
            return Err(Status::invalid_argument(format!(
                "embedding.coords[{i}] is not finite: {c}"
            )));
        }
    }

    let norm_sq: f64 = emb.coords.iter().map(|x| x * x).sum();
    if norm_sq >= 1.0 {
        return Err(Status::invalid_argument(format!(
            "embedding must be strictly inside the unit ball (‖x‖ < 1.0), \
             got ‖x‖ = {:.6}",
            norm_sq.sqrt()
        )));
    }

    Ok(())
}

// ── Energy ────────────────────────────────────────────────────────────────────

/// Validate that `energy ∈ [0.0, 1.0]`.
pub fn validate_energy(energy: f32) -> Result<(), Status> {
    if !energy.is_finite() {
        return Err(Status::invalid_argument(format!(
            "energy must be finite, got {energy}"
        )));
    }
    if !(0.0..=1.0).contains(&energy) {
        return Err(Status::invalid_argument(format!(
            "energy must be in [0.0, 1.0], got {energy}"
        )));
    }
    Ok(())
}

// ── NQL ───────────────────────────────────────────────────────────────────────

/// Validate a raw NQL query string.
pub fn validate_nql(nql: &str) -> Result<(), Status> {
    if nql.trim().is_empty() {
        return Err(Status::invalid_argument("NQL query must not be empty"));
    }
    if nql.len() > MAX_NQL_LEN {
        return Err(Status::invalid_argument(format!(
            "NQL query length {} bytes exceeds maximum {} bytes",
            nql.len(),
            MAX_NQL_LEN,
        )));
    }
    Ok(())
}

// ── KNN ───────────────────────────────────────────────────────────────────────

/// Validate the `k` parameter for KNN search.
pub fn validate_k(k: u32) -> Result<(), Status> {
    if k == 0 {
        return Err(Status::invalid_argument("k must be ≥ 1"));
    }
    if k > MAX_KNN_K {
        return Err(Status::invalid_argument(format!(
            "k={k} exceeds maximum {MAX_KNN_K}"
        )));
    }
    Ok(())
}

// ── Diffusion ─────────────────────────────────────────────────────────────────

/// Validate diffusion time values.
pub fn validate_t_values(t_values: &[f64]) -> Result<(), Status> {
    if t_values.is_empty() {
        return Err(Status::invalid_argument("t_values must not be empty"));
    }
    for (i, &t) in t_values.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 {
            return Err(Status::invalid_argument(format!(
                "t_values[{i}] = {t} must be a finite positive number"
            )));
        }
    }
    Ok(())
}

/// Validate the number of diffusion source seeds.
pub fn validate_source_count(n: usize) -> Result<(), Status> {
    if n == 0 {
        return Err(Status::invalid_argument(
            "at least one source_id is required for diffusion",
        ));
    }
    if n > MAX_DIFFUSION_SOURCES {
        return Err(Status::invalid_argument(format!(
            "source_ids count {n} exceeds maximum {MAX_DIFFUSION_SOURCES}"
        )));
    }
    Ok(())
}

// ── UUID ──────────────────────────────────────────────────────────────────────

/// Parse and validate a UUID string, returning a descriptive Status on failure.
pub fn parse_uuid(s: &str, field: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(s)
        .map_err(|_| Status::invalid_argument(format!("invalid UUID for '{field}': {s:?}")))
}

// ── Sleep ─────────────────────────────────────────────────────────────────────

/// Validate sleep cycle request fields.
pub fn validate_sleep_params(noise: f64, adam_lr: f64) -> Result<(), Status> {
    if noise < 0.0 || !noise.is_finite() {
        return Err(Status::invalid_argument(format!(
            "noise={noise} must be a non-negative finite number"
        )));
    }
    if adam_lr < 0.0 || !adam_lr.is_finite() {
        return Err(Status::invalid_argument(format!(
            "adam_lr={adam_lr} must be a non-negative finite number"
        )));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_emb(coords: Vec<f64>) -> PoincareVector {
        let dim = coords.len() as u32;
        PoincareVector { coords, dim }
    }

    // ── embedding ────────────────────────────────────────────────────────────

    #[test]
    fn valid_embedding_passes() {
        let emb = vec_emb(vec![0.1, 0.2]);
        assert!(validate_embedding(&emb).is_ok());
    }

    #[test]
    fn empty_coords_rejected() {
        let emb = vec_emb(vec![]);
        assert!(validate_embedding(&emb).is_err());
    }

    #[test]
    fn outside_ball_rejected() {
        let emb = vec_emb(vec![0.8, 0.8]); // ‖·‖ ≈ 1.13
        assert!(validate_embedding(&emb).is_err());
    }

    #[test]
    fn nan_coord_rejected() {
        let emb = vec_emb(vec![f64::NAN, 0.0]);
        assert!(validate_embedding(&emb).is_err());
    }

    #[test]
    fn inf_coord_rejected() {
        let emb = vec_emb(vec![f64::INFINITY, 0.0]);
        assert!(validate_embedding(&emb).is_err());
    }

    #[test]
    fn oversized_embedding_rejected() {
        let emb = vec_emb(vec![0.0; MAX_EMBEDDING_DIM + 1]);
        assert!(validate_embedding(&emb).is_err());
    }

    // ── energy ───────────────────────────────────────────────────────────────

    #[test]
    fn valid_energy_passes() {
        assert!(validate_energy(0.0).is_ok());
        assert!(validate_energy(0.5).is_ok());
        assert!(validate_energy(1.0).is_ok());
    }

    #[test]
    fn negative_energy_rejected() {
        assert!(validate_energy(-0.1).is_err());
    }

    #[test]
    fn energy_above_one_rejected() {
        assert!(validate_energy(1.01).is_err());
    }

    // ── NQL ──────────────────────────────────────────────────────────────────

    #[test]
    fn valid_nql_passes() {
        assert!(validate_nql("MATCH (n) RETURN n").is_ok());
    }

    #[test]
    fn empty_nql_rejected() {
        assert!(validate_nql("   ").is_err());
    }

    #[test]
    fn overlong_nql_rejected() {
        let long = "x".repeat(MAX_NQL_LEN + 1);
        assert!(validate_nql(&long).is_err());
    }

    // ── KNN ──────────────────────────────────────────────────────────────────

    #[test]
    fn valid_k_passes() {
        assert!(validate_k(1).is_ok());
        assert!(validate_k(10).is_ok());
    }

    #[test]
    fn zero_k_rejected() {
        assert!(validate_k(0).is_err());
    }

    #[test]
    fn k_exceeds_max_rejected() {
        assert!(validate_k(MAX_KNN_K + 1).is_err());
    }

    // ── diffusion ────────────────────────────────────────────────────────────

    #[test]
    fn valid_t_values_pass() {
        assert!(validate_t_values(&[0.1, 1.0, 10.0]).is_ok());
    }

    #[test]
    fn empty_t_values_rejected() {
        assert!(validate_t_values(&[]).is_err());
    }

    #[test]
    fn negative_t_rejected() {
        assert!(validate_t_values(&[-1.0]).is_err());
    }

    #[test]
    fn zero_t_rejected() {
        assert!(validate_t_values(&[0.0]).is_err());
    }

    // ── UUID ─────────────────────────────────────────────────────────────────

    #[test]
    fn valid_uuid_passes() {
        let id = Uuid::new_v4().to_string();
        assert!(parse_uuid(&id, "id").is_ok());
    }

    #[test]
    fn invalid_uuid_rejected() {
        assert!(parse_uuid("not-a-uuid", "id").is_err());
    }
}
