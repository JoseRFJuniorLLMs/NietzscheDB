//! gRPC API key authentication interceptor.
//!
//! Reads `NIETZSCHE_API_KEY` at startup. If set, all gRPC requests must include
//! a matching `x-api-key` metadata header. The key is stored as a SHA-256 hash;
//! comparison uses constant-time equality to prevent timing attacks.
//!
//! If `NIETZSCHE_API_KEY` is not set, authentication is disabled (all requests pass).

use sha2::{Digest, Sha256};
use tonic::{service::Interceptor, Request, Status};

#[derive(Clone)]
pub struct AuthInterceptor {
    expected_hash: Option<String>,
}

impl AuthInterceptor {
    /// Create a new interceptor.  Pass `Some(key)` to enable auth, `None` to disable.
    pub fn new(api_key: Option<String>) -> Self {
        let expected_hash = api_key.map(|key| {
            let mut hasher = Sha256::new();
            hasher.update(key.as_bytes());
            hex::encode(hasher.finalize())
        });
        if expected_hash.is_some() {
            tracing::info!("gRPC API key authentication enabled");
        } else {
            tracing::warn!("gRPC API key authentication disabled (NIETZSCHE_API_KEY not set)");
        }
        Self { expected_hash }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(expected) = &self.expected_hash else {
            return Ok(request);
        };

        match request.metadata().get("x-api-key") {
            Some(t) => {
                let token_str = t
                    .to_str()
                    .map_err(|_| Status::unauthenticated("invalid API key encoding"))?;

                let mut hasher = Sha256::new();
                hasher.update(token_str.as_bytes());
                let request_hash = hex::encode(hasher.finalize());

                if constant_time_eq(request_hash.as_bytes(), expected.as_bytes()) {
                    Ok(request)
                } else {
                    Err(Status::unauthenticated("invalid API key"))
                }
            }
            None => Err(Status::unauthenticated("missing x-api-key header")),
        }
    }
}

/// Constant-time byte comparison (prevents timing attacks).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}
