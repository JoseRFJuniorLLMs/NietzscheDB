//! gRPC RBAC authentication interceptor.
//!
//! Supports multiple API keys with different roles:
//!
//! | Environment variable          | Role     | Permissions                          |
//! |-------------------------------|----------|--------------------------------------|
//! | `NIETZSCHE_API_KEY`           | Admin    | All operations (backward compat)     |
//! | `NIETZSCHE_API_KEY_ADMIN`     | Admin    | All operations                       |
//! | `NIETZSCHE_API_KEY_WRITER`    | Writer   | Read + write (no admin ops)          |
//! | `NIETZSCHE_API_KEY_READER`    | Reader   | Read-only                            |
//!
//! If no API key env vars are set, authentication is disabled (all requests pass as Admin).
//!
//! The interceptor injects [`nietzsche_api::Role`] into `request.extensions()` so
//! that downstream RPC handlers can call `require_writer` / `require_admin`.

use sha2::{Digest, Sha256};
use tonic::{service::Interceptor, Request, Status};

// Re-use Role from nietzsche-api so interceptor and server share the same type.
use nietzsche_api::Role;

// ─────────────────────────────────────────────
// AuthInterceptor
// ─────────────────────────────────────────────

#[derive(Clone)]
pub struct AuthInterceptor {
    /// (SHA-256 hex hash, Role) pairs. Empty = auth disabled.
    key_roles: Vec<(String, Role)>,
}

impl AuthInterceptor {
    /// Build the interceptor from environment variables.
    ///
    /// Reads up to 4 env vars. If none are set, auth is disabled and all
    /// requests pass through as `Role::Admin`.
    pub fn from_env() -> Self {
        let mut key_roles = Vec::new();

        for (env_var, role) in [
            ("NIETZSCHE_API_KEY", Role::Admin),
            ("NIETZSCHE_API_KEY_ADMIN", Role::Admin),
            ("NIETZSCHE_API_KEY_WRITER", Role::Writer),
            ("NIETZSCHE_API_KEY_READER", Role::Reader),
        ] {
            if let Ok(key) = std::env::var(env_var) {
                if !key.is_empty() {
                    let hash = sha256_hex(&key);
                    key_roles.push((hash, role));
                    tracing::info!(env = env_var, role = ?role, "API key registered");
                }
            }
        }

        if key_roles.is_empty() {
            tracing::warn!("gRPC authentication disabled — no API key env vars set");
        } else {
            tracing::info!(keys = key_roles.len(), "gRPC RBAC authentication enabled");
        }

        Self { key_roles }
    }

    /// Legacy constructor for backward compatibility.
    pub fn new(api_key: Option<String>) -> Self {
        match api_key {
            Some(key) if !key.is_empty() => {
                let hash = sha256_hex(&key);
                tracing::info!("gRPC API key authentication enabled (legacy single-key)");
                Self { key_roles: vec![(hash, Role::Admin)] }
            }
            _ => Self::from_env(),
        }
    }
}

impl Interceptor for AuthInterceptor {
    fn call(&mut self, mut request: Request<()>) -> Result<Request<()>, Status> {
        // No keys configured → auth disabled, grant Admin
        if self.key_roles.is_empty() {
            request.extensions_mut().insert(Role::Admin);
            return Ok(request);
        }

        match request.metadata().get("x-api-key") {
            Some(t) => {
                let token_str = t
                    .to_str()
                    .map_err(|_| Status::unauthenticated("invalid API key encoding"))?;

                let request_hash = sha256_hex(token_str);

                for (expected_hash, role) in &self.key_roles {
                    if constant_time_eq(request_hash.as_bytes(), expected_hash.as_bytes()) {
                        request.extensions_mut().insert(*role);
                        return Ok(request);
                    }
                }

                Err(Status::unauthenticated("invalid API key"))
            }
            None => Err(Status::unauthenticated("missing x-api-key header")),
        }
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_is_deterministic() {
        let h1 = sha256_hex("test-key-123");
        let h2 = sha256_hex("test-key-123");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"short", b"longer"));
    }

    #[test]
    fn from_env_with_no_keys_disables_auth() {
        for var in ["NIETZSCHE_API_KEY", "NIETZSCHE_API_KEY_ADMIN", "NIETZSCHE_API_KEY_WRITER", "NIETZSCHE_API_KEY_READER"] {
            std::env::remove_var(var);
        }
        let interceptor = AuthInterceptor::from_env();
        assert!(interceptor.key_roles.is_empty());
    }
}
