//! Application-level encryption at-rest using AES-256-CTR.
//!
//! Provides transparent encrypt/decrypt for data stored in RocksDB.
//! Master key is loaded from `NIETZSCHE_ENCRYPTION_KEY` (base64-encoded 32 bytes).
//! Per-item keys are derived via HKDF-SHA256 using a unique salt (e.g., CF name + key).
//!
//! ## Design
//!
//! Since the Rust RocksDB bindings don't expose the `Env` encryption layer,
//! we encrypt at the application level: data is encrypted before `put_cf()`
//! and decrypted after `get_cf()`. The energy/meta index CFs use sortable
//! keys and are NOT encrypted (they contain no user data — just node IDs
//! and numeric sort keys).
//!
//! ## Key derivation
//!
//! ```text
//! master_key (32 bytes from env)
//!     │
//!     ├─ HKDF-SHA256(info = "nietzsche:nodes") → derived_key for CF_NODES
//!     ├─ HKDF-SHA256(info = "nietzsche:embeddings") → derived_key for CF_EMBEDDINGS
//!     ├─ HKDF-SHA256(info = "nietzsche:edges") → derived_key for CF_EDGES
//!     └─ HKDF-SHA256(info = "nietzsche:sensory") → derived_key for CF_SENSORY
//! ```

use aes::Aes256;
use ctr::cipher::{KeyIvInit, StreamCipher};
use hkdf::Hkdf;
use sha2::Sha256;

type Aes256Ctr = ctr::Ctr128BE<Aes256>;

/// Configuration for at-rest encryption.
#[derive(Clone)]
pub struct EncryptionConfig {
    /// Whether encryption is enabled.
    pub enabled: bool,
    /// Master key (32 bytes). `None` if encryption is disabled.
    master_key: Option<[u8; 32]>,
}

impl std::fmt::Debug for EncryptionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionConfig")
            .field("enabled", &self.enabled)
            .field("master_key", &if self.master_key.is_some() { "[REDACTED]" } else { "None" })
            .finish()
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self { enabled: false, master_key: None }
    }
}

impl EncryptionConfig {
    /// Create a new encryption config with the given master key.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self { enabled: true, master_key: Some(master_key) }
    }

    /// Load encryption config from environment.
    ///
    /// Reads `NIETZSCHE_ENCRYPTION_KEY` as a base64-encoded 32-byte key.
    /// If the env var is absent or empty, encryption is disabled.
    pub fn from_env() -> Self {
        match std::env::var("NIETZSCHE_ENCRYPTION_KEY") {
            Ok(b64) if !b64.is_empty() => {
                // Simple base64 decode (no padding required)
                match decode_base64(&b64) {
                    Some(key) if key.len() == 32 => {
                        let mut master = [0u8; 32];
                        master.copy_from_slice(&key);
                        Self::new(master)
                    }
                    _ => {
                        eprintln!("WARNING: NIETZSCHE_ENCRYPTION_KEY is not a valid 32-byte base64 key; encryption disabled");
                        Self::default()
                    }
                }
            }
            _ => Self::default(),
        }
    }

    /// Derive a per-CF encryption key using HKDF-SHA256.
    fn derive_key(&self, info: &[u8]) -> [u8; 32] {
        let master = self.master_key.expect("derive_key called with no master key");
        let hk = Hkdf::<Sha256>::new(None, &master);
        let mut derived = [0u8; 32];
        hk.expand(info, &mut derived).expect("HKDF expand failed");
        derived
    }

    /// Encrypt data for a specific column family.
    ///
    /// The IV is derived from the first 16 bytes of `item_key` (e.g. node UUID).
    /// This makes encryption deterministic per key, which is acceptable since
    /// each node ID is unique (UUIDv4) and the key is never reused with different data.
    pub fn encrypt(&self, cf_name: &str, item_key: &[u8], plaintext: &[u8]) -> Vec<u8> {
        if !self.enabled {
            return plaintext.to_vec();
        }
        let derived = self.derive_key(format!("nietzsche:{cf_name}").as_bytes());
        let iv = derive_iv(item_key);
        let mut buf = plaintext.to_vec();
        let mut cipher = Aes256Ctr::new(derived.as_ref().into(), iv.as_ref().into());
        cipher.apply_keystream(&mut buf);
        buf
    }

    /// Decrypt data from a specific column family.
    pub fn decrypt(&self, cf_name: &str, item_key: &[u8], ciphertext: &[u8]) -> Vec<u8> {
        // AES-CTR is symmetric: encrypt == decrypt with the same keystream
        self.encrypt(cf_name, item_key, ciphertext)
    }
}

/// Derive a 16-byte IV from an item key.
/// Uses the first 16 bytes of the key, zero-padded if shorter.
fn derive_iv(key: &[u8]) -> [u8; 16] {
    let mut iv = [0u8; 16];
    let len = key.len().min(16);
    iv[..len].copy_from_slice(&key[..len]);
    iv
}

/// Simple base64 decoder (standard alphabet, no padding required).
fn decode_base64(input: &str) -> Option<Vec<u8>> {
    const TABLE: [u8; 128] = {
        let mut t = [0xFF_u8; 128];
        let mut i = 0u8;
        while i < 26 { t[(b'A' + i) as usize] = i; i += 1; }
        i = 0;
        while i < 26 { t[(b'a' + i) as usize] = 26 + i; i += 1; }
        i = 0;
        while i < 10 { t[(b'0' + i) as usize] = 52 + i; i += 1; }
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        t
    };

    let bytes: Vec<u8> = input.bytes()
        .filter(|&b| b != b'=' && b != b'\n' && b != b'\r')
        .collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let chunks = bytes.chunks(4);
    for chunk in chunks {
        let mut buf = [0u32; 4];
        for (i, &b) in chunk.iter().enumerate() {
            if b as usize >= 128 { return None; }
            let v = TABLE[b as usize];
            if v == 0xFF { return None; }
            buf[i] = v as u32;
        }
        let n = (buf[0] << 18) | (buf[1] << 12) | (buf[2] << 6) | buf[3];
        out.push((n >> 16) as u8);
        if chunk.len() > 2 { out.push((n >> 8) as u8); }
        if chunk.len() > 3 { out.push(n as u8); }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        let plaintext = b"hello world, this is sensitive node data";
        let item_key = uuid::Uuid::new_v4();

        let encrypted = config.encrypt("nodes", item_key.as_bytes(), plaintext);
        assert_ne!(encrypted, plaintext.to_vec());

        let decrypted = config.decrypt("nodes", item_key.as_bytes(), &encrypted);
        assert_eq!(decrypted, plaintext.to_vec());
    }

    #[test]
    fn disabled_is_passthrough() {
        let config = EncryptionConfig::default();
        let data = b"plaintext data";
        let key = [0u8; 16];
        assert_eq!(config.encrypt("nodes", &key, data), data.to_vec());
    }

    #[test]
    fn different_cfs_produce_different_ciphertext() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        let data = b"same data";
        let item_key = [0xAA; 16];

        let enc1 = config.encrypt("nodes", &item_key, data);
        let enc2 = config.encrypt("embeddings", &item_key, data);
        assert_ne!(enc1, enc2);
    }
}
