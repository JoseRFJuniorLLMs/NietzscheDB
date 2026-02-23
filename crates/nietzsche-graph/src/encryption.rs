//! Application-level authenticated encryption at-rest using AES-256-GCM.
//!
//! Provides transparent encrypt/decrypt for data stored in RocksDB.
//! Master key is loaded from `NIETZSCHE_ENCRYPTION_KEY` (base64-encoded 32 bytes).
//! Per-CF keys are derived via HKDF-SHA256. Each ciphertext is prepended with a
//! random 12-byte nonce and includes a 16-byte GCM authentication tag for
//! integrity verification.
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
//!
//! ## Ciphertext format
//!
//! ```text
//! [version: u8 = 0x02][nonce: 12 bytes][ciphertext + GCM tag: N + 16 bytes]
//! ```
//!
//! Version 0x01 (legacy AES-CTR) is supported for reading during migration.

use aes::Aes256;
use aes_gcm::{Aes256Gcm, Nonce as GcmNonce};
use aes_gcm::aead::{Aead, KeyInit};
use ctr::cipher::{KeyIvInit, StreamCipher};
use hkdf::Hkdf;
use sha2::Sha256;

type Aes256Ctr = ctr::Ctr128BE<Aes256>;

/// Version byte for the new AES-256-GCM format.
const ENCRYPTION_VERSION_GCM: u8 = 0x02;

/// GCM nonce size (96 bits as recommended by NIST SP 800-38D).
const GCM_NONCE_SIZE: usize = 12;

/// Configuration for at-rest encryption.
pub struct EncryptionConfig {
    /// Whether encryption is enabled.
    pub enabled: bool,
    /// Master key (32 bytes). `None` if encryption is disabled.
    /// Wrapped in a zeroing type for memory safety.
    master_key: Option<[u8; 32]>,
}

impl Clone for EncryptionConfig {
    fn clone(&self) -> Self {
        Self {
            enabled: self.enabled,
            master_key: self.master_key,
        }
    }
}

impl Drop for EncryptionConfig {
    fn drop(&mut self) {
        // Zeroize master key on drop to prevent key material lingering in memory
        if let Some(ref mut key) = self.master_key {
            for byte in key.iter_mut() {
                // Use volatile write to prevent compiler from optimizing away the zeroization
                unsafe { std::ptr::write_volatile(byte, 0) };
            }
        }
    }
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
    /// **Panics** if the env var is set but contains an invalid key (fail-fast).
    pub fn from_env() -> Self {
        match std::env::var("NIETZSCHE_ENCRYPTION_KEY") {
            Ok(b64) if !b64.is_empty() => {
                match decode_base64(&b64) {
                    Some(key) if key.len() == 32 => {
                        let mut master = [0u8; 32];
                        master.copy_from_slice(&key);
                        Self::new(master)
                    }
                    _ => {
                        panic!(
                            "FATAL: NIETZSCHE_ENCRYPTION_KEY is set but is not a valid 32-byte \
                             base64 key. Refusing to start with invalid encryption config. \
                             Remove the variable to disable encryption, or provide a valid key."
                        );
                    }
                }
            }
            _ => Self::default(),
        }
    }

    /// Derive a per-CF encryption key using HKDF-SHA256 with a fixed salt.
    fn derive_key(&self, info: &[u8]) -> [u8; 32] {
        let master = self.master_key.expect("derive_key called with no master key");
        // Use a fixed salt for domain separation (better than None)
        let salt = b"nietzsche-db-encryption-v2";
        let hk = Hkdf::<Sha256>::new(Some(salt), &master);
        let mut derived = [0u8; 32];
        hk.expand(info, &mut derived).expect("HKDF expand failed");
        derived
    }

    /// Encrypt data for a specific column family using AES-256-GCM.
    ///
    /// Returns: `[version_byte | random_nonce(12B) | ciphertext + tag]`
    pub fn encrypt(&self, cf_name: &str, _item_key: &[u8], plaintext: &[u8]) -> Vec<u8> {
        if !self.enabled {
            return plaintext.to_vec();
        }
        let derived = self.derive_key(format!("nietzsche:{cf_name}").as_bytes());

        // Generate random nonce (critical: must be unique per encryption)
        let mut nonce_bytes = [0u8; GCM_NONCE_SIZE];
        getrandom::getrandom(&mut nonce_bytes)
            .expect("failed to generate random nonce");

        let cipher = Aes256Gcm::new(derived.as_ref().into());
        let nonce = GcmNonce::from_slice(&nonce_bytes);
        let ciphertext = cipher.encrypt(nonce, plaintext)
            .expect("AES-256-GCM encryption should not fail");

        // Format: [version | nonce | ciphertext+tag]
        let mut out = Vec::with_capacity(1 + GCM_NONCE_SIZE + ciphertext.len());
        out.push(ENCRYPTION_VERSION_GCM);
        out.extend_from_slice(&nonce_bytes);
        out.extend_from_slice(&ciphertext);
        out
    }

    /// Decrypt data from a specific column family.
    ///
    /// Supports both V2 (AES-256-GCM) and legacy V1 (AES-256-CTR) formats.
    pub fn decrypt(&self, cf_name: &str, item_key: &[u8], data: &[u8]) -> Vec<u8> {
        if !self.enabled {
            return data.to_vec();
        }

        // Check version byte to determine format
        if !data.is_empty() && data[0] == ENCRYPTION_VERSION_GCM {
            // V2: AES-256-GCM
            self.decrypt_gcm(cf_name, &data[1..])
        } else {
            // Legacy V1: AES-256-CTR (for migration compatibility)
            self.decrypt_ctr_legacy(cf_name, item_key, data)
        }
    }

    /// Decrypt using new AES-256-GCM format.
    fn decrypt_gcm(&self, cf_name: &str, data: &[u8]) -> Vec<u8> {
        if data.len() < GCM_NONCE_SIZE + 16 {
            // Too short to contain nonce + tag; return as-is (likely unencrypted)
            return data.to_vec();
        }
        let derived = self.derive_key(format!("nietzsche:{cf_name}").as_bytes());
        let nonce = GcmNonce::from_slice(&data[..GCM_NONCE_SIZE]);
        let ciphertext = &data[GCM_NONCE_SIZE..];

        let cipher = Aes256Gcm::new(derived.as_ref().into());
        match cipher.decrypt(nonce, ciphertext) {
            Ok(plaintext) => plaintext,
            Err(_) => {
                tracing::error!(cf = cf_name, "AES-GCM decryption failed — data integrity violation!");
                // Return empty rather than corrupted data
                Vec::new()
            }
        }
    }

    /// Decrypt using legacy AES-256-CTR format (backward compatibility).
    fn decrypt_ctr_legacy(&self, cf_name: &str, item_key: &[u8], ciphertext: &[u8]) -> Vec<u8> {
        // Legacy: derive key without salt for backward compat
        let master = self.master_key.expect("decrypt called with no master key");
        let hk = Hkdf::<Sha256>::new(None, &master);
        let mut derived = [0u8; 32];
        hk.expand(format!("nietzsche:{cf_name}").as_bytes(), &mut derived)
            .expect("HKDF expand failed");

        let iv = derive_iv_legacy(item_key);
        let mut buf = ciphertext.to_vec();
        let mut cipher = Aes256Ctr::new(derived.as_ref().into(), iv.as_ref().into());
        cipher.apply_keystream(&mut buf);
        buf
    }
}

/// Legacy IV derivation (first 16 bytes of item key).
fn derive_iv_legacy(key: &[u8]) -> [u8; 16] {
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
    fn encrypt_decrypt_roundtrip_gcm() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        let plaintext = b"hello world, this is sensitive node data";
        let item_key = uuid::Uuid::new_v4();

        let encrypted = config.encrypt("nodes", item_key.as_bytes(), plaintext);
        // GCM format: version(1) + nonce(12) + ciphertext + tag(16)
        assert_eq!(encrypted[0], ENCRYPTION_VERSION_GCM);
        assert!(encrypted.len() > 1 + GCM_NONCE_SIZE + 16);
        assert_ne!(&encrypted[1 + GCM_NONCE_SIZE..], plaintext.as_slice());

        let decrypted = config.decrypt("nodes", item_key.as_bytes(), &encrypted);
        assert_eq!(decrypted, plaintext.to_vec());
    }

    #[test]
    fn tampered_ciphertext_detected() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        let plaintext = b"sensitive data";
        let item_key = [0xBB; 16];

        let mut encrypted = config.encrypt("nodes", &item_key, plaintext);
        // Flip a bit in the ciphertext
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;

        let decrypted = config.decrypt("nodes", &item_key, &encrypted);
        // GCM should detect tampering and return empty
        assert!(decrypted.is_empty());
    }

    #[test]
    fn same_plaintext_produces_different_ciphertext() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        let plaintext = b"same data";
        let item_key = [0xAA; 16];

        let enc1 = config.encrypt("nodes", &item_key, plaintext);
        let enc2 = config.encrypt("nodes", &item_key, plaintext);
        // Random nonce ensures different ciphertext each time
        assert_ne!(enc1, enc2);

        // But both decrypt to the same plaintext
        assert_eq!(config.decrypt("nodes", &item_key, &enc1), plaintext.to_vec());
        assert_eq!(config.decrypt("nodes", &item_key, &enc2), plaintext.to_vec());
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

    #[test]
    fn zeroize_on_drop() {
        let key = [0x42u8; 32];
        let config = EncryptionConfig::new(key);
        // After drop, the key should be zeroed (we can't easily test this
        // without unsafe, but at least test that drop doesn't panic)
        drop(config);
    }
}
