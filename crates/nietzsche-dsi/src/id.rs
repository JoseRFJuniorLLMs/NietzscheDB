use serde::{Deserialize, Serialize};

/// Hierarchical semantic identifier for a node.
/// Each element is a discrete latent code (e.g. from VQ-VAE).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SemanticId(pub Vec<u16>);

impl SemanticId {
    pub fn new(codes: Vec<u16>) -> Self {
        Self(codes)
    }

    pub fn to_string(&self) -> String {
        self.0.iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(".")
    }

    pub fn parent(&self) -> Option<Self> {
        if self.0.len() <= 1 {
            None
        } else {
            let mut parent_codes = self.0.clone();
            parent_codes.pop();
            Some(Self(parent_codes))
        }
    }

    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn prefix(&self, level: usize) -> Option<Self> {
        if level > self.0.len() {
            None
        } else {
            Some(Self(self.0[..level].to_vec()))
        }
    }

    /// Returns a prefix-friendly byte representation (each u16 as little-endian, no length prefix).
    pub fn to_prefix_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.0.len() * 2);
        for &code in &self.0 {
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        bytes
    }

    /// Decodes from prefix-friendly bytes.
    pub fn from_prefix_bytes(bytes: &[u8]) -> Self {
        let codes = bytes.chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        Self(codes)
    }
}
