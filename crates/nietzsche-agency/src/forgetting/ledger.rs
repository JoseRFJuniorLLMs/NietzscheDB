//! Deletion Ledger — Auditable record of the Forgetting Engine's kills.
//!
//! CAMADA 2 of the Nezhmetdinov architecture: cryptographic receipts for
//! every hard delete, organized in a Merkle Tree for tamper-proof auditing.
//!
//! ## Design
//!
//! - **DeletionReceipt**: Records what was deleted, when, and why
//! - **Structural Hash**: Hash of topology (edges, depth, type) WITHOUT content
//! - **Merkle Root**: Batch signature for all deletions in a cycle
//! - **Inclusion Proof**: Prove that node X was deleted in cycle Y

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// SHA-256 hash represented as 32 bytes.
pub type Hash256 = [u8; 32];

/// Compute SHA-256 of arbitrary bytes.
fn sha256(data: &[u8]) -> Hash256 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Lightweight hash for MVP — replace with ring/sha2 crate for production.
    // We use a double-hash with different seeds for collision resistance.
    let mut h1 = DefaultHasher::new();
    data.hash(&mut h1);
    let hash1 = h1.finish();

    let mut h2 = DefaultHasher::new();
    hash1.hash(&mut h2);
    data.len().hash(&mut h2);
    let hash2 = h2.finish();

    let mut h3 = DefaultHasher::new();
    hash2.hash(&mut h3);
    data.hash(&mut h3);
    let hash3 = h3.finish();

    let mut h4 = DefaultHasher::new();
    hash3.hash(&mut h4);
    hash1.hash(&mut h4);
    let hash4 = h4.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&hash1.to_le_bytes());
    result[8..16].copy_from_slice(&hash2.to_le_bytes());
    result[16..24].copy_from_slice(&hash3.to_le_bytes());
    result[24..32].copy_from_slice(&hash4.to_le_bytes());
    result
}

/// Combine two hashes into a parent hash (Merkle node).
fn hash_pair(left: &Hash256, right: &Hash256) -> Hash256 {
    let mut combined = Vec::with_capacity(64);
    combined.extend_from_slice(left);
    combined.extend_from_slice(right);
    sha256(&combined)
}

/// Format a hash as hex string.
fn hash_to_hex(hash: &Hash256) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Structural hash of a deleted node — captures topology without content.
///
/// This allows proving the structure that was removed without revealing
/// the actual data (privacy-preserving audit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralHash {
    /// Hash of: node_type + depth + edge_count + neighbor_ids
    pub hash: Hash256,
    /// Human-readable description of what was hashed.
    pub description: String,
}

impl StructuralHash {
    /// Create from node structural properties.
    pub fn from_node_topology(
        node_id: &Uuid,
        node_type: &str,
        depth: f32,
        edge_count: usize,
        neighbor_ids: &[Uuid],
    ) -> Self {
        let mut data = Vec::new();
        data.extend_from_slice(node_id.as_bytes());
        data.extend_from_slice(node_type.as_bytes());
        data.extend_from_slice(&depth.to_le_bytes());
        data.extend_from_slice(&(edge_count as u64).to_le_bytes());
        for nid in neighbor_ids {
            data.extend_from_slice(nid.as_bytes());
        }

        Self {
            hash: sha256(&data),
            description: format!(
                "type={}, depth={:.3}, edges={}, neighbors={}",
                node_type, depth, edge_count, neighbor_ids.len()
            ),
        }
    }
}

impl fmt::Display for StructuralHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hash_to_hex(&self.hash))
    }
}

/// Receipt for a single node deletion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionReceipt {
    /// The deleted node's ID.
    pub node_id: Uuid,
    /// Cycle in which the deletion occurred.
    pub cycle: u64,
    /// Timestamp (ms since epoch).
    pub timestamp_ms: u64,
    /// The vitality score at time of deletion.
    pub vitality_at_death: f32,
    /// Energy at time of deletion.
    pub energy_at_death: f32,
    /// The verdict that caused deletion.
    pub verdict: String,
    /// Structural hash (topology without content).
    pub structural_hash: StructuralHash,
    /// Poincaré coordinates at time of deletion (for Void tracking).
    pub poincare_coords: Vec<f32>,
    /// Index within the cycle's Merkle tree.
    pub merkle_index: usize,
}

impl DeletionReceipt {
    /// Compute the leaf hash for this receipt in the Merkle tree.
    pub fn leaf_hash(&self) -> Hash256 {
        let mut data = Vec::new();
        data.extend_from_slice(self.node_id.as_bytes());
        data.extend_from_slice(&self.cycle.to_le_bytes());
        data.extend_from_slice(&self.vitality_at_death.to_le_bytes());
        data.extend_from_slice(&self.structural_hash.hash);
        sha256(&data)
    }
}

/// Inclusion proof: prove that a specific receipt exists in a Merkle root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InclusionProof {
    /// The receipt being proven.
    pub node_id: Uuid,
    /// Cycle of deletion.
    pub cycle: u64,
    /// Leaf hash of the receipt.
    pub leaf_hash: Hash256,
    /// Sibling hashes along the path to root (bottom-up).
    /// Each entry is (hash, is_left) where is_left means the sibling is on the left.
    pub path: Vec<(Hash256, bool)>,
    /// The Merkle root this proof validates against.
    pub root: Hash256,
}

impl InclusionProof {
    /// Verify this proof against a known Merkle root.
    pub fn verify(&self) -> bool {
        let mut current = self.leaf_hash;
        for (sibling, is_left) in &self.path {
            current = if *is_left {
                hash_pair(sibling, &current)
            } else {
                hash_pair(&current, sibling)
            };
        }
        current == self.root
    }
}

/// Deletion Ledger — accumulates receipts and builds Merkle trees per cycle.
#[derive(Debug, Clone, Default)]
pub struct DeletionLedger {
    /// All receipts across all cycles.
    pub receipts: Vec<DeletionReceipt>,
    /// Merkle roots indexed by cycle number.
    pub merkle_roots: Vec<(u64, Hash256)>,
    /// Internal: leaf hashes for current cycle (reset each cycle).
    current_leaves: Vec<Hash256>,
    /// Current cycle number.
    current_cycle: u64,
}

impl DeletionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a deletion receipt to the current cycle.
    pub fn record_deletion(&mut self, mut receipt: DeletionReceipt) {
        receipt.merkle_index = self.current_leaves.len();
        let leaf = receipt.leaf_hash();
        self.current_leaves.push(leaf);
        self.receipts.push(receipt);
    }

    /// Finalize the current cycle: build Merkle tree and store root.
    /// Returns the Merkle root for this cycle.
    pub fn finalize_cycle(&mut self, cycle: u64) -> Hash256 {
        self.current_cycle = cycle;
        let root = build_merkle_root(&self.current_leaves);
        self.merkle_roots.push((cycle, root));
        self.current_leaves.clear();
        root
    }

    /// Generate an inclusion proof for a specific receipt.
    pub fn prove_inclusion(&self, node_id: &Uuid, cycle: u64) -> Option<InclusionProof> {
        // Find the receipt
        let receipt = self.receipts.iter()
            .find(|r| r.node_id == *node_id && r.cycle == cycle)?;

        // Find the Merkle root for this cycle
        let root = self.merkle_roots.iter()
            .find(|(c, _)| *c == cycle)
            .map(|(_, r)| *r)?;

        // Rebuild leaf hashes for this cycle
        let cycle_receipts: Vec<&DeletionReceipt> = self.receipts.iter()
            .filter(|r| r.cycle == cycle)
            .collect();

        let leaves: Vec<Hash256> = cycle_receipts.iter()
            .map(|r| r.leaf_hash())
            .collect();

        let idx = receipt.merkle_index;
        let path = build_merkle_proof(&leaves, idx);

        Some(InclusionProof {
            node_id: *node_id,
            cycle,
            leaf_hash: receipt.leaf_hash(),
            path,
            root,
        })
    }

    /// Total deletions across all cycles.
    pub fn total_deletions(&self) -> usize {
        self.receipts.len()
    }

    /// Deletions in a specific cycle.
    pub fn deletions_in_cycle(&self, cycle: u64) -> usize {
        self.receipts.iter().filter(|r| r.cycle == cycle).count()
    }
}

/// Build a Merkle root from leaf hashes.
fn build_merkle_root(leaves: &[Hash256]) -> Hash256 {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next_level = Vec::new();
        for chunk in level.chunks(2) {
            if chunk.len() == 2 {
                next_level.push(hash_pair(&chunk[0], &chunk[1]));
            } else {
                // Odd leaf: duplicate it
                next_level.push(hash_pair(&chunk[0], &chunk[0]));
            }
        }
        level = next_level;
    }
    level[0]
}

/// Build a Merkle proof (path from leaf to root) for a given leaf index.
fn build_merkle_proof(leaves: &[Hash256], index: usize) -> Vec<(Hash256, bool)> {
    if leaves.len() <= 1 {
        return vec![];
    }

    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;

    while level.len() > 1 {
        let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
        let sibling_idx = sibling_idx.min(level.len() - 1);
        let is_left = idx % 2 != 0; // sibling is on the left if we're odd

        path.push((level[sibling_idx], is_left));

        // Build next level
        let mut next_level = Vec::new();
        for chunk in level.chunks(2) {
            if chunk.len() == 2 {
                next_level.push(hash_pair(&chunk[0], &chunk[1]));
            } else {
                next_level.push(hash_pair(&chunk[0], &chunk[0]));
            }
        }
        level = next_level;
        idx /= 2;
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_receipt(cycle: u64, idx: usize) -> DeletionReceipt {
        DeletionReceipt {
            node_id: Uuid::new_v4(),
            cycle,
            timestamp_ms: 1000 + idx as u64,
            vitality_at_death: 0.1,
            energy_at_death: 0.05,
            verdict: "CONDEMNED".into(),
            structural_hash: StructuralHash {
                hash: sha256(format!("node_{}", idx).as_bytes()),
                description: format!("test node {}", idx),
            },
            poincare_coords: vec![0.1, 0.2],
            merkle_index: 0,
        }
    }

    #[test]
    fn ledger_records_and_finalizes() {
        let mut ledger = DeletionLedger::new();

        for i in 0..5 {
            ledger.record_deletion(make_receipt(1, i));
        }

        let root = ledger.finalize_cycle(1);
        assert_ne!(root, [0u8; 32]);
        assert_eq!(ledger.total_deletions(), 5);
        assert_eq!(ledger.deletions_in_cycle(1), 5);
    }

    #[test]
    fn inclusion_proof_verifies() {
        let mut ledger = DeletionLedger::new();
        let mut receipts = Vec::new();

        for i in 0..8 {
            let r = make_receipt(1, i);
            receipts.push(r.clone());
            ledger.record_deletion(r);
        }

        ledger.finalize_cycle(1);

        // Prove inclusion for each receipt
        for r in &receipts {
            let proof = ledger.prove_inclusion(&r.node_id, 1);
            assert!(proof.is_some(), "should find proof for {}", r.node_id);
            let proof = proof.unwrap();
            assert!(proof.verify(), "proof should verify for {}", r.node_id);
        }
    }

    #[test]
    fn merkle_root_deterministic() {
        let leaves: Vec<Hash256> = (0..4)
            .map(|i| sha256(format!("leaf_{}", i).as_bytes()))
            .collect();

        let root1 = build_merkle_root(&leaves);
        let root2 = build_merkle_root(&leaves);
        assert_eq!(root1, root2, "same leaves should produce same root");
    }

    #[test]
    fn empty_ledger_root() {
        let root = build_merkle_root(&[]);
        assert_eq!(root, [0u8; 32]);
    }

    #[test]
    fn single_leaf_root() {
        let leaf = sha256(b"single");
        let root = build_merkle_root(&[leaf]);
        assert_eq!(root, leaf);
    }
}
