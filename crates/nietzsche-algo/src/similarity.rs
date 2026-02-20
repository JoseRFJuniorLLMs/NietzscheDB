//! Node similarity: Jaccard.

use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use nietzsche_graph::{GraphStorage, AdjacencyIndex, GraphError};

pub struct SimilarityPair {
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub score: f64,
}

/// Jaccard node similarity: `|N(u) ∩ N(v)| / |N(u) ∪ N(v)|`.
///
/// Returns pairs with similarity >= `threshold`, sorted descending, limited to `top_k`.
pub fn jaccard_similarity(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    top_k: usize,
    threshold: f64,
) -> Result<Vec<SimilarityPair>, GraphError> {
    let node_ids: Vec<Uuid> = storage
        .scan_nodes_meta()?
        .into_iter()
        .map(|n| n.id)
        .collect();
    let n = node_ids.len();

    let id_to_idx: HashMap<Uuid, usize> = node_ids.iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    // Build neighbor sets (undirected)
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, id) in node_ids.iter().enumerate() {
        for entry in adjacency.entries_out(id) {
            if let Some(&j) = id_to_idx.get(&entry.neighbor_id) {
                neighbors[i].insert(j);
                neighbors[j].insert(i);
            }
        }
    }

    let mut pairs = Vec::new();

    // Compare all pairs with a neighbor-of-neighbor optimization:
    // only compare nodes that share at least one neighbor
    for u in 0..n {
        let mut candidates: HashSet<usize> = HashSet::new();
        for &v in &neighbors[u] {
            for &w in &neighbors[v] {
                if w > u { candidates.insert(w); }
            }
        }

        for v in candidates {
            let intersection = neighbors[u].intersection(&neighbors[v]).count();
            if intersection == 0 { continue; }
            let union_size = neighbors[u].len() + neighbors[v].len() - intersection;
            if union_size == 0 { continue; }
            let score = intersection as f64 / union_size as f64;
            if score >= threshold {
                pairs.push(SimilarityPair {
                    node_a: node_ids[u],
                    node_b: node_ids[v],
                    score,
                });
            }
        }
    }

    // Sort descending by score, take top_k
    pairs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);

    Ok(pairs)
}
