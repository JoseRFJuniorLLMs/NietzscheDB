//! Coverage metric.
//!
//! Measures how well the embeddings utilize the Poincaré ball space.
//! High coverage = concepts are well-distributed across depth levels
//! and angular sectors. Low coverage = everything clustered in one region.

use nietzsche_graph::GraphStorage;
use uuid::Uuid;

/// Number of radial bins for coverage estimation.
const RADIAL_BINS: usize = 10;

/// Compute radial coverage: how uniformly nodes are distributed across depths.
///
/// Returns [0.0, 1.0] where 1.0 = perfectly uniform distribution across
/// depth levels, 0.0 = all nodes at the same depth.
pub fn radial_coverage(
    storage: &GraphStorage,
    node_ids: &[Uuid],
) -> f32 {
    if node_ids.is_empty() {
        return 0.0;
    }

    let mut bins = [0u32; RADIAL_BINS];

    for nid in node_ids {
        let meta = match storage.get_node_meta(nid) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        // Depth ∈ [0, 1), map to bin index
        let bin = ((meta.depth * RADIAL_BINS as f32) as usize).min(RADIAL_BINS - 1);
        bins[bin] += 1;
    }

    // Compute uniformity via normalized entropy
    let total: u32 = bins.iter().sum();
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0f64;
    for &count in &bins {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }

    // Max entropy = ln(RADIAL_BINS)
    let max_entropy = (RADIAL_BINS as f64).ln();
    if max_entropy == 0.0 {
        return 1.0;
    }

    (entropy / max_entropy) as f32
}

/// Compute energy distribution: how uniformly energy is distributed.
///
/// Returns [0.0, 1.0] where 1.0 = all nodes have similar energy,
/// 0.0 = extreme energy inequality.
pub fn energy_coverage(
    storage: &GraphStorage,
    node_ids: &[Uuid],
) -> f32 {
    if node_ids.is_empty() {
        return 0.0;
    }

    let mut energies = Vec::with_capacity(node_ids.len());
    for nid in node_ids {
        let meta = match storage.get_node_meta(nid) {
            Ok(Some(m)) => m,
            _ => continue,
        };
        energies.push(meta.energy);
    }

    if energies.is_empty() {
        return 0.0;
    }

    let mean = energies.iter().sum::<f32>() / energies.len() as f32;
    if mean == 0.0 {
        return 0.0;
    }

    // Coefficient of variation (lower = more uniform)
    let variance = energies.iter()
        .map(|&e| (e - mean).powi(2))
        .sum::<f32>() / energies.len() as f32;
    let std_dev = variance.sqrt();
    let cv = std_dev / mean;

    // Convert CV to a 0-1 score (CV of 0 = perfect, CV > 1 = poor)
    (1.0 - cv.min(1.0)).max(0.0)
}

/// Combined coverage score.
pub fn combined_coverage(
    storage: &GraphStorage,
    node_ids: &[Uuid],
) -> f32 {
    let radial = radial_coverage(storage, node_ids);
    let energy = energy_coverage(storage, node_ids);
    (radial + energy) / 2.0
}
