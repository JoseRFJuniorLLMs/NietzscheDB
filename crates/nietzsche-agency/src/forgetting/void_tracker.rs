//! Void Tracker — Maps the abysses left by the Forgetting Engine.
//!
//! CAMADA 3 of the Nezhmetdinov architecture: the Voids left by deleted nodes
//! become coordinates for generative creation. Where memory dies, new hypotheses
//! can be born.
//!
//! ## Design
//!
//! Each deleted node leaves a "Void" — its Poincaré coordinates, depth, and
//! structural context. These voids are the seeds for the VQ-VAE Decoder
//! (future implementation) to generate new knowledge.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A void left by a deleted node — the coordinates of absence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoidCoordinate {
    /// Unique ID for this void.
    pub void_id: Uuid,
    /// The original node that was deleted.
    pub original_node_id: Uuid,
    /// Poincaré ball coordinates where the node existed.
    pub poincare_coords: Vec<f32>,
    /// Depth in the hierarchy (norm of Poincaré coords).
    pub depth: f32,
    /// The cycle in which the node was deleted.
    pub creation_cycle: u64,
    /// IDs of neighbors that the deleted node was connected to.
    /// These surviving neighbors define the "context" of the void.
    pub surviving_neighbor_ids: Vec<Uuid>,
    /// Whether this void has been consumed by a generative process.
    pub consumed: bool,
    /// Structural plausibility score: how "interesting" is this void
    /// as a seed for new knowledge? Higher = more potential.
    pub plausibility: f32,
}

impl VoidCoordinate {
    /// Create from deletion context.
    pub fn from_deletion(
        original_node_id: Uuid,
        poincare_coords: Vec<f32>,
        creation_cycle: u64,
        surviving_neighbor_ids: Vec<Uuid>,
    ) -> Self {
        let depth = poincare_coords.iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;

        // Plausibility heuristic: voids with more surviving neighbors
        // and at intermediate depths are more interesting.
        let neighbor_factor = (surviving_neighbor_ids.len() as f32).min(10.0) / 10.0;
        let depth_factor = 1.0 - (depth - 0.5).abs() * 2.0; // peaks at depth 0.5
        let plausibility = (neighbor_factor * 0.6 + depth_factor.max(0.0) * 0.4).clamp(0.0, 1.0);

        Self {
            void_id: Uuid::new_v4(),
            original_node_id,
            poincare_coords,
            depth,
            creation_cycle,
            surviving_neighbor_ids,
            consumed: false,
            plausibility,
        }
    }

    /// Mark this void as consumed by a generative process.
    pub fn consume(&mut self) {
        self.consumed = true;
    }
}

/// Tracks all voids across cycles and provides seeds for generation.
#[derive(Debug, Clone, Default)]
pub struct VoidTracker {
    /// All tracked voids (including consumed ones for history).
    pub voids: Vec<VoidCoordinate>,
    /// Maximum age (in cycles) before a void expires.
    pub max_void_age: u64,
}

impl VoidTracker {
    pub fn new(max_void_age: u64) -> Self {
        Self {
            voids: Vec::new(),
            max_void_age,
        }
    }

    /// Record a new void from a deletion.
    pub fn record_void(&mut self, void: VoidCoordinate) {
        self.voids.push(void);
    }

    /// Get unconsumed voids sorted by plausibility (highest first).
    pub fn available_voids(&self) -> Vec<&VoidCoordinate> {
        let mut available: Vec<&VoidCoordinate> = self.voids.iter()
            .filter(|v| !v.consumed)
            .collect();
        available.sort_by(|a, b| b.plausibility.partial_cmp(&a.plausibility).unwrap_or(std::cmp::Ordering::Equal));
        available
    }

    /// Get the top N most plausible unconsumed voids.
    pub fn top_voids(&self, n: usize) -> Vec<&VoidCoordinate> {
        self.available_voids().into_iter().take(n).collect()
    }

    /// Expire old voids that have aged beyond the maximum.
    pub fn expire_old_voids(&mut self, current_cycle: u64) {
        self.voids.retain(|v| {
            v.consumed || (current_cycle - v.creation_cycle) <= self.max_void_age
        });
    }

    /// Mark a void as consumed. Returns false if not found.
    pub fn consume_void(&mut self, void_id: &Uuid) -> bool {
        if let Some(v) = self.voids.iter_mut().find(|v| v.void_id == *void_id) {
            v.consumed = true;
            true
        } else {
            false
        }
    }

    /// Total unconsumed voids.
    pub fn available_count(&self) -> usize {
        self.voids.iter().filter(|v| !v.consumed).count()
    }

    /// Total voids ever recorded.
    pub fn total_count(&self) -> usize {
        self.voids.len()
    }

    /// Compute the centroid of available voids (for TGC intrinsic analysis).
    ///
    /// Returns None if no voids are available.
    pub fn void_centroid(&self) -> Option<Vec<f32>> {
        let available: Vec<&VoidCoordinate> = self.voids.iter()
            .filter(|v| !v.consumed)
            .collect();

        if available.is_empty() {
            return None;
        }

        let dim = available[0].poincare_coords.len();
        let mut centroid = vec![0.0f32; dim];
        let n = available.len() as f32;

        for v in &available {
            for (i, coord) in v.poincare_coords.iter().enumerate() {
                if i < dim {
                    centroid[i] += coord / n;
                }
            }
        }

        Some(centroid)
    }

    /// Mean plausibility of available voids.
    pub fn mean_plausibility(&self) -> f32 {
        let available: Vec<f32> = self.voids.iter()
            .filter(|v| !v.consumed)
            .map(|v| v.plausibility)
            .collect();

        if available.is_empty() {
            return 0.0;
        }

        available.iter().sum::<f32>() / available.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_void(cycle: u64, depth: f32, neighbors: usize) -> VoidCoordinate {
        let neighbor_ids: Vec<Uuid> = (0..neighbors).map(|_| Uuid::new_v4()).collect();
        VoidCoordinate::from_deletion(
            Uuid::new_v4(),
            vec![depth * 0.7, depth * 0.3],
            cycle,
            neighbor_ids,
        )
    }

    #[test]
    fn void_creation_and_tracking() {
        let mut tracker = VoidTracker::new(100);
        let void = make_void(1, 0.5, 5);
        tracker.record_void(void);

        assert_eq!(tracker.total_count(), 1);
        assert_eq!(tracker.available_count(), 1);
    }

    #[test]
    fn void_consumption() {
        let mut tracker = VoidTracker::new(100);
        let void = make_void(1, 0.5, 5);
        let vid = void.void_id;
        tracker.record_void(void);

        assert!(tracker.consume_void(&vid));
        assert_eq!(tracker.available_count(), 0);
        assert_eq!(tracker.total_count(), 1);
    }

    #[test]
    fn void_expiration() {
        let mut tracker = VoidTracker::new(10);
        tracker.record_void(make_void(1, 0.5, 3));
        tracker.record_void(make_void(5, 0.5, 3));
        tracker.record_void(make_void(15, 0.5, 3));

        tracker.expire_old_voids(20);
        // Void from cycle 1: age 19 > 10, expired
        // Void from cycle 5: age 15 > 10, expired
        // Void from cycle 15: age 5 <= 10, kept
        assert_eq!(tracker.available_count(), 1);
    }

    #[test]
    fn plausibility_sorting() {
        let mut tracker = VoidTracker::new(100);
        // Low plausibility: shallow, no neighbors
        tracker.record_void(make_void(1, 0.05, 0));
        // High plausibility: medium depth, many neighbors
        tracker.record_void(make_void(1, 0.5, 8));

        let top = tracker.top_voids(1);
        assert_eq!(top.len(), 1);
        assert!(top[0].plausibility > 0.3);
    }

    #[test]
    fn void_centroid() {
        let mut tracker = VoidTracker::new(100);
        tracker.record_void(VoidCoordinate::from_deletion(
            Uuid::new_v4(), vec![0.1, 0.2], 1, vec![],
        ));
        tracker.record_void(VoidCoordinate::from_deletion(
            Uuid::new_v4(), vec![0.3, 0.4], 1, vec![],
        ));

        let centroid = tracker.void_centroid().unwrap();
        assert!((centroid[0] - 0.2).abs() < 0.01);
        assert!((centroid[1] - 0.3).abs() < 0.01);
    }
}
