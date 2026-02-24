//! Causal Immunity — Minkowski Spacetime Edge Protection.
//!
//! From the Nezhmetdinov spec:
//! "Se apagar esta memória for destruir a lógica de um raciocínio que veio
//! antes ou depois dela (Cadeia de Minkowski), o Carrasco guarda a lâmina
//! e poupa a memória. O espaço-tempo é sagrado."
//!
//! κ(n) = count of Minkowski timelike/lightlike edges incident to node n.
//! If κ(n) > 0, the node is part of a verified causal chain and is IMMUNE
//! to deletion regardless of its vitality score.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Causal immunity status for a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalStatus {
    /// κ(n) = 0: No causal chains. Node can be deleted if other conditions met.
    Unanchored,
    /// κ(n) > 0: Part of at least one causal chain. IMMUNE to deletion.
    CausallyAnchored {
        /// Number of timelike edges.
        timelike_count: usize,
        /// Number of lightlike edges.
        lightlike_count: usize,
    },
}

impl CausalStatus {
    /// Total causal edge count κ(n).
    pub fn kappa(&self) -> usize {
        match self {
            CausalStatus::Unanchored => 0,
            CausalStatus::CausallyAnchored { timelike_count, lightlike_count } => {
                timelike_count + lightlike_count
            }
        }
    }

    /// Is this node immune to deletion?
    pub fn is_immune(&self) -> bool {
        self.kappa() > 0
    }
}

/// Causal chain analysis for a set of nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CausalAnalysis {
    /// Total nodes analyzed.
    pub total_nodes: usize,
    /// Nodes with causal immunity (κ > 0).
    pub immune_count: usize,
    /// Nodes without causal anchoring (κ = 0).
    pub unanchored_count: usize,
    /// Total timelike edges found.
    pub total_timelike: usize,
    /// Total lightlike edges found.
    pub total_lightlike: usize,
    /// Total spacelike edges (causally independent).
    pub total_spacelike: usize,
    /// Fraction of nodes with immunity.
    pub immunity_ratio: f32,
}

impl CausalAnalysis {
    /// Build from a list of causal statuses.
    pub fn from_statuses(statuses: &[CausalStatus]) -> Self {
        let total_nodes = statuses.len();
        let mut immune = 0usize;
        let mut unanchored = 0usize;
        let mut total_tl = 0usize;
        let mut total_ll = 0usize;

        for s in statuses {
            match s {
                CausalStatus::Unanchored => unanchored += 1,
                CausalStatus::CausallyAnchored { timelike_count, lightlike_count } => {
                    immune += 1;
                    total_tl += timelike_count;
                    total_ll += lightlike_count;
                }
            }
        }

        Self {
            total_nodes,
            immune_count: immune,
            unanchored_count: unanchored,
            total_timelike: total_tl,
            total_lightlike: total_ll,
            total_spacelike: 0, // Would need edge data
            immunity_ratio: if total_nodes > 0 {
                immune as f32 / total_nodes as f32
            } else {
                0.0
            },
        }
    }
}

/// Determines if a Minkowski interval classifies an edge as causal.
///
/// ds² = -c²Δt² + ||Δx||²
/// - Timelike: ds² < 0 (source caused target)
/// - Lightlike: ds² ≈ 0 (light cone boundary)
/// - Spacelike: ds² > 0 (causally independent)
pub fn classify_minkowski_interval(ds_squared: f32) -> &'static str {
    const LIGHTLIKE_EPSILON: f32 = 0.01;
    if ds_squared < -LIGHTLIKE_EPSILON {
        "timelike"
    } else if ds_squared.abs() <= LIGHTLIKE_EPSILON {
        "lightlike"
    } else {
        "spacelike"
    }
}

/// Check if a Minkowski interval represents a causal relationship.
pub fn is_causal_edge(ds_squared: f32) -> bool {
    const LIGHTLIKE_EPSILON: f32 = 0.01;
    ds_squared <= LIGHTLIKE_EPSILON // timelike or lightlike
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unanchored_is_not_immune() {
        let status = CausalStatus::Unanchored;
        assert_eq!(status.kappa(), 0);
        assert!(!status.is_immune());
    }

    #[test]
    fn anchored_is_immune() {
        let status = CausalStatus::CausallyAnchored {
            timelike_count: 3,
            lightlike_count: 1,
        };
        assert_eq!(status.kappa(), 4);
        assert!(status.is_immune());
    }

    #[test]
    fn analysis_from_mixed_statuses() {
        let statuses = vec![
            CausalStatus::Unanchored,
            CausalStatus::CausallyAnchored { timelike_count: 2, lightlike_count: 0 },
            CausalStatus::Unanchored,
            CausalStatus::CausallyAnchored { timelike_count: 1, lightlike_count: 1 },
            CausalStatus::Unanchored,
        ];
        let analysis = CausalAnalysis::from_statuses(&statuses);
        assert_eq!(analysis.total_nodes, 5);
        assert_eq!(analysis.immune_count, 2);
        assert_eq!(analysis.unanchored_count, 3);
        assert_eq!(analysis.total_timelike, 3);
        assert_eq!(analysis.total_lightlike, 1);
        assert!((analysis.immunity_ratio - 0.4).abs() < 0.01);
    }

    #[test]
    fn minkowski_classification() {
        assert_eq!(classify_minkowski_interval(-1.0), "timelike");
        assert_eq!(classify_minkowski_interval(0.0), "lightlike");
        assert_eq!(classify_minkowski_interval(1.0), "spacelike");
    }

    #[test]
    fn causal_edge_detection() {
        assert!(is_causal_edge(-1.0));
        assert!(is_causal_edge(0.0));
        assert!(!is_causal_edge(1.0));
    }
}
