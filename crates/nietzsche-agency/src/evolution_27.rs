//! Phase 27 — Epistemic Evolution Analysis.
//!
//! Implements an autonomous knowledge evolution loop within the Agency Engine.
//! Inspired by Karpathy's `autoresearch` pattern, adapted for epistemic
//! mutations on hyperbolic knowledge graphs.
//!
//! ## How it works
//!
//! 1. Sample a set of nodes and evaluate their epistemic quality
//! 2. Identify low-scoring regions (poor hierarchy, low coherence)
//! 3. Propose mutations (new edges, concept nodes, reclassifications)
//! 4. Predict whether the mutation would improve the composite score
//! 5. Emit `AgencyIntent::EpistemicMutation` for the server to apply
//!
//! ## Integration
//!
//! Called from `AgencyEngine::tick()` every `evolution_27_interval` ticks.
//! Produces `AgencyIntent::EpistemicMutation` for each proposed action.
//! The server handles accept/reject via snapshot-based rollback.

use nietzsche_graph::{AdjacencyIndex, GraphStorage};
use nietzsche_epistemics::{
    evaluate_subgraph, EpistemicScore, ScoreWeights,
};
use uuid::Uuid;

// ── Configuration ──────────────────────────────────────────────

/// Configuration for Phase 27 — Epistemic Evolution.
#[derive(Debug, Clone)]
pub struct Evolution27Config {
    /// Maximum nodes to evaluate per tick.
    pub max_eval_nodes: usize,
    /// Minimum epistemic composite score — nodes below this are candidates
    /// for improvement.
    pub quality_floor: f32,
    /// Maximum mutations to propose per tick.
    pub max_proposals: usize,
    /// Minimum energy for nodes to participate in evolution.
    pub min_energy: f32,
    /// Hierarchy consistency threshold — below this, propose reclassification.
    pub hierarchy_threshold: f32,
    /// Coherence threshold — below this, propose new edges.
    pub coherence_threshold: f32,
}

impl Default for Evolution27Config {
    fn default() -> Self {
        Self {
            max_eval_nodes: 500,
            quality_floor: 0.4,
            max_proposals: 5,
            min_energy: 0.05,
            hierarchy_threshold: 0.6,
            coherence_threshold: 0.5,
        }
    }
}

/// Build config from the parent AgencyConfig.
pub fn build_evolution_27_config(cfg: &crate::config::AgencyConfig) -> Evolution27Config {
    Evolution27Config {
        max_eval_nodes: cfg.evolution_27_max_eval,
        quality_floor: cfg.evolution_27_quality_floor,
        max_proposals: cfg.evolution_27_max_proposals,
        min_energy: cfg.evolution_27_min_energy,
        hierarchy_threshold: 0.6,
        coherence_threshold: 0.5,
    }
}

// ── Report ─────────────────────────────────────────────────────

/// Report from Phase 27 epistemic evolution analysis.
#[derive(Debug, Clone)]
pub struct Evolution27Report {
    /// Number of nodes evaluated.
    pub nodes_evaluated: usize,
    /// Current epistemic score of sampled subgraph.
    pub epistemic_score: EpistemicScore,
    /// Number of mutations proposed.
    pub mutations_proposed: usize,
    /// Proposals emitted.
    pub proposals: Vec<EvolutionProposal>,
}

/// A single evolution proposal.
#[derive(Debug, Clone)]
pub struct EvolutionProposal {
    /// Type of mutation.
    pub mutation_type: EvolutionMutationType,
    /// Affected node IDs.
    pub node_ids: Vec<Uuid>,
    /// Reason for the proposal.
    pub reason: String,
    /// Estimated improvement in composite score.
    pub estimated_delta: f32,
}

/// Types of epistemic mutations.
#[derive(Debug, Clone, PartialEq)]
pub enum EvolutionMutationType {
    /// Connect two nodes that should be linked.
    ProposeEdge,
    /// Reclassify a node to a different depth level.
    Reclassify,
    /// Boost energy of an undervalued node.
    EnergyBoost,
    /// Prune a low-quality edge.
    PruneEdge,
}

// ── Main scan function ─────────────────────────────────────────

/// Run the epistemic evolution analysis.
///
/// Scans the graph, evaluates quality, and proposes improvements.
pub fn run_evolution_27_scan(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    config: &Evolution27Config,
) -> Evolution27Report {
    // 1. Sample nodes with sufficient energy
    let node_ids = sample_active_nodes(storage, config.max_eval_nodes, config.min_energy);

    if node_ids.is_empty() {
        return Evolution27Report {
            nodes_evaluated: 0,
            epistemic_score: EpistemicScore {
                hierarchy_consistency: 1.0,
                coherence: 1.0,
                coverage: 0.0,
                redundancy: 0.0,
                energy_avg: 0.0,
                hausdorff_avg: 1.0,
                node_count: 0,
                composite: 0.0,
            },
            mutations_proposed: 0,
            proposals: Vec::new(),
        };
    }

    // 2. Evaluate epistemic quality
    let weights = ScoreWeights::default();
    let score = evaluate_subgraph(storage, adjacency, &node_ids, &weights);

    // 3. Identify weaknesses and propose mutations
    let mut proposals = Vec::new();

    // 3a. If hierarchy consistency is low, find bad edges
    if score.hierarchy_consistency < config.hierarchy_threshold {
        let hierarchy_proposals = propose_hierarchy_fixes(
            storage, adjacency, &node_ids, config.max_proposals,
        );
        proposals.extend(hierarchy_proposals);
    }

    // 3b. If coherence is low, propose connecting distant neighbors
    if score.coherence < config.coherence_threshold && proposals.len() < config.max_proposals {
        let coherence_proposals = propose_coherence_edges(
            storage, adjacency, &node_ids,
            config.max_proposals - proposals.len(),
        );
        proposals.extend(coherence_proposals);
    }

    // 3c. Boost undervalued nodes (high connectivity but low energy)
    if proposals.len() < config.max_proposals {
        let energy_proposals = propose_energy_boosts(
            storage, adjacency, &node_ids,
            config.max_proposals - proposals.len(),
        );
        proposals.extend(energy_proposals);
    }

    // Cap proposals
    proposals.truncate(config.max_proposals);
    let mutations_proposed = proposals.len();

    Evolution27Report {
        nodes_evaluated: node_ids.len(),
        epistemic_score: score,
        mutations_proposed,
        proposals,
    }
}

// ── Internal helpers ───────────────────────────────────────────

fn sample_active_nodes(
    storage: &GraphStorage,
    max_nodes: usize,
    min_energy: f32,
) -> Vec<Uuid> {
    let mut ids = Vec::new();
    for result in storage.iter_nodes_meta() {
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.energy >= min_energy && !meta.is_phantom {
            ids.push(meta.id);
            if ids.len() >= max_nodes {
                break;
            }
        }
    }
    ids
}

/// Find edges where parent.depth > child.depth (hierarchy violation)
/// and propose reclassification.
fn propose_hierarchy_fixes(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
    max_proposals: usize,
) -> Vec<EvolutionProposal> {
    let mut proposals = Vec::new();

    for nid in node_ids {
        if proposals.len() >= max_proposals {
            break;
        }

        let src_meta = match storage.get_node_meta(nid) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        for nbr_id in adjacency.neighbors_out(nid) {
            let dst_meta = match storage.get_node_meta(&nbr_id) {
                Ok(Some(m)) => m,
                _ => continue,
            };

            // Hierarchy violation: source is deeper than destination
            if src_meta.depth > dst_meta.depth + 0.1 {
                proposals.push(EvolutionProposal {
                    mutation_type: EvolutionMutationType::Reclassify,
                    node_ids: vec![*nid, nbr_id],
                    reason: format!(
                        "Hierarchy violation: {:.3} → {:.3} (should be ascending)",
                        src_meta.depth, dst_meta.depth,
                    ),
                    estimated_delta: 0.02,
                });

                if proposals.len() >= max_proposals {
                    break;
                }
            }
        }
    }

    proposals
}

/// Find nodes with high degree but few edges to their
/// nearest embedding neighbors — propose connecting them.
fn propose_coherence_edges(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
    max_proposals: usize,
) -> Vec<EvolutionProposal> {
    let mut proposals = Vec::new();

    // Simple heuristic: find pairs of nodes that are close in depth
    // but not connected.
    for (i, nid_a) in node_ids.iter().enumerate() {
        if proposals.len() >= max_proposals {
            break;
        }

        let meta_a = match storage.get_node_meta(nid_a) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        let neighbors_a: std::collections::HashSet<Uuid> =
            adjacency.neighbors_both(nid_a).into_iter().collect();

        // Check a few subsequent nodes (limited scan)
        for j in (i + 1)..node_ids.len().min(i + 20) {
            let nid_b = &node_ids[j];
            if neighbors_a.contains(nid_b) {
                continue; // already connected
            }

            let meta_b = match storage.get_node_meta(nid_b) {
                Ok(Some(m)) => m,
                _ => continue,
            };

            let depth_diff = (meta_a.depth - meta_b.depth).abs();
            if depth_diff < 0.15 && meta_a.energy > 0.2 && meta_b.energy > 0.2 {
                proposals.push(EvolutionProposal {
                    mutation_type: EvolutionMutationType::ProposeEdge,
                    node_ids: vec![*nid_a, *nid_b],
                    reason: format!(
                        "Similar depth ({:.3} ≈ {:.3}) but unconnected — coherence improvement",
                        meta_a.depth, meta_b.depth,
                    ),
                    estimated_delta: 0.01,
                });
                break; // one proposal per source node
            }
        }
    }

    proposals
}

/// Find nodes with high connectivity but disproportionately low energy.
fn propose_energy_boosts(
    storage: &GraphStorage,
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
    max_proposals: usize,
) -> Vec<EvolutionProposal> {
    let mut proposals = Vec::new();

    for nid in node_ids {
        if proposals.len() >= max_proposals {
            break;
        }

        let meta = match storage.get_node_meta(nid) {
            Ok(Some(m)) => m,
            _ => continue,
        };

        let degree = adjacency.neighbors_both(nid).len();

        // High degree (> 5) but low energy (< 0.2) → undervalued hub
        if degree > 5 && meta.energy < 0.2 {
            proposals.push(EvolutionProposal {
                mutation_type: EvolutionMutationType::EnergyBoost,
                node_ids: vec![*nid],
                reason: format!(
                    "Undervalued hub: degree={} energy={:.3} — should be higher",
                    degree, meta.energy,
                ),
                estimated_delta: 0.005,
            });
        }
    }

    proposals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = Evolution27Config::default();
        assert_eq!(cfg.max_eval_nodes, 500);
        assert_eq!(cfg.max_proposals, 5);
        assert!(cfg.quality_floor > 0.0);
    }
}
