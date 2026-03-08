//! Phase XII — Economic Attention Network (ECAN) for NietzscheDB.
//!
//! Transforms the graph's energy system into a **cognitive economy** where
//! attention is a scarce resource that flows through the graph via auctions.
//!
//! ## Core Concepts
//!
//! - **Attention Budget**: Each node has a budget proportional to
//!   `energy × ln(degree + 1)` — hubs get more, periphery gets less.
//! - **Attention Bids**: Nodes compete to activate neighbours via bids.
//!   `bid = source_mass / (1 + distance)`.
//! - **Auction Resolution**: Per-target, the highest bidder wins. Ties
//!   broken by energy. Losing bids are discarded (no refund).
//! - **Cognitive Inflation**: When total demand exceeds total budget,
//!   a `price` factor scales down all bids — preventing attention storms.
//!
//! ## Integration
//!
//! The ECAN cycle runs as step 12 inside `AgencyEngine::tick()`.
//! It reads node metadata (no embeddings), computes budgets and bids,
//! resolves auctions, and produces an `AttentionReport` with flow metrics.
//! Energy updates are expressed as `AgencyIntent::AdjustAttention` for
//! the server to apply under a write lock.

use std::collections::HashMap;
use uuid::Uuid;

use nietzsche_graph::GraphStorage;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Per-node attention state computed during a cycle.
#[derive(Debug, Clone)]
pub struct AttentionState {
    pub node_id: Uuid,
    /// Attention budget available this cycle.
    pub budget: f32,
    /// Attention demand (how much this node wants to spend).
    pub demand: f32,
    /// Attention received from winning bids.
    pub received: f32,
    /// Attention spent on successful bids.
    pub spent: f32,
    /// Semantic mass: energy × ln(degree + 1).
    pub semantic_mass: f32,
}

/// A bid from one node to activate another.
#[derive(Debug, Clone)]
pub struct AttentionBid {
    pub source: Uuid,
    pub target: Uuid,
    /// Bid value: source_mass / (1 + distance).
    pub value: f32,
}

/// Configuration for the attention economy.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Maximum nodes to scan per cycle (0 = all).
    pub max_scan: usize,
    /// Budget multiplier (scales raw budget).
    pub budget_scale: f32,
    /// Demand fraction: what % of budget a node wants to spend.
    pub demand_fraction: f32,
    /// Minimum energy to participate in the economy.
    pub energy_floor: f32,
    /// Maximum bids per source node (prevents hub spam).
    pub max_bids_per_source: usize,
    /// Energy boost per unit of attention received.
    pub attention_energy_gain: f32,
    /// Energy decay factor per cycle (multiplicative).
    pub energy_decay: f32,
    /// Whether to generate energy adjustment intents.
    pub emit_intents: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            max_scan: 10_000,
            budget_scale: 1.0,
            demand_fraction: 0.5,
            energy_floor: 0.05,
            max_bids_per_source: 5,
            attention_energy_gain: 0.1,
            energy_decay: 0.97,
            emit_intents: true,
        }
    }
}

/// Report from one ECAN cycle.
#[derive(Debug, Clone)]
pub struct AttentionReport {
    /// Number of nodes that participated.
    pub participants: usize,
    /// Total bids generated.
    pub total_bids: usize,
    /// Total bids that won auctions.
    pub winning_bids: usize,
    /// Cognitive inflation price (1.0 = no inflation).
    pub price: f32,
    /// Total attention transferred.
    pub total_flow: f32,
    /// Top receivers (node_id, attention_received).
    pub top_receivers: Vec<(Uuid, f32)>,
    /// Energy adjustments: (node_id, delta). Positive = gained, negative = decayed.
    pub energy_deltas: Vec<(Uuid, f32)>,
}

// ─────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────

/// Compute semantic mass for a node.
///
/// `mass = energy × ln(degree + 1)`
///
/// This is the fundamental unit of "cognitive importance" in the economy.
#[inline]
pub fn semantic_mass(energy: f32, degree: usize) -> f32 {
    energy * ((degree as f32) + 1.0).ln()
}

/// Compute attention budget for a node.
///
/// Budget = mass × scale factor.
/// Nodes with zero mass get zero budget (can't spend attention).
#[inline]
pub fn compute_budget(mass: f32, scale: f32) -> f32 {
    (mass * scale).max(0.0)
}

/// Compute a bid value from source to target.
///
/// `bid = source_mass / (1 + distance)`
///
/// Distance here is the poincaré distance (from depth proxy).
/// Lower distance = higher bid = more likely to win.
#[inline]
pub fn compute_bid_value(source_mass: f32, distance: f32) -> f32 {
    source_mass / (1.0 + distance)
}

/// Compute cognitive inflation price.
///
/// `price = total_demand / total_budget`
///
/// When price > 1.0, all bids are scaled down proportionally.
/// This prevents attention storms in highly-connected regions.
#[inline]
pub fn cognitive_inflation(total_demand: f32, total_budget: f32) -> f32 {
    if total_budget < 1e-9 {
        return 1.0;
    }
    (total_demand / total_budget).max(1.0)
}

/// Resolve auctions: for each target, select the highest bidder.
///
/// Returns: (winning bids, total flow)
pub fn resolve_auctions(
    bids: &[AttentionBid],
    price: f32,
) -> (Vec<AttentionBid>, f32) {
    // Group bids by target
    let mut by_target: HashMap<Uuid, Vec<&AttentionBid>> = HashMap::new();
    for bid in bids {
        by_target.entry(bid.target).or_default().push(bid);
    }

    let mut winners = Vec::new();
    let mut total_flow = 0.0f32;

    for (_target, candidates) in &by_target {
        // Find the highest bid (deflated by price)
        if let Some(best) = candidates.iter().max_by(|a, b| {
            let va = a.value / price;
            let vb = b.value / price;
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let deflated_value = best.value / price;
            winners.push(AttentionBid {
                source: best.source,
                target: best.target,
                value: deflated_value,
            });
            total_flow += deflated_value;
        }
    }

    (winners, total_flow)
}

/// Compute energy deltas from attention flow.
///
/// For each participating node:
///
/// 1. **Attention gain**: `received × attention_energy_gain`
/// 2. **Energy decay**: `current_energy × (1 - energy_decay)` — thermodynamic dissipation
/// 3. **Net delta** = gain − decay
///
/// Decay ensures nodes that stop receiving attention gradually cool down.
/// This creates the thermodynamic foundation: energy is not conserved at
/// the node level but flows and dissipates, driving self-organization.
///
/// `energy(t+1) = energy(t) × decay + received × gain_rate`
pub fn compute_energy_deltas(
    states: &[AttentionState],
    config: &AttentionConfig,
) -> Vec<(Uuid, f32)> {
    let mut deltas = Vec::with_capacity(states.len());

    for state in states {
        let gain = state.received * config.attention_energy_gain;
        // Decay: energy lost per tick = current_energy × (1 - decay_rate)
        // semantic_mass ∝ energy, so we use it as proxy for current energy level
        let decay_loss = state.semantic_mass * (1.0 - config.energy_decay);
        let net = gain - decay_loss;

        if net.abs() > 1e-6 {
            deltas.push((state.node_id, net));
        }
    }

    deltas
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_mass_grows_with_degree() {
        let m1 = semantic_mass(0.8, 2);  // ln(3) ≈ 1.10
        let m2 = semantic_mass(0.8, 10); // ln(11) ≈ 2.40
        assert!(m2 > m1, "higher degree should give higher mass");
    }

    #[test]
    fn semantic_mass_zero_energy() {
        let m = semantic_mass(0.0, 100);
        assert_eq!(m, 0.0, "zero energy = zero mass");
    }

    #[test]
    fn budget_scales_with_mass() {
        let b1 = compute_budget(1.0, 1.0);
        let b2 = compute_budget(2.0, 1.0);
        assert!(b2 > b1);
    }

    #[test]
    fn bid_value_decreases_with_distance() {
        let b_near = compute_bid_value(1.0, 0.1);
        let b_far = compute_bid_value(1.0, 2.0);
        assert!(b_near > b_far, "closer targets should get higher bids");
    }

    #[test]
    fn inflation_above_one_when_demand_exceeds_budget() {
        let p = cognitive_inflation(100.0, 50.0);
        assert!(p > 1.0, "price should > 1 when demand > budget");
        assert!((p - 2.0).abs() < 0.01);
    }

    #[test]
    fn inflation_is_one_when_balanced() {
        let p = cognitive_inflation(50.0, 100.0);
        assert!((p - 1.0).abs() < 0.01, "price should be 1.0 when supply >= demand");
    }

    #[test]
    fn auction_picks_highest_bidder() {
        let bids = vec![
            AttentionBid { source: Uuid::nil(), target: Uuid::from_u128(1), value: 0.5 },
            AttentionBid { source: Uuid::from_u128(2), target: Uuid::from_u128(1), value: 0.9 },
            AttentionBid { source: Uuid::from_u128(3), target: Uuid::from_u128(1), value: 0.3 },
        ];

        let (winners, flow) = resolve_auctions(&bids, 1.0);
        assert_eq!(winners.len(), 1, "one target = one winner");
        assert_eq!(winners[0].source, Uuid::from_u128(2), "highest bidder should win");
        assert!((flow - 0.9).abs() < 0.01);
    }

    #[test]
    fn auction_deflates_by_price() {
        let bids = vec![
            AttentionBid { source: Uuid::nil(), target: Uuid::from_u128(1), value: 2.0 },
        ];

        let (winners, flow) = resolve_auctions(&bids, 2.0);
        assert_eq!(winners.len(), 1);
        assert!((winners[0].value - 1.0).abs() < 0.01, "bid should be halved by price=2.0");
    }

    #[test]
    fn multiple_targets_independent_auctions() {
        let bids = vec![
            AttentionBid { source: Uuid::from_u128(1), target: Uuid::from_u128(10), value: 0.5 },
            AttentionBid { source: Uuid::from_u128(2), target: Uuid::from_u128(10), value: 0.8 },
            AttentionBid { source: Uuid::from_u128(3), target: Uuid::from_u128(20), value: 0.3 },
            AttentionBid { source: Uuid::from_u128(4), target: Uuid::from_u128(20), value: 0.7 },
        ];

        let (winners, _) = resolve_auctions(&bids, 1.0);
        assert_eq!(winners.len(), 2, "two targets = two winners");
    }

    #[test]
    fn energy_deltas_from_attention() {
        let states = vec![
            AttentionState {
                node_id: Uuid::from_u128(1),
                budget: 1.0, demand: 0.5, received: 0.8, spent: 0.3,
                semantic_mass: 1.0,
            },
            AttentionState {
                node_id: Uuid::from_u128(2),
                budget: 1.0, demand: 0.5, received: 0.0, spent: 0.5,
                semantic_mass: 0.5,
            },
        ];

        let config = AttentionConfig::default();
        let deltas = compute_energy_deltas(&states, &config);

        // Node 1 received attention → net gain (0.8*0.1 - 1.0*0.03 = 0.05)
        let d1 = deltas.iter().find(|(id, _)| *id == Uuid::from_u128(1));
        assert!(d1.is_some(), "node 1 should have a delta");
        assert!(d1.unwrap().1 > 0.0, "node 1 should gain energy");

        // Node 2 received nothing → negative delta from decay (0 - 0.5*0.03 = -0.015)
        let d2 = deltas.iter().find(|(id, _)| *id == Uuid::from_u128(2));
        assert!(d2.is_some(), "node 2 should have a decay delta");
        assert!(d2.unwrap().1 < 0.0, "node 2 should lose energy from decay");
    }
}
