//! **Schrödinger Edges** — probabilistic edges with Markov transition probabilities.
//!
//! An association between "Apple" and "Isaac Newton" should not be fixed; it
//! should have a probability of existing that depends on the state of the query.
//! Edges are "superpositions" that collapse only at MATCH time.
//!
//! # How it works
//!
//! Each edge can optionally have a `probability` field in its metadata:
//! ```json
//! { "probability": 0.7, "decay_rate": 0.01, "context_boost": "physics" }
//! ```
//!
//! - `probability` ∈ [0.0, 1.0] — base transition probability
//! - `decay_rate` — per-tick probability decay (edges fade if unused)
//! - `context_boost` — optional context tag; if the query context matches,
//!   the probability is boosted
//!
//! During MATCH/traversal, each Schrödinger edge is "collapsed": a random
//! sample determines whether the edge exists for this particular query.
//! This models the brain's context-dependent associations.

use rand::Rng;
use uuid::Uuid;

use crate::model::Edge;

// ─────────────────────────────────────────────
// Bloch state (local copy — avoids circular dep on nietzsche-agency)
// ─────────────────────────────────────────────

/// Quantum state on the Bloch sphere.
///
/// This is a lightweight, local definition compatible with
/// `nietzsche_agency::quantum::BlochState`. We duplicate it here because
/// `nietzsche-agency` already depends on `nietzsche-graph`, so adding the
/// reverse dependency would create a cycle.
#[derive(Debug, Clone)]
pub struct BlochState {
    /// Polar angle theta in [0, pi].
    pub theta: f64,
    /// Azimuthal angle phi in [0, 2*pi).
    pub phi: f64,
    /// State purity in [0, 1] (1 = pure, 0 = maximally mixed).
    pub purity: f64,
    /// Bloch vector (x, y, z) with ||v|| = purity.
    pub vector: [f64; 3],
}

impl BlochState {
    /// Create from polar coordinates + purity.
    pub fn new(theta: f64, phi: f64, purity: f64) -> Self {
        let p = purity.clamp(0.0, 1.0);
        let vector = [
            p * theta.sin() * phi.cos(),
            p * theta.sin() * phi.sin(),
            p * theta.cos(),
        ];
        Self { theta, phi, purity: p, vector }
    }

    /// Quantum fidelity between two states.
    /// For pure states: F = cos^2(alpha/2) where alpha is the Bloch vector angle.
    pub fn fidelity(&self, other: &BlochState) -> f64 {
        let dot = self.vector[0] * other.vector[0]
            + self.vector[1] * other.vector[1]
            + self.vector[2] * other.vector[2];
        let norm_a = bloch_vec_norm(&self.vector);
        let norm_b = bloch_vec_norm(&other.vector);

        if norm_a < 1e-12 || norm_b < 1e-12 {
            return 0.5; // Maximally mixed
        }

        let cos_angle = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
        (1.0 + cos_angle) / 2.0
    }
}

fn bloch_vec_norm(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Entanglement proxy: average fidelity between two groups of Bloch states.
/// High fidelity means strong coupling between groups.
///
/// This is a local copy of `nietzsche_agency::quantum::entanglement_proxy`
/// to avoid a circular crate dependency.
pub fn entanglement_proxy(group_a: &[BlochState], group_b: &[BlochState]) -> f64 {
    if group_a.is_empty() || group_b.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0u64;
    for a in group_a {
        for b in group_b {
            total += a.fidelity(b);
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { total / count as f64 }
}

// ─────────────────────────────────────────────
// Schrödinger Edge
// ─────────────────────────────────────────────

/// Probabilistic edge wrapper around a standard `Edge`.
#[derive(Debug, Clone)]
pub struct SchrodingerEdge {
    /// The underlying edge.
    pub edge: Edge,
    /// Base probability ∈ [0.0, 1.0]. Default: 1.0 (deterministic).
    pub probability: f32,
    /// Per-tick decay rate. Default: 0.0 (no decay).
    pub decay_rate: f32,
    /// Optional context tag for probability boosting.
    pub context_boost: Option<String>,
    /// Boost multiplier when context matches. Default: 1.5.
    pub boost_factor: f32,
}

impl SchrodingerEdge {
    /// Extract probabilistic parameters from an edge's metadata.
    pub fn from_edge(edge: &Edge) -> Self {
        let probability = edge.metadata.get("probability")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;
        let decay_rate = edge.metadata.get("decay_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let context_boost = edge.metadata.get("context_boost")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let boost_factor = edge.metadata.get("boost_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.5) as f32;

        Self {
            edge: edge.clone(),
            probability,
            decay_rate,
            context_boost,
            boost_factor,
        }
    }

    /// Compute the effective probability given an optional query context.
    pub fn effective_probability(&self, context: Option<&str>) -> f32 {
        let mut p = self.probability;

        // Context boost
        if let (Some(boost_ctx), Some(query_ctx)) = (&self.context_boost, context) {
            if query_ctx.contains(boost_ctx.as_str()) {
                p = (p * self.boost_factor).min(1.0);
            }
        }

        p.clamp(0.0, 1.0)
    }

    /// Collapse the superposition: decide whether this edge "exists" for
    /// the current query.
    ///
    /// Returns `true` if the edge collapses into existence.
    pub fn collapse(&self, context: Option<&str>, rng: &mut impl Rng) -> bool {
        let p = self.effective_probability(context);
        rng.gen::<f32>() < p
    }

    /// Apply per-tick decay to the probability.
    ///
    /// Returns the new probability after decay.
    pub fn decay(&mut self) -> f32 {
        self.probability = (self.probability - self.decay_rate).max(0.0);
        self.probability
    }

    /// Reinforce the edge (increase probability after successful use).
    pub fn reinforce(&mut self, amount: f32) {
        self.probability = (self.probability + amount).min(1.0);
    }

    /// Quantum-enhanced collapse using the entanglement proxy.
    ///
    /// If context `BlochState`s are highly entangled (high fidelity) with the
    /// target state, the edge is forced to materialise -- analogous to
    /// observing one half of an entangled pair forcing the other to collapse.
    ///
    /// When entanglement is below the threshold the method falls back to the
    /// classical probabilistic collapse with the given context string and rng.
    pub fn collapse_with_entanglement(
        &self,
        context_states: &[BlochState],
        target_state: &BlochState,
        entanglement_threshold: f64,
        context: Option<&str>,
        rng: &mut impl Rng,
    ) -> bool {
        let target_group = [target_state.clone()];
        let entanglement = entanglement_proxy(context_states, &target_group);

        if entanglement > entanglement_threshold {
            // Quantum physics dictates: observing one forces the other to materialise
            true
        } else {
            // Fall back to classical probabilistic collapse
            self.collapse(context, rng)
        }
    }

    /// Write the probabilistic state back into the edge's metadata.
    pub fn write_to_edge(&self) -> Edge {
        let mut edge = self.edge.clone();
        edge.metadata.insert(
            "probability".into(),
            serde_json::json!(self.probability),
        );
        edge.metadata.insert(
            "decay_rate".into(),
            serde_json::json!(self.decay_rate),
        );
        if let Some(ctx) = &self.context_boost {
            edge.metadata.insert(
                "context_boost".into(),
                serde_json::json!(ctx),
            );
        }
        edge.metadata.insert(
            "boost_factor".into(),
            serde_json::json!(self.boost_factor),
        );
        edge
    }
}

// ─────────────────────────────────────────────
// Batch operations
// ─────────────────────────────────────────────

/// Filter a set of edges by probabilistic collapse.
///
/// Returns only the edges that "exist" for this particular query context.
pub fn collapse_edges(
    edges: &[Edge],
    context: Option<&str>,
    rng: &mut impl Rng,
) -> Vec<Edge> {
    edges
        .iter()
        .filter(|e| {
            let se = SchrodingerEdge::from_edge(e);
            se.collapse(context, rng)
        })
        .cloned()
        .collect()
}

/// Batch entanglement-aware collapse: for each edge, check whether the
/// corresponding target state is entangled with the query context states.
///
/// `edges` and `target_states` must be the same length — each edge is paired
/// with its target node's `BlochState`. Edges whose target is highly
/// entangled with the context are forced to materialise; the rest fall back
/// to classical probabilistic collapse.
///
/// Returns a `Vec<bool>` aligned with `edges` (true = edge exists).
pub fn collapse_edges_with_entanglement(
    edges: &[Edge],
    context_states: &[BlochState],
    target_states: &[BlochState],
    entanglement_threshold: f64,
    context: Option<&str>,
    rng: &mut impl Rng,
) -> Vec<bool> {
    assert_eq!(
        edges.len(),
        target_states.len(),
        "edges and target_states must have the same length"
    );

    edges
        .iter()
        .zip(target_states.iter())
        .map(|(edge, target)| {
            let se = SchrodingerEdge::from_edge(edge);
            se.collapse_with_entanglement(
                context_states,
                target,
                entanglement_threshold,
                context,
                rng,
            )
        })
        .collect()
}

/// Apply decay to all probabilistic edges.
///
/// Returns `(edge_id, new_probability)` for edges that were decayed.
pub fn decay_all_edges(edges: &[Edge]) -> Vec<(Uuid, Edge)> {
    let mut updates = Vec::new();

    for edge in edges {
        if edge.metadata.contains_key("probability") {
            let mut se = SchrodingerEdge::from_edge(edge);
            let old_p = se.probability;
            se.decay();
            if (se.probability - old_p).abs() > f32::EPSILON {
                updates.push((edge.id, se.write_to_edge()));
            }
        }
    }

    updates
}

/// Create a new edge with Schrödinger probabilistic metadata.
pub fn create_probabilistic_edge(
    from: Uuid,
    to: Uuid,
    edge_type: crate::model::EdgeType,
    weight: f32,
    probability: f32,
    decay_rate: f32,
    context_boost: Option<&str>,
) -> Edge {
    let mut edge = Edge::new(from, to, edge_type, weight);
    edge.metadata.insert("probability".into(), serde_json::json!(probability));
    edge.metadata.insert("decay_rate".into(), serde_json::json!(decay_rate));
    if let Some(ctx) = context_boost {
        edge.metadata.insert("context_boost".into(), serde_json::json!(ctx));
    }
    edge
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::EdgeType;
    use rand::SeedableRng;

    fn test_edge(prob: f32) -> Edge {
        let mut e = Edge::new(Uuid::new_v4(), Uuid::new_v4(), EdgeType::Association, 1.0);
        e.metadata.insert("probability".into(), serde_json::json!(prob));
        e
    }

    #[test]
    fn deterministic_edge_always_collapses() {
        let edge = test_edge(1.0);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            assert!(se.collapse(None, &mut rng), "p=1.0 should always collapse");
        }
    }

    #[test]
    fn zero_probability_never_collapses() {
        let edge = test_edge(0.0);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..100 {
            assert!(!se.collapse(None, &mut rng), "p=0.0 should never collapse");
        }
    }

    #[test]
    fn partial_probability_collapses_sometimes() {
        let edge = test_edge(0.5);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let collapses: usize = (0..1000)
            .filter(|_| se.collapse(None, &mut rng))
            .count();

        // With p=0.5, expect ~500 ± 50
        assert!(collapses > 400 && collapses < 600,
            "p=0.5 should collapse ~50% of the time, got {collapses}/1000");
    }

    #[test]
    fn context_boost_increases_probability() {
        let mut edge = test_edge(0.4);
        edge.metadata.insert("context_boost".into(), serde_json::json!("physics"));
        edge.metadata.insert("boost_factor".into(), serde_json::json!(2.0));

        let se = SchrodingerEdge::from_edge(&edge);

        let p_no_ctx = se.effective_probability(None);
        let p_wrong_ctx = se.effective_probability(Some("biology"));
        let p_right_ctx = se.effective_probability(Some("quantum physics"));

        assert!((p_no_ctx - 0.4).abs() < 1e-6);
        assert!((p_wrong_ctx - 0.4).abs() < 1e-6);
        assert!((p_right_ctx - 0.8).abs() < 1e-6, "physics context should boost to 0.8");
    }

    #[test]
    fn decay_reduces_probability() {
        let mut edge = test_edge(0.5);
        edge.metadata.insert("decay_rate".into(), serde_json::json!(0.1));

        let mut se = SchrodingerEdge::from_edge(&edge);
        se.decay();
        assert!((se.probability - 0.4).abs() < 1e-6);
        se.decay();
        assert!((se.probability - 0.3).abs() < 1e-5);
    }

    #[test]
    fn decay_floors_at_zero() {
        let mut edge = test_edge(0.05);
        edge.metadata.insert("decay_rate".into(), serde_json::json!(0.1));

        let mut se = SchrodingerEdge::from_edge(&edge);
        se.decay();
        assert_eq!(se.probability, 0.0);
    }

    #[test]
    fn reinforce_increases_probability() {
        let edge = test_edge(0.6);
        let mut se = SchrodingerEdge::from_edge(&edge);
        se.reinforce(0.2);
        assert!((se.probability - 0.8).abs() < 1e-6);
    }

    #[test]
    fn reinforce_caps_at_one() {
        let edge = test_edge(0.9);
        let mut se = SchrodingerEdge::from_edge(&edge);
        se.reinforce(0.5);
        assert_eq!(se.probability, 1.0);
    }

    #[test]
    fn write_to_edge_preserves_state() {
        let mut edge = test_edge(0.7);
        edge.metadata.insert("decay_rate".into(), serde_json::json!(0.05));
        edge.metadata.insert("context_boost".into(), serde_json::json!("math"));

        let se = SchrodingerEdge::from_edge(&edge);
        let written = se.write_to_edge();

        let se2 = SchrodingerEdge::from_edge(&written);
        assert!((se2.probability - 0.7).abs() < 1e-6);
        assert!((se2.decay_rate - 0.05).abs() < 1e-6);
        assert_eq!(se2.context_boost.as_deref(), Some("math"));
    }

    #[test]
    fn collapse_edges_filters_batch() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let edges: Vec<Edge> = (0..10)
            .map(|_| test_edge(0.5))
            .collect();

        let collapsed = collapse_edges(&edges, None, &mut rng);
        // With p=0.5, expect roughly 5 (±3)
        assert!(collapsed.len() >= 2 && collapsed.len() <= 8,
            "expected ~5 collapsed, got {}", collapsed.len());
    }

    #[test]
    fn create_probabilistic_edge_has_metadata() {
        let edge = create_probabilistic_edge(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeType::Association,
            1.0,
            0.7,
            0.02,
            Some("science"),
        );

        let se = SchrodingerEdge::from_edge(&edge);
        assert!((se.probability - 0.7).abs() < 1e-6);
        assert!((se.decay_rate - 0.02).abs() < 1e-6);
        assert_eq!(se.context_boost.as_deref(), Some("science"));
    }

    // ── Entanglement-based collapse tests ───────────────────────────

    #[test]
    fn bloch_fidelity_same_state_is_one() {
        let s = BlochState::new(1.0, 0.5, 1.0);
        assert!((s.fidelity(&s) - 1.0).abs() < 0.01);
    }

    #[test]
    fn bloch_fidelity_opposite_poles_near_zero() {
        let zero = BlochState::new(0.0, 0.0, 1.0); // north pole
        let one = BlochState::new(std::f64::consts::PI, 0.0, 1.0); // south pole
        assert!(zero.fidelity(&one) < 0.01, "opposite poles should have fidelity ~0");
    }

    #[test]
    fn entanglement_proxy_identical_groups_high() {
        let states = vec![
            BlochState::new(0.5, 0.5, 1.0),
            BlochState::new(0.6, 0.5, 1.0),
        ];
        let e = entanglement_proxy(&states, &states);
        assert!(e > 0.8, "identical groups should be highly entangled, got {e}");
    }

    #[test]
    fn entanglement_proxy_empty_is_zero() {
        let states = vec![BlochState::new(0.5, 0.5, 1.0)];
        assert_eq!(entanglement_proxy(&[], &states), 0.0);
        assert_eq!(entanglement_proxy(&states, &[]), 0.0);
    }

    #[test]
    fn entanglement_proxy_orthogonal_groups_low() {
        let group_a = vec![BlochState::new(0.0, 0.0, 1.0)]; // north pole
        let group_b = vec![BlochState::new(std::f64::consts::PI, 0.0, 1.0)]; // south pole
        let e = entanglement_proxy(&group_a, &group_b);
        assert!(e < 0.1, "orthogonal groups should have low entanglement, got {e}");
    }

    #[test]
    fn collapse_with_entanglement_forces_true_when_entangled() {
        // Edge with p=0.0 should NEVER collapse classically
        let edge = test_edge(0.0);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        // Context and target are the same state → fidelity = 1.0
        let context = vec![BlochState::new(0.5, 1.0, 1.0)];
        let target = BlochState::new(0.5, 1.0, 1.0);

        for _ in 0..50 {
            assert!(
                se.collapse_with_entanglement(&context, &target, 0.5, None, &mut rng),
                "high entanglement should force collapse even with p=0.0"
            );
        }
    }

    #[test]
    fn collapse_with_entanglement_falls_back_when_low() {
        // Edge with p=0.0 should never collapse when entanglement is also low
        let edge = test_edge(0.0);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        // North pole vs south pole → fidelity ~0
        let context = vec![BlochState::new(0.0, 0.0, 1.0)];
        let target = BlochState::new(std::f64::consts::PI, 0.0, 1.0);

        for _ in 0..50 {
            assert!(
                !se.collapse_with_entanglement(&context, &target, 0.5, None, &mut rng),
                "low entanglement + p=0 should never collapse"
            );
        }
    }

    #[test]
    fn collapse_with_entanglement_respects_threshold() {
        let edge = test_edge(0.0);
        let se = SchrodingerEdge::from_edge(&edge);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Two nearby states — fidelity is high but less than 1.0
        let context = vec![BlochState::new(0.5, 1.0, 1.0)];
        let target = BlochState::new(0.6, 1.0, 1.0);

        let entanglement = entanglement_proxy(&context, &[target.clone()]);

        // With a threshold above the actual entanglement → should NOT force collapse
        let high_threshold = entanglement + 0.1;
        assert!(
            !se.collapse_with_entanglement(&context, &target, high_threshold, None, &mut rng),
            "threshold above entanglement should fall back to classical (p=0)"
        );

        // With a threshold below the actual entanglement → should force collapse
        let low_threshold = entanglement - 0.1;
        assert!(
            se.collapse_with_entanglement(&context, &target, low_threshold, None, &mut rng),
            "threshold below entanglement should force collapse"
        );
    }

    #[test]
    fn batch_collapse_with_entanglement_forces_entangled() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // All edges have p=0 (would never collapse classically)
        let edges: Vec<Edge> = (0..3).map(|_| test_edge(0.0)).collect();

        // Same state for context and all targets → entanglement = 1.0
        let context = vec![BlochState::new(0.5, 1.0, 1.0)];
        let targets = vec![
            BlochState::new(0.5, 1.0, 1.0),
            BlochState::new(0.5, 1.0, 1.0),
            BlochState::new(0.5, 1.0, 1.0),
        ];

        let results = collapse_edges_with_entanglement(
            &edges, &context, &targets, 0.5, None, &mut rng,
        );

        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|&r| r), "all edges should be forced to collapse");
    }

    #[test]
    fn batch_collapse_mixed_entanglement() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // p=0 edges
        let edges: Vec<Edge> = (0..2).map(|_| test_edge(0.0)).collect();

        let context = vec![BlochState::new(0.5, 1.0, 1.0)];
        let targets = vec![
            BlochState::new(0.5, 1.0, 1.0),                             // high entanglement
            BlochState::new(std::f64::consts::PI, 0.0, 1.0),            // low entanglement (opposite pole)
        ];

        let results = collapse_edges_with_entanglement(
            &edges, &context, &targets, 0.5, None, &mut rng,
        );

        assert_eq!(results.len(), 2);
        assert!(results[0], "first edge should collapse (high entanglement)");
        assert!(!results[1], "second edge should NOT collapse (low entanglement + p=0)");
    }

    #[test]
    #[should_panic(expected = "edges and target_states must have the same length")]
    fn batch_collapse_panics_on_mismatched_lengths() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let edges: Vec<Edge> = (0..3).map(|_| test_edge(0.5)).collect();
        let context = vec![BlochState::new(0.5, 1.0, 1.0)];
        let targets = vec![BlochState::new(0.5, 1.0, 1.0)]; // only 1, but 3 edges

        collapse_edges_with_entanglement(&edges, &context, &targets, 0.5, None, &mut rng);
    }
}
