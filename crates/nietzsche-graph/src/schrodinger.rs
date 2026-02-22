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
}
