//! Production rules for the L-System fractal growth engine.
//!
//! A [`ProductionRule`] pairs a [`RuleCondition`] (evaluated per node)
//! with a [`RuleAction`] (applied when the condition is met).
//!
//! ## Semantics
//!
//! On every [`crate::engine::LSystemEngine::tick`] call, each live node is
//! tested against the rule list **in order**. The **first matching rule
//! wins** (deterministic, no rule stacking). Prune rules derived from the
//! global Hausdorff check are applied after user-defined rules.
//!
//! ## Built-in conditions
//!
//! | Condition            | Meaning                                             |
//! |----------------------|-----------------------------------------------------|
//! | `EnergyAbove(t)`     | `node.energy > t` — active, high-energy node        |
//! | `EnergyBelow(t)`     | `node.energy < t` — fading node                     |
//! | `DepthAbove(t)`      | `node.depth > t` — specific / episodic              |
//! | `DepthBelow(t)`      | `node.depth < t` — abstract / semantic              |
//! | `HausdorffAbove(t)`  | local D > t — over-complex neighbourhood            |
//! | `HausdorffBelow(t)`  | local D < t — under-complex neighbourhood           |
//! | `GenerationBelow(n)` | `node.lsystem_generation < n` — growth depth limit  |
//! | `And(a, b)`          | both conditions hold                                |
//! | `Or(a, b)`           | either condition holds                              |

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ─────────────────────────────────────────────
// Condition
// ─────────────────────────────────────────────

/// Guard evaluated against a node's current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleCondition {
    EnergyAbove(f32),
    EnergyBelow(f32),
    DepthAbove(f32),
    DepthBelow(f32),
    HausdorffAbove(f32),
    HausdorffBelow(f32),
    /// Only apply the rule to nodes below generation `n`
    /// (prevents unbounded recursive growth).
    GenerationBelow(u32),
    And(Box<RuleCondition>, Box<RuleCondition>),
    Or(Box<RuleCondition>, Box<RuleCondition>),
    Not(Box<RuleCondition>),
    /// Always fires (useful as a catch-all rule).
    Always,
}

impl RuleCondition {
    /// Convenience: fires when energy > t AND generation < max_gen.
    pub fn growth(energy_threshold: f32, max_gen: u32) -> Self {
        Self::And(
            Box::new(Self::EnergyAbove(energy_threshold)),
            Box::new(Self::GenerationBelow(max_gen)),
        )
    }

    /// Convenience: fires when node is a good pruning candidate.
    pub fn prunable(energy_lo: f32) -> Self {
        Self::EnergyBelow(energy_lo)
    }
}

// ─────────────────────────────────────────────
// Action
// ─────────────────────────────────────────────

/// Mutation applied to the graph when a rule fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    /// Spawn a child node deeper in the Poincaré ball.
    ///
    /// `depth_offset` ∈ (0, 0.3] — hyperbolic displacement toward boundary.
    SpawnChild {
        depth_offset: f64,
        weight:       f32,
        #[serde(default)]
        content:      JsonValue,
    },

    /// Spawn a sibling node at the same depth, offset by `angle` radians.
    ///
    /// `distance` ∈ (0, 0.3] — hyperbolic displacement.
    SpawnSibling {
        angle:    f64,
        distance: f64,
        weight:   f32,
        #[serde(default)]
        content:  JsonValue,
    },

    /// Soft-delete the node (energy → 0, edge_type → Pruned).
    Prune,

    /// Adjust the node's energy by `delta` (clamped to [0, 1]).
    UpdateEnergy { delta: f32 },
}

// ─────────────────────────────────────────────
// Production Rule
// ─────────────────────────────────────────────

/// A single L-System production rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionRule {
    /// Human-readable name used in WAL entries and edge metadata.
    pub name:      String,
    /// Guard: evaluated against each node.
    pub condition: RuleCondition,
    /// Mutation applied when the condition is met.
    pub action:    RuleAction,
}

impl ProductionRule {
    pub fn new(name: impl Into<String>, condition: RuleCondition, action: RuleAction) -> Self {
        Self { name: name.into(), condition, action }
    }

    /// Predefined rule: high-energy nodes spawn a child deeper in the ball.
    pub fn growth_child(name: impl Into<String>, max_gen: u32) -> Self {
        Self::new(
            name,
            RuleCondition::growth(0.7, max_gen),
            RuleAction::SpawnChild {
                depth_offset: 0.08,
                weight:       0.8,
                content:      JsonValue::Null,
            },
        )
    }

    /// Predefined rule: lateral association at 90°.
    pub fn lateral_association(name: impl Into<String>, max_gen: u32) -> Self {
        Self::new(
            name,
            RuleCondition::growth(0.6, max_gen),
            RuleAction::SpawnSibling {
                angle:    std::f64::consts::FRAC_PI_2, // 90°
                distance: 0.06,
                weight:   0.6,
                content:  JsonValue::Null,
            },
        )
    }

    /// Predefined rule: prune fading nodes.
    pub fn prune_fading(name: impl Into<String>, threshold: f32) -> Self {
        Self::new(
            name,
            RuleCondition::prunable(threshold),
            RuleAction::Prune,
        )
    }
}

// ─────────────────────────────────────────────
// Condition evaluation
// ─────────────────────────────────────────────

/// Evaluate `condition` against a node's observable fields.
///
/// Called by the engine — not intended for direct use.
pub fn check_condition(
    energy:      f32,
    depth:       f32,
    hausdorff:   f32,
    generation:  u32,
    condition:   &RuleCondition,
) -> bool {
    match condition {
        RuleCondition::EnergyAbove(t)    => energy > *t,
        RuleCondition::EnergyBelow(t)    => energy < *t,
        RuleCondition::DepthAbove(t)     => depth > *t,
        RuleCondition::DepthBelow(t)     => depth < *t,
        RuleCondition::HausdorffAbove(t) => hausdorff > *t,
        RuleCondition::HausdorffBelow(t) => hausdorff < *t,
        RuleCondition::GenerationBelow(n)=> generation < *n,
        RuleCondition::And(a, b) => {
            check_condition(energy, depth, hausdorff, generation, a)
                && check_condition(energy, depth, hausdorff, generation, b)
        }
        RuleCondition::Or(a, b) => {
            check_condition(energy, depth, hausdorff, generation, a)
                || check_condition(energy, depth, hausdorff, generation, b)
        }
        RuleCondition::Not(c) => !check_condition(energy, depth, hausdorff, generation, c),
        RuleCondition::Always => true,
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check(energy: f32, depth: f32, hausdorff: f32, gen: u32, cond: &RuleCondition) -> bool {
        check_condition(energy, depth, hausdorff, gen, cond)
    }

    #[test]
    fn energy_above_threshold() {
        assert!( check(0.8, 0.5, 1.0, 0, &RuleCondition::EnergyAbove(0.7)));
        assert!(!check(0.5, 0.5, 1.0, 0, &RuleCondition::EnergyAbove(0.7)));
    }

    #[test]
    fn generation_below_limit() {
        assert!( check(1.0, 0.5, 1.0, 2, &RuleCondition::GenerationBelow(5)));
        assert!(!check(1.0, 0.5, 1.0, 5, &RuleCondition::GenerationBelow(5)));
    }

    #[test]
    fn and_requires_both() {
        let cond = RuleCondition::And(
            Box::new(RuleCondition::EnergyAbove(0.5)),
            Box::new(RuleCondition::DepthBelow(0.8)),
        );
        assert!( check(0.9, 0.3, 1.0, 0, &cond));
        assert!(!check(0.3, 0.3, 1.0, 0, &cond)); // energy too low
        assert!(!check(0.9, 0.9, 1.0, 0, &cond)); // depth too high
    }

    #[test]
    fn or_requires_either() {
        let cond = RuleCondition::Or(
            Box::new(RuleCondition::EnergyAbove(0.8)),
            Box::new(RuleCondition::HausdorffBelow(0.4)),
        );
        assert!( check(0.9, 0.5, 1.0, 0, &cond)); // energy match
        assert!( check(0.3, 0.5, 0.3, 0, &cond)); // hausdorff match
        assert!(!check(0.3, 0.5, 1.0, 0, &cond)); // neither
    }

    #[test]
    fn not_inverts() {
        let cond = RuleCondition::Not(Box::new(RuleCondition::EnergyBelow(0.5)));
        assert!( check(0.8, 0.0, 1.0, 0, &cond));
        assert!(!check(0.2, 0.0, 1.0, 0, &cond));
    }

    #[test]
    fn always_fires() {
        assert!(check(0.0, 0.0, 0.0, 999, &RuleCondition::Always));
    }

    #[test]
    fn predefined_growth_child_rule() {
        let rule = ProductionRule::growth_child("grow", 3);
        // High energy, gen < 3 → fires
        assert!(check(0.9, 0.5, 1.0, 2, &rule.condition));
        // Too old
        assert!(!check(0.9, 0.5, 1.0, 3, &rule.condition));
    }
}
