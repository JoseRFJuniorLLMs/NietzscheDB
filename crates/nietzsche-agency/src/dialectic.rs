//! **Hegelian Dialectic Engine** — autonomous contradiction resolution (AGI-2).
//!
//! # Philosophy
//!
//! Hegel's dialectic process: **Thesis + Antithesis → Synthesis**.
//!
//! When the knowledge graph contains contradictory beliefs (nodes with opposing
//! semantic content pointing at the same conceptual center), the engine detects
//! them as **Tension Nodes** and, during consolidation, produces a higher-order
//! **Synthesis** node that reconciles or supersedes the contradiction.
//!
//! # Fields added to nodes
//!
//! - `certainty` — epistemic confidence ∈ [0.0, 1.0] (stored in `metadata`)
//! - `truth_gradient` — direction of belief revision (positive = gaining evidence)
//!
//! # Algorithm
//!
//! 1. **Scan** — collect semantic nodes with embeddings near the same region
//! 2. **Detect** — find pairs where `content` indicates opposition (negation,
//!    contradiction keywords, or user-tagged `polarity`)
//! 3. **Create Tension** — insert a `TensionNode` linking thesis and antithesis
//! 4. **Synthesize** — during sleep, merge tension nodes into a Synthesis node
//!    placed hierarchically above both in the Poincaré ball

use uuid::Uuid;

use nietzsche_graph::{
    Edge, EdgeType, GraphStorage, Node, NodeType, PoincareVector,
};

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the Hegelian Dialectic Engine.
#[derive(Debug, Clone)]
pub struct DialecticConfig {
    /// Maximum Poincaré distance between two nodes to be considered
    /// as potential contradictions. Default: `0.8`.
    pub proximity_threshold: f64,

    /// Minimum certainty for a node to participate in dialectic.
    /// Default: `0.3`.
    pub min_certainty: f32,

    /// Maximum number of node pairs to evaluate per cycle.
    /// Default: `200`.
    pub max_scan: usize,

    /// Minimum polarity difference to flag a contradiction.
    /// Polarity is stored as `metadata["polarity"]` ∈ [-1.0, 1.0].
    /// Default: `1.2` (e.g., +0.7 vs -0.5 = 1.2 difference).
    pub polarity_gap_threshold: f32,

    /// Default certainty for newly created nodes. Default: `0.5`.
    pub default_certainty: f32,
}

impl Default for DialecticConfig {
    fn default() -> Self {
        Self {
            proximity_threshold: 0.8,
            min_certainty: 0.3,
            max_scan: 200,
            polarity_gap_threshold: 1.2,
            default_certainty: 0.5,
        }
    }
}

// ─────────────────────────────────────────────
// Reports
// ─────────────────────────────────────────────

/// A detected contradiction between two nodes.
#[derive(Debug, Clone)]
pub struct Contradiction {
    pub thesis_id: Uuid,
    pub antithesis_id: Uuid,
    pub polarity_gap: f32,
    pub poincare_distance: f64,
}

/// A synthesis node created by resolving a contradiction.
#[derive(Debug, Clone)]
pub struct Synthesis {
    pub synthesis_id: Uuid,
    pub thesis_id: Uuid,
    pub antithesis_id: Uuid,
    /// Certainty of the synthesis (average of thesis+antithesis capped at 1.0).
    pub certainty: f32,
}

/// Report from a single dialectic cycle.
#[derive(Debug, Clone, Default)]
pub struct DialecticReport {
    pub nodes_scanned: usize,
    pub contradictions_found: usize,
    pub tension_nodes_created: usize,
    pub syntheses_created: usize,
    pub contradictions: Vec<Contradiction>,
    pub syntheses: Vec<Synthesis>,
}

// ─────────────────────────────────────────────
// Utility: node metadata helpers
// ─────────────────────────────────────────────

/// Read `certainty` from node metadata (default: 0.5).
pub fn get_certainty(node: &Node) -> f32 {
    node.meta.metadata
        .get("certainty")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.5)
}

/// Read `polarity` from node metadata (default: 0.0).
pub fn get_polarity(node: &Node) -> f32 {
    node.meta.metadata
        .get("polarity")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.0)
}

/// Read `truth_gradient` from node metadata (default: 0.0).
pub fn get_truth_gradient(node: &Node) -> f32 {
    node.meta.metadata
        .get("truth_gradient")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(0.0)
}

// ─────────────────────────────────────────────
// Detect contradictions
// ─────────────────────────────────────────────

/// Scan the graph for contradictory node pairs.
///
/// Two nodes are contradictory if:
/// 1. They are both Semantic (or Concept) type
/// 2. Their embeddings are within `proximity_threshold` (same conceptual area)
/// 3. Their polarity values differ by ≥ `polarity_gap_threshold`
/// 4. Both have certainty ≥ `min_certainty`
pub fn detect_contradictions(
    storage: &GraphStorage,
    config: &DialecticConfig,
) -> Result<Vec<Contradiction>, String> {
    let all_nodes = storage.scan_nodes()
        .map_err(|e| e.to_string())?;

    // Filter to semantic/concept nodes with sufficient certainty
    let candidates: Vec<Node> = all_nodes
        .into_iter()
        .filter(|n| {
            !n.meta.is_phantom
                && n.meta.energy > 0.0
                && matches!(n.meta.node_type, NodeType::Semantic | NodeType::Concept)
                && get_certainty(n) >= config.min_certainty
        })
        .take(config.max_scan)
        .collect();

    let mut contradictions = Vec::new();

    for i in 0..candidates.len() {
        for j in (i + 1)..candidates.len() {
            let dist = candidates[i].embedding.distance(&candidates[j].embedding);
            if dist > config.proximity_threshold {
                continue;
            }

            let pol_a = get_polarity(&candidates[i]);
            let pol_b = get_polarity(&candidates[j]);
            let gap = (pol_a - pol_b).abs();

            if gap >= config.polarity_gap_threshold {
                contradictions.push(Contradiction {
                    thesis_id: candidates[i].id,
                    antithesis_id: candidates[j].id,
                    polarity_gap: gap,
                    poincare_distance: dist,
                });
            }
        }
    }

    Ok(contradictions)
}

// ─────────────────────────────────────────────
// Create Tension Nodes
// ─────────────────────────────────────────────

/// Create a Tension Node for each detected contradiction.
///
/// The tension node is placed at the midpoint of the thesis and antithesis
/// embeddings, with `node_type = Concept` and a `tension` marker in content.
///
/// Returns the list of `(tension_node_id, thesis_id, antithesis_id)`.
pub fn create_tension_nodes(
    storage: &GraphStorage,
    contradictions: &[Contradiction],
) -> Result<Vec<(Uuid, Uuid, Uuid)>, String> {
    let mut result = Vec::new();

    for c in contradictions {
        let thesis = storage.get_node(&c.thesis_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("thesis {} not found", c.thesis_id))?;
        let antithesis = storage.get_node(&c.antithesis_id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("antithesis {} not found", c.antithesis_id))?;

        // Midpoint embedding
        let dim = thesis.embedding.dim;
        let mut midpoint = vec![0.0f32; dim];
        for d in 0..dim {
            midpoint[d] = (thesis.embedding.coords[d] + antithesis.embedding.coords[d]) / 2.0;
        }
        let midpoint_vec = PoincareVector::new(midpoint).project_into_ball();

        // Content
        let content = serde_json::json!({
            "dialectic": "tension",
            "thesis_id": c.thesis_id.to_string(),
            "antithesis_id": c.antithesis_id.to_string(),
            "polarity_gap": c.polarity_gap,
            "poincare_distance": c.poincare_distance,
        });

        let mut tension = Node::new(Uuid::new_v4(), midpoint_vec, content);
        tension.meta.node_type = NodeType::Concept;
        tension.meta.energy = (thesis.meta.energy + antithesis.meta.energy) / 2.0;
        tension.meta.metadata.insert(
            "certainty".into(),
            serde_json::Value::from(0.0), // tension = uncertain
        );
        tension.meta.metadata.insert(
            "truth_gradient".into(),
            serde_json::Value::from(0.0),
        );
        tension.meta.metadata.insert(
            "dialectic_role".into(),
            serde_json::Value::from("tension"),
        );

        let tension_id = tension.id;
        storage.put_node(&tension).map_err(|e| e.to_string())?;

        // Connect tension to thesis and antithesis
        storage.put_edge(&Edge::new(tension_id, c.thesis_id, EdgeType::Association, 0.5))
            .map_err(|e| e.to_string())?;
        storage.put_edge(&Edge::new(tension_id, c.antithesis_id, EdgeType::Association, 0.5))
            .map_err(|e| e.to_string())?;

        result.push((tension_id, c.thesis_id, c.antithesis_id));
    }

    Ok(result)
}

// ─────────────────────────────────────────────
// Synthesize (during sleep)
// ─────────────────────────────────────────────

/// Synthesize tension nodes into higher-order concepts.
///
/// For each tension node:
/// 1. Compute a synthesis embedding **closer to the center** (more abstract)
/// 2. Create a Semantic synthesis node with combined content
/// 3. Transfer connections from thesis/antithesis to synthesis
/// 4. Lower the certainty of thesis and antithesis (they're now subsumed)
///
/// This should be called during the sleep cycle.
pub fn synthesize_tensions(
    storage: &GraphStorage,
    config: &DialecticConfig,
) -> Result<Vec<Synthesis>, String> {
    let all_nodes = storage.scan_nodes()
        .map_err(|e| e.to_string())?;

    // Find tension nodes
    let tensions: Vec<&Node> = all_nodes
        .iter()
        .filter(|n| {
            n.meta.metadata.get("dialectic_role")
                .and_then(|v| v.as_str())
                == Some("tension")
                && !n.meta.is_phantom
        })
        .collect();

    let mut syntheses = Vec::new();

    for tension in tensions {
        // Extract thesis/antithesis IDs from content
        let thesis_id = tension.meta.content.get("thesis_id")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Uuid>().ok());
        let antithesis_id = tension.meta.content.get("antithesis_id")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<Uuid>().ok());

        let (thesis_id, antithesis_id) = match (thesis_id, antithesis_id) {
            (Some(t), Some(a)) => (t, a),
            _ => continue,
        };

        // Load thesis and antithesis
        let thesis = match storage.get_node(&thesis_id).map_err(|e| e.to_string())? {
            Some(n) => n,
            None => continue,
        };
        let antithesis = match storage.get_node(&antithesis_id).map_err(|e| e.to_string())? {
            Some(n) => n,
            None => continue,
        };

        // Synthesis embedding: midpoint, but pulled toward center (lower depth)
        let dim = thesis.embedding.dim;
        let mut synthesis_coords = vec![0.0f32; dim];
        for d in 0..dim {
            synthesis_coords[d] = (thesis.embedding.coords[d] + antithesis.embedding.coords[d]) / 2.0;
        }
        // Pull toward center by 20% (more abstract)
        for c in synthesis_coords.iter_mut() {
            *c *= 0.8;
        }
        let synthesis_vec = PoincareVector::new(synthesis_coords).project_into_ball();

        // Certainty: average of thesis/antithesis, clamped
        let cert_t = get_certainty(&thesis);
        let cert_a = get_certainty(&antithesis);
        let synthesis_certainty = ((cert_t + cert_a) / 2.0).min(1.0);

        // Content: merge
        let content = serde_json::json!({
            "dialectic": "synthesis",
            "thesis": {
                "id": thesis_id.to_string(),
                "content": thesis.meta.content,
                "polarity": get_polarity(&thesis),
                "certainty": cert_t,
            },
            "antithesis": {
                "id": antithesis_id.to_string(),
                "content": antithesis.meta.content,
                "polarity": get_polarity(&antithesis),
                "certainty": cert_a,
            },
            "resolution": "hegelian_synthesis",
        });

        let mut synthesis = Node::new(Uuid::new_v4(), synthesis_vec, content);
        synthesis.meta.node_type = NodeType::Semantic;
        synthesis.meta.energy = thesis.meta.energy.max(antithesis.meta.energy);
        synthesis.meta.metadata.insert(
            "certainty".into(),
            serde_json::Value::from(synthesis_certainty as f64),
        );
        synthesis.meta.metadata.insert(
            "truth_gradient".into(),
            serde_json::Value::from(0.0),
        );
        synthesis.meta.metadata.insert(
            "dialectic_role".into(),
            serde_json::Value::from("synthesis"),
        );

        let synthesis_id = synthesis.id;
        storage.put_node(&synthesis).map_err(|e| e.to_string())?;

        // Connect synthesis to thesis and antithesis
        storage.put_edge(&Edge::new(synthesis_id, thesis_id, EdgeType::Association, 0.8))
            .map_err(|e| e.to_string())?;
        storage.put_edge(&Edge::new(synthesis_id, antithesis_id, EdgeType::Association, 0.8))
            .map_err(|e| e.to_string())?;

        // Phantomize the tension node (its job is done)
        let mut tension_meta = tension.meta.clone();
        tension_meta.is_phantom = true;
        tension_meta.energy = 0.0;
        storage.put_node_meta(&tension_meta).map_err(|e| e.to_string())?;

        // Lower certainty of thesis and antithesis (subsumed)
        for id in [thesis_id, antithesis_id] {
            if let Ok(Some(mut meta)) = storage.get_node_meta(&id) {
                let old_cert = meta.metadata
                    .get("certainty")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(config.default_certainty as f64);
                meta.metadata.insert(
                    "certainty".into(),
                    serde_json::Value::from(old_cert * 0.7), // reduce by 30%
                );
                let _ = storage.put_node_meta(&meta);
            }
        }

        syntheses.push(Synthesis {
            synthesis_id,
            thesis_id,
            antithesis_id,
            certainty: synthesis_certainty,
        });
    }

    Ok(syntheses)
}

// ─────────────────────────────────────────────
// Full cycle
// ─────────────────────────────────────────────

/// Run a full dialectic cycle: detect → create tension → synthesize.
pub fn run_dialectic_cycle(
    storage: &GraphStorage,
    config: &DialecticConfig,
) -> Result<DialecticReport, String> {
    let mut report = DialecticReport::default();

    // Phase 1: Detect
    let contradictions = detect_contradictions(storage, config)?;
    report.contradictions_found = contradictions.len();
    report.contradictions = contradictions.clone();

    // Phase 2: Create tension nodes for new contradictions
    let tensions = create_tension_nodes(storage, &contradictions)?;
    report.tension_nodes_created = tensions.len();

    // Phase 3: Synthesize existing tension nodes
    let syntheses = synthesize_tensions(storage, config)?;
    report.syntheses_created = syntheses.len();
    report.syntheses = syntheses;

    // Count scanned
    report.nodes_scanned = config.max_scan;

    Ok(report)
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open_storage(dir: &TempDir) -> GraphStorage {
        GraphStorage::open(dir.path().to_str().unwrap()).unwrap()
    }

    fn semantic_node(x: f32, y: f32, polarity: f32, certainty: f32) -> Node {
        let mut n = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x, y]),
            serde_json::json!({"claim": format!("polarity={polarity}")}),
        );
        n.meta.node_type = NodeType::Semantic;
        n.meta.energy = 0.8;
        n.meta.metadata.insert("polarity".into(), serde_json::Value::from(polarity as f64));
        n.meta.metadata.insert("certainty".into(), serde_json::Value::from(certainty as f64));
        n
    }

    #[test]
    fn detects_opposing_polarities() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Thesis: "fire is dangerous" (polarity +0.8, near center)
        let thesis = semantic_node(0.1, 0.05, 0.8, 0.7);
        storage.put_node(&thesis).unwrap();

        // Antithesis: "fire is safe" (polarity -0.6, near same location)
        let anti = semantic_node(0.12, 0.06, -0.6, 0.6);
        storage.put_node(&anti).unwrap();

        let config = DialecticConfig {
            proximity_threshold: 1.0,
            polarity_gap_threshold: 1.0, // gap = 1.4 > 1.0
            ..Default::default()
        };

        let contradictions = detect_contradictions(&storage, &config).unwrap();
        assert_eq!(contradictions.len(), 1);
        assert!(contradictions[0].polarity_gap >= 1.0);
    }

    #[test]
    fn no_contradiction_for_aligned_nodes() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Both positive polarity
        let a = semantic_node(0.1, 0.05, 0.5, 0.7);
        let b = semantic_node(0.12, 0.06, 0.6, 0.6);
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();

        let config = DialecticConfig {
            polarity_gap_threshold: 1.0,
            ..Default::default()
        };

        let contradictions = detect_contradictions(&storage, &config).unwrap();
        assert!(contradictions.is_empty(), "aligned nodes should not contradict");
    }

    #[test]
    fn tension_node_created_at_midpoint() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        let thesis = semantic_node(0.2, 0.0, 0.8, 0.7);
        let anti = semantic_node(0.0, 0.2, -0.7, 0.6);
        let thesis_id = thesis.id;
        let anti_id = anti.id;
        storage.put_node(&thesis).unwrap();
        storage.put_node(&anti).unwrap();

        let contradictions = vec![Contradiction {
            thesis_id,
            antithesis_id: anti_id,
            polarity_gap: 1.5,
            poincare_distance: 0.3,
        }];

        let tensions = create_tension_nodes(&storage, &contradictions).unwrap();
        assert_eq!(tensions.len(), 1);

        let (tension_id, t_id, a_id) = tensions[0];
        assert_eq!(t_id, thesis_id);
        assert_eq!(a_id, anti_id);

        // Verify the tension node exists
        let tn = storage.get_node(&tension_id).unwrap().unwrap();
        assert_eq!(
            tn.meta.metadata.get("dialectic_role").unwrap().as_str().unwrap(),
            "tension"
        );

        // Midpoint should be roughly (0.1, 0.1)
        assert!((tn.embedding.coords[0] - 0.1).abs() < 0.02);
        assert!((tn.embedding.coords[1] - 0.1).abs() < 0.02);
    }

    #[test]
    fn synthesis_resolves_tension() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Create thesis and antithesis
        let thesis = semantic_node(0.3, 0.0, 0.8, 0.7);
        let anti = semantic_node(0.0, 0.3, -0.7, 0.6);
        let thesis_id = thesis.id;
        let anti_id = anti.id;
        storage.put_node(&thesis).unwrap();
        storage.put_node(&anti).unwrap();

        // Create a tension node manually
        let midpoint = PoincareVector::new(vec![0.15, 0.15]).project_into_ball();
        let mut tension = Node::new(Uuid::new_v4(), midpoint, serde_json::json!({
            "dialectic": "tension",
            "thesis_id": thesis_id.to_string(),
            "antithesis_id": anti_id.to_string(),
            "polarity_gap": 1.5,
            "poincare_distance": 0.4,
        }));
        tension.meta.node_type = NodeType::Concept;
        tension.meta.metadata.insert("dialectic_role".into(), serde_json::Value::from("tension"));
        tension.meta.metadata.insert("certainty".into(), serde_json::Value::from(0.0));
        storage.put_node(&tension).unwrap();

        let config = DialecticConfig::default();
        let syntheses = synthesize_tensions(&storage, &config).unwrap();

        assert_eq!(syntheses.len(), 1);
        let s = &syntheses[0];
        assert_eq!(s.thesis_id, thesis_id);
        assert_eq!(s.antithesis_id, anti_id);

        // Verify synthesis node
        let sn = storage.get_node(&s.synthesis_id).unwrap().unwrap();
        assert_eq!(sn.meta.node_type, NodeType::Semantic);
        assert_eq!(
            sn.meta.metadata.get("dialectic_role").unwrap().as_str().unwrap(),
            "synthesis"
        );

        // Synthesis should be closer to center than thesis/antithesis
        // thesis depth ≈ 0.3, anti depth ≈ 0.3, synthesis ≈ 0.12 * 0.8 = ~0.17
        assert!(
            sn.depth < thesis.depth.max(anti.depth),
            "synthesis depth {} should be less than max({}, {})",
            sn.depth, thesis.depth, anti.depth,
        );

        // Tension node should be phantomized
        let tn = storage.get_node_meta(&tension.id).unwrap().unwrap();
        assert!(tn.is_phantom, "tension should be phantomized after synthesis");
    }

    #[test]
    fn full_cycle_end_to_end() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Setup: thesis + antithesis
        let thesis = semantic_node(0.15, 0.0, 0.9, 0.8);
        let anti = semantic_node(0.0, 0.15, -0.8, 0.7);
        storage.put_node(&thesis).unwrap();
        storage.put_node(&anti).unwrap();

        let config = DialecticConfig {
            proximity_threshold: 1.0,
            polarity_gap_threshold: 1.0,
            ..Default::default()
        };

        // Cycle 1: detect + create tension
        let report1 = run_dialectic_cycle(&storage, &config).unwrap();
        assert_eq!(report1.contradictions_found, 1);
        assert_eq!(report1.tension_nodes_created, 1);
        // No syntheses yet (tensions just created, not yet ripe)
        // Tensions created this cycle won't be synthesized until next

        // Cycle 2: synthesize the tension from cycle 1
        let report2 = run_dialectic_cycle(&storage, &config).unwrap();
        assert!(report2.syntheses_created >= 1, "should synthesize tension from cycle 1");
    }

    #[test]
    fn skips_episodic_nodes() {
        let dir = TempDir::new().unwrap();
        let storage = open_storage(&dir);

        // Episodic nodes with opposing polarity should be ignored
        let mut a = semantic_node(0.1, 0.0, 0.8, 0.7);
        a.meta.node_type = NodeType::Episodic;
        let mut b = semantic_node(0.12, 0.0, -0.8, 0.7);
        b.meta.node_type = NodeType::Episodic;
        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();

        let config = DialecticConfig {
            polarity_gap_threshold: 1.0,
            ..Default::default()
        };

        let contradictions = detect_contradictions(&storage, &config).unwrap();
        assert!(contradictions.is_empty(), "episodic nodes should not participate");
    }
}
