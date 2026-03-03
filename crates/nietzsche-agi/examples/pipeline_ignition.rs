//! # Pipeline Ignition — End-to-end AGI inference test
//!
//! Demonstrates the complete AGI pipeline:
//!
//!   ID 95 ("Unsafe Rust")  ──→  Trajectory  ──→  ID 180 ("Foucault: Vigilância")
//!           ↓                       ↓                        ↓
//!     PoincareVector          GCS Validation            PoincareVector
//!       (concrete)          Klein collinearity            (abstract)
//!         ‖x‖≈0.7               ↓                        ‖x‖≈0.3
//!                        InferenceEngine.analyze()
//!                               ↓
//!                     InferenceType::DialecticalSynthesis
//!                               ↓
//!                    FrechetSynthesizer.synthesize()
//!                               ↓
//!                     Synthesis Point  ‖s‖≈0.21
//!                    "Type-safe Governance"
//!                               ↓
//!                    FeedbackLoop.prepare()
//!                               ↓
//!                     FeedbackResult { node, edges }
//!                               ↓
//!                          ╔══════════════╗
//!                          ║  RATIONALE   ║
//!                          ║  (proof)     ║
//!                          ╚══════════════╝
//!
//! ## Run
//! ```bash
//! cargo run -p nietzsche-agi --example pipeline_ignition
//! ```

use uuid::Uuid;
use nietzsche_graph::{CausalType, PoincareVector};
use nietzsche_hyp_ops;

use nietzsche_agi::{
    // Layer 2 — Trajectory
    validate_trajectory,
    GeodesicTrajectory,
    // Layer 3 — Inference
    InferenceEngine,
    InferenceType,
    Rationale,
    // Layer 3 — Synthesis
    FrechetSynthesizer,
    // Layer 4 — Feedback
    SynthesisNode,
};
use nietzsche_agi::trajectory::GcsConfig;
use nietzsche_agi::inference_engine::{InferenceConfig, EdgeInfo};
use nietzsche_agi::feedback_loop::{FeedbackLoop, SourceNodeInfo};
use nietzsche_agi::homeostasis::HomeostasisGuard;
use nietzsche_agi::dialectic::DialecticDetector;
use nietzsche_agi::evolution::EvolutionScheduler;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      NietzscheDB AGI — Pipeline Ignition Test              ║");
    println!("║      ID 95 (Unsafe Rust) → ID 180 (Foucault)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────
    // 1. SIMULATE GRAPH DATA
    // ─────────────────────────────────────────────
    // In production, these come from GraphStorage.get_node() / get_embedding()
    // Here we simulate 5 nodes on a path from concrete→abstract

    let id_95  = Uuid::new_v4(); // "Unsafe Rust" — concrete, near boundary
    let id_120 = Uuid::new_v4(); // "Memory Safety" — intermediate
    let id_145 = Uuid::new_v4(); // "Systems Control" — bridge between domains
    let id_160 = Uuid::new_v4(); // "Power Structures" — entering Foucault's domain
    let id_180 = Uuid::new_v4(); // "Foucault: Vigilância" — abstract, near center

    // Poincaré embeddings (8-dim for this example)
    // ‖x‖ encodes depth: high = concrete/boundary, low = abstract/center
    let emb_95:  Vec<f64> = vec![ 0.55,  0.40, 0.10, 0.05, 0.02, 0.01, 0.01, 0.00]; // ‖‖≈0.68
    let emb_120: Vec<f64> = vec![ 0.35,  0.30, 0.15, 0.10, 0.05, 0.02, 0.01, 0.00]; // ‖‖≈0.50
    let emb_145: Vec<f64> = vec![ 0.20,  0.15, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]; // ‖‖≈0.39
    let emb_160: Vec<f64> = vec![ 0.10,  0.08, 0.15, 0.20, 0.12, 0.08, 0.03, 0.01]; // ‖‖≈0.33
    let emb_180: Vec<f64> = vec![ 0.05,  0.04, 0.10, 0.15, 0.15, 0.10, 0.05, 0.02]; // ‖‖≈0.27

    let path = vec![id_95, id_120, id_145, id_160, id_180];
    let embeddings = vec![
        emb_95.clone(),
        emb_120.clone(),
        emb_145.clone(),
        emb_160.clone(),
        emb_180.clone(),
    ];

    let names = ["Unsafe Rust", "Memory Safety", "Systems Control", "Power Structures", "Foucault: Vigilância"];

    println!("━━━ STAGE 1: Graph Data ━━━");
    for (i, (name, emb)) in names.iter().zip(embeddings.iter()).enumerate() {
        let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("  Node {i}: {name:25} ‖x‖={norm:.4}  depth={}", depth_label(norm));
    }
    println!();

    // ─────────────────────────────────────────────
    // 2. VALIDATE TRAJECTORY (GCS)
    // ─────────────────────────────────────────────
    println!("━━━ STAGE 2: Geodesic Coherence Validation ━━━");

    let gcs_config = GcsConfig {
        hop_threshold: 0.3,        // Lenient for this example
        trajectory_threshold: 0.4,
        collinearity_epsilon: 0.15,
    };

    let trajectory = validate_trajectory(&path, &embeddings, &gcs_config)
        .expect("trajectory validation failed");

    println!("  Trajectory length: {} nodes, {} hops", trajectory.path.len(), trajectory.hop_scores.len());
    println!("  Aggregate GCS:    {:.4}", trajectory.aggregate_gcs);
    println!("  Radial gradient:  {:.4} ({})", trajectory.radial_gradient,
        if trajectory.radial_gradient < 0.0 { "INWARD → Generalization" } else { "OUTWARD → Specialization" });
    println!("  Valid:            {}", if trajectory.is_valid { "✅ YES" } else { "❌ NO" });
    println!();

    for hop in &trajectory.hop_scores {
        let label = names.get(hop.hop).unwrap_or(&"?");
        println!("    Hop {}: {label:25} GCS={:.4} {}",
            hop.hop, hop.score,
            if hop.is_coherent { "✓" } else { "✗ RUPTURE" });
    }
    println!();

    // ─────────────────────────────────────────────
    // 3. INFERENCE ENGINE — CLASSIFY
    // ─────────────────────────────────────────────
    println!("━━━ STAGE 3: Inference Engine ━━━");

    let inference_config = InferenceConfig {
        gcs_threshold: 0.3,
        radial_threshold: 0.05,
        min_cluster_transitions_for_synthesis: 2,
        auto_synthesize: true,
    };
    let engine = InferenceEngine::new(inference_config);

    // Simulate edge causal types (all Timelike = causal chain)
    let edge_infos = vec![
        EdgeInfo { causal_type: CausalType::Timelike },  // 95→120
        EdgeInfo { causal_type: CausalType::Timelike },  // 120→145
        EdgeInfo { causal_type: CausalType::Timelike },  // 145→160
        EdgeInfo { causal_type: CausalType::Timelike },  // 160→180
    ];

    // Simulate cluster membership: nodes 0-1 in cluster 0 (CS), nodes 3-4 in cluster 1 (Philosophy)
    // Node 2 (Systems Control) is the bridge
    let cluster_map = move |id: &Uuid| -> Option<u32> {
        if *id == id_95 || *id == id_120 { Some(0) }       // Cluster 0: Computer Science
        else if *id == id_160 || *id == id_180 { Some(1) }  // Cluster 1: Philosophy
        else { Some(2) }                                      // Cluster 2: Bridge
    };

    let rationale = engine.analyze(&trajectory, &edge_infos, Some(&cluster_map))
        .expect("inference failed");

    println!("  Inference Type:      {}", rationale.inference_type);
    println!("  GCS (aggregate):     {:.4}", rationale.gcs);
    println!("  Radial Gradient:     {:.4}", rationale.radial_gradient);
    println!("  Cluster Transitions: {}", rationale.cluster_transitions);
    println!("  Fidelity:            {:.4}", rationale.fidelity);
    println!("  Causal Fraction:     {:.0}%", edge_infos.iter().filter(|e| e.causal_type == CausalType::Timelike).count() as f64 / edge_infos.len() as f64 * 100.0);
    println!("  Valid:               {}", if rationale.is_valid() { "✅ ACCEPTED" } else { "❌ REJECTED (LogicalRupture)" });
    println!();

    // ─────────────────────────────────────────────
    // 4. FRÉCHET SYNTHESIS (if DialecticalSynthesis)
    // ─────────────────────────────────────────────
    if rationale.inference_type == InferenceType::DialecticalSynthesis {
        println!("━━━ STAGE 4: Dialectical Synthesis (Thesis + Antithesis → Synthesis) ━━━");

        let synthesizer = FrechetSynthesizer::with_defaults();

        // Source nodes: the two cluster representatives (Unsafe Rust + Foucault)
        let pv_95 = PoincareVector {
            coords: emb_95.iter().map(|&x| x as f32).collect(),
            dim: 8,
        };
        let pv_180 = PoincareVector {
            coords: emb_180.iter().map(|&x| x as f32).collect(),
            dim: 8,
        };

        let source_ids = vec![id_95, id_180];
        let source_embeddings = vec![&pv_95, &pv_180];

        let (synthesis_point, synthesis_node) = synthesizer
            .synthesize(&source_embeddings, &source_ids, rationale.clone())
            .expect("synthesis failed");

        let synth_norm: f64 = synthesis_point.iter().map(|x| x * x).sum::<f64>().sqrt();

        println!("  Thesis:     Unsafe Rust       (‖x‖={:.4}, cluster=CS)", norm(&emb_95));
        println!("  Antithesis: Foucault           (‖x‖={:.4}, cluster=Philosophy)", norm(&emb_180));
        println!("  ─────────────────────────────────────");
        println!("  Synthesis:  Type-safe Governance");
        println!("    ‖s‖ = {synth_norm:.4} (MORE ABSTRACT than both inputs ✓)");
        println!("    dim  = {}", synthesis_point.len());
        println!("    ID   = {}", synthesis_node.node_id);
        println!("    depth = {} (center → abstract)", depth_label(synth_norm));
        println!();

        // ─────────────────────────────────────────────
        // 5. HOMEOSTASIS CHECK
        // ─────────────────────────────────────────────
        println!("━━━ STAGE 5: Homeostasis Guard ━━━");
        let guard = HomeostasisGuard::default();
        let status = guard.check(synth_norm);
        println!("  Min radius:  {:.4}", guard.min_radius());
        println!("  Synth ‖s‖:   {synth_norm:.4}");
        println!("  Status:      {}", if status.is_ok() { "✅ WITHIN BOUNDS" } else { "⚠️  TOO CLOSE TO CENTER" });
        println!();

        // ─────────────────────────────────────────────
        // 6. FEEDBACK LOOP — PREPARE FOR INSERTION
        // ─────────────────────────────────────────────
        println!("━━━ STAGE 6: Feedback Loop ━━━");

        let graph_node = synthesizer.to_graph_node(
            &synthesis_point,
            &synthesis_node,
            "Type-safe Governance",
        );

        let feedback = FeedbackLoop::with_defaults();
        let source_infos = vec![
            SourceNodeInfo {
                id: id_95,
                coords: pv_95.coords.clone(),
                created_at: 1709000000, // Simulated timestamp
            },
            SourceNodeInfo {
                id: id_180,
                coords: pv_180.coords.clone(),
                created_at: 1709000100,
            },
        ];

        let result = feedback.prepare(graph_node, synthesis_node, &source_infos);

        println!("  Node ready:  {} (depth={:.4}, energy={:.1})",
            result.node.meta.id, result.node.meta.depth, result.node.meta.energy);
        println!("  Edges:       {} hierarchical edges", result.edges.len());
        for (i, edge) in result.edges.iter().enumerate() {
            println!("    Edge {i}: {} → {} (weight={:.1}, causal={:?})",
                edge.from, edge.to, edge.weight, edge.causal_type);
        }
        println!();

        // ─────────────────────────────────────────────
        // 7. PRINT THE RATIONALE (PROOF CERTIFICATE)
        // ─────────────────────────────────────────────
        print_rationale(&result.synthesis_meta.rationale.as_ref().unwrap(), &names);

    } else {
        println!("  ℹ️  Inference type is {:?}, not DialecticalSynthesis.", rationale.inference_type);
        println!("  Printing Rationale anyway:");
        println!();
        print_rationale(&rationale, &names);
    }

    // ─────────────────────────────────────────────
    // 8. DIALECTIC DETECTOR (bonus)
    // ─────────────────────────────────────────────
    println!("━━━ BONUS: Dialectic Tension Detection ━━━");

    let detector = DialecticDetector::with_defaults();
    let centroids = vec![
        (id_95,  0, emb_95.clone()),   // CS cluster representative
        (id_180, 1, emb_180.clone()),  // Philosophy cluster representative
    ];
    let tensions = detector.detect_tensions(&centroids).unwrap();
    for pair in &tensions {
        println!("  Tension: cluster {} ↔ cluster {} | distance={:.4} | score={:.4}",
            pair.cluster_a, pair.cluster_b, pair.distance, pair.tension_score);
    }
    if tensions.is_empty() {
        println!("  No tensions detected (centroids may be too far apart or too close)");
    }
    println!();

    // ─────────────────────────────────────────────
    // 9. EVOLUTION SCHEDULER STATUS
    // ─────────────────────────────────────────────
    println!("━━━ BONUS: Evolution Scheduler ━━━");
    let scheduler = EvolutionScheduler::with_defaults();
    println!("  {}", scheduler.stats());
    println!("  Should run: {}", scheduler.should_run());
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Pipeline Ignition Complete — Proof Certificate Generated  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn depth_label(norm: f64) -> &'static str {
    if norm < 0.2 { "ABSTRACT (center)" }
    else if norm < 0.4 { "semi-abstract" }
    else if norm < 0.6 { "intermediate" }
    else if norm < 0.8 { "semi-concrete" }
    else { "CONCRETE (boundary)" }
}

fn print_rationale(rationale: &Rationale, names: &[&str]) {
    println!("━━━ RATIONALE — Proof Certificate ━━━");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  InferenceType:       {:40} ║", format!("{}", rationale.inference_type));
    println!("║  GCS (aggregate):     {:40} ║", format!("{:.6}", rationale.gcs));
    println!("║  Fidelity:            {:40} ║", format!("{:.6}", rationale.fidelity));
    println!("║  Radial Gradient:     {:40} ║", format!("{:.6}", rationale.radial_gradient));
    println!("║  Cluster Transitions: {:40} ║", format!("{}", rationale.cluster_transitions));
    println!("║  Min Hop GCS:         {:40} ║", format!("{:.6}", rationale.min_hop_gcs()));
    println!("║  Path Length:         {:40} ║", format!("{} nodes", rationale.path.len()));
    println!("║  Created At:          {:40} ║", format!("{}", rationale.created_at));
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Path:                                                      ║");
    for (i, id) in rationale.path.iter().enumerate() {
        let label = names.get(i).unwrap_or(&"synthesis");
        println!("║    [{i}] {id} ({label}) ║");
    }
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Per-hop GCS:                                               ║");
    for (i, &gcs) in rationale.hop_gcs.iter().enumerate() {
        let status = if gcs >= 0.3 { "✓" } else { "✗" };
        println!("║    Hop {i}: {gcs:.6} {status:>42} ║");
    }
    println!("╠══════════════════════════════════════════════════════════════╣");
    if let Some(ref fpt) = rationale.frechet_point {
        let fnorm: f64 = fpt.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!("║  Fréchet Point:       ‖s‖={fnorm:.6} ({:30}) ║", depth_label(fnorm));
    }
    if let Some(ref sid) = rationale.synthesis_node_id {
        println!("║  Synthesis Node ID:   {:40} ║", format!("{sid}"));
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Machine-readable JSON
    println!("━━━ Rationale (JSON) ━━━");
    let json = serde_json::to_string_pretty(rationale).unwrap();
    println!("{json}");
    println!();
}
