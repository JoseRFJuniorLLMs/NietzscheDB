// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase VI.1 — Cortex bootstrap with controlled asymmetry.
//!
//! On first startup (empty "default" collection), seeds the Ur-Cortex node,
//! four manifold lens anchors with asymmetric Poincaré embeddings, geodesic
//! edges, and the `energy_guardian` daemon.
//!
//! The asymmetry forces the first λ₂ oscillation, giving the Zaratustra,
//! L-System, and MetabolicSleepManager engines a gradient to navigate.

use std::collections::HashMap;
use std::sync::Arc;

use tracing::info;
use uuid::Uuid;

/// Generate a deterministic UUID v5 from a namespace + name.
/// This makes bootstrap idempotent across backup/restore cycles —
/// the same logical entity always gets the same UUID.
fn deterministic_uuid(name: &str) -> Uuid {
    // NietzscheDB namespace UUID (randomly generated, fixed forever)
    const NIETZSCHE_NS: Uuid = Uuid::from_bytes([
        0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1,
        0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
    ]);
    Uuid::new_v5(&NIETZSCHE_NS, name.as_bytes())
}

use nietzsche_graph::{
    CollectionManager, Edge, EdgeType, Node, NodeType, PoincareVector,
};
use nietzsche_query::ast::{
    CompOp, Condition, DaemonAction, Expr, MathFunc, MathFuncArg, NodePattern,
};
use nietzsche_wiederkehr::{put_daemon, DaemonDef};

/// Run the cortex bootstrap on the "default" collection.
///
/// Returns `Ok(true)` if the bootstrap was applied, `Ok(false)` if already
/// bootstrapped (cortex_zero exists), or `Err` on failure.
pub async fn run_bootstrap(cm: &Arc<CollectionManager>) -> anyhow::Result<bool> {
    let shared = cm
        .get("default")
        .ok_or_else(|| anyhow::anyhow!("default collection not found"))?;

    let mut db = shared.write().await;

    // ── Idempotency check: does cortex_zero already exist? ──────────────
    let existing = db.find_node_by_content(
        "Concept",
        &serde_json::json!({"id": "cortex_zero"}),
    )?;
    if existing.is_some() {
        return Ok(false);
    }

    // ── Resolve collection dimension for zero-padded embeddings ─────────
    let dim = {
        let info = cm.list();
        info.iter()
            .find(|c| c.name == "default")
            .map(|c| c.dim)
            .unwrap_or(3072)
    };

    // ── Helper: build a PoincareVector with 4D coords zero-padded to dim ──
    let make_embedding = |c0: f32, c1: f32, c2: f32, c3: f32| -> PoincareVector {
        let mut coords = vec![0.0f32; dim];
        if dim >= 4 {
            coords[0] = c0;
            coords[1] = c1;
            coords[2] = c2;
            coords[3] = c3;
        }
        PoincareVector::new(coords)
    };

    // ── 1. Ur-Cortex — Concept node at the origin ───────────────────────
    // Deterministic UUID so bootstrap survives backup/restore cycles.
    let cortex_id = deterministic_uuid("cortex_zero");
    let mut cortex = Node::new(
        cortex_id,
        make_embedding(0.0, 0.0, 0.0, 0.0),
        serde_json::json!({
            "id": "cortex_zero",
            "title": "Ur-Cortex",
            "node_label": "Cortex"
        }),
    );
    cortex.node_type = NodeType::Concept;
    cortex.energy = 0.85;
    cortex.meta.valence = 0.5;
    cortex.meta.arousal = 0.5;
    db.insert_node(cortex)?;
    info!("[BOOT] cortex_zero created (energy=0.85, valence=0.5, arousal=0.5)");

    // ── 2. Manifold lenses with tensorized asymmetry (ε-perturbation) ───
    struct LensDef {
        content_id: &'static str,
        title: &'static str,
        manifold_type: &'static str,
        coords: [f32; 4],
        energy: f32,
        edge_type: EdgeType,
        causal_type: &'static str,
    }

    let lenses = [
        LensDef {
            content_id: "lens_poincare",
            title: "Âncora de Hierarquia",
            manifold_type: "Hyperbolic",
            coords: [0.12, 0.01, 0.00, 0.00],
            energy: 0.8,
            edge_type: EdgeType::Hierarchical,
            causal_type: "Structural",
        },
        LensDef {
            content_id: "lens_klein",
            title: "Âncora de Lógica",
            manifold_type: "Hyperbolic_Straight",
            coords: [0.00, 0.09, 0.02, 0.00],
            energy: 0.8,
            edge_type: EdgeType::Association,
            causal_type: "Interpretative",
        },
        LensDef {
            content_id: "lens_minkowski",
            title: "Âncora de Causalidade",
            manifold_type: "Lorentzian",
            coords: [0.01, 0.00, 0.11, 0.01],
            energy: 0.9,
            edge_type: EdgeType::Association,
            causal_type: "Timelike",
        },
        LensDef {
            content_id: "lens_riemann",
            title: "Âncora de Síntese",
            manifold_type: "Spherical",
            coords: [0.00, 0.02, 0.00, 0.08],
            energy: 0.7,
            edge_type: EdgeType::Association,
            causal_type: "Interpretative",
        },
    ];

    let mut lens_ids = Vec::with_capacity(lenses.len());

    for lens in &lenses {
        let lens_id = deterministic_uuid(lens.content_id);
        let mut node = Node::new(
            lens_id,
            make_embedding(lens.coords[0], lens.coords[1], lens.coords[2], lens.coords[3]),
            serde_json::json!({
                "id": lens.content_id,
                "title": lens.title,
                "type": lens.manifold_type,
                "node_label": "Manifold"
            }),
        );
        node.node_type = NodeType::Semantic;
        node.energy = lens.energy;
        db.insert_node(node)?;
        lens_ids.push(lens_id);
    }
    info!("[BOOT] 4 manifold lenses created (poincare/klein/minkowski/riemann)");

    // ── 3. Geodesic edges with epistemic differentiation ────────────────
    for (i, lens) in lenses.iter().enumerate() {
        let mut edge = Edge::new(cortex_id, lens_ids[i], lens.edge_type.clone(), 1.0);
        edge.metadata.insert(
            "causal_type".into(),
            serde_json::Value::String(lens.causal_type.to_string()),
        );
        db.insert_edge(edge)?;
    }
    info!("[BOOT] 4 geodesic edges created (1 Hierarchical + 3 Association)");

    // ── 4. energy_guardian daemon via Wiederkehr ─────────────────────────
    let daemon_def = DaemonDef {
        name: "energy_guardian".into(),
        on_pattern: NodePattern {
            alias: "n".into(),
            label: Some("Concept".into()),
            semantic_id: None,
        },
        when_cond: Condition::Compare {
            left: Expr::Property { alias: "n".into(), field: "energy".into() },
            op: CompOp::Gt,
            right: Expr::Float(0.90),
        },
        then_action: DaemonAction::Diffuse {
            alias: "n".into(),
            t_values: vec![0.1, 0.5],
            max_hops: 3,
        },
        every: Expr::MathFunc {
            func: MathFunc::Interval,
            args: vec![MathFuncArg::Str("15m".into())],
        },
        energy: 0.8,
        last_run: 0.0,
        interval_secs: 15.0 * 60.0, // 15 minutes
    };
    put_daemon(db.storage(), &daemon_def)
        .map_err(|e| anyhow::anyhow!("daemon store error: {e}"))?;
    info!("[BOOT] daemon energy_guardian registered (interval=15m, threshold=0.90)");

    Ok(true)
}
