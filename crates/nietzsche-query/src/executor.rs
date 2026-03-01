//! NQL executor: maps AST nodes to graph operations.
//!
//! Each `execute_*` function takes references to `GraphStorage` and
//! `AdjacencyIndex` (extracted from `NietzscheDB` via `.storage()` /
//! `.adjacency()`) plus a `Params` map for query parameters.

use std::collections::{HashMap, HashSet, VecDeque};

use uuid::Uuid;
use rayon::prelude::*;
use nietzsche_graph::{
    AdjacencyIndex, AdjEntry, Edge, GraphStorage, Node, PoincareVector,
    traversal::{diffusion_walk, DiffusionConfig},
};
use nietzsche_dsi::SemanticId;

use crate::ast::*;
use crate::error::QueryError;

/// Minimum number of nodes to engage rayon parallel filter.
/// Below this threshold the overhead of rayon outweighs the gain.
const PARALLEL_SCAN_THRESHOLD: usize = 2_000;

// ── Gas Tracking (Industrial Refactors) ──────

pub struct GasTracker {
    pub remaining: usize,
}

impl GasTracker {
    pub fn d_node(&mut self) -> Result<(), QueryError> { self.consume(1) }
    pub fn d_edge(&mut self) -> Result<(), QueryError> { self.consume(2) }
    pub fn d_eval(&mut self) -> Result<(), QueryError> { self.consume(1) }
    
    fn consume(&mut self, amount: usize) -> Result<(), QueryError> {
        if self.remaining < amount {
            return Err(QueryError::Execution("gas limit exceeded: query too complex or infinite recursion detected".into()));
        }
        self.remaining -= amount;
        Ok(())
    }
}

/// Typed parameter values passed at query execution time.
#[derive(Debug, Clone)]
pub enum ParamValue {
    Uuid(Uuid),
    Float(f64),
    Int(i64),
    Str(String),
    /// A Poincaré-ball vector used with `HYPERBOLIC_DIST`.
    Vector(Vec<f64>),
}

/// Parameter map: keys are param names **without** the leading `$`.
pub type Params = HashMap<String, ParamValue>;

// ─────────────────────────────────────────────
// Query results
// ─────────────────────────────────────────────

/// One row returned by an NQL query.
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// Result of `MATCH (n) …`
    Node(Node),
    /// Result of `MATCH (a)-[…]->(b) …`
    NodePair { from: Node, to: Node },
    /// Result of `DIFFUSE FROM …`
    DiffusionPath(Vec<Uuid>),
    /// Phase 11: Result of `RECONSTRUCT …` — returns the node ID, requested
    /// modality, and quality hint.  The caller (gRPC handler) is responsible
    /// for actually loading the latent from SensoryStorage and decoding it.
    ReconstructRequest {
        node_id:  Uuid,
        modality: Option<String>,
        quality:  Option<String>,
    },
    /// Result of `EXPLAIN <query>` — textual execution plan.
    ExplainPlan(String),
    /// Scalar results from aggregation queries (COUNT, SUM, AVG, etc.).
    Scalar(Vec<(String, ScalarValue)>),
    /// Phase C: Result of `INVOKE ZARATUSTRA …` — the server handles execution.
    /// The executor only forwards the parsed parameters to the gRPC layer.
    InvokeZaratustraRequest {
        collection: Option<String>,
        cycles:     Option<u32>,
        alpha:      Option<f64>,
        decay:      Option<f64>,
    },
    /// Phase F: `BEGIN` was parsed — caller must open a new transaction.
    TxBegin,
    /// Phase F: `COMMIT` was parsed — caller must commit the active transaction.
    TxCommit,
    /// Phase F: `ROLLBACK` was parsed — caller must rollback the active transaction.
    TxRollback,
    /// Phase D: Result of `MERGE (n:Type {key: val}) …` — the caller (gRPC handler)
    /// must acquire a write lock and perform the actual upsert.
    MergeNodeRequest {
        node_type:     Option<String>,
        match_keys:    serde_json::Value,
        on_create_set: serde_json::Value,
        on_match_set:  serde_json::Value,
    },
    /// Phase D: Result of `MERGE (a)-[:TYPE]->(b) …` — edge upsert intent.
    MergeEdgeRequest {
        from_match:    serde_json::Value,
        to_match:      serde_json::Value,
        edge_type:     Option<String>,
        on_create_set: serde_json::Value,
        on_match_set:  serde_json::Value,
    },
    /// Result of `CREATE (n:Type {props}) …` — the caller inserts a new node.
    CreateNodeRequest {
        node_type:  Option<String>,
        properties: serde_json::Value,
    },
    /// Result of `MATCH … SET …` — the caller updates matched nodes.
    /// Each entry in `updates` is `(node_id, computed_assignments_json)`.
    SetRequest {
        updates: Vec<(Uuid, serde_json::Value)>,
    },
    /// Result of `MATCH … DELETE …` or `MATCH … DETACH DELETE …`.
    DeleteRequest {
        matched_ids: Vec<Uuid>,
        /// When `true`, also remove all incident edges (DETACH DELETE).
        detach: bool,
    },
    /// Result of `CREATE DAEMON …` — the caller persists the daemon definition.
    CreateDaemonRequest {
        name:        String,
        on_pattern:  NodePattern,
        when_cond:   Condition,
        then_action: DaemonAction,
        every:       Expr,
        energy:      Option<f64>,
    },
    /// Result of `DROP DAEMON …` — the caller removes the daemon definition.
    DropDaemonRequest {
        name: String,
    },
    /// Result of `SHOW DAEMONS` — the caller lists all daemon definitions.
    ShowDaemonsRequest,
    // ── Dream Queries (Phase 15.2) ──────────────────────
    /// `DREAM FROM …` — initiate a dream simulation.
    DreamFromRequest {
        seed_param: Option<String>,
        seed_alias: Option<String>,
        depth:      Option<usize>,
        noise:      Option<f64>,
    },
    /// `APPLY DREAM $id` — persist dream results.
    ApplyDreamRequest { dream_id: String },
    /// `REJECT DREAM $id` — discard dream results.
    RejectDreamRequest { dream_id: String },
    /// `SHOW DREAMS` — list pending dream sessions.
    ShowDreamsRequest,
    // ── Synesthesia (Phase 15.3) ────────────────────────
    /// `TRANSLATE $node FROM mod TO mod [QUALITY q]`
    TranslateRequest {
        node_id:       Option<Uuid>,
        node_alias:    Option<String>,
        from_modality: String,
        to_modality:   String,
        quality:       Option<String>,
    },
    // ── Eternal Return (Phase 15.4) ─────────────────────
    /// `COUNTERFACTUAL SET … MATCH … RETURN …`
    CounterfactualRequest {
        overlays: serde_json::Value,
        inner_results: Vec<QueryResult>,
    },
    // ── Collective Unconscious (Phase 15.6) ─────────────
    /// `SHOW ARCHETYPES`
    ShowArchetypesRequest,
    /// `SHARE ARCHETYPE $node TO "col"`
    ShareArchetypeRequest {
        node_id:           Uuid,
        target_collection: String,
    },
    // ── Narrative Engine (Phase 15.7) ───────────────────
    /// `NARRATE [IN "col"] [WINDOW h] [FORMAT f]`
    NarrateRequest {
        collection:   Option<String>,
        window_hours: Option<u64>,
        format:       Option<String>,
    },
    // ── Psychoanalyze (Lineage) ─────────────────────────
    /// `PSYCHOANALYZE $node` — evolutionary lineage of a concept.
    PsychoanalyzeResult {
        node_id:    Uuid,
        lineage:    serde_json::Value,
    },

    // ═══════════════════════════════════════════════════
    // ── NQL 2.0: New Query Result Types ─────────────
    // ═══════════════════════════════════════════════════

    /// `UNWIND expr AS alias RETURN …` — caller must expand the array.
    UnwindRequest {
        expr:  String,
        alias: String,
    },
    /// `SHORTEST_PATH((src),(dst))` — caller performs BFS/Dijkstra.
    ShortestPathRequest {
        from_alias: String,
        from_label: Option<String>,
        to_alias:   String,
        to_label:   Option<String>,
        limit:      Option<usize>,
    },
    /// `MEASURE TENSION BETWEEN (a) AND (b)` — caller computes dialectical tension.
    MeasureTensionRequest {
        node_a_alias: String,
        node_a_label: Option<String>,
        node_b_alias: String,
        node_b_label: Option<String>,
    },
    /// `FIND NEAREST [IN space] TO $vec [LIMIT n]` — caller performs KNN.
    FindNearestRequest {
        space:  Option<String>,
        target: String,
        limit:  Option<usize>,
    },
}

/// A typed scalar value returned by aggregation or property-projection queries.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Null,
}

// ─────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────

/// Default gas budget for a single NQL query.
///
/// 50,000 units provides roughly:
/// - 50,000 node scans, OR
/// - 25,000 edge scans (2 units each), OR
/// - ~1,200 full edge + condition evaluations on a mid-size graph.
///
/// Override via `execute_with_gas_limit()` for privileged internal queries.
pub const DEFAULT_GAS_LIMIT: usize = 50_000;

pub fn execute(
    query:     &Query,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    let mut gas = GasTracker { remaining: DEFAULT_GAS_LIMIT };
    execute_with_indexes_and_gas(query, storage, adjacency, params, &HashSet::new(), &mut gas)
}

/// Execute an NQL query with a custom gas limit.
///
/// Use this for privileged callers (internal agency queries, admin APIs)
/// that need a higher budget. For anonymous REST endpoints, prefer a
/// *lower* limit (e.g. 10_000) for safety.
///
/// # Example
/// ```rust
/// // Agency daemon queries — allow 5× the default budget
/// let results = execute_with_gas_limit(
///     &query, storage, adjacency, params,
///     DEFAULT_GAS_LIMIT * 5
/// )?;
/// ```
pub fn execute_with_gas_limit(
    query:     &Query,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    gas_limit: usize,
) -> Result<Vec<QueryResult>, QueryError> {
    let mut gas = GasTracker { remaining: gas_limit };
    execute_with_indexes_and_gas(query, storage, adjacency, params, &HashSet::new(), &mut gas)
}

/// Execute a parsed NQL query with secondary index awareness and gas tracking.
pub fn execute_with_indexes_and_gas(
    query:          &Query,
    storage:        &GraphStorage,
    adjacency:      &AdjacencyIndex,
    params:         &Params,
    indexed_fields: &HashSet<String>,
    gas:            &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    match query {
        Query::Match(m)              => execute_match(m, storage, adjacency, params, indexed_fields, gas),
        Query::Diffuse(d)            => execute_diffuse(d, storage, adjacency, params, gas),
        Query::Reconstruct(r)        => execute_reconstruct(r, params),
        Query::Explain(inner)        => execute_explain(inner, storage, adjacency, params),
        Query::InvokeZaratustra(iz)  => Ok(vec![QueryResult::InvokeZaratustraRequest {
            collection: iz.collection.clone(),
            cycles:     iz.cycles,
            alpha:      iz.alpha,
            decay:      iz.decay,
        }]),
        Query::BeginTx               => Ok(vec![QueryResult::TxBegin]),
        Query::CommitTx              => Ok(vec![QueryResult::TxCommit]),
        Query::RollbackTx            => Ok(vec![QueryResult::TxRollback]),
        Query::Merge(m)              => execute_merge(m, params),
        Query::Create(c)             => execute_create(c, params),
        Query::MatchSet(ms)          => execute_match_set(ms, storage, adjacency, params, gas),
        Query::MatchDelete(md)       => execute_match_delete(md, storage, adjacency, params, gas),
        Query::CreateDaemon(cd)      => Ok(vec![QueryResult::CreateDaemonRequest {
            name:        cd.name.clone(),
            on_pattern:  cd.on_pattern.clone(),
            when_cond:   cd.when_cond.clone(),
            then_action: cd.then_action.clone(),
            every:       cd.every.clone(),
            energy:      cd.energy,
        }]),
        Query::DropDaemon(dd)        => Ok(vec![QueryResult::DropDaemonRequest {
            name: dd.name.clone(),
        }]),
        Query::ShowDaemons           => Ok(vec![QueryResult::ShowDaemonsRequest]),
        // ── Dream Queries (Phase 15.2) ──────────────────────
        Query::DreamFrom(d) => {
            let (seed_param, seed_alias) = match &d.seed {
                DiffuseFrom::Param(p) => (Some(p.clone()), None),
                DiffuseFrom::Alias(a) => (None, Some(a.clone())),
            };
            Ok(vec![QueryResult::DreamFromRequest {
                seed_param,
                seed_alias,
                depth: d.depth,
                noise: d.noise,
            }])
        }
        Query::ApplyDream(a) => Ok(vec![QueryResult::ApplyDreamRequest {
            dream_id: a.dream_id.clone(),
        }]),
        Query::RejectDream(r) => Ok(vec![QueryResult::RejectDreamRequest {
            dream_id: r.dream_id.clone(),
        }]),
        Query::ShowDreams => Ok(vec![QueryResult::ShowDreamsRequest]),
        // ── Synesthesia (Phase 15.3) ────────────────────────
        Query::Translate(t) => {
            let (node_id, node_alias) = match &t.target {
                ReconstructTarget::Param(p) => {
                    let uid = params.get(p)
                        .and_then(|v| if let ParamValue::Uuid(u) = v { Some(*u) } else { None });
                    (uid, None)
                }
                ReconstructTarget::Alias(a) => (None, Some(a.clone())),
            };
            Ok(vec![QueryResult::TranslateRequest {
                node_id,
                node_alias,
                from_modality: t.from_modality.clone(),
                to_modality:   t.to_modality.clone(),
                quality:       t.quality.clone(),
            }])
        }
        // ── Eternal Return (Phase 15.4) ─────────────────────
        Query::Counterfactual(cf) => {
            // Serialize overlays as JSON, execute inner match normally
            let overlay_json = serde_json::json!(
                cf.overlays.iter().map(|sa| {
                    serde_json::json!({
                        "alias": sa.alias,
                        "field": sa.field,
                        "value": format!("{:?}", sa.value),
                    })
                }).collect::<Vec<_>>()
            );
            let inner_results = execute_with_indexes_and_gas(
                &Query::Match(cf.inner.clone()),
                storage, adjacency, params,
                indexed_fields, gas
            )?;
            Ok(vec![QueryResult::CounterfactualRequest {
                overlays: overlay_json,
                inner_results,
            }])
        }
        // ── Collective Unconscious (Phase 15.6) ─────────────
        Query::ShowArchetypes => Ok(vec![QueryResult::ShowArchetypesRequest]),
        Query::ShareArchetype(sa) => {
            let uid = params.get(&sa.node_param)
                .and_then(|v| if let ParamValue::Uuid(u) = v { Some(*u) } else { None })
                .ok_or_else(|| QueryError::ParamNotFound { name: sa.node_param.clone() })?;
            Ok(vec![QueryResult::ShareArchetypeRequest {
                node_id:           uid,
                target_collection: sa.target_collection.clone(),
            }])
        }
        // ── Narrative Engine (Phase 15.7) ───────────────────
        Query::Narrate(n) => Ok(vec![QueryResult::NarrateRequest {
            collection:   n.collection.clone(),
            window_hours: n.window_hours,
            format:       n.format.clone(),
        }]),
        // ── Psychoanalyze (Lineage) ─────────────────────────
        Query::Psychoanalyze(pq) => execute_psychoanalyze(pq, storage, adjacency, params),

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: New query types ─────────────────────
        // ═══════════════════════════════════════════════════

        Query::Union(uq) => {
            let mut results = Vec::new();
            for (i, mq) in uq.queries.iter().enumerate() {
                let mut sub_results = execute_match(mq, storage, adjacency, params, indexed_fields, gas)?;
                if i > 0 && !uq.all.get(i - 1).copied().unwrap_or(false) {
                    // UNION (not ALL) — deduplicate by node ID
                    let existing_ids: HashSet<Uuid> = results.iter().filter_map(|r| {
                        if let QueryResult::Node(n) = r { Some(n.id) } else { None }
                    }).collect();
                    sub_results.retain(|r| {
                        if let QueryResult::Node(n) = r { !existing_ids.contains(&n.id) } else { true }
                    });
                }
                results.extend(sub_results);
            }
            Ok(results)
        }

        Query::Unwind(uq) => {
            // UNWIND returns intent — caller must expand the array
            Ok(vec![QueryResult::UnwindRequest {
                expr:  format!("{:?}", uq.expr),
                alias: uq.alias.clone(),
            }])
        }

        Query::ShortestPath(sp) => {
            Ok(vec![QueryResult::ShortestPathRequest {
                from_alias: sp.from.alias.clone(),
                from_label: sp.from.label.clone(),
                to_alias:   sp.to.alias.clone(),
                to_label:   sp.to.label.clone(),
                limit:      sp.limit,
            }])
        }

        Query::MatchElites(me) => {
            // Execute directly: scan all nodes, sort by energy DESC, take top N
            let limit = me.limit.unwrap_or(10);
            let mut nodes: Vec<Node> = storage.scan_nodes()?;
            nodes.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));
            nodes.truncate(limit);
            Ok(nodes.into_iter().map(QueryResult::Node).collect())
        }

        Query::MeasureTension(mt) => {
            Ok(vec![QueryResult::MeasureTensionRequest {
                node_a_alias: mt.node_a.alias.clone(),
                node_a_label: mt.node_a.label.clone(),
                node_b_alias: mt.node_b.alias.clone(),
                node_b_label: mt.node_b.label.clone(),
            }])
        }

        Query::MeasureTgc(_tgc) => {
            // Execute directly: compute global telemetry
            let nodes: Vec<Node> = storage.scan_nodes()?;
            let n = nodes.len() as f64;
            if n == 0.0 {
                return Ok(vec![QueryResult::Scalar(vec![
                    ("total_nodes".into(), ScalarValue::Int(0)),
                    ("avg_energy".into(), ScalarValue::Float(0.0)),
                    ("entropy".into(), ScalarValue::Float(0.0)),
                    ("tgc".into(), ScalarValue::Float(0.0)),
                ])]);
            }
            let total_energy: f64 = nodes.iter().map(|n| n.energy as f64).sum();
            let avg_energy = total_energy / n;
            // Shannon entropy of energy distribution
            let entropy: f64 = nodes.iter().map(|node| {
                let p = (node.energy as f64) / total_energy.max(1e-15);
                if p > 1e-15 { -p * p.ln() } else { 0.0 }
            }).sum();
            // TGC = entropy * avg_energy * ln(N)
            let tgc_val = entropy * avg_energy * n.ln();
            Ok(vec![QueryResult::Scalar(vec![
                ("total_nodes".into(), ScalarValue::Int(nodes.len() as i64)),
                ("avg_energy".into(), ScalarValue::Float(avg_energy)),
                ("entropy".into(), ScalarValue::Float(entropy)),
                ("tgc".into(), ScalarValue::Float(tgc_val)),
            ])])
        }

        Query::FindNearest(fq) => {
            Ok(vec![QueryResult::FindNearestRequest {
                space:  fq.space.clone(),
                target: format!("{:?}", fq.target),
                limit:  fq.limit,
            }])
        }
    }
}

// ─────────────────────────────────────────────
// EXPLAIN executor
// ─────────────────────────────────────────────

fn execute_explain(
    query:     &Query,
    storage:   &GraphStorage,
    _adjacency:&AdjacencyIndex,
    _params:   &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    let cost = crate::cost::estimate_cost(query, storage);
    let plan = match query {
        Query::Match(m) => {
            let scan_type = match &m.pattern {
                Pattern::Node(np) => {
                    let label = np.label.as_deref().unwrap_or("*");
                    format!("NodeScan(label={})", label)
                }
                Pattern::Path(pp) => {
                    let edge = pp.edge_label.as_deref().unwrap_or("*");
                    let dir = match pp.direction {
                        Direction::Out => "-->",
                        Direction::In  => "<--",
                    };
                    if let Some(hr) = &pp.hop_range {
                        format!("BoundedBFS(type={}, dir={}, hops={}..{})", edge, dir, hr.min, hr.max)
                    } else {
                        format!("EdgeScan(type={}, dir={})", edge, dir)
                    }
                }
            };
            let mut parts = vec![scan_type];
            if !m.conditions.is_empty() {
                parts.push(format!("Filter(conditions={})", m.conditions.len()));
            }
            if let Some(ref ob) = m.ret.order_by {
                parts.push(format!("Sort(dir={:?})", ob.dir));
            }
            if let Some(skip) = m.ret.skip {
                parts.push(format!("Skip({})", skip));
            }
            if let Some(limit) = m.ret.limit {
                parts.push(format!("Limit({})", limit));
            }
            if m.ret.distinct {
                parts.push("Distinct".into());
            }
            if !m.ret.group_by.is_empty() {
                parts.push(format!("GroupBy(keys={})", m.ret.group_by.len()));
            }
            parts.join(" -> ")
        }
        Query::Diffuse(d) => {
            format!("DiffusionWalk(max_hops={}, t_scales={})", d.max_hops, d.t_values.len())
        }
        Query::Reconstruct(_)        => "ReconstructLatent".into(),
        Query::Explain(_)            => "Explain(nested — not supported)".into(),
        Query::InvokeZaratustra(iz)  => format!(
            "InvokeZaratustra(collection={}, cycles={}, alpha={}, decay={})",
            iz.collection.as_deref().unwrap_or("default"),
            iz.cycles.unwrap_or(1),
            iz.alpha.unwrap_or(0.10),
            iz.decay.unwrap_or(0.02),
        ),
        Query::BeginTx               => "BeginTransaction".into(),
        Query::CommitTx              => "CommitTransaction".into(),
        Query::RollbackTx            => "RollbackTransaction".into(),
        Query::Merge(m)              => {
            let pat = match &m.pattern {
                crate::ast::MergePattern::Node(np) =>
                    format!("MergeNode(type={})", np.label.as_deref().unwrap_or("*")),
                crate::ast::MergePattern::Edge(ep) =>
                    format!("MergeEdge(type={})", ep.edge_label.as_deref().unwrap_or("*")),
            };
            format!("{} -> OnCreate({}) -> OnMatch({})",
                pat,
                m.on_create.len(),
                m.on_match.len(),
            )
        }
        Query::Create(c) => {
            format!("CreateNode(type={}, props={})",
                c.label.as_deref().unwrap_or("*"),
                c.properties.len(),
            )
        }
        Query::MatchSet(ms) => {
            format!("MatchSet(assignments={}, conditions={})",
                ms.assignments.len(),
                ms.conditions.len(),
            )
        }
        Query::MatchDelete(md) => {
            format!("MatchDelete(targets={}, conditions={})",
                md.targets.len(),
                md.conditions.len(),
            )
        }
        Query::CreateDaemon(cd) => format!("CreateDaemon(name={})", cd.name),
        Query::DropDaemon(dd)   => format!("DropDaemon(name={})", dd.name),
        Query::ShowDaemons      => "ShowDaemons".into(),
        Query::DreamFrom(d)     => format!("DreamFrom(depth={:?}, noise={:?})", d.depth, d.noise),
        Query::ApplyDream(a)    => format!("ApplyDream(id={})", a.dream_id),
        Query::RejectDream(r)   => format!("RejectDream(id={})", r.dream_id),
        Query::ShowDreams       => "ShowDreams".into(),
        Query::Translate(t)     => format!("Translate(from={}, to={})", t.from_modality, t.to_modality),
        Query::Counterfactual(cf) => format!("Counterfactual(overlays={}, inner=Match)", cf.overlays.len()),
        Query::ShowArchetypes   => "ShowArchetypes".into(),
        Query::ShareArchetype(s)=> format!("ShareArchetype(to={})", s.target_collection),
        Query::Narrate(n)       => format!("Narrate(window={:?}, format={:?})", n.window_hours, n.format),
        Query::Psychoanalyze(_) => "Psychoanalyze(MetaScan + AdjacencyScan)".to_string(),
        // ── NQL 2.0 EXPLAIN entries ──────────────────────
        Query::Union(uq) => format!("Union(queries={}, dedup={})",
            uq.queries.len(),
            uq.all.iter().filter(|a| !**a).count(),
        ),
        Query::Unwind(uq) => format!("Unwind(alias={})", uq.alias),
        Query::ShortestPath(sp) => format!("ShortestPath({} -> {}, limit={:?})",
            sp.from.alias, sp.to.alias, sp.limit),
        Query::MatchElites(me) => format!("MatchElites(limit={}, collection={:?})",
            me.limit.unwrap_or(10), me.collection),
        Query::MeasureTension(mt) => format!("MeasureTension({} vs {})",
            mt.node_a.alias, mt.node_b.alias),
        Query::MeasureTgc(tgc) => format!("MeasureTgc(collection={:?})", tgc.collection),
        Query::FindNearest(fq) => format!("FindNearest(space={:?}, limit={:?})",
            fq.space, fq.limit),
    };
    let full_plan = format!("{} | {}", plan, cost);
    Ok(vec![QueryResult::ExplainPlan(full_plan)])
}

// ─────────────────────────────────────────────
// PSYCHOANALYZE executor
// ─────────────────────────────────────────────

fn execute_psychoanalyze(
    pq:        &PsychoanalyzeQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    // Resolve target node ID
    let node_id = match &pq.target {
        ReconstructTarget::Param(p) => {
            params.get(p)
                .and_then(|v| if let ParamValue::Uuid(u) = v { Some(*u) } else { None })
                .ok_or_else(|| QueryError::ParamNotFound { name: p.clone() })?
        }
        ReconstructTarget::Alias(_a) => {
            return Err(QueryError::Execution("PSYCHOANALYZE requires a $param, not an alias".into()));
        }
    };

    // Load NodeMeta
    let meta = storage.get_node_meta(&node_id)
        .map_err(|e| QueryError::Execution(e.to_string()))?
        .ok_or_else(|| QueryError::Execution(format!("node {} not found", node_id)))?;

    // Gather adjacency info
    let neighbors_out = adjacency.neighbors_out(&node_id);
    let neighbors_in  = adjacency.neighbors_in(&node_id);

    // Classify creation source
    let origin = if meta.lsystem_generation > 0 {
        format!("L-System generation {}", meta.lsystem_generation)
    } else {
        "manual insertion".to_string()
    };

    // Classify structural role based on depth
    let structural_role = if meta.depth < 0.2 {
        "core semantic (near center)"
    } else if meta.depth < 0.5 {
        "mid-level concept"
    } else if meta.depth < 0.8 {
        "specific knowledge"
    } else {
        "peripheral / episodic (near boundary)"
    };

    // Classify health status
    let health = if meta.is_phantom {
        "phantom (structural scar — topology preserved, inactive)"
    } else if meta.energy <= 0.0 {
        "depleted (energy = 0, prunable)"
    } else if meta.energy < 0.3 {
        "fading (low energy, at risk)"
    } else if meta.energy > 0.85 {
        "hyperactive (potential inflation)"
    } else {
        "healthy"
    };

    // Check for fusion metadata
    let fusion_info = meta.content.get("fusion")
        .map(|f| f.clone())
        .unwrap_or(serde_json::Value::Null);

    // Build lineage JSON
    let lineage = serde_json::json!({
        "node_id": node_id.to_string(),
        "origin": origin,
        "node_type": format!("{:?}", meta.node_type),
        "created_at": meta.created_at,
        "lsystem_generation": meta.lsystem_generation,
        "structural_role": structural_role,
        "depth": meta.depth,
        "energy": meta.energy,
        "health_status": health,
        "is_phantom": meta.is_phantom,
        "hausdorff_local": meta.hausdorff_local,
        "expires_at": meta.expires_at,
        "connections": {
            "outgoing": neighbors_out.len(),
            "incoming": neighbors_in.len(),
            "total": neighbors_out.len() + neighbors_in.len(),
        },
        "fusion": fusion_info,
        "metadata": meta.metadata,
    });

    Ok(vec![QueryResult::PsychoanalyzeResult { node_id, lineage }])
}

// ─────────────────────────────────────────────
// MATCH executor
// ─────────────────────────────────────────────

fn execute_match(
    query:          &MatchQuery,
    storage:        &GraphStorage,
    adjacency:      &AdjacencyIndex,
    params:         &Params,
    indexed_fields: &HashSet<String>,
    gas:            &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    match &query.pattern {
        Pattern::Node(np) => execute_node_match(query, np, storage, adjacency, params, indexed_fields, gas),
        Pattern::Path(pp) => execute_path_match(query, pp, storage, adjacency, params, gas),
    }
}

// ── Node match ────────────────────────────────

fn execute_node_match(
    query:          &MatchQuery,
    np:             &NodePattern,
    storage:        &GraphStorage,
    adjacency:      &AdjacencyIndex,
    params:         &Params,
    indexed_fields: &HashSet<String>,
    gas:            &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    let alias = &np.alias;

    // ── Fast path 0: DSI Generative Retrieval ──
    if let Some(semantic_str) = &np.semantic_id {
        let codes: Vec<u16> = semantic_str.split('.')
            .map(|s| s.parse::<u16>().map_err(|_| QueryError::Execution(format!("invalid semantic ID part: {}", s))))
            .collect::<Result<_, _>>()?;
        let sid = SemanticId::new(codes);
        let prefix_bytes = sid.to_prefix_bytes();
            
        let ids = storage.scan_nodes_by_dsi_prefix(&prefix_bytes)?;
        let mut dsi_nodes = Vec::with_capacity(ids.len());
        
        for id in ids {
            if let Some(node) = storage.get_node(&id)? {
                gas.d_node()?;
                // If label is specified, check it
                if let Some(label) = &np.label {
                    let label_lc = label.to_lowercase();
                    if format!("{:?}", node.node_type).to_lowercase() != label_lc {
                        continue;
                    }
                }
                // Check WHERE conditions
                let binding = vec![(alias.as_str(), &node)];
                if eval_conditions(&query.conditions, &binding, params, storage, adjacency)? {
                    dsi_nodes.push(QueryResult::Node(node));
                }
            }
        }
        return Ok(dsi_nodes);
    }

    // ── Fast path 1: use energy secondary index for energy range queries ──
    // ── Fast path 2: use metadata secondary index for indexed field queries ──
    let nodes = if let Some((min_e, max_e)) = extract_energy_range_hint(query) {
        storage.scan_nodes_energy_range(min_e, max_e)?
    } else if let Some(indexed_ids) = extract_meta_index_hint(query, indexed_fields, params, storage) {
        // Load only the nodes that matched the index scan
        let mut out = Vec::with_capacity(indexed_ids.len());
        for id in indexed_ids {
            if let Ok(Some(node)) = storage.get_node(&id) {
                out.push(node);
            }
        }
        out
    } else {
        storage.scan_nodes()?
    };

    // Gas consumption for nodes scanned
    // Filter by label (node_type)
    let typed: Vec<Node> = if let Some(label) = &np.label {
        let label_lc = label.to_lowercase();
        nodes.into_iter()
            .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label_lc)
            .collect()
    } else {
        nodes
    };

    // Apply WHERE conditions — parallel for large scans, sequential for small ones
    let filtered: Vec<Node> = if typed.len() >= PARALLEL_SCAN_THRESHOLD {
        // Parallel filter: each thread evaluates conditions independently
        let alias_str: &str = alias.as_str();
        typed.into_par_iter()
            .filter(|node| {
                let binding = vec![(alias_str, node)];
                eval_conditions(&query.conditions, &binding, params, storage, adjacency)
                    .unwrap_or(false)
            })
            .collect()
    } else {
        let mut out = Vec::new();
        for node in typed {
            let binding = vec![(alias.as_str(), &node)];
            if eval_conditions(&query.conditions, &binding, params, storage, adjacency)? {
                out.push(node);
            }
        }
        out
    };
    // Re-bind as `mut` for subsequent transformations
    let mut filtered = filtered;

    // Apply ORDER BY
    if let Some(order) = &query.ret.order_by {
        sort_nodes(&mut filtered, alias, order, params, storage, adjacency)?;
    }

    // Apply DISTINCT before SKIP/LIMIT so pagination is over the deduplicated set.
    if query.ret.distinct {
        let mut seen = std::collections::HashSet::new();
        filtered.retain(|n| seen.insert(n.id));
    }

    // Apply SKIP
    if let Some(skip) = query.ret.skip {
        if skip < filtered.len() {
            filtered = filtered.split_off(skip);
        } else {
            filtered.clear();
        }
    }

    // Apply LIMIT
    if let Some(limit) = query.ret.limit {
        filtered.truncate(limit);
    }

    // Check if this is an aggregation query
    let has_agg = query.ret.items.iter().any(|item| {
        matches!(item.expr, ReturnExpr::Aggregate { .. })
    });

    if has_agg {
        return execute_aggregation(&filtered, alias, &query.ret);
    }

    Ok(filtered.into_iter().map(QueryResult::Node).collect())
}

// ── Path match ────────────────────────────────

fn execute_path_match(
    query:     &MatchQuery,
    pp:        &PathPattern,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    gas:       &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    // Multi-hop BFS when hop_range is present (e.g. *2..4)
    if pp.hop_range.is_some() {
        return execute_multi_hop_path(query, pp, storage, adjacency, params, gas);
    }

    let from_alias = &pp.from.alias;
    let to_alias   = &pp.to.alias;
    let edge_alias = pp.edge_alias.as_deref();

    // Collect (from_node, edge, to_node) triples
    let mut triples: Vec<(Node, Edge, Node)> = Vec::new();
    let edges = storage.scan_edges()?;
    for _ in 0..edges.len() { gas.d_edge()?; }

    for edge in edges {
        // Direction filter
        let (from_id, to_id) = match pp.direction {
            Direction::Out => (edge.from, edge.to),
            Direction::In  => (edge.to,   edge.from),
        };

        // Edge-type filter
        if let Some(label) = &pp.edge_label {
            let type_str = format!("{:?}", edge.edge_type).to_lowercase();
            if type_str != label.to_lowercase() {
                continue;
            }
        }

        // Label filters on from/to nodes
        let from_node = match storage.get_node(&from_id)? {
            Some(n) => n,
            None    => continue,
        };
        if let Some(label) = &pp.from.label {
            if format!("{:?}", from_node.node_type).to_lowercase() != label.to_lowercase() {
                continue;
            }
        }
        let to_node = match storage.get_node(&to_id)? {
            Some(n) => n,
            None    => continue,
        };
        if let Some(label) = &pp.to.label {
            if format!("{:?}", to_node.node_type).to_lowercase() != label.to_lowercase() {
                continue;
            }
        }

        // Evaluate WHERE conditions with node + edge bindings
        let node_binding = vec![
            (from_alias.as_str(), &from_node),
            (to_alias.as_str(),   &to_node),
        ];
        let edge_binding: Vec<(&str, &Edge)> = if let Some(ea) = edge_alias {
            vec![(ea, &edge)]
        } else {
            vec![]
        };
        if eval_conditions_with_edges(&query.conditions, &node_binding, &edge_binding, params, storage, adjacency)? {
            triples.push((from_node, edge, to_node));
        }
    }

    // ORDER BY — supports both node and edge properties
    if let Some(order) = &query.ret.order_by {
        let mut keyed: Vec<(f64, (Node, Edge, Node))> = triples
            .drain(..)
            .map(|(f, e, t)| {
                let key = compute_order_key_with_edge(&order.expr, from_alias, to_alias, edge_alias, &f, &t, &e, params, storage, adjacency)
                    .unwrap_or(f64::MAX);
                (key, (f, e, t))
            })
            .collect();
        if order.dir == OrderDir::Asc {
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        triples.extend(keyed.into_iter().map(|(_, t)| t));
    }

    // DISTINCT — deduplicate by (from_id, to_id) pair
    if query.ret.distinct {
        let mut seen = HashSet::new();
        triples.retain(|(f, _, t)| seen.insert((f.id, t.id)));
    }

    // SKIP
    if let Some(skip) = query.ret.skip {
        if skip < triples.len() {
            triples = triples.split_off(skip);
        } else {
            triples.clear();
        }
    }

    // LIMIT
    if let Some(limit) = query.ret.limit {
        triples.truncate(limit);
    }

    Ok(triples
        .into_iter()
        .map(|(from, _edge, to)| QueryResult::NodePair { from, to })
        .collect())
}

/// Execute a variable-length path match with bounded BFS.
///
/// For a query like `MATCH (a)-[:TYPE*2..4]->(b)`, this performs a BFS
/// from every candidate start node, following edges that match the label
/// and direction, collecting `(start, reached)` pairs where the reached
/// node is at depth `[min_hops, max_hops]` from the start.
fn execute_multi_hop_path(
    query:     &MatchQuery,
    pp:        &PathPattern,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    _gas:      &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    let from_alias = &pp.from.alias;
    let to_alias   = &pp.to.alias;
    let hr = pp.hop_range.as_ref().unwrap();
    let min_hops = hr.min;
    let max_hops = hr.max;

    let edge_label_lc = pp.edge_label.as_ref().map(|l| l.to_lowercase());

    // Collect all candidate start nodes (filtered by from-label if any)
    let all_nodes = storage.scan_nodes()?;
    let start_nodes: Vec<&Node> = if let Some(label) = &pp.from.label {
        let label_lc = label.to_lowercase();
        all_nodes.iter()
            .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label_lc)
            .collect()
    } else {
        all_nodes.iter().collect()
    };

    let mut pairs: Vec<(Node, Node)> = Vec::new();

    for start in &start_nodes {
        // BFS from this start node
        let reached = bounded_bfs(
            start.id,
            min_hops,
            max_hops,
            &edge_label_lc,
            &pp.direction,
            adjacency,
        );

        for to_id in reached {
            // Filter to-node by label if specified
            let to_node = match storage.get_node(&to_id)? {
                Some(n) => n,
                None    => continue,
            };
            if let Some(label) = &pp.to.label {
                let label_lc = label.to_lowercase();
                if format!("{:?}", to_node.node_type).to_lowercase() != label_lc {
                    continue;
                }
            }

            // Evaluate WHERE conditions with both aliases bound
            let binding = vec![
                (from_alias.as_str(), *start),
                (to_alias.as_str(),   &to_node),
            ];
            if eval_conditions(&query.conditions, &binding, params, storage, adjacency)? {
                pairs.push(((*start).clone(), to_node));
            }
        }
    }

    // ORDER BY
    if let Some(order) = &query.ret.order_by {
        let mut keyed: Vec<(f64, (Node, Node))> = pairs
            .drain(..)
            .map(|(f, t)| {
                let key = compute_order_key(&order.expr, to_alias, &t, params, storage, adjacency)
                    .unwrap_or(f64::MAX);
                (key, (f, t))
            })
            .collect();
        if order.dir == OrderDir::Asc {
            keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        pairs.extend(keyed.into_iter().map(|(_, p)| p));
    }

    // DISTINCT
    if query.ret.distinct {
        let mut seen = HashSet::new();
        pairs.retain(|(f, t)| seen.insert((f.id, t.id)));
    }

    // SKIP
    if let Some(skip) = query.ret.skip {
        if skip < pairs.len() {
            pairs = pairs.split_off(skip);
        } else {
            pairs.clear();
        }
    }

    // LIMIT
    if let Some(limit) = query.ret.limit {
        pairs.truncate(limit);
    }

    Ok(pairs
        .into_iter()
        .map(|(from, to)| QueryResult::NodePair { from, to })
        .collect())
}

/// Bounded BFS: from `start`, follow edges matching `edge_label` and
/// `direction` up to `max_hops` steps. Return all node IDs reachable
/// at depth `[min_hops, max_hops]` (inclusive). Cycle-safe via visited set.
fn bounded_bfs(
    start:      Uuid,
    min_hops:   usize,
    max_hops:   usize,
    edge_label: &Option<String>,  // already lowercased
    direction:  &Direction,
    adjacency:  &AdjacencyIndex,
) -> Vec<Uuid> {
    let mut visited: HashSet<Uuid> = HashSet::new();
    visited.insert(start);

    // (node_id, depth)
    let mut queue: VecDeque<(Uuid, usize)> = VecDeque::new();
    queue.push_back((start, 0));

    let mut results: Vec<Uuid> = Vec::new();

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }

        // Get neighbors based on direction
        let entries: Vec<AdjEntry> = match direction {
            Direction::Out => adjacency.entries_out(&current),
            Direction::In  => adjacency.entries_in(&current),
        };

        for entry in entries {
            // Edge-type filter
            if let Some(label) = edge_label {
                let type_str = format!("{:?}", entry.edge_type).to_lowercase();
                if type_str != *label {
                    continue;
                }
            }

            let neighbor = entry.neighbor_id;
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);

            let next_depth = depth + 1;
            if next_depth >= min_hops {
                results.push(neighbor);
            }
            if next_depth < max_hops {
                queue.push_back((neighbor, next_depth));
            }
        }
    }

    results
}

// ─────────────────────────────────────────────
// DIFFUSE executor
// ─────────────────────────────────────────────

fn execute_diffuse(
    query:     &DiffuseQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    gas:       &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    // Basic gas consumption based on walk depth
    gas.consume(query.max_hops as usize * 5)?; // 5 units per hop as a heuristic

    let start_id = resolve_diffuse_from(&query.from, params)?;

    let config = DiffusionConfig {
        steps:       query.max_hops,
        energy_bias: 1.0,
        seed:        None,
    };

    let path = diffusion_walk(storage, adjacency, start_id, &config)?;
    Ok(vec![QueryResult::DiffusionPath(path)])
}

// ─────────────────────────────────────────────
// RECONSTRUCT executor (Phase 11)
// ─────────────────────────────────────────────

fn execute_reconstruct(
    query:  &ReconstructQuery,
    params: &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    let node_id = match &query.target {
        ReconstructTarget::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Uuid(u)) => *u,
            Some(ParamValue::Str(s))  => {
                Uuid::parse_str(s)
                    .map_err(|_| QueryError::Execution(format!("param ${name} is not a valid UUID")))?
            }
            _ => return Err(QueryError::Execution(format!("param ${name} must be a UUID"))),
        },
        ReconstructTarget::Alias(a) => {
            return Err(QueryError::Execution(
                format!("alias binding '{}' not yet supported in RECONSTRUCT", a)
            ));
        }
    };

    Ok(vec![QueryResult::ReconstructRequest {
        node_id,
        modality: query.modality.clone(),
        quality:  query.quality.clone(),
    }])
}

// ─────────────────────────────────────────────
// MERGE executor (Phase D)
// ─────────────────────────────────────────────

fn execute_merge(
    query:  &MergeQuery,
    params: &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    match &query.pattern {
        MergePattern::Node(np) => {
            let match_keys = props_to_json(&np.properties, params)?;
            let on_create  = sets_to_json(&query.on_create, params)?;
            let on_match   = sets_to_json(&query.on_match, params)?;

            Ok(vec![QueryResult::MergeNodeRequest {
                node_type:     np.label.clone(),
                match_keys,
                on_create_set: on_create,
                on_match_set:  on_match,
            }])
        }
        MergePattern::Edge(ep) => {
            let from_match = props_to_json(&ep.from.properties, params)?;
            let to_match   = props_to_json(&ep.to.properties, params)?;
            let on_create  = sets_to_json(&query.on_create, params)?;
            let on_match   = sets_to_json(&query.on_match, params)?;

            Ok(vec![QueryResult::MergeEdgeRequest {
                from_match,
                to_match,
                edge_type: ep.edge_label.clone(),
                on_create_set: on_create,
                on_match_set:  on_match,
            }])
        }
    }
}

// ─────────────────────────────────────────────
// CREATE executor
// ─────────────────────────────────────────────

fn execute_create(
    query:  &CreateQuery,
    params: &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    let properties = props_to_json(&query.properties, params)?;
    Ok(vec![QueryResult::CreateNodeRequest {
        node_type:  query.label.clone(),
        properties,
    }])
}

// ─────────────────────────────────────────────
// MATCH … SET executor
// ─────────────────────────────────────────────

fn execute_match_set(
    query:     &MatchSetQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    gas:       &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    // Resolve matched nodes (need full nodes for per-node arithmetic)
    let matched = resolve_match_nodes(&query.pattern, &query.conditions, storage, adjacency, params, gas)?;

    // Evaluate SET assignments per node (supports n.count = n.count + 1)
    let mut updates = Vec::with_capacity(matched.len());
    for node in &matched {
        let alias = match &query.pattern {
            Pattern::Node(np) => &np.alias,
            Pattern::Path(pp) => &pp.from.alias,
        };
        let binding = vec![(alias.as_str(), node)];
        let mut map = serde_json::Map::new();
        for sa in &query.assignments {
            let val = eval_expr(&sa.value, &binding, params, storage, adjacency)?;
            map.insert(sa.field.clone(), value_to_json(&val));
        }
        updates.push((node.id, serde_json::Value::Object(map)));
    }
    Ok(vec![QueryResult::SetRequest { updates }])
}

/// Resolve matched nodes (full Node objects) from a MATCH pattern + WHERE.
fn resolve_match_nodes(
    pattern:    &Pattern,
    conditions: &[Condition],
    storage:    &GraphStorage,
    adjacency:  &AdjacencyIndex,
    params:     &Params,
    gas:        &mut GasTracker,
) -> Result<Vec<Node>, QueryError> {
    match pattern {
        Pattern::Node(np) => {
            let nodes = storage.scan_nodes()?;
            
            // Gas consumption for scanning nodes
            for _ in 0..nodes.len() {
                gas.d_node()?;
            }
            let alias = &np.alias;

            let typed: Vec<Node> = if let Some(label) = &np.label {
                let label_lc = label.to_lowercase();
                nodes.into_iter()
                    .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label_lc)
                    .collect()
            } else {
                nodes
            };

            let mut out = Vec::new();
            for node in typed {
                let binding = vec![(alias.as_str(), &node)];
                if eval_conditions(conditions, &binding, params, storage, adjacency)? {
                    out.push(node);
                }
            }
            Ok(out)
        }
        Pattern::Path(_) => {
            Err(QueryError::Execution("SET on path patterns not yet supported".into()))
        }
    }
}

/// Convert an executor Value to serde_json::Value.
fn value_to_json(val: &Value) -> serde_json::Value {
    match val {
        Value::Float(f) => serde_json::json!(*f),
        Value::Str(s)   => serde_json::json!(s),
        Value::Bool(b)  => serde_json::json!(*b),
        Value::Null     => serde_json::Value::Null,
    }
}

// ─────────────────────────────────────────────
// MATCH … DELETE executor
// ─────────────────────────────────────────────

fn execute_match_delete(
    query:     &MatchDeleteQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
    gas:       &mut GasTracker,
) -> Result<Vec<QueryResult>, QueryError> {
    let matched_ids = resolve_match_ids(&query.pattern, &query.conditions, storage, adjacency, params, gas)?;
    Ok(vec![QueryResult::DeleteRequest { matched_ids, detach: query.detach }])
}

/// Resolve matched node IDs from a MATCH pattern + WHERE conditions.
/// Shared by MATCH…SET and MATCH…DELETE executors.
fn resolve_match_ids(
    pattern:    &Pattern,
    conditions: &[Condition],
    storage:    &GraphStorage,
    adjacency:  &AdjacencyIndex,
    params:     &Params,
    gas:        &mut GasTracker,
) -> Result<Vec<Uuid>, QueryError> {
    match pattern {
        Pattern::Node(np) => {
            let nodes = storage.scan_nodes()?;

            // Gas consumption for scanning nodes
            for _ in 0..nodes.len() {
                gas.d_node()?;
            }
            let alias = &np.alias;

            // Filter by label
            let typed: Vec<Node> = if let Some(label) = &np.label {
                let label_lc = label.to_lowercase();
                nodes.into_iter()
                    .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label_lc)
                    .collect()
            } else {
                nodes
            };

            // Apply WHERE conditions
            let mut ids = Vec::new();
            for node in typed {
                let binding = vec![(alias.as_str(), &node)];
                if eval_conditions(conditions, &binding, params, storage, adjacency)? {
                    ids.push(node.id);
                }
            }
            Ok(ids)
        }
        Pattern::Path(_) => {
            Err(QueryError::Execution("SET/DELETE on path patterns not yet supported".into()))
        }
    }
}

/// Convert merge-pattern properties `[(key, Expr)]` into a JSON object.
fn props_to_json(
    props:  &[(String, Expr)],
    params: &Params,
) -> Result<serde_json::Value, QueryError> {
    let mut map = serde_json::Map::new();
    for (key, expr) in props {
        map.insert(key.clone(), expr_to_json(expr, params)?);
    }
    Ok(serde_json::Value::Object(map))
}

/// Convert `SetAssignment` list into a JSON object `{ field: value, … }`.
fn sets_to_json(
    sets:   &[SetAssignment],
    params: &Params,
) -> Result<serde_json::Value, QueryError> {
    let mut map = serde_json::Map::new();
    for sa in sets {
        map.insert(sa.field.clone(), expr_to_json(&sa.value, params)?);
    }
    Ok(serde_json::Value::Object(map))
}

/// Resolve an `Expr` to a `serde_json::Value` for MERGE property maps.
fn expr_to_json(expr: &Expr, params: &Params) -> Result<serde_json::Value, QueryError> {
    match expr {
        Expr::Float(f)  => Ok(serde_json::json!(*f)),
        Expr::Int(i)    => Ok(serde_json::json!(*i)),
        Expr::Str(s)    => Ok(serde_json::json!(s)),
        Expr::Bool(b)   => Ok(serde_json::json!(*b)),
        Expr::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => Ok(serde_json::json!(*f)),
            Some(ParamValue::Int(i))   => Ok(serde_json::json!(*i)),
            Some(ParamValue::Str(s))   => Ok(serde_json::json!(s)),
            Some(ParamValue::Uuid(u))  => Ok(serde_json::json!(u.to_string())),
            Some(ParamValue::Vector(v)) => Ok(serde_json::json!(v)),
            None => Err(QueryError::ParamNotFound { name: name.clone() }),
        },
        _ => Err(QueryError::Execution(
            "MERGE properties only support literals and $params".into()
        )),
    }
}

fn resolve_diffuse_from(from: &DiffuseFrom, params: &Params) -> Result<Uuid, QueryError> {
    match from {
        DiffuseFrom::Param(name) => {
            match params.get(name.as_str()) {
                Some(ParamValue::Uuid(u)) => Ok(*u),
                Some(ParamValue::Str(s))  => {
                    Uuid::parse_str(s)
                        .map_err(|_| QueryError::Execution(format!("param ${name} is not a valid UUID")))
                }
                _ => Err(QueryError::Execution(format!("param ${name} must be a UUID"))),
            }
        }
        DiffuseFrom::Alias(a) => {
            Err(QueryError::Execution(format!("alias binding '{}' not yet supported in DIFFUSE FROM", a)))
        }
    }
}

// ─────────────────────────────────────────────
// Condition evaluation
// ─────────────────────────────────────────────

/// Evaluate all conditions (ANDed together at the top level of the vec).
fn eval_conditions(
    conditions: &[Condition],
    binding:    &[(&str, &Node)],
    params:     &Params,
    storage:    &GraphStorage,
    adjacency:  &AdjacencyIndex,
) -> Result<bool, QueryError> {
    eval_conditions_with_edges(conditions, binding, &[], params, storage, adjacency)
}

/// Evaluate conditions with optional edge bindings for `r.field` access.
fn eval_conditions_with_edges(
    conditions:    &[Condition],
    binding:       &[(&str, &Node)],
    edge_bindings: &[(&str, &Edge)],
    params:        &Params,
    storage:       &GraphStorage,
    adjacency:     &AdjacencyIndex,
) -> Result<bool, QueryError> {
    for cond in conditions {
        if !eval_condition_with_edges(cond, binding, edge_bindings, params, storage, adjacency)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn eval_condition_with_edges(
    cond:          &Condition,
    binding:       &[(&str, &Node)],
    edge_bindings: &[(&str, &Edge)],
    params:        &Params,
    storage:       &GraphStorage,
    adjacency:     &AdjacencyIndex,
) -> Result<bool, QueryError> {
    match cond {
        Condition::Compare { left, op, right } => {
            let l = eval_expr_with_edges(left, binding, edge_bindings, params, storage, adjacency)?;
            let r = eval_expr_with_edges(right, binding, edge_bindings, params, storage, adjacency)?;
            compare_values(&l, op, &r)
        }
        Condition::And(a, b) => {
            Ok(eval_condition_with_edges(a, binding, edge_bindings, params, storage, adjacency)?
               && eval_condition_with_edges(b, binding, edge_bindings, params, storage, adjacency)?)
        }
        Condition::Or(a, b) => {
            Ok(eval_condition_with_edges(a, binding, edge_bindings, params, storage, adjacency)?
               || eval_condition_with_edges(b, binding, edge_bindings, params, storage, adjacency)?)
        }
        Condition::Not(c) => Ok(!eval_condition_with_edges(c, binding, edge_bindings, params, storage, adjacency)?),

        Condition::In { expr, values } => {
            let needle = eval_expr_with_edges(expr, binding, edge_bindings, params, storage, adjacency)?;
            for v in values {
                let candidate = eval_expr_with_edges(v, binding, edge_bindings, params, storage, adjacency)?;
                if values_equal(&needle, &candidate) {
                    return Ok(true);
                }
            }
            Ok(false)
        }

        Condition::Between { expr, low, high } => {
            let val = eval_expr_with_edges(expr, binding, edge_bindings, params, storage, adjacency)?;
            let lo  = eval_expr_with_edges(low, binding, edge_bindings, params, storage, adjacency)?;
            let hi  = eval_expr_with_edges(high, binding, edge_bindings, params, storage, adjacency)?;
            match (&val, &lo, &hi) {
                (Value::Float(v), Value::Float(l), Value::Float(h)) => Ok(*v >= *l && *v <= *h),
                _ => Err(QueryError::TypeMismatch {
                    context:    "BETWEEN".into(),
                    left_type:  val.type_name(),
                    right_type: lo.type_name(),
                }),
            }
        }

        Condition::StringOp { left, op, right } => {
            let l = eval_expr_with_edges(left, binding, edge_bindings, params, storage, adjacency)?;
            let r = eval_expr_with_edges(right, binding, edge_bindings, params, storage, adjacency)?;
            match (&l, &r) {
                (Value::Str(ls), Value::Str(rs)) => Ok(match op {
                    StringCompOp::Contains   => ls.contains(rs.as_str()),
                    StringCompOp::StartsWith => ls.starts_with(rs.as_str()),
                    StringCompOp::EndsWith   => ls.ends_with(rs.as_str()),
                }),
                _ => Err(QueryError::StringOpTypeMismatch {
                    op: format!("{:?}", op),
                }),
            }
        }

        // ── NQL 2.0: IS NULL ──────────────────────────────
        Condition::IsNull { expr } => {
            // Try evaluating the expression; if it resolves to Null or fails (field not found), it's null
            match eval_expr_with_edges(expr, binding, edge_bindings, params, storage, adjacency) {
                Ok(val) => Ok(val.is_null()),
                Err(_) => Ok(true), // field doesn't exist → treat as null
            }
        }

        // ── NQL 2.0: IS NOT NULL ──────────────────────────
        Condition::IsNotNull { expr } => {
            match eval_expr_with_edges(expr, binding, edge_bindings, params, storage, adjacency) {
                Ok(val) => Ok(!val.is_null()),
                Err(_) => Ok(false), // field doesn't exist → is null → IS NOT NULL = false
            }
        }

        // ── NQL 2.0: Regex =~ ─────────────────────────────
        Condition::Regex { expr, pattern } => {
            let val = eval_expr_with_edges(expr, binding, edge_bindings, params, storage, adjacency)?;
            let pat = eval_expr_with_edges(pattern, binding, edge_bindings, params, storage, adjacency)?;
            match (&val, &pat) {
                (Value::Str(text), Value::Str(regex_str)) => {
                    // Compile regex and match
                    let re = regex::Regex::new(regex_str)
                        .map_err(|e| QueryError::Execution(format!("invalid regex '{}': {}", regex_str, e)))?;
                    Ok(re.is_match(text))
                }
                _ => Err(QueryError::Execution(
                    format!("=~ requires string operands, got {} =~ {}", val.type_name(), pat.type_name()),
                )),
            }
        }

        // ── NQL 2.0: EXISTS { MATCH … WHERE … } ──────────
        Condition::Exists { pattern, conditions } => {
            // EXISTS evaluates a subquery pattern against current bindings
            // It returns true if any nodes/edges match the inner pattern + conditions
            match pattern {
                Pattern::Node(np) => {
                    let nodes = storage.scan_nodes()
                        .map_err(|e| QueryError::Execution(e.to_string()))?;
                    let typed: Vec<&Node> = if let Some(label) = &np.label {
                        let label_lc = label.to_lowercase();
                        nodes.iter()
                            .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label_lc)
                            .collect()
                    } else {
                        nodes.iter().collect()
                    };
                    for node in typed {
                        // Extend existing binding with the inner alias
                        let mut inner_binding: Vec<(&str, &Node)> = binding.to_vec();
                        inner_binding.push((np.alias.as_str(), node));
                        if eval_conditions_with_edges(conditions, &inner_binding, edge_bindings, params, storage, adjacency)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
                Pattern::Path(pp) => {
                    let edges = storage.scan_edges()
                        .map_err(|e| QueryError::Execution(e.to_string()))?;
                    for edge in &edges {
                        let (from_id, to_id) = match pp.direction {
                            Direction::Out => (edge.from, edge.to),
                            Direction::In  => (edge.to, edge.from),
                        };
                        if let Some(label) = &pp.edge_label {
                            let type_str = format!("{:?}", edge.edge_type).to_lowercase();
                            if type_str != label.to_lowercase() { continue; }
                        }
                        let from_node = match storage.get_node(&from_id) {
                            Ok(Some(n)) => n,
                            _ => continue,
                        };
                        let to_node = match storage.get_node(&to_id) {
                            Ok(Some(n)) => n,
                            _ => continue,
                        };
                        let mut inner_binding: Vec<(&str, &Node)> = binding.to_vec();
                        inner_binding.push((pp.from.alias.as_str(), &from_node));
                        inner_binding.push((pp.to.alias.as_str(), &to_node));
                        if eval_conditions_with_edges(conditions, &inner_binding, edge_bindings, params, storage, adjacency)? {
                            return Ok(true);
                        }
                    }
                    Ok(false)
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// Expression evaluation
// ─────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Value {
    Float(f64),
    Str(String),
    Bool(bool),
    Null,
}

impl Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Float(_) => "float",
            Value::Str(_)   => "string",
            Value::Bool(_)  => "bool",
            Value::Null     => "null",
        }
    }

    fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

/// Check if two Values are equal (used by IN operator).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null)         => true,
        (Value::Float(l), Value::Float(r)) => (l - r).abs() < 1e-10,
        (Value::Str(l),   Value::Str(r))   => l == r,
        (Value::Bool(l),  Value::Bool(r))  => l == r,
        _ => false,
    }
}

fn eval_expr(
    expr:      &Expr,
    binding:   &[(&str, &Node)],
    params:    &Params,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<Value, QueryError> {
    eval_expr_with_edges(expr, binding, &[], params, storage, adjacency)
}

fn eval_expr_with_edges(
    expr:          &Expr,
    binding:       &[(&str, &Node)],
    edge_bindings: &[(&str, &Edge)],
    params:        &Params,
    storage:       &GraphStorage,
    adjacency:     &AdjacencyIndex,
) -> Result<Value, QueryError> {
    match expr {
        Expr::Float(f) => Ok(Value::Float(*f)),
        Expr::Int(i)   => Ok(Value::Float(*i as f64)),
        Expr::Str(s)   => Ok(Value::Str(s.clone())),
        Expr::Bool(b)  => Ok(Value::Bool(*b)),

        Expr::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => Ok(Value::Float(*f)),
            Some(ParamValue::Int(i))   => Ok(Value::Float(*i as f64)),
            Some(ParamValue::Str(s))   => Ok(Value::Str(s.clone())),
            Some(ParamValue::Uuid(u))  => Ok(Value::Str(u.to_string())),
            Some(ParamValue::Vector(_)) =>
                Err(QueryError::ParamTypeMismatch {
                    name: name.clone(), expected: "scalar", got: "Vector".into(),
                }),
            None =>
                Err(QueryError::ParamNotFound { name: name.clone() }),
        },

        Expr::Property { alias, field } => {
            // Try node bindings first
            if let Ok(node) = find_node(alias, binding) {
                return eval_field(node, field);
            }
            // Fall back to edge bindings (NQL-6: r.field for edge alias)
            if let Some((_, edge)) = edge_bindings.iter().find(|(a, _)| *a == alias) {
                return eval_edge_field(edge, field);
            }
            Err(QueryError::UnknownAlias { alias: alias.to_string() })
        }

        Expr::HyperbolicDist { alias, field, arg } => {
            if field != "embedding" {
                return Err(QueryError::Execution(
                    "HYPERBOLIC_DIST only supports the `.embedding` field".into()
                ));
            }
            let node = find_node(alias, binding)?;
            let query_vec = resolve_hdist_arg(arg, params)?;
            Ok(Value::Float(node.embedding.distance(&query_vec)))
        }

        Expr::SensoryDist { alias, field: _, arg } => {
            // SENSORY_DIST computes distance in embedding space between the node's
            // Poincaré embedding and a query vector supplied as a parameter.
            // In the full pipeline the gRPC handler maps the sensory latent into the
            // same Poincaré ball; here we use the node's primary embedding as proxy.
            let node = find_node(alias, binding)?;
            let query_vec = match params.get(arg.as_str()) {
                Some(ParamValue::Vector(v)) => PoincareVector::from_f64(v.clone()),
                _ => return Err(QueryError::Execution(
                    format!("param ${arg} must be a Vector for SENSORY_DIST")
                )),
            };
            Ok(Value::Float(node.embedding.distance(&query_vec)))
        }

        Expr::MathFunc { func, args } => {
            eval_math_func(func, args, binding, params, storage, adjacency)
        }

        Expr::BinOp { left, op, right } => {
            let lv = eval_expr_with_edges(left, binding, edge_bindings, params, storage, adjacency)?;
            let rv = eval_expr_with_edges(right, binding, edge_bindings, params, storage, adjacency)?;
            let (l, r) = match (&lv, &rv) {
                (Value::Float(a), Value::Float(b)) => (*a, *b),
                _ => return Err(QueryError::Execution(
                    format!("arithmetic requires numeric operands, got {:?} and {:?}", lv, rv),
                )),
            };
            Ok(Value::Float(match op {
                ArithOp::Add => l + r,
                ArithOp::Sub => l - r,
            }))
        }

        // ── NQL 2.0: CASE WHEN ────────────────────────────
        Expr::CaseWhen { branches, else_expr } => {
            for (cond, val_expr) in branches {
                if eval_condition_with_edges(cond, binding, edge_bindings, params, storage, adjacency)? {
                    return eval_expr_with_edges(val_expr, binding, edge_bindings, params, storage, adjacency);
                }
            }
            if let Some(else_val) = else_expr {
                eval_expr_with_edges(else_val, binding, edge_bindings, params, storage, adjacency)
            } else {
                Ok(Value::Null)
            }
        }
    }
}

fn resolve_hdist_arg(arg: &HDistArg, params: &Params) -> Result<PoincareVector, QueryError> {
    match arg {
        HDistArg::Vector(v) => Ok(PoincareVector::from_f64(v.clone())),
        HDistArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Vector(v)) => Ok(PoincareVector::from_f64(v.clone())),
            _ => Err(QueryError::Execution(
                format!("param ${name} must be a Vector for HYPERBOLIC_DIST")
            )),
        },
    }
}

fn find_node<'a>(alias: &str, binding: &[(&str, &'a Node)]) -> Result<&'a Node, QueryError> {
    binding.iter()
        .find(|(a, _)| *a == alias)
        .map(|(_, n)| *n)
        .ok_or_else(|| QueryError::UnknownAlias { alias: alias.to_string() })
}

fn eval_field(node: &Node, field: &str) -> Result<Value, QueryError> {
    match field {
        "energy"            => Ok(Value::Float(node.energy as f64)),
        "depth"             => Ok(Value::Float(node.depth as f64)),
        "hausdorff_local"   => Ok(Value::Float(node.hausdorff_local as f64)),
        "lsystem_generation"=> Ok(Value::Float(node.lsystem_generation as f64)),
        "id"                => Ok(Value::Str(node.id.to_string())),
        "created_at"        => Ok(Value::Float(node.created_at as f64)),
        "node_type"         => Ok(Value::Str(format!("{:?}", node.node_type))),
        "expires_at"        => Ok(Value::Float(node.expires_at.unwrap_or(0) as f64)),
        _ => {
            // Fall back to content JSON (arbitrary fields stored on the node)
            if let Some(val) = node.content.get(field) {
                return json_to_value(val);
            }
            // Fall back to metadata HashMap
            if let Some(val) = node.metadata.get(field) {
                return json_to_value(val);
            }
            Err(QueryError::UnknownField { field: field.to_string() })
        }
    }
}

/// Evaluate an edge field: `r.weight`, `r.edge_type`, `r.created_at`, or metadata.
fn eval_edge_field(edge: &Edge, field: &str) -> Result<Value, QueryError> {
    match field {
        "id"          => Ok(Value::Str(edge.id.to_string())),
        "from"        => Ok(Value::Str(edge.from.to_string())),
        "to"          => Ok(Value::Str(edge.to.to_string())),
        "weight"      => Ok(Value::Float(edge.weight as f64)),
        "edge_type"   => Ok(Value::Str(format!("{:?}", edge.edge_type))),
        "created_at"  => Ok(Value::Float(edge.created_at as f64)),
        _ => {
            // Fall back to edge metadata
            if let Some(val) = edge.metadata.get(field) {
                return json_to_value(val);
            }
            Err(QueryError::UnknownField { field: field.to_string() })
        }
    }
}

/// Convert a `serde_json::Value` to an executor `Value`.
fn json_to_value(val: &serde_json::Value) -> Result<Value, QueryError> {
    match val {
        serde_json::Value::Number(n) => Ok(Value::Float(n.as_f64().unwrap_or(0.0))),
        serde_json::Value::String(s) => Ok(Value::Str(s.clone())),
        serde_json::Value::Bool(b)   => Ok(Value::Bool(*b)),
        serde_json::Value::Null      => Ok(Value::Null),
        _ => Err(QueryError::Execution(format!("field value is not a scalar: {val}"))),
    }
}

// ─────────────────────────────────────────────
// Value comparison
// ─────────────────────────────────────────────

fn compare_values(left: &Value, op: &CompOp, right: &Value) -> Result<bool, QueryError> {
    // NULL comparisons: NULL = NULL is true, NULL != non-null is true, anything else with NULL is false
    if left.is_null() || right.is_null() {
        return Ok(match op {
            CompOp::Eq  => left.is_null() && right.is_null(),
            CompOp::Neq => left.is_null() != right.is_null(),
            _ => false, // NULL < x, NULL > x, etc. are always false
        });
    }
    match (left, right) {
        (Value::Float(l), Value::Float(r)) => Ok(apply_num_op(*l, op, *r)),
        (Value::Str(l),   Value::Str(r))   => Ok(apply_str_op(l, op, r)?),
        (Value::Bool(l),  Value::Bool(r))  => Ok(match op {
            CompOp::Eq  => l == r,
            CompOp::Neq => l != r,
            _ => return Err(QueryError::TypeMismatch {
                context: "bool comparison (only = and != supported)".into(),
                left_type: "bool", right_type: "bool",
            }),
        }),
        _ => Err(QueryError::TypeMismatch {
            context:    "comparison".into(),
            left_type:  left.type_name(),
            right_type: right.type_name(),
        }),
    }
}

fn apply_num_op(l: f64, op: &CompOp, r: f64) -> bool {
    match op {
        CompOp::Lt  => l < r,
        CompOp::Lte => l <= r,
        CompOp::Gt  => l > r,
        CompOp::Gte => l >= r,
        CompOp::Eq  => (l - r).abs() < 1e-10,
        CompOp::Neq => (l - r).abs() >= 1e-10,
    }
}

fn apply_str_op(l: &str, op: &CompOp, r: &str) -> Result<bool, QueryError> {
    match op {
        CompOp::Eq  => Ok(l == r),
        CompOp::Neq => Ok(l != r),
        _ => Err(QueryError::Execution("string comparison supports only = and !=".into())),
    }
}

// ─────────────────────────────────────────────
// Sorting
// ─────────────────────────────────────────────

fn sort_nodes(
    nodes:     &mut Vec<Node>,
    alias:     &str,
    order:     &OrderBy,
    params:    &Params,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<(), QueryError> {
    // Pre-compute sort keys to avoid repeated evaluation errors
    let mut keyed: Vec<(f64, Node)> = nodes
        .drain(..)
        .map(|node| {
            let key = compute_order_key(&order.expr, alias, &node, params, storage, adjacency)
                .unwrap_or(f64::MAX);
            (key, node)
        })
        .collect();

    if order.dir == OrderDir::Asc {
        keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    nodes.extend(keyed.into_iter().map(|(_, n)| n));
    Ok(())
}

fn compute_order_key(
    expr:      &OrderExpr,
    alias:     &str,
    node:      &Node,
    params:    &Params,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<f64, QueryError> {
    match expr {
        OrderExpr::Property(a, field) => {
            if a == alias {
                if let Value::Float(f) = eval_field(node, field)? {
                    return Ok(f);
                }
            }
            Err(QueryError::Execution(format!("ORDER BY: unknown alias '{a}'")))
        }
        OrderExpr::HyperbolicDist { alias: a, field, arg } => {
            if a != alias || field != "embedding" {
                return Err(QueryError::Execution("ORDER BY HYPERBOLIC_DIST: invalid prop".into()));
            }
            let query_vec = resolve_hdist_arg(arg, params)?;
            Ok(node.embedding.distance(&query_vec))
        }
        OrderExpr::SensoryDist { alias: a, field: _, arg } => {
            if a != alias {
                return Err(QueryError::Execution("ORDER BY SENSORY_DIST: unknown alias".into()));
            }
            let query_vec = match params.get(arg.as_str()) {
                Some(ParamValue::Vector(v)) => PoincareVector::from_f64(v.clone()),
                _ => return Err(QueryError::Execution(
                    format!("param ${arg} must be a Vector for SENSORY_DIST")
                )),
            };
            Ok(node.embedding.distance(&query_vec))
        }
        OrderExpr::MathFunc { func, args } => {
            let binding = vec![(alias, node)];
            match eval_math_func(func, args, &binding, params, storage, adjacency)? {
                Value::Float(f) => Ok(f),
                _ => Err(QueryError::Execution(
                    format!("ORDER BY {:?}: result is not numeric", func)
                )),
            }
        }
        OrderExpr::Alias(a) => {
            Err(QueryError::Execution(format!("ORDER BY alias '{a}' is not a sortable scalar")))
        }
    }
}

/// ORDER BY key computation with edge support for path queries.
fn compute_order_key_with_edge(
    expr:       &OrderExpr,
    from_alias: &str,
    to_alias:   &str,
    edge_alias: Option<&str>,
    from_node:  &Node,
    to_node:    &Node,
    edge:       &Edge,
    params:     &Params,
    storage:    &GraphStorage,
    adjacency:  &AdjacencyIndex,
) -> Result<f64, QueryError> {
    match expr {
        OrderExpr::Property(a, field) => {
            // Check node aliases
            if a == from_alias {
                if let Value::Float(f) = eval_field(from_node, field)? { return Ok(f); }
            }
            if a == to_alias {
                if let Value::Float(f) = eval_field(to_node, field)? { return Ok(f); }
            }
            // Check edge alias
            if edge_alias.map_or(false, |ea| a == ea) {
                if let Value::Float(f) = eval_edge_field(edge, field)? { return Ok(f); }
            }
            Err(QueryError::Execution(format!("ORDER BY: unknown alias '{a}'")))
        }
        _ => {
            // Fall back to to_node for non-property expressions
            compute_order_key(expr, to_alias, to_node, params, storage, adjacency)
        }
    }
}

// ─────────────────────────────────────────────
// Mathematician-named functions
// ─────────────────────────────────────────────

fn eval_math_func(
    func:      &MathFunc,
    args:      &[MathFuncArg],
    binding:   &[(&str, &Node)],
    params:    &Params,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<Value, QueryError> {
    match func {
        // ── Poincaré (alias for HYPERBOLIC_DIST) ──────────────
        MathFunc::PoincaNietzscheDBt => {
            let (node, query_vec) = resolve_dist_args(args, binding, params)?;
            Ok(Value::Float(node.embedding.distance(&query_vec)))
        }

        // ── Klein (Beltrami-Klein model distance) ─────────────
        // Maps Poincaré coords to Klein: k = 2p/(1+‖p‖²), then
        // computes Klein-metric distance (same geodesic distance, different formula).
        MathFunc::KleinDist => {
            let (node, query_vec) = resolve_dist_args(args, binding, params)?;
            let c1 = node.embedding.coords_f64();
            let c2 = query_vec.coords_f64();
            let k1 = poincare_to_klein(&c1);
            let k2 = poincare_to_klein(&c2);
            let dot: f64 = k1.iter().zip(&k2).map(|(a, b)| a * b).sum();
            let n1_sq: f64 = k1.iter().map(|x| x * x).sum();
            let n2_sq: f64 = k2.iter().map(|x| x * x).sum();
            let denom = ((1.0 - n1_sq) * (1.0 - n2_sq)).max(1e-15).sqrt();
            let cosh_d = ((1.0 - dot) / denom).max(1.0);
            Ok(Value::Float(cosh_d.acosh()))
        }

        // ── Minkowski (conformal factor λ = 2/(1−‖x‖²)) ──────
        // Higher values mean the node is closer to the boundary (deeper in hierarchy).
        MathFunc::MinkowskiNorm => {
            let node = resolve_single_prop_node(args, binding)?;
            let norm_sq: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let lambda = 2.0 / (1.0 - norm_sq).max(1e-15);
            Ok(Value::Float(lambda))
        }

        // ── Lobachevsky (angle of parallelism) ────────────────
        // Π(d) = 2·arctan(e^(−d)) — the fundamental Lobachevsky relation.
        MathFunc::LobachevskyAngle => {
            let (node, query_vec) = resolve_dist_args(args, binding, params)?;
            let d = node.embedding.distance(&query_vec);
            let angle = 2.0 * (-d).exp().atan();
            Ok(Value::Float(angle))
        }

        // ── Riemann (discrete Ollivier-Ricci curvature) ───────
        // Estimates local curvature from embedding vs graph distance.
        // κ ≈ 1 − (avg_embedding_dist / avg_graph_hops) for 1-hop neighbors.
        MathFunc::RiemannCurvature => {
            let node = resolve_alias_node(args, binding)?;
            let neighbors = adjacency.neighbors_out(&node.id);
            if neighbors.is_empty() {
                return Ok(Value::Float(0.0));
            }
            let mut total_emb_dist = 0.0;
            let mut count = 0usize;
            for nid in &neighbors {
                if let Ok(Some(neighbor)) = storage.get_node(nid) {
                    total_emb_dist += node.embedding.distance(&neighbor.embedding);
                    count += 1;
                }
            }
            if count == 0 { return Ok(Value::Float(0.0)); }
            // avg embedding dist vs 1 hop → curvature estimate
            let avg_dist = total_emb_dist / count as f64;
            let kappa = 1.0 - avg_dist; // flat = 0, curved = deviated
            Ok(Value::Float(kappa))
        }

        // ── Gauss (heat kernel) ───────────────────────────────
        // h_t(x) = exp(−d²/(4t)) where d = ‖embedding‖ (dist from origin).
        MathFunc::GaussKernel => {
            let (node, t) = resolve_alias_and_scalar(args, binding, params)?;
            let norm_sq: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let d_sq = norm_sq; // squared Euclidean distance from origin
            let kernel = (-d_sq / (4.0 * t.max(1e-15))).exp();
            Ok(Value::Float(kernel))
        }

        // ── Chebyshev (T_k(x) at normalized node position) ───
        // Chebyshev polynomial T_k(x) = cos(k·arccos(x)), with
        // x = 2·‖embedding‖ − 1 ∈ [−1, 1].
        MathFunc::ChebyshevCoeff => {
            let (node, k) = resolve_alias_and_scalar(args, binding, params)?;
            let norm: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            let x = (2.0 * norm - 1.0).clamp(-1.0, 1.0);
            let tk = (k * x.acos()).cos();
            Ok(Value::Float(tk))
        }

        // ── Ramanujan (local expansion ratio) ─────────────────
        // Ratio of actual out-degree to Ramanujan bound 2√(d−1).
        // Values ≥ 1.0 indicate expander-like local structure.
        MathFunc::RamanujanExpansion => {
            let node = resolve_alias_node(args, binding)?;
            let out_deg = adjacency.neighbors_out(&node.id).len() as f64;
            let in_deg  = adjacency.neighbors_in(&node.id).len() as f64;
            let total_deg = out_deg + in_deg;
            if total_deg <= 1.0 { return Ok(Value::Float(0.0)); }
            // Ramanujan bound: spectral gap ≥ 2√(d−1) for d-regular
            let ramanujan_bound = 2.0 * (total_deg - 1.0).max(0.0).sqrt();
            let expansion = total_deg / ramanujan_bound.max(1e-15);
            Ok(Value::Float(expansion))
        }

        // ── Hausdorff (local dimension) ───────────────────────
        // Returns the pre-computed hausdorff_local field on the node.
        MathFunc::HausdorffDim => {
            let node = resolve_alias_node(args, binding)?;
            Ok(Value::Float(node.hausdorff_local as f64))
        }

        // ── Euler (local Euler characteristic) ────────────────
        // χ = V − E for the 1-hop ego-graph (node + its neighbors).
        MathFunc::EulerChar => {
            let node = resolve_alias_node(args, binding)?;
            let neighbors = adjacency.neighbors_out(&node.id);
            let v = 1 + neighbors.len(); // node itself + neighbors
            // Count edges among the ego-graph nodes
            let mut ego_edges = neighbors.len(); // edges from node to each neighbor
            let neighbor_set: std::collections::HashSet<Uuid> = neighbors.iter().cloned().collect();
            for nid in &neighbors {
                for nid2 in adjacency.neighbors_out(nid) {
                    if neighbor_set.contains(&nid2) {
                        ego_edges += 1;
                    }
                }
            }
            let chi = v as f64 - ego_edges as f64;
            Ok(Value::Float(chi))
        }

        // ── Laplace (graph Laplacian diagonal score) ──────────
        // L_ii = degree(i) for unweighted graph.
        // Weighted: L_ii = Σ_j w_ij for all edges incident to i.
        MathFunc::LaplacianScore => {
            let node = resolve_alias_node(args, binding)?;
            let out_edges = adjacency.neighbors_out(&node.id);
            let in_edges  = adjacency.neighbors_in(&node.id);
            let degree = (out_edges.len() + in_edges.len()) as f64;
            Ok(Value::Float(degree))
        }

        // ── Fourier (graph Fourier coefficient) ───────────────
        // Approximation: cos(k·π·x) where x = ‖embedding‖ (normalized position).
        MathFunc::FourierCoeff => {
            let (node, k) = resolve_alias_and_scalar(args, binding, params)?;
            let norm: f64 = node.embedding.coords.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            let coeff = (k * std::f64::consts::PI * norm).cos();
            Ok(Value::Float(coeff))
        }

        // ── Dirichlet (local Dirichlet energy) ────────────────
        // E(f) = Σ_{j∈N(i)} (f(i) − f(j))²  where f = energy field.
        MathFunc::DirichletEnergy => {
            let node = resolve_alias_node(args, binding)?;
            let neighbors = adjacency.neighbors_out(&node.id);
            let mut energy_sum = 0.0;
            for nid in &neighbors {
                if let Ok(Some(neighbor)) = storage.get_node(nid) {
                    let diff = node.energy as f64 - neighbor.energy as f64;
                    energy_sum += diff * diff;
                }
            }
            Ok(Value::Float(energy_sum))
        }

        // ── Time functions ─────────────────────────────────────
        MathFunc::Now => {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            Ok(Value::Float(secs))
        }
        MathFunc::EpochMs => {
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as f64;
            Ok(Value::Float(ms))
        }
        MathFunc::Interval => {
            let duration_str = match &args[0] {
                MathFuncArg::Str(s) => s.clone(),
                MathFuncArg::Param(name) => match params.get(name.as_str()) {
                    Some(ParamValue::Str(s)) => s.clone(),
                    _ => return Err(QueryError::Execution(
                        format!("INTERVAL: param ${name} must be a string")
                    )),
                },
                _ => return Err(QueryError::Execution(
                    "INTERVAL expects a string argument like \"1h\", \"7d\", \"30m\"".into()
                )),
            };
            let seconds = parse_interval_str(&duration_str)?;
            Ok(Value::Float(seconds))
        }

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: String Functions ────────────────────
        // ═══════════════════════════════════════════════════

        MathFunc::Upper => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.to_uppercase()))
        }
        MathFunc::Lower => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.to_lowercase()))
        }
        MathFunc::Trim => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.trim().to_string()))
        }
        MathFunc::Ltrim => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.trim_start().to_string()))
        }
        MathFunc::Rtrim => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.trim_end().to_string()))
        }
        MathFunc::Length => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(s.chars().count() as f64))
        }
        MathFunc::Substring => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            let start = resolve_scalar_arg(&args[1], binding, params, storage, adjacency)? as usize;
            let len = resolve_scalar_arg(&args[2], binding, params, storage, adjacency)? as usize;
            let result: String = s.chars().skip(start).take(len).collect();
            Ok(Value::Str(result))
        }
        MathFunc::Replace => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            let search = resolve_string_arg(&args[1], binding, params, storage, adjacency)?;
            let replacement = resolve_string_arg(&args[2], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.replace(&search, &replacement)))
        }
        MathFunc::Concat => {
            let mut result = String::new();
            for arg in args {
                let s = resolve_string_arg(arg, binding, params, storage, adjacency)?;
                result.push_str(&s);
            }
            Ok(Value::Str(result))
        }
        MathFunc::Reverse => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Str(s.chars().rev().collect()))
        }
        MathFunc::Split => {
            let s = resolve_string_arg(&args[0], binding, params, storage, adjacency)?;
            let delim = resolve_string_arg(&args[1], binding, params, storage, adjacency)?;
            // Return as a JSON-like string representation of the array
            let parts: Vec<String> = s.split(&delim).map(|p| p.to_string()).collect();
            Ok(Value::Str(format!("{:?}", parts)))
        }

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: Math Functions ──────────────────────
        // ═══════════════════════════════════════════════════

        MathFunc::Abs => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.abs()))
        }
        MathFunc::Ceil => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.ceil()))
        }
        MathFunc::Floor => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.floor()))
        }
        MathFunc::Round => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.round()))
        }
        MathFunc::Sqrt => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.sqrt()))
        }
        MathFunc::Log => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.ln()))
        }
        MathFunc::Log10 => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(x.log10()))
        }
        MathFunc::Pow => {
            let base = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            let exp = resolve_scalar_arg(&args[1], binding, params, storage, adjacency)?;
            Ok(Value::Float(base.powf(exp)))
        }
        MathFunc::Sign => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            Ok(Value::Float(if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }))
        }
        MathFunc::Mod => {
            let x = resolve_scalar_arg(&args[0], binding, params, storage, adjacency)?;
            let y = resolve_scalar_arg(&args[1], binding, params, storage, adjacency)?;
            Ok(Value::Float(x % y))
        }

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: Type Casting ────────────────────────
        // ═══════════════════════════════════════════════════

        MathFunc::ToInt => {
            let val = resolve_any_arg(&args[0], binding, params, storage, adjacency)?;
            match val {
                Value::Float(f) => Ok(Value::Float((f as i64) as f64)),
                Value::Str(s)   => Ok(Value::Float(s.parse::<i64>().unwrap_or(0) as f64)),
                Value::Bool(b)  => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
                Value::Null     => Ok(Value::Null),
            }
        }
        MathFunc::ToFloat => {
            let val = resolve_any_arg(&args[0], binding, params, storage, adjacency)?;
            match val {
                Value::Float(f) => Ok(Value::Float(f)),
                Value::Str(s)   => Ok(Value::Float(s.parse::<f64>().unwrap_or(0.0))),
                Value::Bool(b)  => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
                Value::Null     => Ok(Value::Null),
            }
        }
        MathFunc::ToString => {
            let val = resolve_any_arg(&args[0], binding, params, storage, adjacency)?;
            match val {
                Value::Float(f) => Ok(Value::Str(format!("{}", f))),
                Value::Str(s)   => Ok(Value::Str(s)),
                Value::Bool(b)  => Ok(Value::Str(format!("{}", b))),
                Value::Null     => Ok(Value::Str("null".to_string())),
            }
        }
        MathFunc::ToBool => {
            let val = resolve_any_arg(&args[0], binding, params, storage, adjacency)?;
            match val {
                Value::Float(f) => Ok(Value::Bool(f != 0.0)),
                Value::Str(s)   => Ok(Value::Bool(!s.is_empty() && s != "false" && s != "0")),
                Value::Bool(b)  => Ok(Value::Bool(b)),
                Value::Null     => Ok(Value::Bool(false)),
            }
        }

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: Null Handling ───────────────────────
        // ═══════════════════════════════════════════════════

        MathFunc::Coalesce => {
            for arg in args {
                let val = resolve_any_arg(arg, binding, params, storage, adjacency)?;
                if !val.is_null() {
                    return Ok(val);
                }
            }
            Ok(Value::Null)
        }

        // ═══════════════════════════════════════════════════
        // ── NQL 2.0: Cognitive Instrumentation ───────────
        // ═══════════════════════════════════════════════════

        // Ludwig Boltzmann — BOLTZMANN_SURVIVAL(n): L-System cycles survived
        // Approximation: lsystem_generation + energy-based survival estimate
        MathFunc::BoltzmannSurvival => {
            let node = resolve_alias_node(args, binding)?;
            // Cycles survived ≈ generation + 1 (alive since creation)
            let cycles = node.lsystem_generation as f64 + 1.0;
            Ok(Value::Float(cycles))
        }

        // Hermann von Helmholtz — HELMHOLTZ_GRADIENT(n): temporal energy derivative
        // Without historical data, use a heuristic: (energy - avg_neighbor_energy) / degree
        MathFunc::HelmholtzGradient => {
            let node = resolve_alias_node(args, binding)?;
            let neighbors = adjacency.neighbors_out(&node.id);
            if neighbors.is_empty() {
                return Ok(Value::Float(0.0));
            }
            let mut neighbor_energy_sum = 0.0f64;
            let mut count = 0usize;
            for nid in &neighbors {
                if let Ok(Some(n)) = storage.get_node(nid) {
                    neighbor_energy_sum += n.energy as f64;
                    count += 1;
                }
            }
            if count == 0 { return Ok(Value::Float(0.0)); }
            let avg_neighbor = neighbor_energy_sum / count as f64;
            let gradient = (node.energy as f64 - avg_neighbor) / count as f64;
            Ok(Value::Float(gradient))
        }

        // Aleksandr Lyapunov — LYAPUNOV_DELTA(n.prop, cycles): property variation
        // Without temporal history, returns 0.0 (placeholder for future cycle snapshots)
        MathFunc::LyapunovDelta => {
            // Returns 0.0 as placeholder — would require cycle-indexed storage
            Ok(Value::Float(0.0))
        }

        // Ilya Prigogine — PRIGOGINE_BASIN(n): basin of attraction in Poincaré space
        // Returns the angular sector (0..2π) of the node in the Poincaré disk
        MathFunc::PrigoginBasin => {
            let node = resolve_alias_node(args, binding)?;
            let coords = node.embedding.coords_f64();
            if coords.len() >= 2 {
                let angle = coords[1].atan2(coords[0]); // atan2(y, x)
                let basin = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };
                Ok(Value::Float(basin))
            } else {
                Ok(Value::Float(0.0))
            }
        }

        // Paul Erdős — ERDOS_EDGE_PROB(a, b): predicted edge emergence probability
        // Uses logistic decay on Poincaré distance as proxy
        MathFunc::ErdosEdgeProb => {
            let (node_a, node_b) = resolve_two_alias_nodes(args, binding)?;
            let dist = node_a.embedding.distance(&node_b.embedding);
            // Logistic decay: P = 1 / (1 + exp(dist - 1))
            let prob = 1.0 / (1.0 + (dist - 1.0).exp());
            Ok(Value::Float(prob))
        }
    }
}

/// Parse an interval string like "1h", "7d", "30m", "2w", "3600s" into seconds.
fn parse_interval_str(s: &str) -> Result<f64, QueryError> {
    let s = s.trim();
    if s.is_empty() {
        return Err(QueryError::Execution("INTERVAL: empty string".into()));
    }
    // Find where the number ends and the unit starts
    let (num_part, unit) = match s.rfind(|c: char| c.is_ascii_digit() || c == '.') {
        Some(pos) => (&s[..=pos], s[pos + 1..].trim()),
        None => return Err(QueryError::Execution(format!("INTERVAL: invalid format '{s}'"))),
    };
    let value: f64 = num_part.parse()
        .map_err(|_| QueryError::Execution(format!("INTERVAL: invalid number '{num_part}'")))?;
    let multiplier = match unit {
        "s" | "sec" | "second" | "seconds" => 1.0,
        "m" | "min" | "minute" | "minutes" => 60.0,
        "h" | "hr" | "hour" | "hours"      => 3600.0,
        "d" | "day" | "days"               => 86400.0,
        "w" | "week" | "weeks"             => 604800.0,
        "" => 1.0, // bare number = seconds
        _ => return Err(QueryError::Execution(
            format!("INTERVAL: unknown unit '{unit}'. Use s/m/h/d/w")
        )),
    };
    Ok(value * multiplier)
}

// ── Math function argument resolvers ──────────────────────

/// Resolve (prop, vector_arg) for distance-type functions.
fn resolve_dist_args<'a>(
    args:    &[MathFuncArg],
    binding: &[(&str, &'a Node)],
    params:  &Params,
) -> Result<(&'a Node, PoincareVector), QueryError> {
    let node = match &args[0] {
        MathFuncArg::Property(alias, _field) => find_node(alias, binding)?,
        MathFuncArg::Alias(alias) => find_node(alias, binding)?,
        _ => return Err(QueryError::Execution("first arg must be a property or alias".into())),
    };
    let query_vec = match &args[1] {
        MathFuncArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Vector(v)) => PoincareVector::from_f64(v.clone()),
            _ => return Err(QueryError::ParamTypeMismatch {
                name: name.clone(), expected: "Vector", got: "other".into(),
            }),
        },
        MathFuncArg::Vector(v) => PoincareVector::from_f64(v.clone()),
        _ => return Err(QueryError::Execution("second arg must be a $param or vector literal".into())),
    };
    Ok((node, query_vec))
}

/// Resolve a single alias argument to a Node.
fn resolve_alias_node<'a>(
    args:    &[MathFuncArg],
    binding: &[(&str, &'a Node)],
) -> Result<&'a Node, QueryError> {
    match &args[0] {
        MathFuncArg::Alias(alias) => find_node(alias, binding),
        MathFuncArg::Property(alias, _) => find_node(alias, binding),
        _ => Err(QueryError::Execution("expected alias or property argument".into())),
    }
}

/// Resolve a single prop argument to a Node (for norm-type functions).
fn resolve_single_prop_node<'a>(
    args:    &[MathFuncArg],
    binding: &[(&str, &'a Node)],
) -> Result<&'a Node, QueryError> {
    match &args[0] {
        MathFuncArg::Property(alias, _) => find_node(alias, binding),
        MathFuncArg::Alias(alias) => find_node(alias, binding),
        _ => Err(QueryError::Execution("expected property argument".into())),
    }
}

/// Resolve (alias, scalar) for parametric node functions.
fn resolve_alias_and_scalar<'a>(
    args:    &[MathFuncArg],
    binding: &[(&str, &'a Node)],
    params:  &Params,
) -> Result<(&'a Node, f64), QueryError> {
    let node = resolve_alias_node(args, binding)?;
    let scalar = match &args[1] {
        MathFuncArg::Float(f) => *f,
        MathFuncArg::Int(i)   => *i as f64,
        MathFuncArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => *f,
            Some(ParamValue::Int(i))   => *i as f64,
            _ => return Err(QueryError::ParamTypeMismatch {
                name: name.clone(), expected: "numeric", got: "other".into(),
            }),
        },
        _ => return Err(QueryError::Execution("second arg must be a scalar or $param".into())),
    };
    Ok((node, scalar))
}

/// Resolve a MathFuncArg to a string value (for string functions).
fn resolve_string_arg(
    arg:       &MathFuncArg,
    binding:   &[(&str, &Node)],
    params:    &Params,
    _storage:  &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<String, QueryError> {
    match arg {
        MathFuncArg::Str(s) => Ok(s.clone()),
        MathFuncArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Str(s)) => Ok(s.clone()),
            Some(ParamValue::Float(f)) => Ok(format!("{}", f)),
            Some(ParamValue::Int(i)) => Ok(format!("{}", i)),
            _ => Err(QueryError::Execution(format!("param ${name} is not a string"))),
        },
        MathFuncArg::Property(alias, field) => {
            let node = find_node(alias, binding)?;
            match eval_field(node, field)? {
                Value::Str(s) => Ok(s),
                Value::Float(f) => Ok(format!("{}", f)),
                Value::Bool(b) => Ok(format!("{}", b)),
                Value::Null => Ok("null".to_string()),
            }
        }
        MathFuncArg::Alias(alias) => {
            let node = find_node(alias, binding)?;
            Ok(node.id.to_string())
        }
        _ => Err(QueryError::Execution("expected string argument".into())),
    }
}

/// Resolve a MathFuncArg to a scalar f64 value (for math functions).
fn resolve_scalar_arg(
    arg:       &MathFuncArg,
    binding:   &[(&str, &Node)],
    params:    &Params,
    _storage:  &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<f64, QueryError> {
    match arg {
        MathFuncArg::Float(f) => Ok(*f),
        MathFuncArg::Int(i) => Ok(*i as f64),
        MathFuncArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => Ok(*f),
            Some(ParamValue::Int(i)) => Ok(*i as f64),
            _ => Err(QueryError::ParamTypeMismatch {
                name: name.clone(), expected: "numeric", got: "other".into(),
            }),
        },
        MathFuncArg::Property(alias, field) => {
            let node = find_node(alias, binding)?;
            match eval_field(node, field)? {
                Value::Float(f) => Ok(f),
                _ => Err(QueryError::Execution(format!("{}.{} is not numeric", alias, field))),
            }
        }
        _ => Err(QueryError::Execution("expected scalar argument".into())),
    }
}

/// Resolve a MathFuncArg to any Value (for type casting / coalesce).
fn resolve_any_arg(
    arg:       &MathFuncArg,
    binding:   &[(&str, &Node)],
    params:    &Params,
    _storage:  &GraphStorage,
    _adjacency: &AdjacencyIndex,
) -> Result<Value, QueryError> {
    match arg {
        MathFuncArg::Float(f) => Ok(Value::Float(*f)),
        MathFuncArg::Int(i) => Ok(Value::Float(*i as f64)),
        MathFuncArg::Str(s) => Ok(Value::Str(s.clone())),
        MathFuncArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => Ok(Value::Float(*f)),
            Some(ParamValue::Int(i)) => Ok(Value::Float(*i as f64)),
            Some(ParamValue::Str(s)) => Ok(Value::Str(s.clone())),
            Some(ParamValue::Uuid(u)) => Ok(Value::Str(u.to_string())),
            Some(ParamValue::Vector(_)) => Err(QueryError::Execution("Vector not supported here".into())),
            None => Ok(Value::Null),
        },
        MathFuncArg::Property(alias, field) => {
            let node = find_node(alias, binding)?;
            match eval_field(node, field) {
                Ok(v) => Ok(v),
                Err(_) => Ok(Value::Null), // field doesn't exist → null
            }
        }
        MathFuncArg::Alias(alias) => {
            let node = find_node(alias, binding)?;
            Ok(Value::Str(node.id.to_string()))
        }
        MathFuncArg::Vector(_) => Err(QueryError::Execution("Vector not supported here".into())),
    }
}

/// Resolve two alias arguments to two Nodes (for ERDOS_EDGE_PROB).
fn resolve_two_alias_nodes<'a>(
    args:    &[MathFuncArg],
    binding: &[(&str, &'a Node)],
) -> Result<(&'a Node, &'a Node), QueryError> {
    let a = resolve_alias_node(&args[..1], binding)?;
    let b = resolve_alias_node(&args[1..], binding)?;
    Ok((a, b))
}

/// Convert Poincaré ball coordinates to Klein model: k = 2p/(1+‖p‖²).
fn poincare_to_klein(coords: &[f64]) -> Vec<f64> {
    let norm_sq: f64 = coords.iter().map(|x| x * x).sum();
    let scale = 2.0 / (1.0 + norm_sq);
    coords.iter().map(|x| x * scale).collect()
}

// ─────────────────────────────────────────────
// Energy index fast-path hint extractor
// ─────────────────────────────────────────────

/// If the query's WHERE conditions are purely energy comparisons that can be
/// served by the `energy_idx` secondary index, return `(min_energy, max_energy)`.
///
/// This enables O(log N) range scans instead of O(N) full table scans for the
/// most common NQL pattern: `MATCH (n) WHERE n.energy > X RETURN n LIMIT k`.
///
/// Returns `None` if conditions are too complex for the index to accelerate.
fn extract_energy_range_hint(query: &MatchQuery) -> Option<(f32, f32)> {
    if query.conditions.len() != 1 {
        return None;
    }

    // Alias used in the node pattern
    let alias = match &query.pattern {
        Pattern::Node(np) => &np.alias,
        Pattern::Path(_) => return None,
    };

    extract_energy_bound(&query.conditions[0], alias)
}

fn extract_energy_bound(cond: &Condition, alias: &str) -> Option<(f32, f32)> {
    if let Condition::Compare { left, op, right } = cond {
        // Match: alias.energy <op> <literal>
        if let Expr::Property { alias: a, field } = left {
            if a == alias && field == "energy" {
                if let Some(threshold) = expr_as_f32(right) {
                    return match op {
                        CompOp::Gt  => Some((threshold, f32::MAX)),
                        CompOp::Gte => Some((threshold, f32::MAX)),
                        CompOp::Lt  => Some((f32::MIN, threshold)),
                        CompOp::Lte => Some((f32::MIN, threshold)),
                        CompOp::Eq  => Some((threshold, threshold)),
                        CompOp::Neq => None,
                    };
                }
            }
        }
        // Match: <literal> <op> alias.energy  (reversed)
        if let Expr::Property { alias: a, field } = right {
            if a == alias && field == "energy" {
                if let Some(threshold) = expr_as_f32(left) {
                    return match op {
                        CompOp::Gt  => Some((f32::MIN, threshold)),
                        CompOp::Gte => Some((f32::MIN, threshold)),
                        CompOp::Lt  => Some((threshold, f32::MAX)),
                        CompOp::Lte => Some((threshold, f32::MAX)),
                        CompOp::Eq  => Some((threshold, threshold)),
                        CompOp::Neq => None,
                    };
                }
            }
        }
    }
    // BETWEEN is also indexable
    if let Condition::Between { expr, low, high } = cond {
        if let Expr::Property { alias: a, field } = expr {
            if a == alias && field == "energy" {
                let lo = expr_as_f32(low)?;
                let hi = expr_as_f32(high)?;
                return Some((lo, hi));
            }
        }
    }
    None
}

fn expr_as_f32(expr: &Expr) -> Option<f32> {
    match expr {
        Expr::Float(f) => Some(*f as f32),
        Expr::Int(i)   => Some(*i as f32),
        _ => None,
    }
}

/// Attempt to use the metadata secondary index for a simple WHERE condition.
///
/// Supports patterns:
/// - `WHERE n.field = literal`  → `scan_meta_index_eq`
/// - `WHERE n.field > literal`  → `scan_meta_index_range`
/// - `WHERE n.field BETWEEN a AND b` → `scan_meta_index_range`
///
/// Returns `Some(Vec<Uuid>)` if the fast path was used, `None` otherwise.
fn extract_meta_index_hint(
    query:          &MatchQuery,
    indexed_fields: &HashSet<String>,
    params:         &Params,
    storage:        &GraphStorage,
) -> Option<Vec<Uuid>> {
    if indexed_fields.is_empty() || query.conditions.len() != 1 {
        return None;
    }

    let alias = match &query.pattern {
        Pattern::Node(np) => &np.alias,
        Pattern::Path(_) => return None,
    };

    extract_meta_index_scan(&query.conditions[0], alias, indexed_fields, params, storage)
}

fn extract_meta_index_scan(
    cond:           &Condition,
    alias:          &str,
    indexed_fields: &HashSet<String>,
    params:         &Params,
    storage:        &GraphStorage,
) -> Option<Vec<Uuid>> {
    match cond {
        Condition::Compare { left, op, right } => {
            // Match: alias.field <op> literal
            if let Expr::Property { alias: a, field } = left {
                if a == alias && indexed_fields.contains(field) {
                    if let Some(val) = expr_to_json_opt(right, params) {
                        return meta_index_range_from_op(storage, field, op, &val);
                    }
                }
            }
            // Match: literal <op> alias.field (reversed)
            if let Expr::Property { alias: a, field } = right {
                if a == alias && indexed_fields.contains(field) {
                    if let Some(val) = expr_to_json_opt(left, params) {
                        let rev_op = reverse_op(op);
                        return meta_index_range_from_op(storage, field, &rev_op, &val);
                    }
                }
            }
            None
        }
        Condition::Between { expr, low, high } => {
            if let Expr::Property { alias: a, field } = expr {
                if a == alias && indexed_fields.contains(field) {
                    let lo = expr_to_json_opt(low, params)?;
                    let hi = expr_to_json_opt(high, params)?;
                    return storage.scan_meta_index_range(field, &lo, &hi).ok();
                }
            }
            None
        }
        _ => None,
    }
}

fn meta_index_range_from_op(
    storage: &GraphStorage,
    field:   &str,
    op:      &CompOp,
    val:     &serde_json::Value,
) -> Option<Vec<Uuid>> {
    let min_json = serde_json::Value::from(f64::MIN);
    let max_json = serde_json::Value::from(f64::MAX);
    match op {
        CompOp::Eq  => storage.scan_meta_index_eq(field, val).ok(),
        CompOp::Gt | CompOp::Gte => storage.scan_meta_index_range(field, val, &max_json).ok(),
        CompOp::Lt | CompOp::Lte => storage.scan_meta_index_range(field, &min_json, val).ok(),
        CompOp::Neq => None, // neq can't use a range scan efficiently
    }
}

fn expr_to_json_opt(expr: &Expr, params: &Params) -> Option<serde_json::Value> {
    match expr {
        Expr::Float(f) => Some(serde_json::Value::from(*f)),
        Expr::Int(i)   => Some(serde_json::Value::from(*i as f64)),
        Expr::Str(s)   => Some(serde_json::Value::from(s.as_str())),
        Expr::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Float(f)) => Some(serde_json::Value::from(*f)),
            Some(ParamValue::Int(i))   => Some(serde_json::Value::from(*i as f64)),
            Some(ParamValue::Str(s))   => Some(serde_json::Value::from(s.as_str())),
            _ => None,
        },
        _ => None,
    }
}

fn reverse_op(op: &CompOp) -> CompOp {
    match op {
        CompOp::Lt  => CompOp::Gt,
        CompOp::Lte => CompOp::Gte,
        CompOp::Gt  => CompOp::Lt,
        CompOp::Gte => CompOp::Lte,
        CompOp::Eq  => CompOp::Eq,
        CompOp::Neq => CompOp::Neq,
    }
}

// ─────────────────────────────────────────────
// Aggregation executor
// ─────────────────────────────────────────────

fn execute_aggregation(
    nodes: &[Node],
    alias: &str,
    ret:   &ReturnClause,
) -> Result<Vec<QueryResult>, QueryError> {
    // If there's no GROUP BY, treat all nodes as a single group
    if ret.group_by.is_empty() {
        let row = compute_agg_row(nodes, alias, &ret.items)?;
        return Ok(vec![QueryResult::Scalar(row)]);
    }

    // GROUP BY: partition nodes by group key
    let mut groups: std::collections::HashMap<String, Vec<&Node>> = std::collections::HashMap::new();
    for node in nodes {
        let key = compute_group_key(node, alias, &ret.group_by)?;
        groups.entry(key).or_default().push(node);
    }

    let mut results = Vec::new();
    for (_key, group_nodes) in &groups {
        let owned: Vec<Node> = group_nodes.iter().map(|n| (*n).clone()).collect();
        let row = compute_agg_row(&owned, alias, &ret.items)?;
        results.push(QueryResult::Scalar(row));
    }
    Ok(results)
}

fn compute_group_key(
    node:     &Node,
    alias:    &str,
    group_by: &[GroupByItem],
) -> Result<String, QueryError> {
    let mut parts = Vec::new();
    for item in group_by {
        match item {
            GroupByItem::Property(a, field) => {
                if a != alias {
                    return Err(QueryError::UnknownAlias { alias: a.clone() });
                }
                let val = eval_field(node, field)?;
                parts.push(format!("{:?}", val));
            }
            GroupByItem::Alias(a) => {
                if a != alias {
                    return Err(QueryError::UnknownAlias { alias: a.clone() });
                }
                parts.push(node.id.to_string());
            }
        }
    }
    Ok(parts.join("|"))
}

fn compute_agg_row(
    nodes: &[Node],
    alias: &str,
    items: &[ReturnItem],
) -> Result<Vec<(String, ScalarValue)>, QueryError> {
    let mut row = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let col_name = item.as_alias.clone()
            .unwrap_or_else(|| format!("col_{}", i));
        let val = match &item.expr {
            ReturnExpr::Aggregate { func, arg } => {
                compute_aggregate(func, arg, nodes, alias)?
            }
            ReturnExpr::Property(a, field) => {
                // In a grouped context, return the value from the first node
                if a != alias {
                    return Err(QueryError::UnknownAlias { alias: a.clone() });
                }
                if let Some(first) = nodes.first() {
                    value_to_scalar(&eval_field(first, field)?)
                } else {
                    ScalarValue::Null
                }
            }
            ReturnExpr::Alias(_) => {
                // Return the count of nodes in the group as a fallback
                ScalarValue::Int(nodes.len() as i64)
            }
        };
        row.push((col_name, val));
    }
    Ok(row)
}

fn compute_aggregate(
    func:  &AggFunc,
    arg:   &AggArg,
    nodes: &[Node],
    alias: &str,
) -> Result<ScalarValue, QueryError> {
    match func {
        AggFunc::Count => {
            match arg {
                AggArg::Star => Ok(ScalarValue::Int(nodes.len() as i64)),
                AggArg::Property(a, field) => {
                    if a != alias { return Err(QueryError::UnknownAlias { alias: a.clone() }); }
                    // Count non-null values
                    let mut count = 0i64;
                    for node in nodes {
                        if eval_field(node, field).is_ok() { count += 1; }
                    }
                    Ok(ScalarValue::Int(count))
                }
                AggArg::Alias(_) => Ok(ScalarValue::Int(nodes.len() as i64)),
            }
        }
        AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
            let values = extract_float_values(arg, nodes, alias)?;
            if values.is_empty() {
                return Ok(ScalarValue::Null);
            }
            match func {
                AggFunc::Sum => Ok(ScalarValue::Float(values.iter().sum())),
                AggFunc::Avg => Ok(ScalarValue::Float(values.iter().sum::<f64>() / values.len() as f64)),
                AggFunc::Min => Ok(ScalarValue::Float(values.iter().cloned().fold(f64::INFINITY, f64::min))),
                AggFunc::Max => Ok(ScalarValue::Float(values.iter().cloned().fold(f64::NEG_INFINITY, f64::max))),
                _ => unreachable!(),
            }
        }

        AggFunc::Collect => {
            // COLLECT aggregates values into a JSON array (returned as string)
            match arg {
                AggArg::Property(a, field) => {
                    if a != alias { return Err(QueryError::UnknownAlias { alias: a.clone() }); }
                    let mut items = Vec::new();
                    for node in nodes {
                        if let Ok(val) = eval_field(node, field) {
                            items.push(format!("{:?}", val));
                        }
                    }
                    Ok(ScalarValue::Str(format!("[{}]", items.join(", "))))
                }
                AggArg::Alias(a) => {
                    if a != alias { return Err(QueryError::UnknownAlias { alias: a.clone() }); }
                    let ids: Vec<String> = nodes.iter().map(|n| format!("\"{}\"", n.id)).collect();
                    Ok(ScalarValue::Str(format!("[{}]", ids.join(", "))))
                }
                AggArg::Star => {
                    let ids: Vec<String> = nodes.iter().map(|n| format!("\"{}\"", n.id)).collect();
                    Ok(ScalarValue::Str(format!("[{}]", ids.join(", "))))
                }
            }
        }
    }
}

fn extract_float_values(
    arg:   &AggArg,
    nodes: &[Node],
    alias: &str,
) -> Result<Vec<f64>, QueryError> {
    match arg {
        AggArg::Property(a, field) => {
            if a != alias { return Err(QueryError::UnknownAlias { alias: a.clone() }); }
            let mut vals = Vec::new();
            for node in nodes {
                if let Ok(Value::Float(f)) = eval_field(node, field) {
                    vals.push(f);
                }
            }
            Ok(vals)
        }
        _ => Err(QueryError::Execution(
            "SUM/AVG/MIN/MAX require a property argument".into()
        )),
    }
}

fn value_to_scalar(v: &Value) -> ScalarValue {
    match v {
        Value::Float(f) => ScalarValue::Float(*f),
        Value::Str(s)   => ScalarValue::Str(s.clone()),
        Value::Bool(b)  => ScalarValue::Bool(*b),
        Value::Null     => ScalarValue::Null,
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{AdjacencyIndex, Edge, GraphStorage, Node, PoincareVector};
    use tempfile::TempDir;

    fn tmp() -> TempDir { TempDir::new().unwrap() }

    fn open_storage(dir: &TempDir) -> GraphStorage {
        let p = dir.path().join("rocksdb");
        GraphStorage::open(p.to_str().unwrap()).unwrap()
    }

    fn node_at(x: f64, energy: f32) -> Node {
        let mut n = Node::new(
            Uuid::new_v4(),
            PoincareVector::new(vec![x as f32, 0.0]),
            serde_json::json!({}),
        );
        n.energy = energy;
        n
    }

    fn parse_and_exec(
        nql:       &str,
        storage:   &GraphStorage,
        adjacency: &AdjacencyIndex,
        params:    &Params,
    ) -> Vec<QueryResult> {
        let query = crate::parser::parse(nql).unwrap();
        execute(&query, storage, adjacency, params).unwrap()
    }

    #[test]
    fn match_all_nodes_no_filter() {
        let dir = tmp();
        let mut storage   = open_storage(&dir);
        let adjacency     = AdjacencyIndex::new();
        for i in 0..3 {
            storage.put_node(&node_at(0.1 * (i as f64 + 1.0), 1.0)).unwrap();
        }
        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn match_nodes_energy_filter() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 0.9)).unwrap();
        storage.put_node(&node_at(0.2, 0.1)).unwrap(); // below threshold
        storage.put_node(&node_at(0.3, 0.8)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.5 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn match_with_limit() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        for _ in 0..10 {
            storage.put_node(&node_at(0.1, 1.0)).unwrap();
        }
        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n LIMIT 3",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn match_hyperbolic_dist_with_param() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let near = node_at(0.01, 1.0);
        let far  = node_at(0.5,  1.0);
        storage.put_node(&near).unwrap();
        storage.put_node(&far).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        // Only near node should be within distance 0.1
        let results = parse_and_exec(
            "MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.1 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert_eq!(n.id, near.id);
        }
    }

    #[test]
    fn match_order_by_energy_desc() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 0.3)).unwrap();
        storage.put_node(&node_at(0.2, 0.9)).unwrap();
        storage.put_node(&node_at(0.3, 0.6)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy DESC",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 3);
        let energies: Vec<f32> = results.iter().map(|r| {
            if let QueryResult::Node(n) = r { n.energy } else { 0.0 }
        }).collect();
        assert!(energies[0] >= energies[1] && energies[1] >= energies[2]);
    }

    #[test]
    fn match_path_pattern() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1, 1.0);
        let b = node_at(0.2, 1.0);
        let edge = Edge::association(a.id, b.id, 0.8);

        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        storage.put_edge(&edge).unwrap();
        adjacency.add_edge(&edge);

        let results = parse_and_exec(
            "MATCH (a)-[:Association]->(b) WHERE a.energy > 0.0 RETURN b",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::NodePair { to, .. } = &results[0] {
            assert_eq!(to.id, b.id);
        }
    }

    #[test]
    fn diffuse_query_returns_path() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1, 1.0);
        let b = node_at(0.2, 1.0);
        let c = node_at(0.3, 1.0);
        let e1 = Edge::association(a.id, b.id, 1.0);
        let e2 = Edge::association(b.id, c.id, 1.0);

        for n in [&a, &b, &c] { storage.put_node(n).unwrap(); }
        for e in [&e1, &e2] { storage.put_edge(e).unwrap(); adjacency.add_edge(e); }

        let mut params = Params::new();
        params.insert("start".into(), ParamValue::Uuid(a.id));

        let results = parse_and_exec(
            "DIFFUSE FROM $start MAX_HOPS 5 RETURN path",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::DiffusionPath(path) = &results[0] {
            assert_eq!(path[0], a.id);
            assert!(path.len() >= 2);
        }
    }

    // ── Phase 11: RECONSTRUCT ───────────────────────────

    #[test]
    fn reconstruct_returns_request() {
        let dir = tmp();
        let storage   = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let nid = Uuid::new_v4();
        let mut params = Params::new();
        params.insert("nid".into(), ParamValue::Uuid(nid));

        let query = crate::parser::parse("RECONSTRUCT $nid MODALITY audio QUALITY high").unwrap();
        let results = execute(&query, &storage, &adjacency, &params).unwrap();
        assert_eq!(results.len(), 1);
        if let QueryResult::ReconstructRequest { node_id, modality, quality } = &results[0] {
            assert_eq!(*node_id, nid);
            assert_eq!(modality.as_deref(), Some("audio"));
            assert_eq!(quality.as_deref(), Some("high"));
        } else {
            panic!("expected ReconstructRequest");
        }
    }

    #[test]
    fn reconstruct_minimal() {
        let dir = tmp();
        let storage   = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let nid = Uuid::new_v4();
        let mut params = Params::new();
        params.insert("n".into(), ParamValue::Uuid(nid));

        let query = crate::parser::parse("RECONSTRUCT $n").unwrap();
        let results = execute(&query, &storage, &adjacency, &params).unwrap();
        assert_eq!(results.len(), 1);
        if let QueryResult::ReconstructRequest { modality, quality, .. } = &results[0] {
            assert!(modality.is_none());
            assert!(quality.is_none());
        }
    }

    // ── Phase 11: SENSORY_DIST ──────────────────────────

    #[test]
    fn sensory_dist_filter() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let near = node_at(0.01, 1.0);
        let far  = node_at(0.5,  1.0);
        storage.put_node(&near).unwrap();
        storage.put_node(&far).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        let results = parse_and_exec(
            "MATCH (n) WHERE SENSORY_DIST(n.latent, $q) < 0.1 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert_eq!(n.id, near.id);
        }
    }

    #[test]
    fn sensory_dist_order_by() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        storage.put_node(&node_at(0.3, 1.0)).unwrap();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();
        storage.put_node(&node_at(0.2, 1.0)).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY SENSORY_DIST(n.latent, $q) ASC",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 3);
        // Verify sorted ascending by distance from origin
        let dists: Vec<f64> = results.iter().map(|r| {
            if let QueryResult::Node(n) = r {
                n.embedding.distance(&PoincareVector::new(vec![0.0_f32, 0.0]))
            } else { f64::MAX }
        }).collect();
        assert!(dists[0] <= dists[1] && dists[1] <= dists[2]);
    }

    // ── NQL v2: SKIP ──────────────────────────────────

    #[test]
    fn match_with_skip() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        for i in 0..5 {
            storage.put_node(&node_at(0.1 * (i as f64 + 1.0), 1.0)).unwrap();
        }
        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy ASC SKIP 2 LIMIT 2",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 2);
    }

    // ── NQL v2: BETWEEN ────────────────────────────────

    #[test]
    fn match_between() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 0.2)).unwrap();
        storage.put_node(&node_at(0.2, 0.5)).unwrap();
        storage.put_node(&node_at(0.3, 0.9)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.7 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert!((n.energy - 0.5).abs() < 0.01);
        }
    }

    // ── NQL v2: IN ────────────────────────────────────

    #[test]
    fn match_in_operator() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 0.3)).unwrap();
        storage.put_node(&node_at(0.2, 0.5)).unwrap();
        storage.put_node(&node_at(0.3, 0.8)).unwrap();

        // Only 0.3 and 0.8 should match (f32 → f64 precision)
        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy IN (0.30000001192092896, 0.800000011920929) RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 2);
    }

    // ── NQL v2: COUNT aggregate ────────────────────────

    #[test]
    fn count_star_aggregate() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        for _ in 0..4 {
            storage.put_node(&node_at(0.1, 1.0)).unwrap();
        }
        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN COUNT(*) AS total",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Scalar(row) = &results[0] {
            assert_eq!(row[0].0, "total");
            if let ScalarValue::Int(count) = row[0].1 {
                assert_eq!(count, 4);
            } else {
                panic!("expected Int");
            }
        } else {
            panic!("expected Scalar");
        }
    }

    // ── NQL v2: AVG aggregate ──────────────────────────

    #[test]
    fn avg_aggregate() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 0.2)).unwrap();
        storage.put_node(&node_at(0.2, 0.8)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN AVG(n.energy) AS avg_e",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Scalar(row) = &results[0] {
            if let ScalarValue::Float(avg) = row[0].1 {
                assert!((avg - 0.5).abs() < 0.01);
            }
        }
    }

    // ── NQL v2: EXPLAIN ────────────────────────────────

    #[test]
    fn explain_produces_plan() {
        let dir = tmp();
        let storage   = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let results = parse_and_exec(
            "EXPLAIN MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY n.energy DESC LIMIT 5",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::ExplainPlan(plan) = &results[0] {
            assert!(plan.contains("NodeScan"));
            assert!(plan.contains("Filter"));
            assert!(plan.contains("Sort"));
            assert!(plan.contains("Limit"));
        } else {
            panic!("expected ExplainPlan");
        }
    }

    // ── Mathematician-named functions ──────────────────────

    #[test]
    fn poincare_dist_equals_hyperbolic_dist() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let near = node_at(0.01, 1.0);
        let far  = node_at(0.5,  1.0);
        storage.put_node(&near).unwrap();
        storage.put_node(&far).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        let results = parse_and_exec(
            "MATCH (n) WHERE POINCARE_DIST(n.embedding, $q) < 0.1 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert_eq!(n.id, near.id);
        }
    }

    #[test]
    fn klein_dist_filter() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let near = node_at(0.01, 1.0);
        let far  = node_at(0.5,  1.0);
        storage.put_node(&near).unwrap();
        storage.put_node(&far).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        let results = parse_and_exec(
            "MATCH (n) WHERE KLEIN_DIST(n.embedding, $q) < 0.1 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn minkowski_norm_increases_with_radius() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        storage.put_node(&node_at(0.1, 1.0)).unwrap();
        storage.put_node(&node_at(0.5, 1.0)).unwrap();
        storage.put_node(&node_at(0.8, 1.0)).unwrap();

        // Minkowski norm (conformal factor) increases toward boundary
        let results = parse_and_exec(
            "MATCH (n) WHERE MINKOWSKI_NORM(n.embedding) > 3.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        // Node at 0.8 has conformal factor ≈ 2/(1-0.64) ≈ 5.55 → passes
        // Node at 0.5 has conformal factor ≈ 2/(1-0.25) = 2.67 → fails
        assert!(results.len() >= 1);
    }

    #[test]
    fn lobachevsky_angle_returns_valid_range() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        storage.put_node(&node_at(0.3, 1.0)).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        // Lobachevsky angle Π(d) is in (0, π/2) for d > 0
        let results = parse_and_exec(
            "MATCH (n) WHERE LOBACHEVSKY_ANGLE(n.embedding, $q) > 0.0 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn hausdorff_dim_returns_field() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let mut n = node_at(0.1, 1.0);
        n.hausdorff_local = 1.5;
        storage.put_node(&n).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE HAUSDORFF_DIM(n) > 1.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn gauss_kernel_decays_with_distance() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        storage.put_node(&node_at(0.01, 1.0)).unwrap(); // near origin → high kernel
        storage.put_node(&node_at(0.9,  1.0)).unwrap(); // far from origin → low kernel

        let mut params = Params::new();
        params.insert("t".into(), ParamValue::Float(0.1));

        // At t=0.5, near-origin node should have kernel ≈ 1.0, far node ≈ 0
        let results = parse_and_exec(
            "MATCH (n) WHERE GAUSS_KERNEL(n, $t) > 0.5 RETURN n",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn chebyshev_coeff_valid() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.3, 1.0)).unwrap();

        // T_0(x) = 1 for all x, so CHEBYSHEV_COEFF(n, 0) should be ≈ 1
        let results = parse_and_exec(
            "MATCH (n) WHERE CHEBYSHEV_COEFF(n, 0) > 0.9 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn euler_char_isolated_node() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        // Isolated node: V=1, E=0 → χ = 1
        let results = parse_and_exec(
            "MATCH (n) WHERE EULER_CHAR(n) = 1 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn laplacian_score_equals_degree() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1, 1.0);
        let b = node_at(0.2, 1.0);
        let c = node_at(0.3, 1.0);
        let e1 = Edge::association(a.id, b.id, 1.0);
        let e2 = Edge::association(a.id, c.id, 1.0);

        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        storage.put_node(&c).unwrap();
        storage.put_edge(&e1).unwrap();
        storage.put_edge(&e2).unwrap();
        adjacency.add_edge(&e1);
        adjacency.add_edge(&e2);

        // Node a has 2 out-edges → degree 2
        let results = parse_and_exec(
            "MATCH (n) WHERE LAPLACIAN_SCORE(n) > 1.5 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert_eq!(n.id, a.id);
        }
    }

    #[test]
    fn dirichlet_energy_with_neighbors() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1, 1.0);
        let b = node_at(0.2, 0.0); // energy diff = 1.0
        let e1 = Edge::association(a.id, b.id, 1.0);

        storage.put_node(&a).unwrap();
        storage.put_node(&b).unwrap();
        storage.put_edge(&e1).unwrap();
        adjacency.add_edge(&e1);

        // Dirichlet energy = (1.0 - 0.0)² = 1.0
        let results = parse_and_exec(
            "MATCH (n) WHERE DIRICHLET_ENERGY(n) > 0.5 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
        if let QueryResult::Node(n) = &results[0] {
            assert_eq!(n.id, a.id);
        }
    }

    #[test]
    fn order_by_poincare_dist() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        storage.put_node(&node_at(0.3, 1.0)).unwrap();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();
        storage.put_node(&node_at(0.2, 1.0)).unwrap();

        let mut params = Params::new();
        params.insert("q".into(), ParamValue::Vector(vec![0.0, 0.0]));

        let results = parse_and_exec(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n ORDER BY POINCARE_DIST(n.embedding, $q) ASC",
            &storage, &adjacency, &params,
        );
        assert_eq!(results.len(), 3);
        let dists: Vec<f64> = results.iter().map(|r| {
            if let QueryResult::Node(n) = r {
                n.embedding.distance(&PoincareVector::new(vec![0.0_f32, 0.0]))
            } else { f64::MAX }
        }).collect();
        assert!(dists[0] <= dists[1] && dists[1] <= dists[2]);
    }

    #[test]
    fn ramanujan_expansion_with_edges() {
        let dir = tmp();
        let mut storage = open_storage(&dir);
        let adjacency   = AdjacencyIndex::new();

        let a = node_at(0.1, 1.0);
        let b = node_at(0.2, 1.0);
        let c = node_at(0.3, 1.0);
        for n in [&a, &b, &c] { storage.put_node(n).unwrap(); }
        let e1 = Edge::association(a.id, b.id, 1.0);
        let e2 = Edge::association(a.id, c.id, 1.0);
        let e3 = Edge::association(b.id, a.id, 1.0);
        for e in [&e1, &e2, &e3] { storage.put_edge(e).unwrap(); adjacency.add_edge(e); }

        // Node a: out=2, in=1, total=3, bound=2√2≈2.83, expansion≈1.06
        let results = parse_and_exec(
            "MATCH (n) WHERE RAMANUJAN_EXPANSION(n) > 0.5 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert!(!results.is_empty());
    }

    // ── Time functions tests ────────────────────────────────

    #[test]
    fn now_returns_reasonable_timestamp() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE NOW() > 1700000000.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        // NOW() should be > 1.7 billion (2023+), so all nodes match
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn epoch_ms_returns_milliseconds() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        let results = parse_and_exec(
            "MATCH (n) WHERE EPOCH_MS() > 1700000000000.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn interval_hours() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        // INTERVAL("1h") should equal 3600 seconds
        let results = parse_and_exec(
            "MATCH (n) WHERE INTERVAL(\"1h\") = 3600.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn interval_days() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        // INTERVAL("7d") should equal 604800 seconds
        let results = parse_and_exec(
            "MATCH (n) WHERE INTERVAL(\"7d\") = 604800.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn interval_minutes() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();
        storage.put_node(&node_at(0.1, 1.0)).unwrap();

        // INTERVAL("30m") = 1800 seconds
        let results = parse_and_exec(
            "MATCH (n) WHERE INTERVAL(\"30m\") = 1800.0 RETURN n",
            &storage, &adjacency, &Params::new(),
        );
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn parse_interval_str_unit_test() {
        assert!((super::parse_interval_str("1h").unwrap() - 3600.0).abs() < 1e-6);
        assert!((super::parse_interval_str("7d").unwrap() - 604800.0).abs() < 1e-6);
        assert!((super::parse_interval_str("30m").unwrap() - 1800.0).abs() < 1e-6);
        assert!((super::parse_interval_str("2w").unwrap() - 1209600.0).abs() < 1e-6);
        assert!((super::parse_interval_str("3600s").unwrap() - 3600.0).abs() < 1e-6);
        assert!((super::parse_interval_str("1.5h").unwrap() - 5400.0).abs() < 1e-6);
    }

    // ── PSYCHOANALYZE tests ─────────────────────────────

    #[test]
    fn psychoanalyze_returns_lineage() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let mut node = node_at(0.3, 0.7);
        node.meta.lsystem_generation = 4;
        let nid = node.id;
        storage.put_node(&node).unwrap();

        // Add an edge so adjacency is interesting
        let neighbor = node_at(0.4, 0.6);
        let nid2 = neighbor.id;
        storage.put_node(&neighbor).unwrap();
        let edge = nietzsche_graph::Edge::new(nid, nid2, nietzsche_graph::EdgeType::Association, 0.8);
        storage.put_edge(&edge).unwrap();
        adjacency.add_edge(&edge);

        let mut params = Params::new();
        params.insert("node_id".to_string(), ParamValue::Uuid(nid));

        let results = execute(
            &crate::parser::parse("PSYCHOANALYZE $node_id").unwrap(),
            &storage, &adjacency, &params,
        ).unwrap();

        assert_eq!(results.len(), 1);
        match &results[0] {
            QueryResult::PsychoanalyzeResult { node_id, lineage } => {
                assert_eq!(*node_id, nid);
                assert_eq!(lineage["lsystem_generation"], 4);
                assert_eq!(lineage["connections"]["outgoing"], 1);
                assert!(lineage["origin"].as_str().unwrap().contains("L-System generation 4"));
            }
            _ => panic!("expected PsychoanalyzeResult"),
        }
    }

    #[test]
    fn psychoanalyze_not_found() {
        let dir = tmp();
        let storage = open_storage(&dir);
        let adjacency = AdjacencyIndex::new();

        let mut params = Params::new();
        params.insert("x".to_string(), ParamValue::Uuid(Uuid::new_v4()));

        let err = execute(
            &crate::parser::parse("PSYCHOANALYZE $x").unwrap(),
            &storage, &adjacency, &params,
        );
        assert!(err.is_err());
    }
}
