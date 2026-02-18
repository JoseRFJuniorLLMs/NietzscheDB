//! NQL executor: maps AST nodes to graph operations.
//!
//! Each `execute_*` function takes references to `GraphStorage` and
//! `AdjacencyIndex` (extracted from `NietzscheDB` via `.storage()` /
//! `.adjacency()`) plus a `Params` map for query parameters.

use std::collections::HashMap;

use uuid::Uuid;
use nietzsche_graph::{
    AdjacencyIndex, GraphError, GraphStorage, Node, PoincareVector,
    traversal::{diffusion_walk, DiffusionConfig},
};

use crate::ast::*;
use crate::error::QueryError;

// ─────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────

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

/// Execute a parsed NQL [`Query`] and return the result rows.
pub fn execute(
    query:     &Query,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    match query {
        Query::Match(m)       => execute_match(m, storage, adjacency, params),
        Query::Diffuse(d)     => execute_diffuse(d, storage, adjacency, params),
        Query::Reconstruct(r) => execute_reconstruct(r, params),
        Query::Explain(inner) => execute_explain(inner, storage, adjacency, params),
    }
}

// ─────────────────────────────────────────────
// EXPLAIN executor
// ─────────────────────────────────────────────

fn execute_explain(
    query:     &Query,
    _storage:  &GraphStorage,
    _adjacency:&AdjacencyIndex,
    _params:   &Params,
) -> Result<Vec<QueryResult>, QueryError> {
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
                    format!("EdgeScan(type={}, dir={})", edge, dir)
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
        Query::Reconstruct(_) => "ReconstructLatent".into(),
        Query::Explain(_)     => "Explain(nested — not supported)".into(),
    };
    Ok(vec![QueryResult::ExplainPlan(plan)])
}

// ─────────────────────────────────────────────
// MATCH executor
// ─────────────────────────────────────────────

fn execute_match(
    query:     &MatchQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    match &query.pattern {
        Pattern::Node(np) => execute_node_match(query, np, storage, adjacency, params),
        Pattern::Path(pp) => execute_path_match(query, pp, storage, adjacency, params),
    }
}

// ── Node match ────────────────────────────────

fn execute_node_match(
    query:     &MatchQuery,
    np:        &NodePattern,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
    let nodes = storage.scan_nodes()?;
    let alias = &np.alias;

    // Filter by label (node_type)
    let typed: Vec<Node> = if let Some(label) = &np.label {
        nodes.into_iter()
            .filter(|n| format!("{:?}", n.node_type).to_lowercase() == label.to_lowercase())
            .collect()
    } else {
        nodes
    };

    // Apply WHERE conditions
    let mut filtered = Vec::new();
    for node in typed {
        let binding = vec![(alias.as_str(), &node)];
        if eval_conditions(&query.conditions, &binding, params, storage, adjacency)? {
            filtered.push(node);
        }
    }

    // Apply ORDER BY
    if let Some(order) = &query.ret.order_by {
        sort_nodes(&mut filtered, alias, order, params, storage, adjacency)?;
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

    // Apply DISTINCT (by node ID)
    if query.ret.distinct {
        let mut seen = std::collections::HashSet::new();
        filtered.retain(|n| seen.insert(n.id));
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
) -> Result<Vec<QueryResult>, QueryError> {
    let from_alias = &pp.from.alias;
    let to_alias   = &pp.to.alias;

    let mut results = Vec::new();
    let edges = storage.scan_edges()?;

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

        let from_node = match storage.get_node(&from_id)? {
            Some(n) => n,
            None    => continue,
        };
        let to_node = match storage.get_node(&to_id)? {
            Some(n) => n,
            None    => continue,
        };

        // Evaluate WHERE conditions with both aliases bound
        let binding = vec![
            (from_alias.as_str(), &from_node),
            (to_alias.as_str(),   &to_node),
        ];
        if eval_conditions(&query.conditions, &binding, params, storage, adjacency)? {
            results.push(QueryResult::NodePair {
                from: from_node,
                to:   to_node,
            });
        }

        if let Some(limit) = query.ret.limit {
            if results.len() >= limit {
                break;
            }
        }
    }

    Ok(results)
}

// ─────────────────────────────────────────────
// DIFFUSE executor
// ─────────────────────────────────────────────

fn execute_diffuse(
    query:     &DiffuseQuery,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
    params:    &Params,
) -> Result<Vec<QueryResult>, QueryError> {
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
    for cond in conditions {
        if !eval_condition(cond, binding, params, storage, adjacency)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn eval_condition(
    cond:      &Condition,
    binding:   &[(&str, &Node)],
    params:    &Params,
    storage:   &GraphStorage,
    adjacency: &AdjacencyIndex,
) -> Result<bool, QueryError> {
    match cond {
        Condition::Compare { left, op, right } => {
            let l = eval_expr(left, binding, params, storage, adjacency)?;
            let r = eval_expr(right, binding, params, storage, adjacency)?;
            compare_values(&l, op, &r)
        }
        Condition::And(a, b) => {
            Ok(eval_condition(a, binding, params, storage, adjacency)?
               && eval_condition(b, binding, params, storage, adjacency)?)
        }
        Condition::Or(a, b) => {
            Ok(eval_condition(a, binding, params, storage, adjacency)?
               || eval_condition(b, binding, params, storage, adjacency)?)
        }
        Condition::Not(c) => Ok(!eval_condition(c, binding, params, storage, adjacency)?),

        Condition::In { expr, values } => {
            let needle = eval_expr(expr, binding, params, storage, adjacency)?;
            for v in values {
                let candidate = eval_expr(v, binding, params, storage, adjacency)?;
                if values_equal(&needle, &candidate) {
                    return Ok(true);
                }
            }
            Ok(false)
        }

        Condition::Between { expr, low, high } => {
            let val = eval_expr(expr, binding, params, storage, adjacency)?;
            let lo  = eval_expr(low, binding, params, storage, adjacency)?;
            let hi  = eval_expr(high, binding, params, storage, adjacency)?;
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
            let l = eval_expr(left, binding, params, storage, adjacency)?;
            let r = eval_expr(right, binding, params, storage, adjacency)?;
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
}

impl Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Float(_) => "float",
            Value::Str(_)   => "string",
            Value::Bool(_)  => "bool",
        }
    }
}

/// Check if two Values are equal (used by IN operator).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
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
            let node = find_node(alias, binding)?;
            eval_field(node, field)
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
                Some(ParamValue::Vector(v)) => PoincareVector::new(v.clone()),
                _ => return Err(QueryError::Execution(
                    format!("param ${arg} must be a Vector for SENSORY_DIST")
                )),
            };
            Ok(Value::Float(node.embedding.distance(&query_vec)))
        }

        Expr::MathFunc { func, args } => {
            eval_math_func(func, args, binding, params, storage, adjacency)
        }
    }
}

fn resolve_hdist_arg(arg: &HDistArg, params: &Params) -> Result<PoincareVector, QueryError> {
    match arg {
        HDistArg::Vector(v) => Ok(PoincareVector::new(v.clone())),
        HDistArg::Param(name) => match params.get(name.as_str()) {
            Some(ParamValue::Vector(v)) => Ok(PoincareVector::new(v.clone())),
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
        _ => Err(QueryError::UnknownField { field: field.to_string() }),
    }
}

// ─────────────────────────────────────────────
// Value comparison
// ─────────────────────────────────────────────

fn compare_values(left: &Value, op: &CompOp, right: &Value) -> Result<bool, QueryError> {
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
                Some(ParamValue::Vector(v)) => PoincareVector::new(v.clone()),
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
        MathFunc::PoincareDist => {
            let (node, query_vec) = resolve_dist_args(args, binding, params)?;
            Ok(Value::Float(node.embedding.distance(&query_vec)))
        }

        // ── Klein (Beltrami-Klein model distance) ─────────────
        // Maps Poincaré coords to Klein: k = 2p/(1+‖p‖²), then
        // computes Klein-metric distance (same geodesic distance, different formula).
        MathFunc::KleinDist => {
            let (node, query_vec) = resolve_dist_args(args, binding, params)?;
            let k1 = poincare_to_klein(&node.embedding.coords);
            let k2 = poincare_to_klein(&query_vec.coords);
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
            let norm_sq: f64 = node.embedding.coords.iter().map(|x| x * x).sum();
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
            let norm_sq: f64 = node.embedding.coords.iter().map(|x| x * x).sum();
            let d_sq = norm_sq; // squared Euclidean distance from origin
            let kernel = (-d_sq / (4.0 * t.max(1e-15))).exp();
            Ok(Value::Float(kernel))
        }

        // ── Chebyshev (T_k(x) at normalized node position) ───
        // Chebyshev polynomial T_k(x) = cos(k·arccos(x)), with
        // x = 2·‖embedding‖ − 1 ∈ [−1, 1].
        MathFunc::ChebyshevCoeff => {
            let (node, k) = resolve_alias_and_scalar(args, binding, params)?;
            let norm: f64 = node.embedding.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
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
            let norm: f64 = node.embedding.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
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
    }
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
            Some(ParamValue::Vector(v)) => PoincareVector::new(v.clone()),
            _ => return Err(QueryError::ParamTypeMismatch {
                name: name.clone(), expected: "Vector", got: "other".into(),
            }),
        },
        MathFuncArg::Vector(v) => PoincareVector::new(v.clone()),
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

/// Convert Poincaré ball coordinates to Klein model: k = 2p/(1+‖p‖²).
fn poincare_to_klein(coords: &[f64]) -> Vec<f64> {
    let norm_sq: f64 = coords.iter().map(|x| x * x).sum();
    let scale = 2.0 / (1.0 + norm_sq);
    coords.iter().map(|x| x * scale).collect()
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
            PoincareVector::new(vec![x, 0.0]),
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
                n.embedding.distance(&PoincareVector::new(vec![0.0, 0.0]))
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
        storage.put_node(&node_at(0.8,  1.0)).unwrap(); // far from origin → low kernel

        let mut params = Params::new();
        params.insert("t".into(), ParamValue::Float(0.5));

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
                n.embedding.distance(&PoincareVector::new(vec![0.0, 0.0]))
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
}
