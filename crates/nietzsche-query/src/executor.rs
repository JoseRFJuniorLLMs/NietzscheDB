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
        Query::Match(m)   => execute_match(m, storage, adjacency, params),
        Query::Diffuse(d) => execute_diffuse(d, storage, adjacency, params),
    }
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
        Pattern::Node(np) => execute_node_match(query, np, storage, params),
        Pattern::Path(pp) => execute_path_match(query, pp, storage, adjacency, params),
    }
}

// ── Node match ────────────────────────────────

fn execute_node_match(
    query:   &MatchQuery,
    np:      &NodePattern,
    storage: &GraphStorage,
    params:  &Params,
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
        if eval_conditions(&query.conditions, &binding, params)? {
            filtered.push(node);
        }
    }

    // Apply ORDER BY
    if let Some(order) = &query.ret.order_by {
        sort_nodes(&mut filtered, alias, order, params)?;
    }

    // Apply LIMIT
    if let Some(limit) = query.ret.limit {
        filtered.truncate(limit);
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

        let from_node = match storage.get_node(from_id)? {
            Some(n) => n,
            None    => continue,
        };
        let to_node = match storage.get_node(to_id)? {
            Some(n) => n,
            None    => continue,
        };

        // Evaluate WHERE conditions with both aliases bound
        let binding = vec![
            (from_alias.as_str(), &from_node),
            (to_alias.as_str(),   &to_node),
        ];
        if eval_conditions(&query.conditions, &binding, params)? {
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
) -> Result<bool, QueryError> {
    for cond in conditions {
        if !eval_condition(cond, binding, params)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn eval_condition(
    cond:    &Condition,
    binding: &[(&str, &Node)],
    params:  &Params,
) -> Result<bool, QueryError> {
    match cond {
        Condition::Compare { left, op, right } => {
            let l = eval_expr(left, binding, params)?;
            let r = eval_expr(right, binding, params)?;
            compare_values(&l, op, &r)
        }
        Condition::And(a, b) => {
            Ok(eval_condition(a, binding, params)? && eval_condition(b, binding, params)?)
        }
        Condition::Or(a, b) => {
            Ok(eval_condition(a, binding, params)? || eval_condition(b, binding, params)?)
        }
        Condition::Not(c) => Ok(!eval_condition(c, binding, params)?),
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

fn eval_expr(
    expr:    &Expr,
    binding: &[(&str, &Node)],
    params:  &Params,
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
                Err(QueryError::Execution(format!("param ${name}: use HYPERBOLIC_DIST for vector params"))),
            None =>
                Err(QueryError::Execution(format!("param ${name} not provided"))),
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
        .ok_or_else(|| QueryError::Execution(format!("unknown alias '{alias}' in condition")))
}

fn eval_field(node: &Node, field: &str) -> Result<Value, QueryError> {
    match field {
        "energy"            => Ok(Value::Float(node.energy as f64)),
        "depth"             => Ok(Value::Float(node.depth as f64)),
        "hausdorff_local"   => Ok(Value::Float(node.hausdorff_local as f64)),
        "lsystem_generation"=> Ok(Value::Float(node.lsystem_generation as f64)),
        "id"                => Ok(Value::Str(node.id.to_string())),
        "created_at"        => Ok(Value::Float(node.created_at as f64)),
        _ => Err(QueryError::Execution(format!("unknown node field '{field}'"))),
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
            _ => return Err(QueryError::Execution("bool comparison supports only = and !=".into())),
        }),
        _ => Err(QueryError::Execution("type mismatch in comparison".into())),
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
    nodes:  &mut Vec<Node>,
    alias:  &str,
    order:  &OrderBy,
    params: &Params,
) -> Result<(), QueryError> {
    // Pre-compute sort keys to avoid repeated evaluation errors
    let mut keyed: Vec<(f64, Node)> = nodes
        .drain(..)
        .map(|node| {
            let key = compute_order_key(&order.expr, alias, &node, params).unwrap_or(f64::MAX);
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

fn compute_order_key(expr: &OrderExpr, alias: &str, node: &Node, params: &Params) -> Result<f64, QueryError> {
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
        OrderExpr::Alias(a) => {
            Err(QueryError::Execution(format!("ORDER BY alias '{a}' is not a sortable scalar")))
        }
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
}
