use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::ast::*;
use crate::error::QueryError;

// ── Pest parser derive ─────────────────────────────────────

#[derive(Parser)]
#[grammar = "src/nql.pest"]
pub struct NqlParser;

// ── Public entry point ────────────────────────────────────

/// Parse an NQL query string into an AST [`Query`].
pub fn parse(input: &str) -> Result<Query, QueryError> {
    let pairs = NqlParser::parse(Rule::query, input)
        .map_err(|e| QueryError::Parse(e.to_string()))?;

    let query_pair = pairs.into_iter().next()
        .ok_or_else(|| QueryError::Parse("empty input".into()))?;

    parse_query(query_pair)
}

// ── Top-level ─────────────────────────────────────────────

fn parse_query(pair: Pair<Rule>) -> Result<Query, QueryError> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_query   => return Ok(Query::Match(parse_match_query(inner)?)),
            Rule::diffuse_query => return Ok(Query::Diffuse(parse_diffuse_query(inner)?)),
            Rule::EOI           => {}
            r => return Err(QueryError::Parse(format!("unexpected rule: {r:?}"))),
        }
    }
    Err(QueryError::Parse("empty query".into()))
}

// ── MATCH ─────────────────────────────────────────────────

fn parse_match_query(pair: Pair<Rule>) -> Result<MatchQuery, QueryError> {
    let mut pattern    = None;
    let mut conditions = Vec::new();
    let mut ret        = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_clause  => { pattern = Some(parse_pattern(inner)?); }
            Rule::where_clause  => { conditions = parse_where_clause(inner)?; }
            Rule::return_clause => { ret = Some(parse_return_clause(inner)?); }
            r => return Err(QueryError::Parse(format!("unexpected in match_query: {r:?}"))),
        }
    }

    Ok(MatchQuery {
        pattern:    pattern.ok_or_else(|| QueryError::Parse("missing MATCH pattern".into()))?,
        conditions,
        ret:        ret.ok_or_else(|| QueryError::Parse("missing RETURN clause".into()))?,
    })
}

fn parse_pattern(match_clause_pair: Pair<Rule>) -> Result<Pattern, QueryError> {
    let inner = match_clause_pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("missing pattern".into()))?;

    match inner.as_rule() {
        Rule::pattern => parse_pattern_inner(inner),
        r => Err(QueryError::Parse(format!("expected pattern, got {r:?}"))),
    }
}

fn parse_pattern_inner(pair: Pair<Rule>) -> Result<Pattern, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty pattern".into()))?;

    match inner.as_rule() {
        Rule::path_pattern => Ok(Pattern::Path(parse_path_pattern(inner)?)),
        Rule::node_pattern => Ok(Pattern::Node(parse_node_pattern(inner)?)),
        r => Err(QueryError::Parse(format!("expected path or node pattern, got {r:?}"))),
    }
}

fn parse_node_pattern(pair: Pair<Rule>) -> Result<NodePattern, QueryError> {
    let mut parts = pair.into_inner();
    let alias = parts.next()
        .ok_or_else(|| QueryError::Parse("missing node alias".into()))?
        .as_str().to_string();
    let label = parts.next().map(|p| p.as_str().to_string());
    Ok(NodePattern { alias, label })
}

fn parse_path_pattern(pair: Pair<Rule>) -> Result<PathPattern, QueryError> {
    let mut inner = pair.into_inner();

    let from = parse_node_pattern(
        inner.next().ok_or_else(|| QueryError::Parse("missing path 'from' node".into()))?
    )?;

    let edge_dir_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("missing edge direction".into()))?;
    let (direction, edge_label) = parse_edge_dir(edge_dir_pair)?;

    let to = parse_node_pattern(
        inner.next().ok_or_else(|| QueryError::Parse("missing path 'to' node".into()))?
    )?;

    Ok(PathPattern { from, edge_label, direction, to })
}

fn parse_edge_dir(pair: Pair<Rule>) -> Result<(Direction, Option<String>), QueryError> {
    // edge_dir = { edge_out | edge_in }
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty edge_dir".into()))?;

    let direction = match inner.as_rule() {
        Rule::edge_out => Direction::Out,
        Rule::edge_in  => Direction::In,
        r => return Err(QueryError::Parse(format!("unexpected edge rule: {r:?}"))),
    };

    // edge_out inner might contain edge_label or be empty (for -->)
    let edge_label = inner.into_inner()
        .find(|p| p.as_rule() == Rule::edge_label)
        .map(|p| p.as_str().to_string());

    Ok((direction, edge_label))
}

// ── WHERE ─────────────────────────────────────────────────

fn parse_where_clause(pair: Pair<Rule>) -> Result<Vec<Condition>, QueryError> {
    // where_clause = { kw_where ~ conditions }
    // kw_where is silent, so inner = [conditions]
    let conditions_pair = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty WHERE".into()))?;
    let cond = parse_or_cond(
        conditions_pair.into_inner().next()
            .ok_or_else(|| QueryError::Parse("empty conditions".into()))?
    )?;
    Ok(vec![cond])
}

fn parse_or_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // or_cond = { and_cond ~ (kw_or ~ and_cond)* }
    // kw_or is silent, so inner = [and_cond, and_cond, ...]
    let mut iter = pair.into_inner();
    let first = parse_and_cond(
        iter.next().ok_or_else(|| QueryError::Parse("empty or_cond".into()))?
    )?;
    let mut result = first;
    for next_pair in iter {
        let next = parse_and_cond(next_pair)?;
        result = Condition::Or(Box::new(result), Box::new(next));
    }
    Ok(result)
}

fn parse_and_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // and_cond = { primary_cond ~ (kw_and ~ primary_cond)* }
    let mut iter = pair.into_inner();
    let first = parse_primary_cond(
        iter.next().ok_or_else(|| QueryError::Parse("empty and_cond".into()))?
    )?;
    let mut result = first;
    for next_pair in iter {
        let next = parse_primary_cond(next_pair)?;
        result = Condition::And(Box::new(result), Box::new(next));
    }
    Ok(result)
}

fn parse_primary_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty primary_cond".into()))?;
    match inner.as_rule() {
        Rule::not_cond => {
            // not_cond = { kw_not ~ primary_cond }  — kw_not is silent
            let sub = parse_primary_cond(
                inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("empty NOT".into()))?
            )?;
            Ok(Condition::Not(Box::new(sub)))
        }
        Rule::paren_cond => {
            // paren_cond = { "(" ~ or_cond ~ ")" }
            let or = inner.into_inner().next()
                .ok_or_else(|| QueryError::Parse("empty paren".into()))?;
            parse_or_cond(or)
        }
        Rule::comparison => parse_comparison(inner),
        r => Err(QueryError::Parse(format!("unexpected primary_cond inner: {r:?}"))),
    }
}

fn parse_comparison(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // comparison = { atom ~ comp_op ~ atom }
    let mut inner = pair.into_inner();
    let left  = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("missing left atom".into()))?)?;
    let op    = parse_comp_op(inner.next().ok_or_else(|| QueryError::Parse("missing comp_op".into()))?)?;
    let right = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("missing right atom".into()))?)?;
    Ok(Condition::Compare { left, op, right })
}

fn parse_comp_op(pair: Pair<Rule>) -> Result<CompOp, QueryError> {
    match pair.as_str() {
        "<=" => Ok(CompOp::Lte),
        ">=" => Ok(CompOp::Gte),
        "!=" => Ok(CompOp::Neq),
        "<"  => Ok(CompOp::Lt),
        ">"  => Ok(CompOp::Gt),
        "="  => Ok(CompOp::Eq),
        s    => Err(QueryError::Parse(format!("unknown operator: {s}"))),
    }
}

fn parse_atom(pair: Pair<Rule>) -> Result<Expr, QueryError> {
    // atom = { hyperbolic_dist | boolean | float | integer | string | param | prop }
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty atom".into()))?;
    match inner.as_rule() {
        Rule::hyperbolic_dist => parse_hyperbolic_dist_expr(inner),
        Rule::boolean => {
            let b = inner.as_str() == "true";
            Ok(Expr::Bool(b))
        }
        Rule::float => {
            let f: f64 = inner.as_str().parse()
                .map_err(|_| QueryError::Parse(format!("bad float: {}", inner.as_str())))?;
            Ok(Expr::Float(f))
        }
        Rule::integer => {
            let i: i64 = inner.as_str().parse()
                .map_err(|_| QueryError::Parse(format!("bad integer: {}", inner.as_str())))?;
            Ok(Expr::Int(i))
        }
        Rule::string => {
            // strip surrounding quotes
            let s = inner.as_str();
            Ok(Expr::Str(s[1..s.len()-1].to_string()))
        }
        Rule::param => {
            // strip leading $
            Ok(Expr::Param(inner.as_str()[1..].to_string()))
        }
        Rule::prop => {
            let (alias, field) = parse_prop(inner)?;
            Ok(Expr::Property { alias, field })
        }
        r => Err(QueryError::Parse(format!("unexpected atom inner: {r:?}"))),
    }
}

fn parse_hyperbolic_dist_expr(pair: Pair<Rule>) -> Result<Expr, QueryError> {
    // hyperbolic_dist = { "HYPERBOLIC_DIST" ~ "(" ~ prop ~ "," ~ hdist_arg ~ ")" }
    let mut inner = pair.into_inner();
    let prop_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("HYPERBOLIC_DIST missing prop".into()))?;
    let (alias, field) = parse_prop(prop_pair)?;

    let arg_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("HYPERBOLIC_DIST missing arg".into()))?;
    let arg = parse_hdist_arg(arg_pair)?;

    Ok(Expr::HyperbolicDist { alias, field, arg })
}

fn parse_hdist_arg(pair: Pair<Rule>) -> Result<HDistArg, QueryError> {
    // hdist_arg = { param | num_list }
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty hdist_arg".into()))?;
    match inner.as_rule() {
        Rule::param    => Ok(HDistArg::Param(inner.as_str()[1..].to_string())),
        Rule::num_list => Ok(HDistArg::Vector(parse_num_list(inner)?)),
        r => Err(QueryError::Parse(format!("unexpected hdist_arg: {r:?}"))),
    }
}

fn parse_num_list(pair: Pair<Rule>) -> Result<Vec<f64>, QueryError> {
    let mut vals = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::number => {
                // number = { float | integer }
                let num_inner = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("empty number".into()))?;
                let f: f64 = num_inner.as_str().parse()
                    .map_err(|_| QueryError::Parse(format!("bad number: {}", num_inner.as_str())))?;
                vals.push(f);
            }
            r => return Err(QueryError::Parse(format!("unexpected in num_list: {r:?}"))),
        }
    }
    Ok(vals)
}

fn parse_prop(pair: Pair<Rule>) -> Result<(String, String), QueryError> {
    // prop = ${ ident ~ "." ~ ident }  — inner: [ident, ident]
    let mut inner = pair.into_inner();
    let alias = inner.next()
        .ok_or_else(|| QueryError::Parse("prop missing alias".into()))?
        .as_str().to_string();
    let field = inner.next()
        .ok_or_else(|| QueryError::Parse("prop missing field".into()))?
        .as_str().to_string();
    Ok((alias, field))
}

// ── RETURN ────────────────────────────────────────────────

fn parse_return_clause(pair: Pair<Rule>) -> Result<ReturnClause, QueryError> {
    let mut items    = Vec::new();
    let mut order_by = None;
    let mut limit    = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::return_item  => items.push(parse_return_item(inner)?),
            Rule::order_by     => order_by = Some(parse_order_by(inner)?),
            Rule::limit_clause => {
                let n: usize = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("LIMIT missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad LIMIT value".into()))?;
                limit = Some(n);
            }
            r => return Err(QueryError::Parse(format!("unexpected in return_clause: {r:?}"))),
        }
    }

    if items.is_empty() {
        return Err(QueryError::Parse("RETURN requires at least one item".into()));
    }

    Ok(ReturnClause { items, order_by, limit })
}

fn parse_return_item(pair: Pair<Rule>) -> Result<ReturnItem, QueryError> {
    // return_item = { prop | ident }
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty return_item".into()))?;
    match inner.as_rule() {
        Rule::prop  => {
            let (alias, field) = parse_prop(inner)?;
            Ok(ReturnItem::Property(alias, field))
        }
        Rule::ident => Ok(ReturnItem::Alias(inner.as_str().to_string())),
        r => Err(QueryError::Parse(format!("unexpected return_item inner: {r:?}"))),
    }
}

fn parse_order_by(pair: Pair<Rule>) -> Result<OrderBy, QueryError> {
    // order_by = { kw_order ~ kw_by ~ order_expr ~ order_dir? }
    let mut inner    = pair.into_inner();
    let expr_pair    = inner.next()
        .ok_or_else(|| QueryError::Parse("ORDER BY missing expression".into()))?;
    let order_expr   = parse_order_expr(expr_pair)?;
    let dir = inner.next()
        .map(|d| if d.as_str() == "DESC" { OrderDir::Desc } else { OrderDir::Asc })
        .unwrap_or(OrderDir::Asc);

    Ok(OrderBy { expr: order_expr, dir })
}

fn parse_order_expr(pair: Pair<Rule>) -> Result<OrderExpr, QueryError> {
    // order_expr = { hyperbolic_dist | prop | ident }
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty order_expr".into()))?;
    match inner.as_rule() {
        Rule::hyperbolic_dist => {
            let mut hd = inner.into_inner();
            let (alias, field) = parse_prop(
                hd.next().ok_or_else(|| QueryError::Parse("hdist missing prop".into()))?
            )?;
            let arg = parse_hdist_arg(
                hd.next().ok_or_else(|| QueryError::Parse("hdist missing arg".into()))?
            )?;
            Ok(OrderExpr::HyperbolicDist { alias, field, arg })
        }
        Rule::prop  => {
            let (alias, field) = parse_prop(inner)?;
            Ok(OrderExpr::Property(alias, field))
        }
        Rule::ident => Ok(OrderExpr::Alias(inner.as_str().to_string())),
        r => Err(QueryError::Parse(format!("unexpected order_expr inner: {r:?}"))),
    }
}

// ── DIFFUSE ───────────────────────────────────────────────

fn parse_diffuse_query(pair: Pair<Rule>) -> Result<DiffuseQuery, QueryError> {
    let mut from     = None;
    let mut t_values = vec![1.0_f64];
    let mut max_hops = 10_usize;
    let mut ret      = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::diffuse_from => {
                let src = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("DIFFUSE FROM missing value".into()))?;
                from = Some(match src.as_rule() {
                    Rule::param => DiffuseFrom::Param(src.as_str()[1..].to_string()),
                    Rule::ident => DiffuseFrom::Alias(src.as_str().to_string()),
                    r => return Err(QueryError::Parse(format!("unexpected diffuse_from: {r:?}"))),
                });
            }
            Rule::diffuse_t => {
                let list_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("WITH t= missing list".into()))?;
                t_values = parse_num_list(list_pair)?;
            }
            Rule::diffuse_hops => {
                max_hops = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("MAX_HOPS missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad MAX_HOPS value".into()))?;
            }
            Rule::return_clause => { ret = Some(parse_return_clause(inner)?); }
            r => return Err(QueryError::Parse(format!("unexpected in diffuse_query: {r:?}"))),
        }
    }

    Ok(DiffuseQuery {
        from:     from.ok_or_else(|| QueryError::Parse("DIFFUSE FROM is required".into()))?,
        t_values,
        max_hops,
        ret,
    })
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_node_match() {
        let q = parse("MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 10").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(m.pattern, Pattern::Node(_)));
        assert_eq!(m.conditions.len(), 1);
        assert_eq!(m.ret.limit, Some(10));
    }

    #[test]
    fn parse_node_match_with_label() {
        let q = parse("MATCH (n:Memory) WHERE n.depth < 0.5 RETURN n").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Pattern::Node(np) = &m.pattern {
            assert_eq!(np.alias, "n");
            assert_eq!(np.label.as_deref(), Some("Memory"));
        } else {
            panic!("expected NodePattern");
        }
    }

    #[test]
    fn parse_path_match() {
        let q = parse("MATCH (a)-[:ASSOCIATION]->(b) WHERE a.energy > 0.5 RETURN b").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Pattern::Path(pp) = &m.pattern {
            assert_eq!(pp.from.alias, "a");
            assert_eq!(pp.to.alias, "b");
            assert_eq!(pp.direction, Direction::Out);
            assert_eq!(pp.edge_label.as_deref(), Some("ASSOCIATION"));
        } else {
            panic!("expected PathPattern");
        }
    }

    #[test]
    fn parse_anonymous_path() {
        let q = parse("MATCH (a)-->(b) RETURN b").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(m.pattern, Pattern::Path(_)));
    }

    #[test]
    fn parse_hyperbolic_dist_condition() {
        let q = parse(
            "MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, $q) < 0.5 RETURN n LIMIT 5"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Condition::Compare { left, op, right } = &m.conditions[0] {
            assert!(matches!(left, Expr::HyperbolicDist { .. }));
            assert_eq!(*op, CompOp::Lt);
            assert!(matches!(right, Expr::Float(f) if (*f - 0.5).abs() < 1e-10));
        } else {
            panic!("expected Compare condition");
        }
    }

    #[test]
    fn parse_and_condition() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.3 AND n.depth < 0.8 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(&m.conditions[0], Condition::And(..)));
    }

    #[test]
    fn parse_not_condition() {
        let q = parse("MATCH (n) WHERE NOT n.energy > 0.9 RETURN n").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(&m.conditions[0], Condition::Not(_)));
    }

    #[test]
    fn parse_order_by_hyperbolic_dist() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.1 RETURN n ORDER BY HYPERBOLIC_DIST(n.embedding, $q) ASC LIMIT 3"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        let order = m.ret.order_by.unwrap();
        assert!(matches!(order.expr, OrderExpr::HyperbolicDist { .. }));
        assert_eq!(order.dir, OrderDir::Asc);
    }

    #[test]
    fn parse_diffuse_query() {
        let q = parse(
            "DIFFUSE FROM $node WITH t = [0.1, 1.0, 10.0] MAX_HOPS 5 RETURN activated_nodes"
        ).unwrap();
        let Query::Diffuse(d) = q else { panic!("expected Diffuse") };
        assert!(matches!(d.from, DiffuseFrom::Param(_)));
        assert_eq!(d.t_values.len(), 3);
        assert_eq!(d.max_hops, 5);
    }

    #[test]
    fn parse_diffuse_minimal() {
        let q = parse("DIFFUSE FROM $n RETURN path").unwrap();
        assert!(matches!(q, Query::Diffuse(_)));
    }

    #[test]
    fn parse_inline_vector_hdist() {
        let q = parse(
            "MATCH (n) WHERE HYPERBOLIC_DIST(n.embedding, [0.1, 0.2]) < 0.9 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            if let Expr::HyperbolicDist { arg: HDistArg::Vector(v), .. } = left {
                assert_eq!(v.len(), 2);
            } else { panic!("expected Vector arg") }
        } else { panic!("expected Compare") }
    }

    #[test]
    fn parse_error_on_invalid_input() {
        assert!(parse("SELECT * FROM nodes").is_err());
        assert!(parse("").is_err());
    }
}
