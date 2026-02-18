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
            Rule::match_query       => return Ok(Query::Match(parse_match_query(inner)?)),
            Rule::diffuse_query     => return Ok(Query::Diffuse(parse_diffuse_query(inner)?)),
            Rule::reconstruct_query => return Ok(Query::Reconstruct(parse_reconstruct_query(inner)?)),
            Rule::explain_query     => return parse_explain_query(inner),
            Rule::EOI               => {}
            r => return Err(QueryError::Parse(format!("unexpected rule: {r:?}"))),
        }
    }
    Err(QueryError::Parse("empty query".into()))
}

fn parse_explain_query(pair: Pair<Rule>) -> Result<Query, QueryError> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_query       => return Ok(Query::Explain(Box::new(Query::Match(parse_match_query(inner)?)))),
            Rule::diffuse_query     => return Ok(Query::Explain(Box::new(Query::Diffuse(parse_diffuse_query(inner)?)))),
            Rule::reconstruct_query => return Ok(Query::Explain(Box::new(Query::Reconstruct(parse_reconstruct_query(inner)?)))),
            r => return Err(QueryError::Parse(format!("unexpected in explain_query: {r:?}"))),
        }
    }
    Err(QueryError::Parse("EXPLAIN requires a query".into()))
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
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty edge_dir".into()))?;

    let direction = match inner.as_rule() {
        Rule::edge_out => Direction::Out,
        Rule::edge_in  => Direction::In,
        r => return Err(QueryError::Parse(format!("unexpected edge rule: {r:?}"))),
    };

    let edge_label = inner.into_inner()
        .find(|p| p.as_rule() == Rule::edge_label)
        .map(|p| p.as_str().to_string());

    Ok((direction, edge_label))
}

// ── WHERE ─────────────────────────────────────────────────

fn parse_where_clause(pair: Pair<Rule>) -> Result<Vec<Condition>, QueryError> {
    let conditions_pair = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty WHERE".into()))?;
    let cond = parse_or_cond(
        conditions_pair.into_inner().next()
            .ok_or_else(|| QueryError::Parse("empty conditions".into()))?
    )?;
    Ok(vec![cond])
}

fn parse_or_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
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
            let sub = parse_primary_cond(
                inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("empty NOT".into()))?
            )?;
            Ok(Condition::Not(Box::new(sub)))
        }
        Rule::paren_cond => {
            let or = inner.into_inner().next()
                .ok_or_else(|| QueryError::Parse("empty paren".into()))?;
            parse_or_cond(or)
        }
        Rule::in_cond      => parse_in_cond(inner),
        Rule::between_cond => parse_between_cond(inner),
        Rule::string_cond  => parse_string_cond(inner),
        Rule::comparison   => parse_comparison(inner),
        r => Err(QueryError::Parse(format!("unexpected primary_cond inner: {r:?}"))),
    }
}

fn parse_comparison(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    let mut inner = pair.into_inner();
    let left  = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("missing left atom".into()))?)?;
    let op    = parse_comp_op(inner.next().ok_or_else(|| QueryError::Parse("missing comp_op".into()))?)?;
    let right = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("missing right atom".into()))?)?;
    Ok(Condition::Compare { left, op, right })
}

fn parse_in_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // in_cond = { atom ~ kw_in ~ in_list }
    let mut inner = pair.into_inner();
    let expr = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("IN missing expression".into()))?)?;
    let list_pair = inner.next().ok_or_else(|| QueryError::Parse("IN missing list".into()))?;
    let values: Vec<Expr> = list_pair.into_inner()
        .map(|p| parse_atom(p))
        .collect::<Result<_, _>>()?;
    Ok(Condition::In { expr, values })
}

fn parse_between_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // between_cond = { atom ~ kw_between ~ atom ~ kw_and ~ atom }
    let mut inner = pair.into_inner();
    let expr = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("BETWEEN missing expression".into()))?)?;
    let low  = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("BETWEEN missing low".into()))?)?;
    let high = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("BETWEEN missing high".into()))?)?;
    Ok(Condition::Between { expr, low, high })
}

fn parse_string_cond(pair: Pair<Rule>) -> Result<Condition, QueryError> {
    // string_cond = { atom ~ string_op ~ atom }
    let mut inner = pair.into_inner();
    let left  = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("string op missing left".into()))?)?;
    let op_pair = inner.next().ok_or_else(|| QueryError::Parse("missing string_op".into()))?;
    let op = match op_pair.as_str() {
        "CONTAINS"    => StringCompOp::Contains,
        "STARTS_WITH" => StringCompOp::StartsWith,
        "ENDS_WITH"   => StringCompOp::EndsWith,
        s => return Err(QueryError::Parse(format!("unknown string op: {s}"))),
    };
    let right = parse_atom(inner.next().ok_or_else(|| QueryError::Parse("string op missing right".into()))?)?;
    Ok(Condition::StringOp { left, op, right })
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
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty atom".into()))?;
    match inner.as_rule() {
        Rule::math_func       => parse_math_func_expr(inner),
        Rule::hyperbolic_dist => parse_hyperbolic_dist_expr(inner),
        Rule::sensory_dist    => parse_sensory_dist_expr(inner),
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
            let s = inner.as_str();
            Ok(Expr::Str(s[1..s.len()-1].to_string()))
        }
        Rule::param => {
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
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty hdist_arg".into()))?;
    match inner.as_rule() {
        Rule::param    => Ok(HDistArg::Param(inner.as_str()[1..].to_string())),
        Rule::num_list => Ok(HDistArg::Vector(parse_num_list(inner)?)),
        r => Err(QueryError::Parse(format!("unexpected hdist_arg: {r:?}"))),
    }
}

fn parse_sensory_dist_expr(pair: Pair<Rule>) -> Result<Expr, QueryError> {
    let mut inner = pair.into_inner();
    let prop_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("SENSORY_DIST missing prop".into()))?;
    let (alias, field) = parse_prop(prop_pair)?;
    let arg_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("SENSORY_DIST missing arg".into()))?;
    let param = parse_sdist_arg(arg_pair)?;
    Ok(Expr::SensoryDist { alias, field, arg: param })
}

fn parse_sdist_arg(pair: Pair<Rule>) -> Result<String, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty sdist_arg".into()))?;
    match inner.as_rule() {
        Rule::param => Ok(inner.as_str()[1..].to_string()),
        r => Err(QueryError::Parse(format!("expected param in sdist_arg, got {r:?}"))),
    }
}

fn parse_num_list(pair: Pair<Rule>) -> Result<Vec<f64>, QueryError> {
    let mut vals = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::number => {
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
    let mut inner = pair.into_inner();
    let alias = inner.next()
        .ok_or_else(|| QueryError::Parse("prop missing alias".into()))?
        .as_str().to_string();
    let field = inner.next()
        .ok_or_else(|| QueryError::Parse("prop missing field".into()))?
        .as_str().to_string();
    Ok((alias, field))
}

// ── Mathematician-named functions ──────────────────────────

fn parse_math_func_expr(pair: Pair<Rule>) -> Result<Expr, QueryError> {
    let (func, args) = parse_math_func_parts(pair)?;
    Ok(Expr::MathFunc { func, args })
}

fn parse_math_func_parts(pair: Pair<Rule>) -> Result<(MathFunc, Vec<MathFuncArg>), QueryError> {
    let mut inner = pair.into_inner();
    let name_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("math_func missing name".into()))?;
    let func = parse_math_func_name(name_pair.as_str())?;

    let mut args = Vec::new();
    for arg_pair in inner {
        if arg_pair.as_rule() == Rule::math_func_arg {
            args.push(parse_math_func_arg(arg_pair)?);
        }
    }

    // Arity validation
    validate_math_func_arity(&func, args.len())?;

    Ok((func, args))
}

fn parse_math_func_name(s: &str) -> Result<MathFunc, QueryError> {
    match s {
        "POINCARE_DIST"        => Ok(MathFunc::PoincareDist),
        "KLEIN_DIST"           => Ok(MathFunc::KleinDist),
        "MINKOWSKI_NORM"       => Ok(MathFunc::MinkowskiNorm),
        "LOBACHEVSKY_ANGLE"    => Ok(MathFunc::LobachevskyAngle),
        "RIEMANN_CURVATURE"    => Ok(MathFunc::RiemannCurvature),
        "GAUSS_KERNEL"         => Ok(MathFunc::GaussKernel),
        "CHEBYSHEV_COEFF"      => Ok(MathFunc::ChebyshevCoeff),
        "RAMANUJAN_EXPANSION"  => Ok(MathFunc::RamanujanExpansion),
        "HAUSDORFF_DIM"        => Ok(MathFunc::HausdorffDim),
        "EULER_CHAR"           => Ok(MathFunc::EulerChar),
        "LAPLACIAN_SCORE"      => Ok(MathFunc::LaplacianScore),
        "FOURIER_COEFF"        => Ok(MathFunc::FourierCoeff),
        "DIRICHLET_ENERGY"     => Ok(MathFunc::DirichletEnergy),
        s => Err(QueryError::Parse(format!("unknown math function: {s}"))),
    }
}

fn parse_math_func_arg(pair: Pair<Rule>) -> Result<MathFuncArg, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty math_func_arg".into()))?;
    match inner.as_rule() {
        Rule::param    => Ok(MathFuncArg::Param(inner.as_str()[1..].to_string())),
        Rule::num_list => Ok(MathFuncArg::Vector(parse_num_list(inner)?)),
        Rule::number   => {
            let num_inner = inner.into_inner().next()
                .ok_or_else(|| QueryError::Parse("empty number in math_func_arg".into()))?;
            let f: f64 = num_inner.as_str().parse()
                .map_err(|_| QueryError::Parse(format!("bad number: {}", num_inner.as_str())))?;
            Ok(MathFuncArg::Float(f))
        }
        Rule::prop => {
            let (alias, field) = parse_prop(inner)?;
            Ok(MathFuncArg::Property(alias, field))
        }
        Rule::ident => Ok(MathFuncArg::Alias(inner.as_str().to_string())),
        r => Err(QueryError::Parse(format!("unexpected math_func_arg inner: {r:?}"))),
    }
}

fn validate_math_func_arity(func: &MathFunc, n: usize) -> Result<(), QueryError> {
    let (min, max) = match func {
        // dist(prop, arg) — 2 args
        MathFunc::PoincareDist     => (2, 2),
        MathFunc::KleinDist        => (2, 2),
        MathFunc::LobachevskyAngle => (2, 2),
        // norm(prop) — 1 arg
        MathFunc::MinkowskiNorm    => (1, 1),
        // node(alias) — 1 arg
        MathFunc::RiemannCurvature   => (1, 1),
        MathFunc::HausdorffDim       => (1, 1),
        MathFunc::EulerChar          => (1, 1),
        MathFunc::LaplacianScore     => (1, 1),
        MathFunc::DirichletEnergy    => (1, 1),
        MathFunc::RamanujanExpansion => (1, 1),
        // node(alias, scalar) — 2 args
        MathFunc::GaussKernel    => (2, 2),
        MathFunc::ChebyshevCoeff => (2, 2),
        MathFunc::FourierCoeff   => (2, 2),
    };
    if n < min || n > max {
        return Err(QueryError::Parse(format!(
            "{:?} expects {} argument(s), got {n}", func, if min == max { format!("{min}") } else { format!("{min}-{max}") }
        )));
    }
    Ok(())
}

// ── RETURN ────────────────────────────────────────────────

fn parse_return_clause(pair: Pair<Rule>) -> Result<ReturnClause, QueryError> {
    let mut distinct = false;
    let mut items    = Vec::new();
    let mut group_by = Vec::new();
    let mut order_by = None;
    let mut limit    = None;
    let mut skip     = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::distinct_kw  => { distinct = true; }
            Rule::return_item  => items.push(parse_return_item(inner)?),
            Rule::group_by     => { group_by = parse_group_by(inner)?; }
            Rule::order_by     => order_by = Some(parse_order_by(inner)?),
            Rule::limit_clause => {
                let n: usize = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("LIMIT missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad LIMIT value".into()))?;
                limit = Some(n);
            }
            Rule::skip_clause => {
                let n: usize = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("SKIP missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad SKIP value".into()))?;
                skip = Some(n);
            }
            r => return Err(QueryError::Parse(format!("unexpected in return_clause: {r:?}"))),
        }
    }

    if items.is_empty() {
        return Err(QueryError::Parse("RETURN requires at least one item".into()));
    }

    Ok(ReturnClause { distinct, items, group_by, order_by, limit, skip })
}

fn parse_return_item(pair: Pair<Rule>) -> Result<ReturnItem, QueryError> {
    let mut expr     = None;
    let mut as_alias = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::agg_func => {
                expr = Some(parse_agg_func(inner)?);
            }
            Rule::prop => {
                let (alias, field) = parse_prop(inner)?;
                expr = Some(ReturnExpr::Property(alias, field));
            }
            Rule::ident => {
                if expr.is_none() {
                    expr = Some(ReturnExpr::Alias(inner.as_str().to_string()));
                }
            }
            Rule::as_alias => {
                let alias_ident = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("AS missing alias name".into()))?;
                as_alias = Some(alias_ident.as_str().to_string());
            }
            r => return Err(QueryError::Parse(format!("unexpected return_item inner: {r:?}"))),
        }
    }

    Ok(ReturnItem {
        expr: expr.ok_or_else(|| QueryError::Parse("empty return_item".into()))?,
        as_alias,
    })
}

fn parse_agg_func(pair: Pair<Rule>) -> Result<ReturnExpr, QueryError> {
    let mut inner = pair.into_inner();
    let func_name = inner.next()
        .ok_or_else(|| QueryError::Parse("aggregate missing function name".into()))?;
    let func = match func_name.as_str() {
        "COUNT" => AggFunc::Count,
        "SUM"   => AggFunc::Sum,
        "AVG"   => AggFunc::Avg,
        "MIN"   => AggFunc::Min,
        "MAX"   => AggFunc::Max,
        s => return Err(QueryError::Parse(format!("unknown aggregate: {s}"))),
    };

    let arg_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("aggregate missing argument".into()))?;
    let arg = parse_agg_arg(arg_pair)?;

    Ok(ReturnExpr::Aggregate { func, arg })
}

fn parse_agg_arg(pair: Pair<Rule>) -> Result<AggArg, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty agg_arg".into()))?;
    match inner.as_rule() {
        Rule::star  => Ok(AggArg::Star),
        Rule::prop  => {
            let (alias, field) = parse_prop(inner)?;
            Ok(AggArg::Property(alias, field))
        }
        Rule::ident => Ok(AggArg::Alias(inner.as_str().to_string())),
        r => Err(QueryError::Parse(format!("unexpected agg_arg inner: {r:?}"))),
    }
}

fn parse_group_by(pair: Pair<Rule>) -> Result<Vec<GroupByItem>, QueryError> {
    let mut items = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::group_by_item => {
                let sub = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("empty group_by_item".into()))?;
                match sub.as_rule() {
                    Rule::prop => {
                        let (alias, field) = parse_prop(sub)?;
                        items.push(GroupByItem::Property(alias, field));
                    }
                    Rule::ident => items.push(GroupByItem::Alias(sub.as_str().to_string())),
                    r => return Err(QueryError::Parse(format!("unexpected group_by_item: {r:?}"))),
                }
            }
            r => return Err(QueryError::Parse(format!("unexpected in group_by: {r:?}"))),
        }
    }
    Ok(items)
}

fn parse_order_by(pair: Pair<Rule>) -> Result<OrderBy, QueryError> {
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
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty order_expr".into()))?;
    match inner.as_rule() {
        Rule::math_func => {
            let (func, args) = parse_math_func_parts(inner)?;
            Ok(OrderExpr::MathFunc { func, args })
        }
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
        Rule::sensory_dist => {
            let mut sd = inner.into_inner();
            let (alias, field) = parse_prop(
                sd.next().ok_or_else(|| QueryError::Parse("sdist missing prop".into()))?
            )?;
            let arg_pair = sd.next()
                .ok_or_else(|| QueryError::Parse("sdist missing arg".into()))?;
            let param = parse_sdist_arg(arg_pair)?;
            Ok(OrderExpr::SensoryDist { alias, field, arg: param })
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

// ── RECONSTRUCT (Phase 11) ────────────────────────────────

fn parse_reconstruct_query(pair: Pair<Rule>) -> Result<ReconstructQuery, QueryError> {
    let mut target   = None;
    let mut modality = None;
    let mut quality  = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::reconstruct_target => {
                let src = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("RECONSTRUCT missing target".into()))?;
                target = Some(match src.as_rule() {
                    Rule::param => ReconstructTarget::Param(src.as_str()[1..].to_string()),
                    Rule::ident => ReconstructTarget::Alias(src.as_str().to_string()),
                    r => return Err(QueryError::Parse(format!("unexpected reconstruct_target: {r:?}"))),
                });
            }
            Rule::modality_name => {
                modality = Some(inner.as_str().to_string());
            }
            Rule::quality_level => {
                quality = Some(inner.as_str().to_string());
            }
            r => return Err(QueryError::Parse(format!("unexpected in reconstruct_query: {r:?}"))),
        }
    }

    Ok(ReconstructQuery {
        target:   target.ok_or_else(|| QueryError::Parse("RECONSTRUCT target is required".into()))?,
        modality,
        quality,
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

    // ── Phase 11: SENSORY_DIST ──────────────────────────

    #[test]
    fn parse_sensory_dist_in_where() {
        let q = parse(
            "MATCH (n) WHERE SENSORY_DIST(n.latent, $q) < 0.3 RETURN n LIMIT 5"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Condition::Compare { left, op, .. } = &m.conditions[0] {
            assert!(matches!(left, Expr::SensoryDist { .. }));
            assert_eq!(*op, CompOp::Lt);
        } else {
            panic!("expected Compare condition");
        }
    }

    #[test]
    fn parse_sensory_dist_order_by() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.1 RETURN n ORDER BY SENSORY_DIST(n.latent, $q) ASC LIMIT 3"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        let order = m.ret.order_by.unwrap();
        assert!(matches!(order.expr, OrderExpr::SensoryDist { .. }));
        assert_eq!(order.dir, OrderDir::Asc);
    }

    // ── Phase 11: RECONSTRUCT ───────────────────────────

    #[test]
    fn parse_reconstruct_full() {
        let q = parse("RECONSTRUCT $node_id MODALITY audio QUALITY high").unwrap();
        let Query::Reconstruct(r) = q else { panic!("expected Reconstruct") };
        assert!(matches!(r.target, ReconstructTarget::Param(ref s) if s == "node_id"));
        assert_eq!(r.modality.as_deref(), Some("audio"));
        assert_eq!(r.quality.as_deref(), Some("high"));
    }

    #[test]
    fn parse_reconstruct_minimal() {
        let q = parse("RECONSTRUCT $nid").unwrap();
        let Query::Reconstruct(r) = q else { panic!("expected Reconstruct") };
        assert!(matches!(r.target, ReconstructTarget::Param(_)));
        assert!(r.modality.is_none());
        assert!(r.quality.is_none());
    }

    #[test]
    fn parse_reconstruct_modality_only() {
        let q = parse("RECONSTRUCT $n MODALITY text").unwrap();
        let Query::Reconstruct(r) = q else { panic!("expected Reconstruct") };
        assert_eq!(r.modality.as_deref(), Some("text"));
        assert!(r.quality.is_none());
    }

    // ── v2: IN operator ─────────────────────────────────

    #[test]
    fn parse_in_condition() {
        let q = parse(
            "MATCH (n) WHERE n.energy IN (0.3, 0.5, 0.9) RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Condition::In { values, .. } = &m.conditions[0] {
            assert_eq!(values.len(), 3);
        } else {
            panic!("expected In condition");
        }
    }

    // ── v2: BETWEEN operator ────────────────────────────

    #[test]
    fn parse_between_condition() {
        let q = parse(
            "MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.8 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(&m.conditions[0], Condition::Between { .. }));
    }

    #[test]
    fn parse_between_with_outer_and() {
        let q = parse(
            "MATCH (n) WHERE n.energy BETWEEN 0.3 AND 0.8 AND n.depth < 1.0 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        // Should be And(Between(...), Compare(...))
        assert!(matches!(&m.conditions[0], Condition::And(..)));
    }

    // ── v2: String operators ────────────────────────────

    #[test]
    fn parse_contains_condition() {
        let q = parse(
            "MATCH (n) WHERE n.id CONTAINS \"abc\" RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let Condition::StringOp { op, .. } = &m.conditions[0] {
            assert_eq!(*op, StringCompOp::Contains);
        } else {
            panic!("expected StringOp");
        }
    }

    #[test]
    fn parse_starts_with_condition() {
        let q = parse(
            "MATCH (n) WHERE n.id STARTS_WITH \"prefix\" RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(&m.conditions[0], Condition::StringOp { op: StringCompOp::StartsWith, .. }));
    }

    #[test]
    fn parse_ends_with_condition() {
        let q = parse(
            "MATCH (n) WHERE n.id ENDS_WITH \"suffix\" RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(matches!(&m.conditions[0], Condition::StringOp { op: StringCompOp::EndsWith, .. }));
    }

    // ── v2: DISTINCT + SKIP ─────────────────────────────

    #[test]
    fn parse_distinct_return() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN DISTINCT n LIMIT 5"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(m.ret.distinct);
    }

    #[test]
    fn parse_skip_clause() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n LIMIT 10 SKIP 5"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert_eq!(m.ret.limit, Some(10));
        assert_eq!(m.ret.skip, Some(5));
    }

    // ── v2: AS alias ────────────────────────────────────

    #[test]
    fn parse_return_as_alias() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n.energy AS e"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert_eq!(m.ret.items[0].as_alias.as_deref(), Some("e"));
    }

    // ── v2: Aggregations ────────────────────────────────

    #[test]
    fn parse_count_star() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN COUNT(*) AS total"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        if let ReturnExpr::Aggregate { func, arg } = &m.ret.items[0].expr {
            assert_eq!(*func, AggFunc::Count);
            assert!(matches!(arg, AggArg::Star));
        } else {
            panic!("expected Aggregate");
        }
        assert_eq!(m.ret.items[0].as_alias.as_deref(), Some("total"));
    }

    #[test]
    fn parse_avg_with_group_by() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN n.node_type, AVG(n.energy) AS avg_e GROUP BY n.node_type"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert_eq!(m.ret.items.len(), 2);
        assert!(matches!(&m.ret.items[1].expr, ReturnExpr::Aggregate { func: AggFunc::Avg, .. }));
        assert_eq!(m.ret.group_by.len(), 1);
    }

    #[test]
    fn parse_min_max_sum() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.0 RETURN MIN(n.energy), MAX(n.energy), SUM(n.depth)"
        ).unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert_eq!(m.ret.items.len(), 3);
        assert!(matches!(&m.ret.items[0].expr, ReturnExpr::Aggregate { func: AggFunc::Min, .. }));
        assert!(matches!(&m.ret.items[1].expr, ReturnExpr::Aggregate { func: AggFunc::Max, .. }));
        assert!(matches!(&m.ret.items[2].expr, ReturnExpr::Aggregate { func: AggFunc::Sum, .. }));
    }

    // ── v2: EXPLAIN ─────────────────────────────────────

    #[test]
    fn parse_explain_match() {
        let q = parse(
            "EXPLAIN MATCH (n) WHERE n.energy > 0.3 RETURN n LIMIT 10"
        ).unwrap();
        assert!(matches!(q, Query::Explain(inner) if matches!(*inner, Query::Match(_))));
    }

    #[test]
    fn parse_explain_diffuse() {
        let q = parse(
            "EXPLAIN DIFFUSE FROM $n RETURN path"
        ).unwrap();
        assert!(matches!(q, Query::Explain(inner) if matches!(*inner, Query::Diffuse(_))));
    }

    // ── Mathematician-named functions ──────────────────────

    #[test]
    fn parse_poincare_dist() {
        let q = parse(
            "MATCH (n) WHERE POINCARE_DIST(n.embedding, $q) < 0.5 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            if let Expr::MathFunc { func, args } = left {
                assert_eq!(*func, MathFunc::PoincareDist);
                assert_eq!(args.len(), 2);
            } else { panic!("expected MathFunc") }
        } else { panic!("expected Compare") }
    }

    #[test]
    fn parse_minkowski_norm() {
        let q = parse(
            "MATCH (n) WHERE MINKOWSKI_NORM(n.embedding) > 1.5 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            assert!(matches!(left, Expr::MathFunc { func: MathFunc::MinkowskiNorm, .. }));
        } else { panic!() }
    }

    #[test]
    fn parse_riemann_curvature() {
        let q = parse(
            "MATCH (n) WHERE RIEMANN_CURVATURE(n) > 0.0 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            assert!(matches!(left, Expr::MathFunc { func: MathFunc::RiemannCurvature, .. }));
        } else { panic!() }
    }

    #[test]
    fn parse_gauss_kernel() {
        let q = parse(
            "MATCH (n) WHERE GAUSS_KERNEL(n, $t) > 0.01 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            if let Expr::MathFunc { func, args } = left {
                assert_eq!(*func, MathFunc::GaussKernel);
                assert_eq!(args.len(), 2);
            } else { panic!() }
        } else { panic!() }
    }

    #[test]
    fn parse_hausdorff_dim() {
        let q = parse(
            "MATCH (n) WHERE HAUSDORFF_DIM(n) > 1.0 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            assert!(matches!(left, Expr::MathFunc { func: MathFunc::HausdorffDim, .. }));
        } else { panic!() }
    }

    #[test]
    fn parse_euler_char() {
        let q = parse(
            "MATCH (n) WHERE EULER_CHAR(n) > 0 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        assert!(matches!(&m.conditions[0], Condition::Compare { .. }));
    }

    #[test]
    fn parse_chebyshev_coeff() {
        let q = parse(
            "MATCH (n) WHERE CHEBYSHEV_COEFF(n, 3) > 0.5 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            if let Expr::MathFunc { func, args } = left {
                assert_eq!(*func, MathFunc::ChebyshevCoeff);
                assert_eq!(args.len(), 2);
            } else { panic!() }
        } else { panic!() }
    }

    #[test]
    fn parse_ramanujan_expansion() {
        let q = parse(
            "MATCH (n) WHERE RAMANUJAN_EXPANSION(n) > 0.5 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        assert!(matches!(&m.conditions[0], Condition::Compare { .. }));
    }

    #[test]
    fn parse_klein_dist_inline_vector() {
        let q = parse(
            "MATCH (n) WHERE KLEIN_DIST(n.embedding, [0.1, 0.2]) < 1.0 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            if let Expr::MathFunc { func, args } = left {
                assert_eq!(*func, MathFunc::KleinDist);
                assert!(matches!(&args[1], MathFuncArg::Vector(_)));
            } else { panic!() }
        } else { panic!() }
    }

    #[test]
    fn parse_order_by_poincare_dist() {
        let q = parse(
            "MATCH (n) WHERE n.energy > 0.1 RETURN n ORDER BY POINCARE_DIST(n.embedding, $q) ASC LIMIT 5"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        let order = m.ret.order_by.unwrap();
        assert!(matches!(order.expr, OrderExpr::MathFunc { func: MathFunc::PoincareDist, .. }));
        assert_eq!(order.dir, OrderDir::Asc);
    }

    #[test]
    fn parse_lobachevsky_angle() {
        let q = parse(
            "MATCH (n) WHERE LOBACHEVSKY_ANGLE(n.embedding, $q) < 0.8 RETURN n"
        ).unwrap();
        let Query::Match(m) = q else { panic!() };
        if let Condition::Compare { left, .. } = &m.conditions[0] {
            assert!(matches!(left, Expr::MathFunc { func: MathFunc::LobachevskyAngle, .. }));
        } else { panic!() }
    }

    #[test]
    fn parse_all_graph_funcs() {
        // Verify all single-arg node functions parse correctly
        for func_name in &[
            "LAPLACIAN_SCORE", "DIRICHLET_ENERGY", "FOURIER_COEFF",
        ] {
            if *func_name == "FOURIER_COEFF" {
                let nql = format!("MATCH (n) WHERE {}(n, 3) > 0.0 RETURN n", func_name);
                assert!(parse(&nql).is_ok(), "failed to parse {func_name}");
            } else {
                let nql = format!("MATCH (n) WHERE {}(n) > 0.0 RETURN n", func_name);
                assert!(parse(&nql).is_ok(), "failed to parse {func_name}");
            }
        }
    }
}
