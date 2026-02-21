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
            Rule::match_query              => return parse_match_or_mutate_query(inner),
            Rule::create_query             => return Ok(Query::Create(parse_create_query(inner)?)),
            Rule::merge_query              => return Ok(Query::Merge(parse_merge_query(inner)?)),
            Rule::diffuse_query            => return Ok(Query::Diffuse(parse_diffuse_query(inner)?)),
            Rule::reconstruct_query        => return Ok(Query::Reconstruct(parse_reconstruct_query(inner)?)),
            Rule::explain_query            => return parse_explain_query(inner),
            Rule::invoke_zaratustra_query  => return Ok(Query::InvokeZaratustra(parse_invoke_zaratustra(inner)?)),
            Rule::begin_tx_query           => return Ok(Query::BeginTx),
            Rule::commit_tx_query          => return Ok(Query::CommitTx),
            Rule::rollback_tx_query        => return Ok(Query::RollbackTx),
            Rule::create_daemon_query      => return Ok(Query::CreateDaemon(parse_create_daemon(inner)?)),
            Rule::drop_daemon_query        => return Ok(Query::DropDaemon(parse_drop_daemon(inner)?)),
            Rule::show_daemons_query       => return Ok(Query::ShowDaemons),
            // Dream Queries (Phase 15.2)
            Rule::dream_from_query         => return Ok(Query::DreamFrom(parse_dream_from(inner)?)),
            Rule::apply_dream_query        => return Ok(Query::ApplyDream(parse_apply_dream(inner)?)),
            Rule::reject_dream_query       => return Ok(Query::RejectDream(parse_reject_dream(inner)?)),
            Rule::show_dreams_query        => return Ok(Query::ShowDreams),
            // Synesthesia (Phase 15.3)
            Rule::translate_query          => return Ok(Query::Translate(parse_translate(inner)?)),
            // Eternal Return (Phase 15.4)
            Rule::counterfactual_query     => return Ok(Query::Counterfactual(parse_counterfactual(inner)?)),
            // Collective Unconscious (Phase 15.6)
            Rule::show_archetypes_query    => return Ok(Query::ShowArchetypes),
            Rule::share_archetype_query    => return Ok(Query::ShareArchetype(parse_share_archetype(inner)?)),
            // Narrative Engine (Phase 15.7)
            Rule::narrate_query            => return Ok(Query::Narrate(parse_narrate(inner)?)),
            Rule::EOI                      => {}
            r => return Err(QueryError::Parse(format!("unexpected rule: {r:?}"))),
        }
    }
    Err(QueryError::Parse("empty query".into()))
}

fn parse_explain_query(pair: Pair<Rule>) -> Result<Query, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("EXPLAIN requires a query".into()))?;
    match inner.as_rule() {
        Rule::match_query       => Ok(Query::Explain(Box::new(parse_match_or_mutate_query(inner)?))),
        Rule::diffuse_query     => Ok(Query::Explain(Box::new(Query::Diffuse(parse_diffuse_query(inner)?)))),
        Rule::reconstruct_query => Ok(Query::Explain(Box::new(Query::Reconstruct(parse_reconstruct_query(inner)?)))),
        r => Err(QueryError::Parse(format!("unexpected in explain_query: {r:?}"))),
    }
}

// ── MATCH (with optional SET / DELETE) ──────────────────────

/// Parse a unified match_query which may contain SET or DELETE clauses.
/// Returns Query::Match, Query::MatchSet, or Query::MatchDelete.
fn parse_match_or_mutate_query(pair: Pair<Rule>) -> Result<Query, QueryError> {
    let mut pattern      = None;
    let mut conditions   = Vec::new();
    let mut ret          = None;
    let mut assignments  = Vec::new();
    let mut targets      = Vec::new();
    let mut has_set      = false;
    let mut has_delete   = false;
    let mut as_of_cycle  = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::match_clause  => { pattern = Some(parse_pattern(inner)?); }
            Rule::where_clause  => { conditions = parse_where_clause(inner)?; }
            Rule::return_clause => { ret = Some(parse_return_clause(inner)?); }
            Rule::as_of_cycle_clause => {
                let n: u32 = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("AS OF CYCLE missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad AS OF CYCLE value".into()))?;
                as_of_cycle = Some(n);
            }
            Rule::set_clause    => {
                has_set = true;
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::set_assignment {
                        assignments.push(parse_set_assignment(child)?);
                    }
                }
            }
            Rule::delete_clause => {
                has_delete = true;
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::delete_target {
                        let alias = child.into_inner().next()
                            .ok_or_else(|| QueryError::Parse("DELETE target missing ident".into()))?
                            .as_str().to_string();
                        targets.push(alias);
                    }
                }
            }
            r => return Err(QueryError::Parse(format!("unexpected in match_query: {r:?}"))),
        }
    }

    let pat = pattern.ok_or_else(|| QueryError::Parse("missing MATCH pattern".into()))?;

    if has_set {
        Ok(Query::MatchSet(MatchSetQuery {
            pattern:     pat,
            conditions,
            assignments,
            ret,
        }))
    } else if has_delete {
        if targets.is_empty() {
            return Err(QueryError::Parse("DELETE requires at least one target".into()));
        }
        Ok(Query::MatchDelete(MatchDeleteQuery {
            pattern:    pat,
            conditions,
            targets,
        }))
    } else {
        Ok(Query::Match(MatchQuery {
            pattern:      pat,
            conditions,
            ret:          ret.ok_or_else(|| QueryError::Parse("missing RETURN clause".into()))?,
            as_of_cycle,
        }))
    }
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
    let (direction, edge_label, hop_range) = parse_edge_dir(edge_dir_pair)?;

    let to = parse_node_pattern(
        inner.next().ok_or_else(|| QueryError::Parse("missing path 'to' node".into()))?
    )?;

    Ok(PathPattern { from, edge_label, direction, to, hop_range })
}

fn parse_edge_dir(pair: Pair<Rule>) -> Result<(Direction, Option<String>, Option<HopRange>), QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty edge_dir".into()))?;

    let direction = match inner.as_rule() {
        Rule::edge_out => Direction::Out,
        Rule::edge_in  => Direction::In,
        r => return Err(QueryError::Parse(format!("unexpected edge rule: {r:?}"))),
    };

    let mut edge_label = None;
    let mut hop_range  = None;

    for child in inner.into_inner() {
        match child.as_rule() {
            Rule::edge_label => {
                edge_label = Some(child.as_str().to_string());
            }
            Rule::hop_range => {
                hop_range = Some(parse_hop_range(child)?);
            }
            r => return Err(QueryError::Parse(format!("unexpected in edge: {r:?}"))),
        }
    }

    Ok((direction, edge_label, hop_range))
}

fn parse_hop_range(pair: Pair<Rule>) -> Result<HopRange, QueryError> {
    let mut inner = pair.into_inner();
    let min: usize = inner.next()
        .ok_or_else(|| QueryError::Parse("hop_range missing min".into()))?
        .as_str().parse()
        .map_err(|_| QueryError::Parse("bad hop_range min".into()))?;
    let max = match inner.next() {
        Some(p) => p.as_str().parse()
            .map_err(|_| QueryError::Parse("bad hop_range max".into()))?,
        None => min,
    };
    Ok(HopRange { min, max })
}

// ── CREATE ─────────────────────────────────────────────────

fn parse_create_query(pair: Pair<Rule>) -> Result<CreateQuery, QueryError> {
    let mut alias      = None;
    let mut label      = None;
    let mut properties = Vec::new();
    let mut ret        = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::create_node_pattern => {
                let mut parts = inner.into_inner();
                alias = Some(
                    parts.next()
                        .ok_or_else(|| QueryError::Parse("CREATE node missing alias".into()))?
                        .as_str().to_string()
                );
                for child in parts {
                    match child.as_rule() {
                        Rule::ident    => { label = Some(child.as_str().to_string()); }
                        Rule::prop_map => { properties = parse_prop_map(child)?; }
                        r => return Err(QueryError::Parse(format!("unexpected in create_node_pattern: {r:?}"))),
                    }
                }
            }
            Rule::return_clause => { ret = Some(parse_return_clause(inner)?); }
            r => return Err(QueryError::Parse(format!("unexpected in create_query: {r:?}"))),
        }
    }

    Ok(CreateQuery {
        alias:      alias.ok_or_else(|| QueryError::Parse("CREATE missing node alias".into()))?,
        label,
        properties,
        ret,
    })
}

// ── MERGE ─────────────────────────────────────────────────

fn parse_merge_query(pair: Pair<Rule>) -> Result<MergeQuery, QueryError> {
    let mut pattern   = None;
    let mut on_create = Vec::new();
    let mut on_match  = Vec::new();
    let mut ret       = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::merge_pattern   => { pattern = Some(parse_merge_pattern(inner)?); }
            Rule::on_create_clause => { on_create = parse_on_clause(inner)?; }
            Rule::on_match_clause  => { on_match = parse_on_clause(inner)?; }
            Rule::return_clause    => { ret = Some(parse_return_clause(inner)?); }
            r => return Err(QueryError::Parse(format!("unexpected in merge_query: {r:?}"))),
        }
    }

    Ok(MergeQuery {
        pattern:   pattern.ok_or_else(|| QueryError::Parse("missing MERGE pattern".into()))?,
        on_create,
        on_match,
        ret,
    })
}

fn parse_merge_pattern(pair: Pair<Rule>) -> Result<MergePattern, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty merge_pattern".into()))?;

    match inner.as_rule() {
        Rule::merge_edge_pattern => Ok(MergePattern::Edge(parse_merge_edge_pattern(inner)?)),
        Rule::merge_node_pattern => Ok(MergePattern::Node(parse_merge_node_pattern(inner)?)),
        r => Err(QueryError::Parse(format!("unexpected in merge_pattern: {r:?}"))),
    }
}

fn parse_merge_node_pattern(pair: Pair<Rule>) -> Result<MergeNodePattern, QueryError> {
    let mut inner = pair.into_inner();
    let alias = inner.next()
        .ok_or_else(|| QueryError::Parse("merge node missing alias".into()))?
        .as_str().to_string();

    let mut label      = None;
    let mut properties = Vec::new();

    for child in inner {
        match child.as_rule() {
            Rule::ident    => { label = Some(child.as_str().to_string()); }
            Rule::prop_map => { properties = parse_prop_map(child)?; }
            r => return Err(QueryError::Parse(format!("unexpected in merge_node_pattern: {r:?}"))),
        }
    }

    Ok(MergeNodePattern { alias, label, properties })
}

fn parse_merge_edge_pattern(pair: Pair<Rule>) -> Result<MergeEdgePattern, QueryError> {
    let mut inner = pair.into_inner();

    let from = parse_merge_node_pattern(
        inner.next().ok_or_else(|| QueryError::Parse("merge edge missing 'from' node".into()))?
    )?;

    let edge_dir_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("merge edge missing direction".into()))?;
    let (direction, edge_label, _hop_range) = parse_edge_dir(edge_dir_pair)?;

    let to = parse_merge_node_pattern(
        inner.next().ok_or_else(|| QueryError::Parse("merge edge missing 'to' node".into()))?
    )?;

    Ok(MergeEdgePattern { from, edge_label, direction, to })
}

fn parse_prop_map(pair: Pair<Rule>) -> Result<Vec<(String, Expr)>, QueryError> {
    let mut props = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::prop_pair {
            let mut pp = child.into_inner();
            let key = pp.next()
                .ok_or_else(|| QueryError::Parse("prop_pair missing key".into()))?
                .as_str().to_string();
            let val_pair = pp.next()
                .ok_or_else(|| QueryError::Parse("prop_pair missing value".into()))?;
            let val = parse_atom(val_pair)?;
            props.push((key, val));
        }
    }
    Ok(props)
}

fn parse_on_clause(pair: Pair<Rule>) -> Result<Vec<SetAssignment>, QueryError> {
    let mut assignments = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::set_clause {
            for sc_child in child.into_inner() {
                if sc_child.as_rule() == Rule::set_assignment {
                    assignments.push(parse_set_assignment(sc_child)?);
                }
            }
        }
    }
    Ok(assignments)
}

fn parse_set_assignment(pair: Pair<Rule>) -> Result<SetAssignment, QueryError> {
    let mut inner = pair.into_inner();
    let prop_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("set_assignment missing prop".into()))?;
    let (alias, field) = parse_prop(prop_pair)?;
    let val_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("set_assignment missing value".into()))?;
    let value = parse_atom(val_pair)?;
    Ok(SetAssignment { alias, field, value })
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
        "NOW"                  => Ok(MathFunc::Now),
        "EPOCH_MS"             => Ok(MathFunc::EpochMs),
        "INTERVAL"             => Ok(MathFunc::Interval),
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
        Rule::string => {
            let raw = inner.as_str();
            // Strip surrounding quotes
            let s = &raw[1..raw.len() - 1];
            Ok(MathFuncArg::Str(s.to_string()))
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
        // time functions
        MathFunc::Now     => (0, 0),
        MathFunc::EpochMs => (0, 0),
        MathFunc::Interval => (1, 1),
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

// ── INVOKE ZARATUSTRA (Phase C) ───────────────────────────

fn parse_invoke_zaratustra(pair: Pair<Rule>) -> Result<InvokeZaratustraQuery, QueryError> {
    let mut collection = None;
    let mut cycles     = None;
    let mut alpha      = None;
    let mut decay      = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::zaratustra_in => {
                // zaratustra_in = { kw_in ~ string }
                let s = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("INVOKE ZARATUSTRA IN missing string".into()))?
                    .as_str();
                // strip surrounding quotes
                collection = Some(s[1..s.len()-1].to_string());
            }
            Rule::zaratustra_cycles => {
                let n: u32 = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("CYCLES missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad CYCLES value".into()))?;
                cycles = Some(n);
            }
            Rule::zaratustra_alpha => {
                let f: f64 = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("ALPHA missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad ALPHA value".into()))?;
                alpha = Some(f);
            }
            Rule::zaratustra_decay => {
                let f: f64 = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("DECAY missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad DECAY value".into()))?;
                decay = Some(f);
            }
            r => return Err(QueryError::Parse(format!("unexpected in invoke_zaratustra_query: {r:?}"))),
        }
    }

    Ok(InvokeZaratustraQuery { collection, cycles, alpha, decay })
}

// ── DAEMON Agents ─────────────────────────────────────────

fn parse_create_daemon(pair: Pair<Rule>) -> Result<CreateDaemonQuery, QueryError> {
    let mut name       = None;
    let mut on_pattern = None;
    let mut when_cond  = None;
    let mut then_action = None;
    let mut every      = None;
    let mut energy     = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::daemon_name => {
                name = Some(inner.as_str().to_string());
            }
            Rule::node_pattern => {
                on_pattern = Some(parse_node_pattern(inner)?);
            }
            Rule::daemon_when => {
                let cond_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("WHEN missing condition".into()))?;
                when_cond = Some(parse_or_cond(
                    cond_pair.into_inner().next()
                        .ok_or_else(|| QueryError::Parse("empty WHEN conditions".into()))?
                )?);
            }
            Rule::daemon_then => {
                let action_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("THEN missing action".into()))?;
                then_action = Some(parse_daemon_action(action_pair)?);
            }
            Rule::daemon_every => {
                let atom_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("EVERY missing expression".into()))?;
                every = Some(parse_atom(atom_pair)?);
            }
            Rule::daemon_energy => {
                let num_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("ENERGY missing value".into()))?;
                let f: f64 = num_pair.as_str().parse()
                    .map_err(|_| QueryError::Parse("bad ENERGY value".into()))?;
                energy = Some(f);
            }
            r => return Err(QueryError::Parse(format!("unexpected in create_daemon_query: {r:?}"))),
        }
    }

    Ok(CreateDaemonQuery {
        name:        name.ok_or_else(|| QueryError::Parse("CREATE DAEMON missing name".into()))?,
        on_pattern:  on_pattern.ok_or_else(|| QueryError::Parse("CREATE DAEMON missing ON pattern".into()))?,
        when_cond:   when_cond.ok_or_else(|| QueryError::Parse("CREATE DAEMON missing WHEN condition".into()))?,
        then_action: then_action.ok_or_else(|| QueryError::Parse("CREATE DAEMON missing THEN action".into()))?,
        every:       every.ok_or_else(|| QueryError::Parse("CREATE DAEMON missing EVERY interval".into()))?,
        energy,
    })
}

fn parse_daemon_action(pair: Pair<Rule>) -> Result<DaemonAction, QueryError> {
    let inner = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("empty daemon_action".into()))?;
    match inner.as_rule() {
        Rule::daemon_delete_action => {
            let alias = inner.into_inner().next()
                .ok_or_else(|| QueryError::Parse("DELETE action missing alias".into()))?
                .as_str().to_string();
            Ok(DaemonAction::Delete { alias })
        }
        Rule::daemon_set_action => {
            let mut assignments = Vec::new();
            for child in inner.into_inner() {
                if child.as_rule() == Rule::set_assignment {
                    assignments.push(parse_set_assignment(child)?);
                }
            }
            Ok(DaemonAction::Set { assignments })
        }
        Rule::daemon_diffuse_action => {
            let mut alias    = None;
            let mut t_values = vec![1.0_f64];
            let mut max_hops = 10_usize;

            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::ident => { alias = Some(child.as_str().to_string()); }
                    Rule::diffuse_t => {
                        let list_pair = child.into_inner().next()
                            .ok_or_else(|| QueryError::Parse("DIFFUSE WITH t= missing list".into()))?;
                        t_values = parse_num_list(list_pair)?;
                    }
                    Rule::diffuse_hops => {
                        max_hops = child.into_inner().next()
                            .ok_or_else(|| QueryError::Parse("MAX_HOPS missing value".into()))?
                            .as_str().parse()
                            .map_err(|_| QueryError::Parse("bad MAX_HOPS value".into()))?;
                    }
                    r => return Err(QueryError::Parse(format!("unexpected in daemon_diffuse_action: {r:?}"))),
                }
            }

            Ok(DaemonAction::Diffuse {
                alias:    alias.ok_or_else(|| QueryError::Parse("DIFFUSE action missing alias".into()))?,
                t_values,
                max_hops,
            })
        }
        r => Err(QueryError::Parse(format!("unexpected daemon_action inner: {r:?}"))),
    }
}

fn parse_drop_daemon(pair: Pair<Rule>) -> Result<DropDaemonQuery, QueryError> {
    let name = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("DROP DAEMON missing name".into()))?
        .as_str().to_string();
    Ok(DropDaemonQuery { name })
}

// ── Dream Queries (Phase 15.2) ────────────────────────────

fn parse_dream_from(pair: Pair<Rule>) -> Result<DreamFromQuery, QueryError> {
    let mut seed  = None;
    let mut depth = None;
    let mut noise = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::dream_seed => {
                let src = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("DREAM FROM missing seed".into()))?;
                seed = Some(match src.as_rule() {
                    Rule::param => DiffuseFrom::Param(src.as_str()[1..].to_string()),
                    Rule::ident => DiffuseFrom::Alias(src.as_str().to_string()),
                    r => return Err(QueryError::Parse(format!("unexpected dream_seed: {r:?}"))),
                });
            }
            Rule::dream_depth => {
                let n: usize = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("DEPTH missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad DEPTH value".into()))?;
                depth = Some(n);
            }
            Rule::dream_noise => {
                let n_pair = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("NOISE missing value".into()))?;
                let f: f64 = n_pair.as_str().parse()
                    .map_err(|_| QueryError::Parse("bad NOISE value".into()))?;
                noise = Some(f);
            }
            r => return Err(QueryError::Parse(format!("unexpected in dream_from_query: {r:?}"))),
        }
    }

    Ok(DreamFromQuery {
        seed:  seed.ok_or_else(|| QueryError::Parse("DREAM FROM is required".into()))?,
        depth,
        noise,
    })
}

fn parse_apply_dream(pair: Pair<Rule>) -> Result<ApplyDreamQuery, QueryError> {
    let param = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("APPLY DREAM missing param".into()))?;
    Ok(ApplyDreamQuery { dream_id: param.as_str()[1..].to_string() })
}

fn parse_reject_dream(pair: Pair<Rule>) -> Result<RejectDreamQuery, QueryError> {
    let param = pair.into_inner().next()
        .ok_or_else(|| QueryError::Parse("REJECT DREAM missing param".into()))?;
    Ok(RejectDreamQuery { dream_id: param.as_str()[1..].to_string() })
}

// ── Synesthesia (Phase 15.3) ──────────────────────────────

fn parse_translate(pair: Pair<Rule>) -> Result<TranslateQuery, QueryError> {
    let mut target        = None;
    let mut from_modality = None;
    let mut to_modality   = None;
    let mut quality       = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::translate_target => {
                let src = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("TRANSLATE missing target".into()))?;
                target = Some(match src.as_rule() {
                    Rule::param => ReconstructTarget::Param(src.as_str()[1..].to_string()),
                    Rule::ident => ReconstructTarget::Alias(src.as_str().to_string()),
                    r => return Err(QueryError::Parse(format!("unexpected translate_target: {r:?}"))),
                });
            }
            Rule::translate_from => {
                let modality = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("FROM missing modality".into()))?
                    .as_str().to_string();
                from_modality = Some(modality);
            }
            Rule::translate_to => {
                let modality = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("TO missing modality".into()))?
                    .as_str().to_string();
                to_modality = Some(modality);
            }
            Rule::quality_level => {
                quality = Some(inner.as_str().to_string());
            }
            r => return Err(QueryError::Parse(format!("unexpected in translate_query: {r:?}"))),
        }
    }

    Ok(TranslateQuery {
        target:        target.ok_or_else(|| QueryError::Parse("TRANSLATE target required".into()))?,
        from_modality: from_modality.ok_or_else(|| QueryError::Parse("FROM modality required".into()))?,
        to_modality:   to_modality.ok_or_else(|| QueryError::Parse("TO modality required".into()))?,
        quality,
    })
}

// ── Counterfactual (Phase 15.4) ───────────────────────────

fn parse_counterfactual(pair: Pair<Rule>) -> Result<CounterfactualQuery, QueryError> {
    let mut overlays = Vec::new();
    let mut inner_match = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::set_clause => {
                for child in inner.into_inner() {
                    if child.as_rule() == Rule::set_assignment {
                        overlays.push(parse_set_assignment(child)?);
                    }
                }
            }
            Rule::match_query => {
                let mq = parse_match_or_mutate_query(inner)?;
                if let Query::Match(m) = mq {
                    inner_match = Some(m);
                } else {
                    return Err(QueryError::Parse("COUNTERFACTUAL inner must be a MATCH query".into()));
                }
            }
            r => return Err(QueryError::Parse(format!("unexpected in counterfactual_query: {r:?}"))),
        }
    }

    Ok(CounterfactualQuery {
        overlays,
        inner: inner_match.ok_or_else(|| QueryError::Parse("COUNTERFACTUAL missing MATCH".into()))?,
    })
}

// ── Collective Unconscious (Phase 15.6) ───────────────────

fn parse_share_archetype(pair: Pair<Rule>) -> Result<ShareArchetypeQuery, QueryError> {
    let mut inner = pair.into_inner();
    let param = inner.next()
        .ok_or_else(|| QueryError::Parse("SHARE ARCHETYPE missing param".into()))?;
    let node_param = param.as_str()[1..].to_string();
    let string_pair = inner.next()
        .ok_or_else(|| QueryError::Parse("SHARE ARCHETYPE missing target collection".into()))?;
    let raw = string_pair.as_str();
    let target_collection = raw[1..raw.len()-1].to_string();
    Ok(ShareArchetypeQuery { node_param, target_collection })
}

// ── Narrative Engine (Phase 15.7) ─────────────────────────

fn parse_narrate(pair: Pair<Rule>) -> Result<NarrateQuery, QueryError> {
    let mut collection   = None;
    let mut window_hours = None;
    let mut format       = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::narrate_in => {
                let s = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("NARRATE IN missing string".into()))?
                    .as_str();
                collection = Some(s[1..s.len()-1].to_string());
            }
            Rule::narrate_window => {
                let n: u64 = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("WINDOW missing value".into()))?
                    .as_str().parse()
                    .map_err(|_| QueryError::Parse("bad WINDOW value".into()))?;
                window_hours = Some(n);
            }
            Rule::narrate_format => {
                let f = inner.into_inner().next()
                    .ok_or_else(|| QueryError::Parse("FORMAT missing value".into()))?
                    .as_str().to_string();
                format = Some(f);
            }
            r => return Err(QueryError::Parse(format!("unexpected in narrate_query: {r:?}"))),
        }
    }

    Ok(NarrateQuery { collection, window_hours, format })
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

    // ── Time function parser tests ──────────────────────

    #[test]
    fn parse_now() {
        let nql = r#"MATCH (n) WHERE NOW() > 1700000000.0 RETURN n"#;
        let query = parse(nql).unwrap();
        if let Query::Match(m) = query {
            assert!(!m.conditions.is_empty());
            if let Condition::Compare { left, .. } = &m.conditions[0] {
                assert!(matches!(left, Expr::MathFunc { func: MathFunc::Now, .. }));
            } else { panic!("expected Compare") }
        } else { panic!("expected Match") }
    }

    #[test]
    fn parse_epoch_ms() {
        let nql = r#"MATCH (n) WHERE EPOCH_MS() > 0.0 RETURN n"#;
        let query = parse(nql).unwrap();
        if let Query::Match(m) = query {
            if let Condition::Compare { left, .. } = &m.conditions[0] {
                assert!(matches!(left, Expr::MathFunc { func: MathFunc::EpochMs, .. }));
            } else { panic!("expected Compare") }
        } else { panic!("expected Match") }
    }

    #[test]
    fn parse_interval() {
        let nql = r#"MATCH (n) WHERE INTERVAL("1h") = 3600.0 RETURN n"#;
        let query = parse(nql).unwrap();
        if let Query::Match(m) = query {
            if let Condition::Compare { left, .. } = &m.conditions[0] {
                if let Expr::MathFunc { func, args } = left {
                    assert_eq!(*func, MathFunc::Interval);
                    assert!(matches!(&args[0], MathFuncArg::Str(s) if s == "1h"));
                } else { panic!("expected MathFunc") }
            } else { panic!("expected Compare") }
        } else { panic!("expected Match") }
    }

    #[test]
    fn parse_now_minus_interval() {
        // This tests that NOW() and INTERVAL() can both appear in expressions
        let nql = r#"MATCH (n) WHERE NOW() > INTERVAL("7d") RETURN n"#;
        assert!(parse(nql).is_ok());
    }

    // ── DAEMON Agents ─────────────────────────────────────

    #[test]
    fn parse_create_daemon_full() {
        let nql = r#"CREATE DAEMON guardian ON (n:Memory) WHEN n.energy > 0.8 THEN DIFFUSE FROM n WITH t=[0.1, 1.0] MAX_HOPS 5 EVERY INTERVAL("1h") ENERGY 0.8"#;
        let q = parse(nql).unwrap();
        let Query::CreateDaemon(d) = q else { panic!("expected CreateDaemon") };
        assert_eq!(d.name, "guardian");
        assert_eq!(d.on_pattern.alias, "n");
        assert_eq!(d.on_pattern.label.as_deref(), Some("Memory"));
        assert!(matches!(d.then_action, DaemonAction::Diffuse { ref alias, .. } if alias == "n"));
        if let DaemonAction::Diffuse { t_values, max_hops, .. } = &d.then_action {
            assert_eq!(t_values.len(), 2);
            assert_eq!(*max_hops, 5);
        }
        assert!((d.energy.unwrap() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn parse_create_daemon_delete_action() {
        let nql = r#"CREATE DAEMON reaper ON (n:Memory) WHEN n.energy < 0.01 THEN DELETE n EVERY INTERVAL("30m") ENERGY 0.5"#;
        let q = parse(nql).unwrap();
        let Query::CreateDaemon(d) = q else { panic!("expected CreateDaemon") };
        assert_eq!(d.name, "reaper");
        assert!(matches!(d.then_action, DaemonAction::Delete { ref alias } if alias == "n"));
    }

    #[test]
    fn parse_create_daemon_set_action() {
        let nql = r#"CREATE DAEMON tagger ON (n:Memory) WHEN n.energy > 0.9 THEN SET n.tagged = true EVERY INTERVAL("2h")"#;
        let q = parse(nql).unwrap();
        let Query::CreateDaemon(d) = q else { panic!("expected CreateDaemon") };
        assert_eq!(d.name, "tagger");
        if let DaemonAction::Set { assignments } = &d.then_action {
            assert_eq!(assignments.len(), 1);
            assert_eq!(assignments[0].field, "tagged");
        } else {
            panic!("expected Set action");
        }
        // No ENERGY clause → default None
        assert!(d.energy.is_none());
    }

    #[test]
    fn parse_create_daemon_default_energy() {
        let nql = r#"CREATE DAEMON watcher ON (n) WHEN n.depth > 0.5 THEN DELETE n EVERY INTERVAL("1h")"#;
        let q = parse(nql).unwrap();
        let Query::CreateDaemon(d) = q else { panic!("expected CreateDaemon") };
        assert!(d.energy.is_none());
        assert!(d.on_pattern.label.is_none());
    }

    #[test]
    fn parse_create_daemon_no_label() {
        let nql = r#"CREATE DAEMON sweep ON (n) WHEN n.energy < 0.1 THEN DELETE n EVERY INTERVAL("5m")"#;
        let q = parse(nql).unwrap();
        let Query::CreateDaemon(d) = q else { panic!("expected CreateDaemon") };
        assert!(d.on_pattern.label.is_none());
    }

    #[test]
    fn parse_drop_daemon() {
        let q = parse("DROP DAEMON guardian").unwrap();
        let Query::DropDaemon(d) = q else { panic!("expected DropDaemon") };
        assert_eq!(d.name, "guardian");
    }

    #[test]
    fn parse_show_daemons() {
        let q = parse("SHOW DAEMONS").unwrap();
        assert!(matches!(q, Query::ShowDaemons));
    }

    #[test]
    fn parse_daemon_reject_invalid() {
        assert!(parse("CREATE DAEMON").is_err());
        assert!(parse("CREATE DAEMON x").is_err());
        assert!(parse("DROP DAEMON").is_err());
    }

    // ── Dream Queries (Phase 15.2) ──────────────────────

    #[test]
    fn parse_dream_from_full() {
        let q = parse("DREAM FROM $node DEPTH 3 NOISE 0.1").unwrap();
        let Query::DreamFrom(d) = q else { panic!("expected DreamFrom") };
        assert!(matches!(d.seed, DiffuseFrom::Param(ref s) if s == "node"));
        assert_eq!(d.depth, Some(3));
        assert!((d.noise.unwrap() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn parse_dream_from_minimal() {
        let q = parse("DREAM FROM $n").unwrap();
        let Query::DreamFrom(d) = q else { panic!("expected DreamFrom") };
        assert!(d.depth.is_none());
        assert!(d.noise.is_none());
    }

    #[test]
    fn parse_apply_dream() {
        let q = parse("APPLY DREAM $dream123").unwrap();
        let Query::ApplyDream(d) = q else { panic!("expected ApplyDream") };
        assert_eq!(d.dream_id, "dream123");
    }

    #[test]
    fn parse_reject_dream() {
        let q = parse("REJECT DREAM $d1").unwrap();
        let Query::RejectDream(d) = q else { panic!("expected RejectDream") };
        assert_eq!(d.dream_id, "d1");
    }

    #[test]
    fn parse_show_dreams() {
        let q = parse("SHOW DREAMS").unwrap();
        assert!(matches!(q, Query::ShowDreams));
    }

    // ── Synesthesia (Phase 15.3) ────────────────────────

    #[test]
    fn parse_translate_full() {
        let q = parse("TRANSLATE $node FROM audio TO text QUALITY high").unwrap();
        let Query::Translate(t) = q else { panic!("expected Translate") };
        assert!(matches!(t.target, ReconstructTarget::Param(ref s) if s == "node"));
        assert_eq!(t.from_modality, "audio");
        assert_eq!(t.to_modality, "text");
        assert_eq!(t.quality.as_deref(), Some("high"));
    }

    #[test]
    fn parse_translate_minimal() {
        let q = parse("TRANSLATE $n FROM image TO text").unwrap();
        let Query::Translate(t) = q else { panic!("expected Translate") };
        assert!(t.quality.is_none());
    }

    // ── Eternal Return (Phase 15.4) ─────────────────────

    #[test]
    fn parse_match_as_of_cycle() {
        let q = parse("MATCH (n) WHERE n.energy > 0.5 AS OF CYCLE 3 RETURN n").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert_eq!(m.as_of_cycle, Some(3));
    }

    #[test]
    fn parse_match_no_cycle() {
        let q = parse("MATCH (n) WHERE n.energy > 0.5 RETURN n").unwrap();
        let Query::Match(m) = q else { panic!("expected Match") };
        assert!(m.as_of_cycle.is_none());
    }

    #[test]
    fn parse_counterfactual() {
        let q = parse("COUNTERFACTUAL SET n.energy = 0.5 MATCH (n) WHERE n.depth > 0.3 RETURN n").unwrap();
        let Query::Counterfactual(c) = q else { panic!("expected Counterfactual") };
        assert_eq!(c.overlays.len(), 1);
        assert_eq!(c.overlays[0].field, "energy");
        assert!(!c.inner.conditions.is_empty());
    }

    // ── Collective Unconscious (Phase 15.6) ─────────────

    #[test]
    fn parse_show_archetypes() {
        let q = parse("SHOW ARCHETYPES").unwrap();
        assert!(matches!(q, Query::ShowArchetypes));
    }

    #[test]
    fn parse_share_archetype() {
        let q = parse(r#"SHARE ARCHETYPE $node_id TO "memories""#).unwrap();
        let Query::ShareArchetype(s) = q else { panic!("expected ShareArchetype") };
        assert_eq!(s.node_param, "node_id");
        assert_eq!(s.target_collection, "memories");
    }

    // ── Narrative Engine (Phase 15.7) ───────────────────

    #[test]
    fn parse_narrate_full() {
        let q = parse(r#"NARRATE IN "memories" WINDOW 24 FORMAT json"#).unwrap();
        let Query::Narrate(n) = q else { panic!("expected Narrate") };
        assert_eq!(n.collection.as_deref(), Some("memories"));
        assert_eq!(n.window_hours, Some(24));
        assert_eq!(n.format.as_deref(), Some("json"));
    }

    #[test]
    fn parse_narrate_minimal() {
        let q = parse("NARRATE").unwrap();
        let Query::Narrate(n) = q else { panic!("expected Narrate") };
        assert!(n.collection.is_none());
        assert!(n.window_hours.is_none());
        assert!(n.format.is_none());
    }
}
