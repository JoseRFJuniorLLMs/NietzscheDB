//! Evaluate an AST [`Condition`] against a node's metadata.
//!
//! This is used by the daemon engine to check WHEN clauses
//! without going through the full NQL executor pipeline.

use nietzsche_graph::Node;
use nietzsche_query::ast::*;

use crate::error::DaemonError;

/// Evaluate a condition against a single node.
///
/// The `alias` is the variable name used in the daemon's ON pattern
/// (e.g. `"n"` in `ON (n:Memory)`).
pub fn evaluate_condition(
    cond:  &Condition,
    node:  &Node,
    alias: &str,
) -> Result<bool, DaemonError> {
    match cond {
        Condition::Compare { left, op, right } => {
            let lv = eval_expr(left, node, alias)?;
            let rv = eval_expr(right, node, alias)?;
            Ok(compare_values(&lv, op, &rv))
        }
        Condition::And(a, b) => {
            Ok(evaluate_condition(a, node, alias)? && evaluate_condition(b, node, alias)?)
        }
        Condition::Or(a, b) => {
            Ok(evaluate_condition(a, node, alias)? || evaluate_condition(b, node, alias)?)
        }
        Condition::Not(inner) => {
            Ok(!evaluate_condition(inner, node, alias)?)
        }
        Condition::In { expr, values } => {
            let ev = eval_expr(expr, node, alias)?;
            for v in values {
                let vv = eval_expr(v, node, alias)?;
                if values_equal(&ev, &vv) {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Condition::Between { expr, low, high } => {
            let ev = eval_expr(expr, node, alias)?;
            let lo = eval_expr(low, node, alias)?;
            let hi = eval_expr(high, node, alias)?;
            Ok(compare_values(&lo, &CompOp::Lte, &ev) && compare_values(&ev, &CompOp::Lte, &hi))
        }
        Condition::StringOp { left, op, right } => {
            let lv = eval_expr(left, node, alias)?;
            let rv = eval_expr(right, node, alias)?;
            match (&lv, &rv) {
                (EvalValue::Str(a), EvalValue::Str(b)) => {
                    Ok(match op {
                        StringCompOp::Contains   => a.contains(b.as_str()),
                        StringCompOp::StartsWith => a.starts_with(b.as_str()),
                        StringCompOp::EndsWith   => a.ends_with(b.as_str()),
                    })
                }
                _ => Ok(false),
            }
        }
        // ── NQL 2.0 conditions — simplified evaluation for daemon context ──
        Condition::IsNull { expr } => {
            match eval_expr(expr, node, alias) {
                Err(_) => Ok(true),   // field not found → null
                Ok(_)  => Ok(false),
            }
        }
        Condition::IsNotNull { expr } => {
            match eval_expr(expr, node, alias) {
                Err(_) => Ok(false),
                Ok(_)  => Ok(true),
            }
        }
        Condition::Regex { expr, pattern } => {
            let ev = eval_expr(expr, node, alias)?;
            let pv = eval_expr(pattern, node, alias)?;
            match (&ev, &pv) {
                (EvalValue::Str(text), EvalValue::Str(re_str)) => {
                    match regex::Regex::new(re_str) {
                        Ok(re) => Ok(re.is_match(text)),
                        Err(_) => Ok(false),
                    }
                }
                _ => Ok(false),
            }
        }
        Condition::Exists { .. } => {
            // EXISTS requires full graph scan — not available in daemon context
            Ok(false)
        }
    }
}

// ── Internal value type for evaluation ──────────────────

#[derive(Debug, Clone)]
enum EvalValue {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Null,
}

fn eval_expr(expr: &Expr, node: &Node, alias: &str) -> Result<EvalValue, DaemonError> {
    match expr {
        Expr::Property { alias: a, field } if a == alias => {
            Ok(resolve_node_field(node, field))
        }
        Expr::Property { alias: a, field } => {
            Err(DaemonError::Eval(format!(
                "unknown alias '{}' (expected '{}'). field='{}'", a, alias, field
            )))
        }
        Expr::Float(f) => Ok(EvalValue::Float(*f)),
        Expr::Int(i)   => Ok(EvalValue::Int(*i)),
        Expr::Str(s)   => Ok(EvalValue::Str(s.clone())),
        Expr::Bool(b)  => Ok(EvalValue::Bool(*b)),
        Expr::MathFunc { func, args } => eval_math_func(func, args),
        Expr::Param(_) => {
            // Params aren't resolved in daemon context — treat as Null
            Ok(EvalValue::Null)
        }
        _ => Ok(EvalValue::Null),
    }
}

fn resolve_node_field(node: &Node, field: &str) -> EvalValue {
    match field {
        "energy"     => EvalValue::Float(node.energy as f64),
        "depth"      => EvalValue::Float(node.depth as f64),
        "created_at" => EvalValue::Float(node.created_at as f64),
        "node_type"  => EvalValue::Str(format!("{:?}", node.node_type)),
        "hausdorff_local" => EvalValue::Float(node.hausdorff_local as f64),
        "lsystem_generation" => EvalValue::Int(node.lsystem_generation as i64),
        _ => {
            // Try content JSON fields
            if let Some(val) = node.content.get(field) {
                json_to_eval(val)
            } else {
                EvalValue::Null
            }
        }
    }
}

fn json_to_eval(val: &serde_json::Value) -> EvalValue {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() { EvalValue::Float(f) }
            else if let Some(i) = n.as_i64() { EvalValue::Int(i) }
            else { EvalValue::Null }
        }
        serde_json::Value::String(s) => EvalValue::Str(s.clone()),
        serde_json::Value::Bool(b)   => EvalValue::Bool(*b),
        _ => EvalValue::Null,
    }
}

fn eval_math_func(func: &MathFunc, _args: &[MathFuncArg]) -> Result<EvalValue, DaemonError> {
    match func {
        MathFunc::Now => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            Ok(EvalValue::Float(now))
        }
        MathFunc::EpochMs => {
            let ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as f64;
            Ok(EvalValue::Float(ms))
        }
        MathFunc::Interval => {
            // INTERVAL("1h") → seconds
            if let Some(MathFuncArg::Str(s)) = _args.first() {
                Ok(EvalValue::Float(parse_interval_str(s)))
            } else {
                Ok(EvalValue::Null)
            }
        }
        // Other math functions require graph context — return Null in daemon evaluator
        _ => Ok(EvalValue::Null),
    }
}

/// Parse an interval string like "1h", "30m", "7d" into seconds.
pub fn parse_interval_str(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() { return 0.0; }

    let (num_part, unit) = s.split_at(s.len() - 1);
    let n: f64 = num_part.parse().unwrap_or(0.0);

    match unit {
        "s" => n,
        "m" => n * 60.0,
        "h" => n * 3600.0,
        "d" => n * 86400.0,
        "w" => n * 604800.0,
        _   => 0.0,
    }
}

fn compare_values(left: &EvalValue, op: &CompOp, right: &EvalValue) -> bool {
    match (left, right) {
        (EvalValue::Float(a), EvalValue::Float(b)) => cmp_f64(*a, op, *b),
        (EvalValue::Int(a), EvalValue::Int(b))     => cmp_i64(*a, op, *b),
        (EvalValue::Float(a), EvalValue::Int(b))   => cmp_f64(*a, op, *b as f64),
        (EvalValue::Int(a), EvalValue::Float(b))   => cmp_f64(*a as f64, op, *b),
        (EvalValue::Str(a), EvalValue::Str(b))     => cmp_str(a, op, b),
        (EvalValue::Bool(a), EvalValue::Bool(b))    => match op {
            CompOp::Eq  => a == b,
            CompOp::Neq => a != b,
            _ => false,
        },
        _ => false,
    }
}

fn values_equal(a: &EvalValue, b: &EvalValue) -> bool {
    compare_values(a, &CompOp::Eq, b)
}

fn cmp_f64(a: f64, op: &CompOp, b: f64) -> bool {
    match op {
        CompOp::Lt  => a < b,
        CompOp::Lte => a <= b,
        CompOp::Gt  => a > b,
        CompOp::Gte => a >= b,
        CompOp::Eq  => (a - b).abs() < 1e-12,
        CompOp::Neq => (a - b).abs() >= 1e-12,
    }
}

fn cmp_i64(a: i64, op: &CompOp, b: i64) -> bool {
    match op {
        CompOp::Lt  => a < b,
        CompOp::Lte => a <= b,
        CompOp::Gt  => a > b,
        CompOp::Gte => a >= b,
        CompOp::Eq  => a == b,
        CompOp::Neq => a != b,
    }
}

fn cmp_str(a: &str, op: &CompOp, b: &str) -> bool {
    match op {
        CompOp::Lt  => a < b,
        CompOp::Lte => a <= b,
        CompOp::Gt  => a > b,
        CompOp::Gte => a >= b,
        CompOp::Eq  => a == b,
        CompOp::Neq => a != b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nietzsche_graph::{Node, NodeMeta, NodeType, PoincareVector};
    use uuid::Uuid;
    use std::collections::HashMap;

    fn make_node(energy: f32, depth: f32, created_at: i64) -> Node {
        Node {
            meta: NodeMeta {
                id: Uuid::new_v4(),
                node_type: NodeType::Semantic,
                energy,
                depth,
                created_at,
                content: serde_json::json!({"title": "test"}),
                lsystem_generation: 0,
                hausdorff_local: 1.0,
                expires_at: None,
                metadata: HashMap::new(),
                is_phantom: false,
            },
            embedding: PoincareVector::origin(8),
        }
    }

    #[test]
    fn eval_energy_gt() {
        let node = make_node(0.9, 0.5, 1000);
        let cond = Condition::Compare {
            left:  Expr::Property { alias: "n".into(), field: "energy".into() },
            op:    CompOp::Gt,
            right: Expr::Float(0.8),
        };
        assert!(evaluate_condition(&cond, &node, "n").unwrap());
    }

    #[test]
    fn eval_energy_lt_fails() {
        let node = make_node(0.5, 0.5, 1000);
        let cond = Condition::Compare {
            left:  Expr::Property { alias: "n".into(), field: "energy".into() },
            op:    CompOp::Gt,
            right: Expr::Float(0.8),
        };
        assert!(!evaluate_condition(&cond, &node, "n").unwrap());
    }

    #[test]
    fn eval_and_condition() {
        let node = make_node(0.9, 0.3, 1000);
        let cond = Condition::And(
            Box::new(Condition::Compare {
                left:  Expr::Property { alias: "n".into(), field: "energy".into() },
                op:    CompOp::Gt,
                right: Expr::Float(0.8),
            }),
            Box::new(Condition::Compare {
                left:  Expr::Property { alias: "n".into(), field: "depth".into() },
                op:    CompOp::Lt,
                right: Expr::Float(0.5),
            }),
        );
        assert!(evaluate_condition(&cond, &node, "n").unwrap());
    }

    #[test]
    fn eval_now_greater_than_created_at() {
        // created_at = 1000 seconds from epoch (way in the past), NOW() >> 1000
        let node = make_node(0.5, 0.5, 1000);
        let cond = Condition::Compare {
            left:  Expr::MathFunc {
                func: MathFunc::Now,
                args: vec![],
            },
            op:    CompOp::Gt,
            right: Expr::Property { alias: "n".into(), field: "created_at".into() },
        };
        assert!(evaluate_condition(&cond, &node, "n").unwrap());
    }

    #[test]
    fn eval_not_condition() {
        let node = make_node(0.5, 0.5, 1000);
        let cond = Condition::Not(Box::new(Condition::Compare {
            left:  Expr::Property { alias: "n".into(), field: "energy".into() },
            op:    CompOp::Gt,
            right: Expr::Float(0.8),
        }));
        assert!(evaluate_condition(&cond, &node, "n").unwrap());
    }
}
