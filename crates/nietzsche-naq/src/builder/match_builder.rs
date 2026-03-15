// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Builder for Match queries — zero text parsing.

use nietzsche_query::ast::*;
use nietzsche_query::Query;

// ─────────────────────────────────────────────────────────────
// Condition builder helpers
// ─────────────────────────────────────────────────────────────

/// Parse "alias.field" or just "field" (defaults alias to "n").
fn split_prop(s: &str) -> (String, String) {
    if let Some(dot) = s.find('.') {
        (s[..dot].to_string(), s[dot + 1..].to_string())
    } else {
        ("n".to_string(), s.to_string())
    }
}

fn prop_expr(s: &str) -> Expr {
    let (alias, field) = split_prop(s);
    Expr::Property { alias, field }
}

fn cond_compare(field: &str, op: CompOp, val: f64) -> Condition {
    Condition::Compare {
        left: prop_expr(field),
        op,
        right: Expr::Float(val),
    }
}

fn cond_contains(field: &str, val: &str) -> Condition {
    Condition::StringOp {
        left: prop_expr(field),
        op: StringCompOp::Contains,
        right: Expr::Str(val.to_string()),
    }
}

fn cond_starts_with(field: &str, val: &str) -> Condition {
    Condition::StringOp {
        left: prop_expr(field),
        op: StringCompOp::StartsWith,
        right: Expr::Str(val.to_string()),
    }
}

fn cond_ends_with(field: &str, val: &str) -> Condition {
    Condition::StringOp {
        left: prop_expr(field),
        op: StringCompOp::EndsWith,
        right: Expr::Str(val.to_string()),
    }
}

fn cond_between(field: &str, low: f64, high: f64) -> Condition {
    Condition::Between {
        expr: prop_expr(field),
        low:  Expr::Float(low),
        high: Expr::Float(high),
    }
}

fn cond_in(field: &str, values: &[f64]) -> Condition {
    Condition::In {
        expr:   prop_expr(field),
        values: values.iter().map(|v| Expr::Float(*v)).collect(),
    }
}

fn cond_eq_str(field: &str, val: &str) -> Condition {
    Condition::Compare {
        left: prop_expr(field),
        op: CompOp::Eq,
        right: Expr::Str(val.to_string()),
    }
}

// ─────────────────────────────────────────────────────────────
// MatchBuilder
// ─────────────────────────────────────────────────────────────

/// Builds a `Query::Match` with node pattern, conditions, return, order, limit.
pub struct MatchBuilder {
    alias:      String,
    label:      Option<String>,
    conditions: Vec<Condition>,
    order_by:   Option<OrderBy>,
    limit:      Option<usize>,
    skip:       Option<usize>,
    distinct:   bool,
    ret_items:  Vec<ReturnItem>,
    group_by:   Vec<GroupByItem>,
    having:     Option<Condition>,
}

impl MatchBuilder {
    pub fn new() -> Self {
        Self {
            alias:      "n".to_string(),
            label:      None,
            conditions: Vec::new(),
            order_by:   None,
            limit:      None,
            skip:       None,
            distinct:   false,
            ret_items:  Vec::new(),
            group_by:   Vec::new(),
            having:     None,
        }
    }

    /// Set the node alias (default: "n").
    pub fn alias(mut self, alias: &str) -> Self {
        self.alias = alias.to_string();
        self
    }

    /// Set the node label: `(n:Label)`.
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    // ── WHERE conditions ─────────────────────────────────────

    /// `WHERE field > value`
    pub fn where_gt(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Gt, value));
        self
    }

    /// `WHERE field >= value`
    pub fn where_gte(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Gte, value));
        self
    }

    /// `WHERE field < value`
    pub fn where_lt(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Lt, value));
        self
    }

    /// `WHERE field <= value`
    pub fn where_lte(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Lte, value));
        self
    }

    /// `WHERE field = value` (string)
    pub fn where_eq(mut self, field: &str, value: &str) -> Self {
        self.conditions.push(cond_eq_str(field, value));
        self
    }

    /// `WHERE field = value` (float)
    pub fn where_eq_f64(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Eq, value));
        self
    }

    /// `WHERE field CONTAINS value`
    pub fn where_contains(mut self, field: &str, value: &str) -> Self {
        self.conditions.push(cond_contains(field, value));
        self
    }

    /// `WHERE field STARTS_WITH value`
    pub fn where_starts_with(mut self, field: &str, value: &str) -> Self {
        self.conditions.push(cond_starts_with(field, value));
        self
    }

    /// `WHERE field ENDS_WITH value`
    pub fn where_ends_with(mut self, field: &str, value: &str) -> Self {
        self.conditions.push(cond_ends_with(field, value));
        self
    }

    /// `WHERE field BETWEEN low AND high`
    pub fn where_between(mut self, field: &str, low: f64, high: f64) -> Self {
        self.conditions.push(cond_between(field, low, high));
        self
    }

    /// `WHERE field IN (values...)`
    pub fn where_in(mut self, field: &str, values: &[f64]) -> Self {
        self.conditions.push(cond_in(field, values));
        self
    }

    /// Add a raw pre-built condition.
    pub fn where_cond(mut self, cond: Condition) -> Self {
        self.conditions.push(cond);
        self
    }

    // ── RETURN ───────────────────────────────────────────────

    /// `RETURN DISTINCT`
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    /// `RETURN alias.field AS alias`
    pub fn return_field(mut self, field: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.ret_items.push(ReturnItem {
            expr: ReturnExpr::Property(alias, f),
            as_alias: None,
        });
        self
    }

    /// `RETURN alias.field AS name`
    pub fn return_field_as(mut self, field: &str, name: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.ret_items.push(ReturnItem {
            expr: ReturnExpr::Property(alias, f),
            as_alias: Some(name.to_string()),
        });
        self
    }

    /// `RETURN COUNT(*) AS name`
    pub fn return_count_as(mut self, name: &str) -> Self {
        self.ret_items.push(ReturnItem {
            expr: ReturnExpr::Aggregate {
                func: AggFunc::Count,
                arg:  AggArg::Star,
            },
            as_alias: Some(name.to_string()),
        });
        self
    }

    /// `RETURN COUNT(n)`
    pub fn return_count_alias(mut self) -> Self {
        self.ret_items.push(ReturnItem {
            expr: ReturnExpr::Aggregate {
                func: AggFunc::Count,
                arg:  AggArg::Alias("n".to_string()),
            },
            as_alias: None,
        });
        self
    }

    /// `RETURN AVG(field) AS name`
    pub fn return_avg_as(mut self, field: &str, name: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.ret_items.push(ReturnItem {
            expr: ReturnExpr::Aggregate {
                func: AggFunc::Avg,
                arg:  AggArg::Property(alias, f),
            },
            as_alias: Some(name.to_string()),
        });
        self
    }

    /// `GROUP BY field`
    pub fn group_by(mut self, field: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.group_by.push(GroupByItem::Property(alias, f));
        self
    }

    // ── ORDER BY ─────────────────────────────────────────────

    /// `ORDER BY field DESC`
    pub fn order_desc(mut self, field: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.order_by = Some(OrderBy {
            expr: OrderExpr::Property(alias, f),
            dir:  OrderDir::Desc,
        });
        self
    }

    /// `ORDER BY field ASC`
    pub fn order_asc(mut self, field: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.order_by = Some(OrderBy {
            expr: OrderExpr::Property(alias, f),
            dir:  OrderDir::Asc,
        });
        self
    }

    // ── LIMIT / SKIP ─────────────────────────────────────────

    /// `LIMIT n`
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// `SKIP n`
    pub fn skip(mut self, n: usize) -> Self {
        self.skip = Some(n);
        self
    }

    // ── Build ────────────────────────────────────────────────

    /// Consume the builder and produce a `Query::Match`.
    pub fn build(self) -> Query {
        let pattern = Pattern::Node(NodePattern {
            alias:       self.alias.clone(),
            label:       self.label,
            semantic_id: None,
        });

        // Default return: RETURN n (the whole node)
        let items = if self.ret_items.is_empty() {
            vec![ReturnItem {
                expr:     ReturnExpr::Alias(self.alias.clone()),
                as_alias: None,
            }]
        } else {
            self.ret_items
        };

        let ret = ReturnClause {
            distinct: self.distinct,
            items,
            group_by: self.group_by,
            having:   self.having,
            order_by: self.order_by,
            limit:    self.limit,
            skip:     self.skip,
        };

        Query::Match(MatchQuery {
            pattern,
            optional_matches: Vec::new(),
            conditions:       self.conditions,
            ret,
            as_of_cycle:      None,
            lateral:          None,
        })
    }
}

impl Default for MatchBuilder {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────
// EdgeMatchBuilder
// ─────────────────────────────────────────────────────────────

/// Builds a `Query::Match` with a path pattern: `(a)-[:TYPE]->(b)`.
pub struct EdgeMatchBuilder {
    from_alias: String,
    from_label: Option<String>,
    to_alias:   String,
    to_label:   Option<String>,
    edge_label: Option<String>,
    edge_alias: Option<String>,
    direction:  Direction,
    conditions: Vec<Condition>,
    order_by:   Option<OrderBy>,
    limit:      Option<usize>,
    skip:       Option<usize>,
    ret_items:  Vec<ReturnItem>,
}

impl EdgeMatchBuilder {
    pub fn new(edge_type: &str) -> Self {
        Self {
            from_alias: "a".to_string(),
            from_label: None,
            to_alias:   "b".to_string(),
            to_label:   None,
            edge_label: Some(edge_type.to_string()),
            edge_alias: None,
            direction:  Direction::Out,
            conditions: Vec::new(),
            order_by:   None,
            limit:      None,
            skip:       None,
            ret_items:  Vec::new(),
        }
    }

    /// Anonymous edge: `(a)-[]->(b)`.
    pub fn anonymous() -> Self {
        Self {
            from_alias: "a".to_string(),
            from_label: None,
            to_alias:   "b".to_string(),
            to_label:   None,
            edge_label: None,
            edge_alias: None,
            direction:  Direction::Out,
            conditions: Vec::new(),
            order_by:   None,
            limit:      None,
            skip:       None,
            ret_items:  Vec::new(),
        }
    }

    pub fn from_label(mut self, label: &str) -> Self {
        self.from_label = Some(label.to_string());
        self
    }

    pub fn to_label(mut self, label: &str) -> Self {
        self.to_label = Some(label.to_string());
        self
    }

    // ── WHERE ────────────────────────────────────────────────

    pub fn where_gt(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Gt, value));
        self
    }

    pub fn where_lt(mut self, field: &str, value: f64) -> Self {
        self.conditions.push(cond_compare(field, CompOp::Lt, value));
        self
    }

    pub fn where_contains(mut self, field: &str, value: &str) -> Self {
        self.conditions.push(cond_contains(field, value));
        self
    }

    // ── RETURN ───────────────────────────────────────────────

    /// `RETURN b` — return the target node.
    pub fn return_target(mut self) -> Self {
        self.ret_items = vec![ReturnItem {
            expr:     ReturnExpr::Alias(self.to_alias.clone()),
            as_alias: None,
        }];
        self
    }

    /// `RETURN COUNT(a)` — count edges.
    pub fn return_count(mut self) -> Self {
        self.ret_items = vec![ReturnItem {
            expr: ReturnExpr::Aggregate {
                func: AggFunc::Count,
                arg:  AggArg::Alias(self.from_alias.clone()),
            },
            as_alias: None,
        }];
        self
    }

    // ── ORDER / LIMIT ────────────────────────────────────────

    pub fn order_desc(mut self, field: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.order_by = Some(OrderBy {
            expr: OrderExpr::Property(alias, f),
            dir:  OrderDir::Desc,
        });
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    // ── Build ────────────────────────────────────────────────

    pub fn build(self) -> Query {
        let pattern = Pattern::Path(PathPattern {
            from: NodePattern {
                alias:       self.from_alias.clone(),
                label:       self.from_label,
                semantic_id: None,
            },
            edge_label: self.edge_label,
            edge_alias: self.edge_alias,
            direction:  self.direction,
            to: NodePattern {
                alias:       self.to_alias.clone(),
                label:       self.to_label,
                semantic_id: None,
            },
            hop_range: None,
        });

        // Default return: RETURN a, b
        let items = if self.ret_items.is_empty() {
            vec![
                ReturnItem { expr: ReturnExpr::Alias(self.from_alias.clone()), as_alias: None },
                ReturnItem { expr: ReturnExpr::Alias(self.to_alias.clone()),   as_alias: None },
            ]
        } else {
            self.ret_items
        };

        let ret = ReturnClause {
            distinct: false,
            items,
            group_by: Vec::new(),
            having:   None,
            order_by: self.order_by,
            limit:    self.limit,
            skip:     self.skip,
        };

        Query::Match(MatchQuery {
            pattern,
            optional_matches: Vec::new(),
            conditions:       self.conditions,
            ret,
            as_of_cycle:      None,
            lateral:          None,
        })
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_simple() {
        let q = MatchBuilder::new().limit(10).build();
        match q {
            Query::Match(m) => {
                assert_eq!(m.ret.limit, Some(10));
                assert!(m.conditions.is_empty());
                match &m.pattern {
                    Pattern::Node(n) => assert_eq!(n.alias, "n"),
                    _ => panic!("expected node pattern"),
                }
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_with_label_and_filters() {
        let q = MatchBuilder::new()
            .label("Semantic")
            .where_gt("energy", 0.5)
            .where_contains("content", "physics")
            .order_desc("energy")
            .limit(10)
            .build();

        match q {
            Query::Match(m) => {
                assert_eq!(m.conditions.len(), 2);
                assert_eq!(m.ret.limit, Some(10));
                assert!(m.ret.order_by.is_some());
                match &m.pattern {
                    Pattern::Node(n) => assert_eq!(n.label.as_deref(), Some("Semantic")),
                    _ => panic!("expected node pattern"),
                }
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_edges_anonymous() {
        let q = EdgeMatchBuilder::anonymous().limit(50).build();
        match q {
            Query::Match(m) => {
                assert_eq!(m.ret.limit, Some(50));
                match &m.pattern {
                    Pattern::Path(p) => {
                        assert!(p.edge_label.is_none());
                        assert_eq!(p.from.alias, "a");
                        assert_eq!(p.to.alias, "b");
                    }
                    _ => panic!("expected path pattern"),
                }
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn edge_count() {
        let q = EdgeMatchBuilder::anonymous().return_count().build();
        match q {
            Query::Match(m) => {
                assert_eq!(m.ret.items.len(), 1);
                match &m.ret.items[0].expr {
                    ReturnExpr::Aggregate { func, .. } => assert_eq!(*func, AggFunc::Count),
                    _ => panic!("expected aggregate"),
                }
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_with_between() {
        let q = MatchBuilder::new()
            .where_between("energy", 0.3, 0.8)
            .build();
        match q {
            Query::Match(m) => {
                assert_eq!(m.conditions.len(), 1);
                match &m.conditions[0] {
                    Condition::Between { .. } => {}
                    _ => panic!("expected Between condition"),
                }
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_return_count() {
        let q = MatchBuilder::new()
            .where_gt("energy", 0.0)
            .return_count_as("total")
            .build();
        match q {
            Query::Match(m) => {
                assert_eq!(m.ret.items.len(), 1);
                assert_eq!(m.ret.items[0].as_alias.as_deref(), Some("total"));
            }
            _ => panic!("expected Match"),
        }
    }
}
