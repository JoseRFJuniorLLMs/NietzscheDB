// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Builders for CREATE, SET, DELETE queries — zero text parsing.

use nietzsche_query::ast::*;
use nietzsche_query::Query;

// ─────────────────────────────────────────────────────────────
// CreateBuilder
// ─────────────────────────────────────────────────────────────

/// Builds a `Query::Create`: `CREATE (n:Label {key: val, ...})`.
pub struct CreateBuilder {
    alias:      String,
    label:      Option<String>,
    properties: Vec<(String, Expr)>,
    ret:        Option<ReturnClause>,
}

impl CreateBuilder {
    pub fn new() -> Self {
        Self {
            alias:      "n".to_string(),
            label:      None,
            properties: Vec::new(),
            ret:        None,
        }
    }

    pub fn alias(mut self, alias: &str) -> Self {
        self.alias = alias.to_string();
        self
    }

    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Set a string property.
    pub fn prop(mut self, key: &str, val: &str) -> Self {
        self.properties.push((key.to_string(), Expr::Str(val.to_string())));
        self
    }

    /// Set a float property.
    pub fn prop_f64(mut self, key: &str, val: f64) -> Self {
        self.properties.push((key.to_string(), Expr::Float(val)));
        self
    }

    /// Set an integer property.
    pub fn prop_i64(mut self, key: &str, val: i64) -> Self {
        self.properties.push((key.to_string(), Expr::Int(val)));
        self
    }

    /// Set a boolean property.
    pub fn prop_bool(mut self, key: &str, val: bool) -> Self {
        self.properties.push((key.to_string(), Expr::Bool(val)));
        self
    }

    /// Add RETURN clause.
    pub fn returning(mut self) -> Self {
        self.ret = Some(ReturnClause {
            distinct: false,
            items: vec![ReturnItem {
                expr:     ReturnExpr::Alias(self.alias.clone()),
                as_alias: None,
            }],
            group_by: Vec::new(),
            having:   None,
            order_by: None,
            limit:    None,
            skip:     None,
        });
        self
    }

    pub fn build(self) -> Query {
        Query::Create(CreateQuery {
            alias:      self.alias,
            label:      self.label,
            properties: self.properties,
            ret:        self.ret,
        })
    }
}

impl Default for CreateBuilder {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────
// MatchSetBuilder
// ─────────────────────────────────────────────────────────────

/// Builds a `Query::MatchSet`: `MATCH (n) WHERE ... SET n.x = val`.
pub struct MatchSetBuilder {
    alias:       String,
    label:       Option<String>,
    conditions:  Vec<Condition>,
    assignments: Vec<SetAssignment>,
    ret:         Option<ReturnClause>,
}

impl MatchSetBuilder {
    pub fn new() -> Self {
        Self {
            alias:       "n".to_string(),
            label:       None,
            conditions:  Vec::new(),
            assignments: Vec::new(),
            ret:         None,
        }
    }

    pub fn alias(mut self, alias: &str) -> Self {
        self.alias = alias.to_string();
        self
    }

    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    // ── WHERE ────────────────────────────────────────────────

    pub fn where_gt(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.conditions.push(Condition::Compare {
            left: Expr::Property { alias, field: f },
            op: CompOp::Gt,
            right: Expr::Float(value),
        });
        self
    }

    pub fn where_lt(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.conditions.push(Condition::Compare {
            left: Expr::Property { alias, field: f },
            op: CompOp::Lt,
            right: Expr::Float(value),
        });
        self
    }

    pub fn where_eq_f64(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.conditions.push(Condition::Compare {
            left: Expr::Property { alias, field: f },
            op: CompOp::Eq,
            right: Expr::Float(value),
        });
        self
    }

    // ── SET ──────────────────────────────────────────────────

    /// `SET alias.field = float_value`
    pub fn set(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.assignments.push(SetAssignment {
            alias,
            field: f,
            value: Expr::Float(value),
        });
        self
    }

    /// `SET alias.field = string_value`
    pub fn set_str(mut self, field: &str, value: &str) -> Self {
        let (alias, f) = split_prop(field);
        self.assignments.push(SetAssignment {
            alias,
            field: f,
            value: Expr::Str(value.to_string()),
        });
        self
    }

    /// `SET alias.field = alias.field + delta` (increment)
    pub fn set_increment(mut self, field: &str, delta: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.assignments.push(SetAssignment {
            alias: alias.clone(),
            field: f.clone(),
            value: Expr::BinOp {
                left:  Box::new(Expr::Property { alias, field: f }),
                op:    ArithOp::Add,
                right: Box::new(Expr::Float(delta)),
            },
        });
        self
    }

    /// `SET alias.field = alias.field - delta` (decrement)
    pub fn set_decrement(mut self, field: &str, delta: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.assignments.push(SetAssignment {
            alias: alias.clone(),
            field: f.clone(),
            value: Expr::BinOp {
                left:  Box::new(Expr::Property { alias, field: f }),
                op:    ArithOp::Sub,
                right: Box::new(Expr::Float(delta)),
            },
        });
        self
    }

    pub fn build(self) -> Query {
        let pattern = Pattern::Node(NodePattern {
            alias:       self.alias,
            label:       self.label,
            semantic_id: None,
        });

        Query::MatchSet(MatchSetQuery {
            pattern,
            conditions:  self.conditions,
            assignments: self.assignments,
            ret:         self.ret,
        })
    }
}

impl Default for MatchSetBuilder {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────
// MatchDeleteBuilder
// ─────────────────────────────────────────────────────────────

/// Builds a `Query::MatchDelete`: `MATCH (n) WHERE ... [DETACH] DELETE n`.
pub struct MatchDeleteBuilder {
    alias:      String,
    label:      Option<String>,
    conditions: Vec<Condition>,
    detach:     bool,
    ret:        Option<ReturnClause>,
}

impl MatchDeleteBuilder {
    pub fn new() -> Self {
        Self {
            alias:      "n".to_string(),
            label:      None,
            conditions: Vec::new(),
            detach:     false,
            ret:        None,
        }
    }

    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn where_gt(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.conditions.push(Condition::Compare {
            left: Expr::Property { alias, field: f },
            op: CompOp::Gt,
            right: Expr::Float(value),
        });
        self
    }

    pub fn where_lt(mut self, field: &str, value: f64) -> Self {
        let (alias, f) = split_prop(field);
        self.conditions.push(Condition::Compare {
            left: Expr::Property { alias, field: f },
            op: CompOp::Lt,
            right: Expr::Float(value),
        });
        self
    }

    /// `DETACH DELETE` — remove node + all incident edges.
    pub fn detach(mut self) -> Self {
        self.detach = true;
        self
    }

    pub fn build(self) -> Query {
        let alias = self.alias.clone();
        let pattern = Pattern::Node(NodePattern {
            alias:       self.alias,
            label:       self.label,
            semantic_id: None,
        });

        Query::MatchDelete(MatchDeleteQuery {
            pattern,
            conditions: self.conditions,
            targets:    vec![alias],
            detach:     self.detach,
            ret:        self.ret,
        })
    }
}

impl Default for MatchDeleteBuilder {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────
// Helper
// ─────────────────────────────────────────────────────────────

fn split_prop(s: &str) -> (String, String) {
    if let Some(dot) = s.find('.') {
        (s[..dot].to_string(), s[dot + 1..].to_string())
    } else {
        ("n".to_string(), s.to_string())
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_node() {
        let q = CreateBuilder::new()
            .label("Semantic")
            .prop("name", "quark")
            .prop_f64("energy", 0.8)
            .build();
        match q {
            Query::Create(c) => {
                assert_eq!(c.label.as_deref(), Some("Semantic"));
                assert_eq!(c.properties.len(), 2);
            }
            _ => panic!("expected Create"),
        }
    }

    #[test]
    fn match_set_energy() {
        let q = MatchSetBuilder::new()
            .where_lt("energy", 0.1)
            .set("energy", 0.0)
            .build();
        match q {
            Query::MatchSet(ms) => {
                assert_eq!(ms.conditions.len(), 1);
                assert_eq!(ms.assignments.len(), 1);
                assert_eq!(ms.assignments[0].field, "energy");
            }
            _ => panic!("expected MatchSet"),
        }
    }

    #[test]
    fn match_set_increment() {
        let q = MatchSetBuilder::new()
            .set_increment("energy", 0.1)
            .build();
        match q {
            Query::MatchSet(ms) => {
                assert_eq!(ms.assignments.len(), 1);
                match &ms.assignments[0].value {
                    Expr::BinOp { op: ArithOp::Add, .. } => {}
                    _ => panic!("expected BinOp Add"),
                }
            }
            _ => panic!("expected MatchSet"),
        }
    }

    #[test]
    fn match_delete_detach() {
        let q = MatchDeleteBuilder::new()
            .where_lt("energy", 0.1)
            .detach()
            .build();
        match q {
            Query::MatchDelete(md) => {
                assert!(md.detach);
                assert_eq!(md.targets, vec!["n"]);
            }
            _ => panic!("expected MatchDelete"),
        }
    }
}
