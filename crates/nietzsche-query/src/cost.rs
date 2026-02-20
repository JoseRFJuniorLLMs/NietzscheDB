//! Query cost estimator for NQL EXPLAIN plans.
//!
//! Produces cardinality estimates and scan-type classifications by
//! inspecting the query AST and collecting lightweight statistics from
//! the storage layer (RocksDB `estimate-num-keys`).

use std::fmt;

use nietzsche_graph::GraphStorage;

use crate::ast::*;

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// How the executor will scan the data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScanType {
    /// Full CF_NODES scan.
    FullScan,
    /// Secondary index on `energy` (CF_ENERGY_IDX range scan).
    EnergyIndexScan,
    /// Secondary index on an arbitrary metadata field (CF_META_IDX).
    MetaIndexScan(String),
    /// Adjacency-driven edge scan (CF_ADJ_OUT / CF_ADJ_IN).
    EdgeScan,
    /// Bounded BFS traversal (multi-hop path).
    BoundedBFS { min_hops: usize, max_hops: usize },
    /// Diffusion random walk.
    DiffusionWalk { max_hops: usize },
    /// O(1) operations (BEGIN, COMMIT, CREATE, etc.).
    Constant,
}

impl fmt::Display for ScanType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScanType::FullScan            => write!(f, "FullScan"),
            ScanType::EnergyIndexScan     => write!(f, "EnergyIndexScan"),
            ScanType::MetaIndexScan(fld)  => write!(f, "MetaIndexScan({})", fld),
            ScanType::EdgeScan            => write!(f, "EdgeScan"),
            ScanType::BoundedBFS { min_hops, max_hops } =>
                write!(f, "BoundedBFS({}..{})", min_hops, max_hops),
            ScanType::DiffusionWalk { max_hops } =>
                write!(f, "DiffusionWalk(max_hops={})", max_hops),
            ScanType::Constant            => write!(f, "Constant"),
        }
    }
}

/// Cost estimate for a single NQL query.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Estimated number of rows the scan will touch before filtering.
    pub estimated_rows: u64,
    /// The scan strategy the executor will use.
    pub scan_type: ScanType,
    /// Whether a secondary index is used to reduce the scan.
    pub index_used: Option<String>,
    /// Rough estimated wall-clock cost in microseconds.
    pub estimated_cost_us: u64,
}

impl fmt::Display for CostEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rows=~{}, scan={}, index={}, cost=~{}µs",
            self.estimated_rows,
            self.scan_type,
            self.index_used.as_deref().unwrap_or("none"),
            self.estimated_cost_us,
        )
    }
}

// ─────────────────────────────────────────────
// Estimation
// ─────────────────────────────────────────────

/// Cost per row for each scan type (microseconds), rough heuristic.
const COST_PER_ROW_FULL_SCAN: u64    = 5;   // deserialize + filter
const COST_PER_ROW_INDEX_SCAN: u64   = 2;   // prefix seek + decode
const COST_PER_ROW_EDGE_SCAN: u64    = 3;   // adjacency lookup + node fetch
const COST_PER_HOP_BFS: u64          = 10;  // BFS expansion per level
const COST_CONSTANT: u64             = 1;   // O(1) operations

/// Estimate the cost of executing the given query.
///
/// This is a best-effort heuristic — it uses `estimate-num-keys` from
/// RocksDB (O(1)) and inspects the AST to determine scan strategy.
pub fn estimate_cost(query: &Query, storage: &GraphStorage) -> CostEstimate {
    match query {
        Query::Match(m)          => estimate_match(m, storage),
        Query::Diffuse(d)        => estimate_diffuse(d, storage),
        Query::Explain(inner)    => estimate_cost(inner, storage),
        Query::MatchSet(ms)      => estimate_match_set(ms, storage),
        Query::MatchDelete(md)   => estimate_match_delete(md, storage),
        _ => CostEstimate {
            estimated_rows: 1,
            scan_type: ScanType::Constant,
            index_used: None,
            estimated_cost_us: COST_CONSTANT,
        },
    }
}

// ── MATCH ────────────────────────────────────

fn estimate_match(query: &MatchQuery, storage: &GraphStorage) -> CostEstimate {
    let node_count = storage.node_count().unwrap_or(0) as u64;
    let edge_count = storage.edge_count().unwrap_or(0) as u64;

    match &query.pattern {
        Pattern::Node(_np) => {
            // Check if energy index can be used
            if has_energy_condition(&query.conditions) {
                let estimated_rows = (node_count / 4).max(1); // ~25% selectivity guess
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::EnergyIndexScan,
                    index_used: Some("CF_ENERGY_IDX".into()),
                    estimated_cost_us: estimated_rows * COST_PER_ROW_INDEX_SCAN,
                }
            } else if let Some(field) = find_meta_index_condition(&query.conditions) {
                let estimated_rows = (node_count / 10).max(1); // ~10% selectivity guess
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::MetaIndexScan(field.clone()),
                    index_used: Some(format!("CF_META_IDX({})", field)),
                    estimated_cost_us: estimated_rows * COST_PER_ROW_INDEX_SCAN,
                }
            } else {
                // Full scan
                let estimated_rows = node_count;
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::FullScan,
                    index_used: None,
                    estimated_cost_us: estimated_rows * COST_PER_ROW_FULL_SCAN,
                }
            }
        }
        Pattern::Path(pp) => {
            if let Some(hr) = &pp.hop_range {
                // BFS: estimated fan-out per level, bounded by total nodes
                let avg_degree = if node_count > 0 { (edge_count / node_count).max(1) } else { 1 };
                let mut estimated_rows = 1u64;
                for _ in 0..hr.max {
                    estimated_rows = (estimated_rows * avg_degree).min(node_count);
                }
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::BoundedBFS { min_hops: hr.min, max_hops: hr.max },
                    index_used: Some("CF_ADJ".into()),
                    estimated_cost_us: estimated_rows * COST_PER_HOP_BFS,
                }
            } else {
                // Single-hop edge scan
                CostEstimate {
                    estimated_rows: edge_count,
                    scan_type: ScanType::EdgeScan,
                    index_used: Some("CF_ADJ".into()),
                    estimated_cost_us: edge_count * COST_PER_ROW_EDGE_SCAN,
                }
            }
        }
    }
}

// ── MATCH SET ────────────────────────────────

fn estimate_match_set(query: &MatchSetQuery, storage: &GraphStorage) -> CostEstimate {
    let node_count = storage.node_count().unwrap_or(0) as u64;
    // Same scan cost as MATCH, plus write cost per matched row
    let base = estimate_match_conditions(&query.pattern, &query.conditions, node_count, storage);
    CostEstimate {
        estimated_cost_us: base.estimated_cost_us + base.estimated_rows * 10, // +10µs per write
        ..base
    }
}

// ── MATCH DELETE ──────────────────────────────

fn estimate_match_delete(query: &MatchDeleteQuery, storage: &GraphStorage) -> CostEstimate {
    let node_count = storage.node_count().unwrap_or(0) as u64;
    let base = estimate_match_conditions(&query.pattern, &query.conditions, node_count, storage);
    CostEstimate {
        estimated_cost_us: base.estimated_cost_us + base.estimated_rows * 15, // +15µs per delete
        ..base
    }
}

// ── DIFFUSE ──────────────────────────────────

fn estimate_diffuse(query: &DiffuseQuery, storage: &GraphStorage) -> CostEstimate {
    let node_count = storage.node_count().unwrap_or(0) as u64;
    let max_hops = query.max_hops as u64;
    let scales = query.t_values.len() as u64;
    let estimated_rows = (max_hops * scales).min(node_count);
    CostEstimate {
        estimated_rows,
        scan_type: ScanType::DiffusionWalk { max_hops: query.max_hops },
        index_used: Some("CF_ADJ".into()),
        estimated_cost_us: estimated_rows * COST_PER_HOP_BFS,
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Shared estimator for MATCH/SET/DELETE that inspects pattern + conditions.
fn estimate_match_conditions(
    pattern: &Pattern,
    conditions: &[Condition],
    node_count: u64,
    storage: &GraphStorage,
) -> CostEstimate {
    let edge_count = storage.edge_count().unwrap_or(0) as u64;

    match pattern {
        Pattern::Node(_) => {
            if has_energy_condition(conditions) {
                let estimated_rows = (node_count / 4).max(1);
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::EnergyIndexScan,
                    index_used: Some("CF_ENERGY_IDX".into()),
                    estimated_cost_us: estimated_rows * COST_PER_ROW_INDEX_SCAN,
                }
            } else if let Some(field) = find_meta_index_condition(conditions) {
                let estimated_rows = (node_count / 10).max(1);
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::MetaIndexScan(field.clone()),
                    index_used: Some(format!("CF_META_IDX({})", field)),
                    estimated_cost_us: estimated_rows * COST_PER_ROW_INDEX_SCAN,
                }
            } else {
                CostEstimate {
                    estimated_rows: node_count,
                    scan_type: ScanType::FullScan,
                    index_used: None,
                    estimated_cost_us: node_count * COST_PER_ROW_FULL_SCAN,
                }
            }
        }
        Pattern::Path(pp) => {
            if let Some(hr) = &pp.hop_range {
                let avg_degree = if node_count > 0 { (edge_count / node_count).max(1) } else { 1 };
                let mut estimated_rows = 1u64;
                for _ in 0..hr.max {
                    estimated_rows = (estimated_rows * avg_degree).min(node_count);
                }
                CostEstimate {
                    estimated_rows,
                    scan_type: ScanType::BoundedBFS { min_hops: hr.min, max_hops: hr.max },
                    index_used: Some("CF_ADJ".into()),
                    estimated_cost_us: estimated_rows * COST_PER_HOP_BFS,
                }
            } else {
                CostEstimate {
                    estimated_rows: edge_count,
                    scan_type: ScanType::EdgeScan,
                    index_used: Some("CF_ADJ".into()),
                    estimated_cost_us: edge_count * COST_PER_ROW_EDGE_SCAN,
                }
            }
        }
    }
}

/// Check if any condition references `alias.energy` with a comparison op.
fn has_energy_condition(conditions: &[Condition]) -> bool {
    conditions.iter().any(|c| condition_references_field(c, "energy"))
}

/// Find the first metadata field referenced in conditions that is *not*
/// a well-known computed field (energy, depth, embedding).
fn find_meta_index_condition(conditions: &[Condition]) -> Option<String> {
    const KNOWN_FIELDS: &[&str] = &["energy", "depth", "embedding", "latent", "id", "node_type"];
    for cond in conditions {
        if let Some(field) = extract_property_field(cond) {
            if !KNOWN_FIELDS.contains(&field.as_str()) {
                return Some(field);
            }
        }
    }
    None
}

/// Recursively check if a condition references a specific field name.
fn condition_references_field(cond: &Condition, field_name: &str) -> bool {
    match cond {
        Condition::Compare { left, right, .. } => {
            expr_has_field(left, field_name) || expr_has_field(right, field_name)
        }
        Condition::And(a, b) | Condition::Or(a, b) => {
            condition_references_field(a, field_name) || condition_references_field(b, field_name)
        }
        Condition::Not(inner) => condition_references_field(inner, field_name),
        Condition::In { expr, .. } => expr_has_field(expr, field_name),
        Condition::Between { expr, .. } => expr_has_field(expr, field_name),
        Condition::StringOp { left, right, .. } => {
            expr_has_field(left, field_name) || expr_has_field(right, field_name)
        }
    }
}

/// Extract the first property field name from a condition (for meta index detection).
fn extract_property_field(cond: &Condition) -> Option<String> {
    match cond {
        Condition::Compare { left, right, .. } => {
            extract_field_from_expr(left).or_else(|| extract_field_from_expr(right))
        }
        Condition::And(a, b) | Condition::Or(a, b) => {
            extract_property_field(a).or_else(|| extract_property_field(b))
        }
        Condition::Not(inner) => extract_property_field(inner),
        Condition::In { expr, .. } => extract_field_from_expr(expr),
        Condition::Between { expr, .. } => extract_field_from_expr(expr),
        Condition::StringOp { left, right, .. } => {
            extract_field_from_expr(left).or_else(|| extract_field_from_expr(right))
        }
    }
}

fn expr_has_field(expr: &Expr, field_name: &str) -> bool {
    matches!(expr, Expr::Property { field, .. } if field == field_name)
}

fn extract_field_from_expr(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Property { field, .. } => Some(field.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_type_display() {
        assert_eq!(ScanType::FullScan.to_string(), "FullScan");
        assert_eq!(ScanType::EnergyIndexScan.to_string(), "EnergyIndexScan");
        assert_eq!(
            ScanType::MetaIndexScan("category".into()).to_string(),
            "MetaIndexScan(category)"
        );
        assert_eq!(
            ScanType::BoundedBFS { min_hops: 1, max_hops: 3 }.to_string(),
            "BoundedBFS(1..3)"
        );
    }

    #[test]
    fn cost_estimate_display() {
        let est = CostEstimate {
            estimated_rows: 1000,
            scan_type: ScanType::FullScan,
            index_used: None,
            estimated_cost_us: 5000,
        };
        assert_eq!(est.to_string(), "rows=~1000, scan=FullScan, index=none, cost=~5000µs");
    }

    #[test]
    fn energy_condition_detection() {
        let cond = Condition::Compare {
            left: Expr::Property { alias: "n".into(), field: "energy".into() },
            op: CompOp::Gt,
            right: Expr::Float(0.5),
        };
        assert!(has_energy_condition(&[cond]));
    }

    #[test]
    fn meta_index_condition_detection() {
        let cond = Condition::Compare {
            left: Expr::Property { alias: "n".into(), field: "category".into() },
            op: CompOp::Eq,
            right: Expr::Str("test".into()),
        };
        assert_eq!(find_meta_index_condition(&[cond]), Some("category".into()));
    }

    #[test]
    fn known_fields_not_detected_as_meta() {
        let cond = Condition::Compare {
            left: Expr::Property { alias: "n".into(), field: "energy".into() },
            op: CompOp::Gt,
            right: Expr::Float(0.5),
        };
        assert_eq!(find_meta_index_condition(&[cond]), None);
    }
}
