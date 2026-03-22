// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Phase 28 — Phantom Reaper
//!
//! Scans for nodes with empty content (metadata HashMap is empty) that are
//! NOT yet marked as `is_phantom`. These are "ghost" nodes created by the
//! L-System or other autonomous processes that never received real content.
//!
//! They pollute KNN results and waste HNSW index space. This phase emits
//! `Phantomize` intents to cleanly remove them from the vector store while
//! preserving topological connections.

use nietzsche_graph::GraphStorage;
use uuid::Uuid;

use crate::config::AgencyConfig;

/// Configuration for the phantom reaper.
#[derive(Debug, Clone)]
pub struct ReapPhantomConfig {
    /// Maximum nodes to scan per tick.
    pub max_scan: usize,
    /// Minimum age in seconds before a node is eligible for reaping.
    /// Prevents reaping nodes that were just created and haven't been
    /// populated yet.
    pub min_age_secs: i64,
}

/// Report from a phantom reap scan.
#[derive(Debug, Clone)]
pub struct ReapPhantomReport {
    /// Total nodes scanned.
    pub nodes_scanned: usize,
    /// Phantoms found (empty content, not yet marked is_phantom).
    pub phantoms_found: usize,
    /// Node IDs to phantomize.
    pub phantom_ids: Vec<Uuid>,
}

/// Build config from AgencyConfig.
pub fn build_reap_phantom_config(cfg: &AgencyConfig) -> ReapPhantomConfig {
    ReapPhantomConfig {
        max_scan: cfg.reap_phantoms_max_scan,
        min_age_secs: cfg.reap_phantoms_min_age,
    }
}

/// Scan storage for phantom nodes (empty metadata, not yet is_phantom).
pub fn scan_reap_phantoms(
    storage: &GraphStorage,
    config: &ReapPhantomConfig,
) -> ReapPhantomReport {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let mut scanned = 0usize;
    let mut phantom_ids = Vec::new();

    for result in storage.iter_nodes_meta() {
        if scanned >= config.max_scan {
            break;
        }
        let meta = match result {
            Ok(m) => m,
            Err(_) => continue,
        };
        scanned += 1;

        // Skip nodes already marked as phantom
        if meta.is_phantom {
            continue;
        }

        // Check age: don't reap nodes younger than min_age_secs
        let age = now - meta.created_at;
        if age < config.min_age_secs {
            continue;
        }

        // A phantom is a node with empty metadata (no real content)
        if meta.metadata.is_empty() {
            phantom_ids.push(meta.id);
        }
    }

    ReapPhantomReport {
        nodes_scanned: scanned,
        phantoms_found: phantom_ids.len(),
        phantom_ids,
    }
}
