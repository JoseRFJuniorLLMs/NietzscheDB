// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! Redundancy metric.
//!
//! Measures path redundancy in the graph. A healthy knowledge graph should have
//! multiple paths between important nodes (redundancy > 0), but not excessive
//! redundancy that indicates duplicate information.

use nietzsche_graph::AdjacencyIndex;
use std::collections::HashSet;
use uuid::Uuid;

/// Maximum BFS depth for path counting.
const MAX_BFS_DEPTH: usize = 4;

/// Compute path diversity between sampled node pairs.
///
/// Returns [0.0, 1.0] where:
/// - 0.0 = no connectivity (isolated nodes)
/// - 0.5 = single paths (tree-like)
/// - 1.0 = multiple paths (well-connected mesh)
///
/// Samples up to `max_pairs` random pairs from the given nodes.
pub fn path_diversity(
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
    max_pairs: usize,
) -> f32 {
    if node_ids.len() < 2 {
        return 0.0;
    }

    let mut connected_count = 0u32;
    let mut multi_path_count = 0u32;
    let mut total_pairs = 0u32;

    // Sample pairs deterministically (first node × limited others)
    let step = (node_ids.len() / max_pairs.max(1)).max(1);

    for (i, src_id) in node_ids.iter().enumerate() {
        if total_pairs as usize >= max_pairs {
            break;
        }
        for j in ((i + 1)..node_ids.len()).step_by(step.max(1)) {
            if total_pairs as usize >= max_pairs {
                break;
            }
            let dst_id = &node_ids[j];
            total_pairs += 1;

            let paths = count_paths_bfs(adjacency, src_id, dst_id, MAX_BFS_DEPTH);
            if paths > 0 {
                connected_count += 1;
                if paths > 1 {
                    multi_path_count += 1;
                }
            }
        }
    }

    if total_pairs == 0 {
        return 0.0;
    }

    let connectivity = connected_count as f32 / total_pairs as f32;
    let multi_path_ratio = if connected_count > 0 {
        multi_path_count as f32 / connected_count as f32
    } else {
        0.0
    };

    // Blend connectivity and multi-path ratio
    connectivity * 0.6 + multi_path_ratio * 0.4
}

/// Count distinct simple paths between src and dst within max_depth.
///
/// Uses iterative BFS with path tracking. Capped at 5 paths to avoid explosion.
fn count_paths_bfs(
    adjacency: &AdjacencyIndex,
    src: &Uuid,
    dst: &Uuid,
    max_depth: usize,
) -> u32 {
    if src == dst {
        return 1;
    }

    let mut count = 0u32;
    // (current_node, visited_set, depth)
    let mut stack: Vec<(Uuid, HashSet<Uuid>, usize)> = Vec::new();

    let mut initial_visited = HashSet::new();
    initial_visited.insert(*src);
    stack.push((*src, initial_visited, 0));

    while let Some((current, visited, depth)) = stack.pop() {
        if depth >= max_depth {
            continue;
        }
        if count >= 5 {
            break; // cap to avoid combinatorial explosion
        }

        for nbr in adjacency.neighbors_both(&current) {
            if nbr == *dst {
                count += 1;
                continue;
            }
            if !visited.contains(&nbr) {
                let mut new_visited = visited.clone();
                new_visited.insert(nbr);
                stack.push((nbr, new_visited, depth + 1));
            }
        }
    }

    count
}

/// Compute local connectivity: average degree of the given nodes.
///
/// Returns average number of neighbors per node.
pub fn average_degree(
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
) -> f32 {
    if node_ids.is_empty() {
        return 0.0;
    }

    let total_degree: usize = node_ids.iter()
        .map(|nid| adjacency.neighbors_both(nid).len())
        .sum();

    total_degree as f32 / node_ids.len() as f32
}
