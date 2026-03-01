//! AUC computation via Wilcoxon-Mann-Whitney U statistic.
//!
//! O(N log N) — sorts once, then linear scan over ranks.

/// Compute AUC from separate positive and negative score arrays.
///
/// Scores should be such that **higher = more likely to be a real edge**.
/// (We use −distance, so closer nodes get higher scores.)
///
/// Returns 0.5 (random guess) if either array is empty.
pub fn calculate_auc(pos_scores: &[f64], neg_scores: &[f64]) -> f64 {
    if pos_scores.is_empty() || neg_scores.is_empty() {
        return 0.5;
    }

    // Label: 1 = positive, 0 = negative
    let mut all: Vec<(f64, u8)> = Vec::with_capacity(pos_scores.len() + neg_scores.len());
    for &s in pos_scores {
        all.push((s, 1));
    }
    for &s in neg_scores {
        all.push((s, 0));
    }

    // Sort ascending by score
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Handle ties: assign average rank to tied groups
    let n = all.len();
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find the end of the tie group
        while j < n && (all[j].0 - all[i].0).abs() < 1e-15 {
            j += 1;
        }
        // Average rank for this tie group (1-indexed)
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    // Sum of ranks for positive samples
    let rank_sum_pos: f64 = all
        .iter()
        .enumerate()
        .filter(|(_, (_, label))| *label == 1)
        .map(|(idx, _)| ranks[idx])
        .sum();

    let p = pos_scores.len() as f64;
    let ng = neg_scores.len() as f64;

    // Mann-Whitney U statistic
    let u = rank_sum_pos - p * (p + 1.0) / 2.0;
    (u / (p * ng)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_separation() {
        // All positives scored higher than all negatives
        let pos = vec![0.8, 0.9, 1.0];
        let neg = vec![0.1, 0.2, 0.3];
        let auc = calculate_auc(&pos, &neg);
        assert!((auc - 1.0).abs() < 1e-10, "expected 1.0, got {auc}");
    }

    #[test]
    fn no_separation() {
        // All scores identical → AUC = 0.5
        let pos = vec![0.5, 0.5, 0.5];
        let neg = vec![0.5, 0.5, 0.5];
        let auc = calculate_auc(&pos, &neg);
        assert!((auc - 0.5).abs() < 1e-10, "expected 0.5, got {auc}");
    }

    #[test]
    fn inverse_separation() {
        // All positives scored LOWER than negatives → AUC ≈ 0.0
        let pos = vec![0.1, 0.2];
        let neg = vec![0.8, 0.9];
        let auc = calculate_auc(&pos, &neg);
        assert!(auc < 0.01, "expected ~0.0, got {auc}");
    }

    #[test]
    fn empty_arrays() {
        assert!((calculate_auc(&[], &[1.0]) - 0.5).abs() < 1e-10);
        assert!((calculate_auc(&[1.0], &[]) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn known_value() {
        // pos = [3, 5], neg = [1, 2, 4]
        // Sorted: 1(N), 2(N), 3(P), 4(N), 5(P) → ranks 1,2,3,4,5
        // rank_sum_pos = 3 + 5 = 8
        // U = 8 - 2*3/2 = 8 - 3 = 5
        // AUC = 5 / (2*3) = 0.833...
        let pos = vec![3.0, 5.0];
        let neg = vec![1.0, 2.0, 4.0];
        let auc = calculate_auc(&pos, &neg);
        assert!(
            (auc - 5.0 / 6.0).abs() < 1e-10,
            "expected {}, got {auc}",
            5.0 / 6.0
        );
    }
}
