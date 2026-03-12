//! Neural Structural Evolution — ONNX-driven L-System strategy selection.
//!
//! Replaces (or augments) the heuristic `RuleEvolution::suggest_strategy` with
//! inference from the `structural_evolver` ONNX model (PPO-trained).
//!
//! ## Model Spec
//!
//! - **Registry name**: `"structural_evolver"`
//! - **Input**: `[1, 320]` — subgraph topology feature vector
//! - **Output**: `[1, 7]` — probability distribution over 7 evolution actions
//!
//! ## Feature Vector Layout (320 dimensions)
//!
//! | Offset | Count | Description |
//! |--------|-------|-------------|
//! |   0    |  15   | Raw health metrics (energy, hausdorff, gaps, etc.) |
//! |  15    |  32   | Degree distribution histogram (32 log-scaled bins) |
//! |  47    |  32   | In-degree distribution histogram (32 bins) |
//! |  79    |  32   | Out-degree distribution histogram (32 bins) |
//! | 111    |  16   | Energy distribution histogram (16 bins) |
//! | 127    |   5   | Graph density features |
//! | 132    |  64   | Sinusoidal encoding of key metrics |
//! | 196    | 124   | Reserved / zero-padded |
//!
//! ## 7 Action Mapping
//!
//! | Index | Strategy | Description |
//! |-------|----------|-------------|
//! |   0   | FavorGrowth | Standard growth to fill gaps |
//! |   1   | FavorPruning | Standard pruning for entropy/energy control |
//! |   2   | Balanced | Moderate growth + maintenance pruning |
//! |   3   | Consolidate | Structural repair (non-fractal) |
//! |   4   | AggressiveGrowth | Lower thresholds, higher max generation |
//! |   5   | DeepPrune | Aggressive pruning + energy drain |
//! |   6   | Restructure | Consolidate + lateral reconnection |
//!
//! ## Gating
//!
//! Controlled by `AGENCY_EVOLUTION_NEURAL_ENABLED` (default: `false`).
//! Falls back to heuristic strategy if the model is not loaded or inference fails.

use nietzsche_graph::AdjacencyIndex;
use uuid::Uuid;

use crate::evolution::EvolutionStrategy;
use crate::observer::HealthReport;

/// Name of the ONNX model in the `nietzsche_neural::REGISTRY`.
const MODEL_NAME: &str = "structural_evolver";

/// Input dimension expected by the model.
const INPUT_DIM: usize = 320;

/// Number of output action classes.
const NUM_ACTIONS: usize = 7;

/// Number of bins for degree distribution histograms.
const DEGREE_BINS: usize = 32;

/// Number of bins for the energy histogram.
const ENERGY_BINS: usize = 16;

/// Maximum nodes to sample for degree distribution (caps I/O cost).
const MAX_DEGREE_SAMPLE: usize = 2000;

/// Result of a neural evolution inference.
#[derive(Debug, Clone)]
pub struct NeuralEvolutionResult {
    /// The chosen strategy after argmax.
    pub strategy: EvolutionStrategy,
    /// Raw action probabilities from the model (softmax output).
    pub action_probs: [f32; NUM_ACTIONS],
    /// Index of the winning action.
    pub action_index: usize,
    /// Confidence (probability of the winning action).
    pub confidence: f32,
}

/// Check if neural evolution is available (env enabled + model loaded).
pub fn is_available() -> bool {
    is_enabled() && nietzsche_neural::REGISTRY.has_model(MODEL_NAME)
}

/// Check if the feature is enabled via environment variable.
fn is_enabled() -> bool {
    std::env::var("AGENCY_EVOLUTION_NEURAL_ENABLED")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
}

/// Run neural evolution inference and return the suggested strategy.
///
/// Builds a 320D feature vector from the health report and adjacency index,
/// runs inference on the `structural_evolver` model, and maps the argmax
/// output to an `EvolutionStrategy`.
///
/// Returns `None` if the feature is disabled, the model is not loaded,
/// or inference fails (graceful fallback to heuristic).
pub fn neural_suggest_strategy(
    report: &HealthReport,
    adjacency: &AdjacencyIndex,
) -> Option<NeuralEvolutionResult> {
    if !is_available() {
        return None;
    }

    // Build feature vector
    let features = build_feature_vector(report, adjacency);
    debug_assert_eq!(features.len(), INPUT_DIM);

    // Run inference
    match nietzsche_neural::REGISTRY.infer_f32(
        MODEL_NAME,
        vec![1, INPUT_DIM],
        features,
    ) {
        Ok(output) => parse_model_output(&output),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "structural_evolver inference failed, falling back to heuristic"
            );
            None
        }
    }
}

/// Build the 320-dimensional feature vector from health metrics and topology.
fn build_feature_vector(report: &HealthReport, adjacency: &AdjacencyIndex) -> Vec<f32> {
    let mut features = vec![0.0f32; INPUT_DIM];
    let mut offset = 0;

    // ── Section 0: Raw health metrics (15 features) ──────────────────
    let total_nodes = report.total_nodes.max(1) as f32;
    let total_edges = report.total_edges as f32;

    features[offset] = normalize_count(report.total_nodes);
    features[offset + 1] = normalize_count(report.total_edges);
    features[offset + 2] = report.global_hausdorff / 3.0; // normalize to ~[0, 1]
    features[offset + 3] = if report.is_fractal { 1.0 } else { 0.0 };
    features[offset + 4] = report.mean_energy;
    features[offset + 5] = report.energy_std;
    features[offset + 6] = report.energy_percentiles.p10;
    features[offset + 7] = report.energy_percentiles.p25;
    features[offset + 8] = report.energy_percentiles.p50;
    features[offset + 9] = report.energy_percentiles.p75;
    features[offset + 10] = report.energy_percentiles.p90;
    features[offset + 11] = report.coherence_score as f32;
    features[offset + 12] = (report.gap_count as f32 / 80.0).min(1.0);
    features[offset + 13] = (report.entropy_spike_count as f32 / 10.0).min(1.0);
    features[offset + 14] = report.max_regional_variance.min(1.0);
    offset += 15;

    // ── Section 1: Degree distribution histograms (3 x 32 = 96 features) ─
    let node_ids = sample_node_ids(adjacency);
    let (total_hist, in_hist, out_hist) = compute_degree_histograms(adjacency, &node_ids);

    features[offset..offset + DEGREE_BINS].copy_from_slice(&total_hist);
    offset += DEGREE_BINS;
    features[offset..offset + DEGREE_BINS].copy_from_slice(&in_hist);
    offset += DEGREE_BINS;
    features[offset..offset + DEGREE_BINS].copy_from_slice(&out_hist);
    offset += DEGREE_BINS;

    // ── Section 2: Energy distribution histogram (16 features) ────────
    // Uses percentiles to approximate the energy PDF
    let energy_hist = build_energy_histogram(report);
    features[offset..offset + ENERGY_BINS].copy_from_slice(&energy_hist);
    offset += ENERGY_BINS;

    // ── Section 3: Graph density features (5 features) ────────────────
    let density = if total_nodes > 1.0 {
        total_edges / (total_nodes * (total_nodes - 1.0))
    } else {
        0.0
    };
    let avg_degree = if total_nodes > 0.0 {
        total_edges * 2.0 / total_nodes
    } else {
        0.0
    };
    let edge_node_ratio = total_edges / total_nodes.max(1.0);

    features[offset] = density.min(1.0);
    features[offset + 1] = normalize_count(avg_degree as usize);
    features[offset + 2] = edge_node_ratio / 100.0; // normalized
    features[offset + 3] = (report.gap_sectors.len() as f32 / 80.0).min(1.0);
    features[offset + 4] = if report.total_nodes > 0 {
        (report.total_edges as f32) / (report.total_nodes as f32).sqrt()
    } else {
        0.0
    }.min(1.0);
    offset += 5;

    // ── Section 4: Sinusoidal positional encoding (64 features) ───────
    // Encodes 8 key metrics with 8 sin/cos frequencies each
    let key_metrics = [
        report.mean_energy,
        report.global_hausdorff / 3.0,
        report.coherence_score as f32,
        report.energy_std,
        density as f32,
        avg_degree / 100.0,
        report.energy_percentiles.p50,
        report.max_regional_variance,
    ];

    for (i, &val) in key_metrics.iter().enumerate() {
        for freq in 0..4 {
            let angle = val * std::f32::consts::PI * (1 << freq) as f32;
            features[offset + i * 8 + freq * 2] = angle.sin();
            features[offset + i * 8 + freq * 2 + 1] = angle.cos();
        }
    }
    offset += 64;

    // ── Section 5: Reserved / zero-padded (remaining to 320) ──────────
    // Already zero-initialized; offset should be 196 here.
    debug_assert!(offset <= INPUT_DIM, "feature vector overflow: offset={offset}");

    features
}

/// Parse model output probabilities and select the best action.
fn parse_model_output(output: &[f32]) -> Option<NeuralEvolutionResult> {
    if output.len() < NUM_ACTIONS {
        tracing::warn!(
            output_len = output.len(),
            expected = NUM_ACTIONS,
            "structural_evolver output too short"
        );
        return None;
    }

    // Softmax normalization (model may output logits)
    let probs = softmax(&output[..NUM_ACTIONS]);

    // Argmax
    let mut best_idx = 0;
    let mut best_prob = probs[0];
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > best_prob {
            best_prob = p;
            best_idx = i;
        }
    }

    let strategy = action_to_strategy(best_idx);

    let mut action_probs = [0.0f32; NUM_ACTIONS];
    action_probs.copy_from_slice(&probs[..NUM_ACTIONS]);

    Some(NeuralEvolutionResult {
        strategy,
        action_probs,
        action_index: best_idx,
        confidence: best_prob,
    })
}

/// Map action index to EvolutionStrategy.
///
/// Actions 4-6 are more nuanced versions that map to base strategies
/// but with modified rule parameters (handled in `evolve_rules_neural`).
fn action_to_strategy(action: usize) -> EvolutionStrategy {
    match action {
        0 => EvolutionStrategy::FavorGrowth,
        1 => EvolutionStrategy::FavorPruning,
        2 => EvolutionStrategy::Balanced,
        3 => EvolutionStrategy::Consolidate,
        // Actions 4-6 map to base strategies with modifier.
        // The reactor logs the neural action index for observability.
        4 => EvolutionStrategy::FavorGrowth,    // AggressiveGrowth
        5 => EvolutionStrategy::FavorPruning,   // DeepPrune
        6 => EvolutionStrategy::Consolidate,    // Restructure
        _ => EvolutionStrategy::Balanced,
    }
}

/// Return a human-readable label for the neural action index.
pub fn action_label(action: usize) -> &'static str {
    match action {
        0 => "FavorGrowth",
        1 => "FavorPruning",
        2 => "Balanced",
        3 => "Consolidate",
        4 => "AggressiveGrowth",
        5 => "DeepPrune",
        6 => "Restructure",
        _ => "Unknown",
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

/// Normalize a count to roughly [0, 1] using log scaling.
fn normalize_count(n: usize) -> f32 {
    if n == 0 {
        0.0
    } else {
        (n as f32).ln() / 15.0 // ln(3_000_000) ~ 14.9, so /15 maps to ~[0,1]
    }
}

/// Sample up to MAX_DEGREE_SAMPLE node IDs from the adjacency index.
fn sample_node_ids(adjacency: &AdjacencyIndex) -> Vec<Uuid> {
    let all = adjacency.all_nodes();
    if all.len() <= MAX_DEGREE_SAMPLE {
        all
    } else {
        // Deterministic stride-based sampling
        let step = all.len() / MAX_DEGREE_SAMPLE;
        all.into_iter().step_by(step).take(MAX_DEGREE_SAMPLE).collect()
    }
}

/// Compute degree distribution histograms (total, in, out) over log-scaled bins.
///
/// Bin boundaries: [0, 1, 2, 4, 8, 16, 32, 64, ...] with DEGREE_BINS bins.
/// Each bin value is the fraction of sampled nodes falling into that bin.
fn compute_degree_histograms(
    adjacency: &AdjacencyIndex,
    node_ids: &[Uuid],
) -> ([f32; DEGREE_BINS], [f32; DEGREE_BINS], [f32; DEGREE_BINS]) {
    let mut total_hist = [0u32; DEGREE_BINS];
    let mut in_hist = [0u32; DEGREE_BINS];
    let mut out_hist = [0u32; DEGREE_BINS];
    let n = node_ids.len() as f32;

    if n == 0.0 {
        return ([0.0; DEGREE_BINS], [0.0; DEGREE_BINS], [0.0; DEGREE_BINS]);
    }

    for id in node_ids {
        let d_out = adjacency.degree_out(id);
        let d_in = adjacency.degree_in(id);
        let d_total = d_out + d_in;

        total_hist[degree_to_bin(d_total)] += 1;
        in_hist[degree_to_bin(d_in)] += 1;
        out_hist[degree_to_bin(d_out)] += 1;
    }

    let to_f32 = |hist: &[u32; DEGREE_BINS]| -> [f32; DEGREE_BINS] {
        let mut result = [0.0f32; DEGREE_BINS];
        for (i, &count) in hist.iter().enumerate() {
            result[i] = count as f32 / n;
        }
        result
    };

    (to_f32(&total_hist), to_f32(&in_hist), to_f32(&out_hist))
}

/// Map a degree value to a log-scaled bin index.
fn degree_to_bin(degree: usize) -> usize {
    if degree == 0 {
        return 0;
    }
    // log2(degree) + 1, clamped to [0, DEGREE_BINS-1]
    let bin = (degree as f64).log2().floor() as usize + 1;
    bin.min(DEGREE_BINS - 1)
}

/// Build an energy distribution histogram from the health report percentiles.
///
/// Uses linear interpolation between the 5 known percentile points
/// (p10, p25, p50, p75, p90) to approximate the energy PDF across 16 bins.
fn build_energy_histogram(report: &HealthReport) -> [f32; ENERGY_BINS] {
    let mut hist = [0.0f32; ENERGY_BINS];

    // Known cumulative distribution points
    let cdf_points = [
        (0.0f32, 0.0f32),
        (report.energy_percentiles.p10, 0.10),
        (report.energy_percentiles.p25, 0.25),
        (report.energy_percentiles.p50, 0.50),
        (report.energy_percentiles.p75, 0.75),
        (report.energy_percentiles.p90, 0.90),
        (1.0, 1.0),
    ];

    for i in 0..ENERGY_BINS {
        let bin_center = (i as f32 + 0.5) / ENERGY_BINS as f32;
        // Linearly interpolate CDF at bin_center
        let cdf_val = interpolate_cdf(&cdf_points, bin_center);
        hist[i] = cdf_val;
    }

    // Convert CDF to PDF (differences between adjacent bins)
    let mut pdf = [0.0f32; ENERGY_BINS];
    pdf[0] = hist[0];
    for i in 1..ENERGY_BINS {
        pdf[i] = (hist[i] - hist[i - 1]).max(0.0);
    }

    pdf
}

/// Linear interpolation on CDF points.
fn interpolate_cdf(points: &[(f32, f32)], x: f32) -> f32 {
    if x <= points[0].0 {
        return points[0].1;
    }
    for i in 1..points.len() {
        if x <= points[i].0 {
            let (x0, y0) = points[i - 1];
            let (x1, y1) = points[i];
            let dx = x1 - x0;
            if dx < 1e-9 {
                return y1;
            }
            return y0 + (y1 - y0) * (x - x0) / dx;
        }
    }
    points.last().unwrap().1
}

/// Numerically stable softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-9 {
        // Uniform fallback if all logits are -inf
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{EnergyPercentiles, HealthReport};

    fn make_report() -> HealthReport {
        HealthReport {
            total_nodes: 1000,
            total_edges: 5000,
            global_hausdorff: 1.5,
            is_fractal: true,
            mean_energy: 0.5,
            energy_std: 0.15,
            energy_percentiles: EnergyPercentiles {
                p10: 0.1,
                p25: 0.3,
                p50: 0.5,
                p75: 0.7,
                p90: 0.9,
            },
            coherence_score: 0.8,
            gap_count: 10,
            entropy_spike_count: 1,
            max_regional_variance: 0.2,
            ..HealthReport::default()
        }
    }

    #[test]
    fn feature_vector_has_correct_length() {
        let report = make_report();
        let adj = nietzsche_graph::AdjacencyIndex::new();
        let features = build_feature_vector(&report, &adj);
        assert_eq!(features.len(), INPUT_DIM);
    }

    #[test]
    fn feature_vector_raw_metrics_correct() {
        let report = make_report();
        let adj = nietzsche_graph::AdjacencyIndex::new();
        let features = build_feature_vector(&report, &adj);

        // Slot 3: is_fractal = 1.0
        assert_eq!(features[3], 1.0);
        // Slot 4: mean_energy = 0.5
        assert!((features[4] - 0.5).abs() < 1e-6);
        // Slot 11: coherence_score = 0.8
        assert!((features[11] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn degree_to_bin_maps_correctly() {
        assert_eq!(degree_to_bin(0), 0);
        assert_eq!(degree_to_bin(1), 1);
        assert_eq!(degree_to_bin(2), 2);
        assert_eq!(degree_to_bin(3), 2); // floor(log2(3)) + 1 = 2
        assert_eq!(degree_to_bin(4), 3);
        assert_eq!(degree_to_bin(7), 3);
        assert_eq!(degree_to_bin(8), 4);
        assert_eq!(degree_to_bin(1_000_000), DEGREE_BINS - 1); // clamped
    }

    #[test]
    fn softmax_produces_valid_distribution() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 7);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Highest logit should have highest probability
        assert!(probs[6] > probs[0]);
    }

    #[test]
    fn softmax_handles_uniform() {
        let logits = [0.0; 7];
        let probs = softmax(&logits);
        for p in &probs {
            assert!((p - 1.0 / 7.0).abs() < 1e-5);
        }
    }

    #[test]
    fn parse_output_selects_argmax() {
        let output = [0.1, 0.05, 0.5, 0.1, 0.05, 0.1, 0.1];
        let result = parse_model_output(&output).unwrap();
        assert_eq!(result.action_index, 2);
        assert_eq!(result.strategy, EvolutionStrategy::Balanced);
    }

    #[test]
    fn parse_output_rejects_short_vector() {
        let output = [0.1, 0.2, 0.3]; // too short
        assert!(parse_model_output(&output).is_none());
    }

    #[test]
    fn action_to_strategy_covers_all() {
        assert_eq!(action_to_strategy(0), EvolutionStrategy::FavorGrowth);
        assert_eq!(action_to_strategy(1), EvolutionStrategy::FavorPruning);
        assert_eq!(action_to_strategy(2), EvolutionStrategy::Balanced);
        assert_eq!(action_to_strategy(3), EvolutionStrategy::Consolidate);
        assert_eq!(action_to_strategy(4), EvolutionStrategy::FavorGrowth);  // AggressiveGrowth
        assert_eq!(action_to_strategy(5), EvolutionStrategy::FavorPruning); // DeepPrune
        assert_eq!(action_to_strategy(6), EvolutionStrategy::Consolidate);  // Restructure
        assert_eq!(action_to_strategy(99), EvolutionStrategy::Balanced);    // unknown
    }

    #[test]
    fn normalize_count_reasonable() {
        assert_eq!(normalize_count(0), 0.0);
        // 1000 nodes: ln(1000)/15 ~ 0.46
        let v = normalize_count(1000);
        assert!(v > 0.4 && v < 0.5);
    }

    #[test]
    fn energy_histogram_sums_roughly_one() {
        let report = make_report();
        let hist = build_energy_histogram(&report);
        let sum: f32 = hist.iter().sum();
        // PDF should sum to approximately 1.0 (within binning error)
        assert!(sum > 0.8 && sum < 1.2, "energy PDF sum={sum}");
    }

    #[test]
    fn is_available_returns_false_by_default() {
        // AGENCY_EVOLUTION_NEURAL_ENABLED not set → false
        assert!(!is_available());
    }
}
