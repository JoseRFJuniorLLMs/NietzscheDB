// Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
//! # HyperbolicHealthMonitor — Geometric Health of the Poincaré Ball
//!
//! Periodically samples embeddings and computes structural health metrics
//! to detect pathological geometries before they degrade search quality.
//!
//! ## Metrics
//!
//! | Metric | What it measures |
//! |--------|-----------------|
//! | RDE | Radial Density Entropy — distribution uniformity |
//! | mean_r / std_r | Boundary crowding detection |
//! | angular_var | Angular compression detection |
//! | radial_peak_count | Ring stratification (L-System layers) |
//! | ring_stability | Temporal stability of radial peaks |
//! | generation_correlation | Pearson(lsystem_gen, ‖x‖) — hierarchy check |
//! | centroid_velocity | d_H(C_t, C_{t-1}) — identity drift |
//!
//! ## Diagnoses
//!
//! | Diagnosis | Condition |
//! |-----------|-----------|
//! | Normal | No anomalies |
//! | HealthyStratification | RPC ≥ 2, mean_r < 0.95, RDE stable |
//! | BoundaryCrowding | mean_r > threshold, RDE falling, RPC ≤ 1 |
//! | SemanticAttractor | Bin density > 3× mean near boundary |

use std::collections::VecDeque;

use nietzsche_graph::GraphStorage;
use nietzsche_hyp_ops::poincare_distance;

use crate::centroid_guardian::CentroidGuardian;
use crate::config::AgencyConfig;
use crate::event_bus::{AgencyEvent, AgencyEventBus};

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────

/// Geometric diagnosis of the Poincaré ball state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HyperbolicDiagnosis {
    /// No anomalies detected.
    Normal,
    /// Ring stratification detected (healthy L-System hierarchy).
    HealthyStratification,
    /// Embeddings concentrating near the boundary.
    BoundaryCrowding,
    /// High-density attractor region detected.
    SemanticAttractor,
}

impl std::fmt::Display for HyperbolicDiagnosis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::HealthyStratification => write!(f, "HealthyStratification"),
            Self::BoundaryCrowding => write!(f, "BoundaryCrowding"),
            Self::SemanticAttractor => write!(f, "SemanticAttractor"),
        }
    }
}

/// Snapshot of geometric health at a point in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HyperbolicHealth {
    /// Radial Density Entropy: H = -Σ p_i ln(p_i).
    pub rde: f64,
    /// Mean Euclidean norm of embeddings (proxy for radial position).
    pub mean_r: f64,
    /// Standard deviation of norms.
    pub std_r: f64,
    /// Angular variance between sampled embedding pairs.
    pub angular_var: f64,
    /// d_H(C_t, C_{t-1}) — centroid drift since last health check.
    pub centroid_velocity: f64,
    /// Number of local maxima in the radial histogram.
    pub radial_peak_count: usize,
    /// Temporal stability of peak positions (1.0 = perfectly stable).
    pub ring_stability: f64,
    /// Pearson correlation between lsystem_generation and ‖x‖.
    pub generation_correlation: f64,
    /// Diagnosis classification.
    pub diagnosis: HyperbolicDiagnosis,
    /// Number of embeddings sampled.
    pub node_count: usize,
    /// Timestamp (Unix ms).
    pub timestamp_ms: u64,
}

// ─────────────────────────────────────────────
// Monitor
// ─────────────────────────────────────────────

/// Periodic health monitor for the hyperbolic embedding space.
///
/// Runs every `hyp_health_interval` ticks, samples embeddings,
/// computes geometric metrics, and emits events on anomalies.
pub struct HyperbolicHealthMonitor {
    /// Rolling history window for trend detection.
    history: VecDeque<HyperbolicHealth>,
    /// Tick counter for frequency gating.
    tick_counter: u64,
    /// Previous radial peak positions (for ring_stability).
    previous_peaks: Vec<f64>,
    /// Previous centroid snapshot (for centroid_velocity).
    previous_centroid: Option<Vec<f64>>,
}

impl HyperbolicHealthMonitor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            tick_counter: 0,
            previous_peaks: Vec::new(),
            previous_centroid: None,
        }
    }

    /// Latest health snapshot (if any).
    pub fn latest(&self) -> Option<&HyperbolicHealth> {
        self.history.back()
    }

    /// Full history window.
    pub fn history(&self) -> &VecDeque<HyperbolicHealth> {
        &self.history
    }

    /// Run one health check cycle. Returns `None` if not yet time.
    pub fn tick(
        &mut self,
        storage: &GraphStorage,
        guardian: &CentroidGuardian,
        bus: &AgencyEventBus,
        config: &AgencyConfig,
    ) -> Option<HyperbolicHealth> {
        self.tick_counter += 1;
        if config.hyp_health_interval == 0
            || self.tick_counter % config.hyp_health_interval != 0
        {
            return None;
        }

        let dim = guardian.dim();
        if dim == 0 {
            return None;
        }

        // 1. Sample embeddings + metadata
        let samples = self.collect_samples(storage, dim, config.hyp_health_max_sample);
        if samples.is_empty() {
            return None;
        }

        let bins = config.hyp_health_bins.max(2);

        // 2. Compute radial statistics
        let norms: Vec<f64> = samples.iter().map(|s| s.norm).collect();
        let (mean_r, std_r) = mean_std(&norms);

        // 3. Radial histogram + RDE
        let histogram = radial_histogram(&norms, bins);
        let rde = shannon_entropy(&histogram, samples.len());

        // 4. Peak detection
        let smoothed = smooth_histogram(&histogram);
        let peaks = detect_peaks(&smoothed, samples.len());
        let radial_peak_count = peaks.len();

        // 5. Ring stability
        let ring_stability = self.compute_ring_stability(&peaks, bins);
        self.previous_peaks = peaks;

        // 6. Angular variance (sampled pairs)
        let angular_var = angular_variance(&samples, 500);

        // 7. Generation correlation
        let generation_correlation = generation_norm_correlation(&samples);

        // 8. Centroid velocity
        let centroid_velocity = match &self.previous_centroid {
            Some(prev) => poincare_distance(guardian.current_centroid(), prev),
            None => 0.0,
        };
        self.previous_centroid = Some(guardian.current_centroid().to_vec());

        // 9. Diagnose
        let diagnosis = self.diagnose(
            rde,
            mean_r,
            radial_peak_count,
            &histogram,
            samples.len(),
            config,
        );

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let health = HyperbolicHealth {
            rde,
            mean_r,
            std_r,
            angular_var,
            centroid_velocity,
            radial_peak_count,
            ring_stability,
            generation_correlation,
            diagnosis,
            node_count: samples.len(),
            timestamp_ms: now_ms,
        };

        // 10. Emit event if anomaly detected
        if diagnosis != HyperbolicDiagnosis::Normal {
            bus.publish(AgencyEvent::HyperbolicHealthAlert {
                diagnosis: diagnosis.to_string(),
                mean_r,
                rde,
                radial_peak_count,
                generation_correlation,
            });

            tracing::warn!(
                diagnosis = %diagnosis,
                mean_r,
                rde,
                radial_peak_count,
                generation_correlation,
                centroid_velocity,
                node_count = samples.len(),
                "hyperbolic health alert"
            );
        } else {
            tracing::info!(
                mean_r,
                rde,
                radial_peak_count,
                generation_correlation,
                node_count = samples.len(),
                "hyperbolic health: normal"
            );
        }

        // 11. Persist to CF_META
        let key = format!("hyp_health:{}", self.tick_counter);
        if let Ok(json) = serde_json::to_vec(&health) {
            let _ = storage.put_meta(&key, &json);
        }

        // 12. Update history
        let max_history = config.hyp_health_history_len.max(1);
        self.history.push_back(health.clone());
        while self.history.len() > max_history {
            self.history.pop_front();
        }

        Some(health)
    }

    /// Classify the geometric state.
    fn diagnose(
        &self,
        rde: f64,
        mean_r: f64,
        radial_peak_count: usize,
        histogram: &[usize],
        total: usize,
        config: &AgencyConfig,
    ) -> HyperbolicDiagnosis {
        // Check boundary crowding first (most urgent)
        if mean_r > config.hyp_health_crowding_r && radial_peak_count <= 1 {
            // Verify RDE is falling (if we have history)
            let rde_falling = self.history.len() >= 3
                && self.history.iter().rev().take(3).all(|h| h.rde >= rde);
            if rde_falling || mean_r > 0.98 {
                return HyperbolicDiagnosis::BoundaryCrowding;
            }
        }

        // Check for semantic attractor (high-density bin near boundary)
        if total > 10 {
            let mean_count = total as f64 / histogram.len() as f64;
            let boundary_start = histogram.len() * 4 / 5; // last 20% of bins
            for &count in &histogram[boundary_start..] {
                if count as f64 > mean_count * 3.0 {
                    return HyperbolicDiagnosis::SemanticAttractor;
                }
            }
        }

        // Check for healthy ring stratification
        if radial_peak_count >= 2 && mean_r < 0.95 {
            return HyperbolicDiagnosis::HealthyStratification;
        }

        HyperbolicDiagnosis::Normal
    }

    /// Compute ring stability: how stable are peak positions over time.
    fn compute_ring_stability(&self, current_peaks: &[f64], bins: usize) -> f64 {
        if self.previous_peaks.is_empty() || current_peaks.is_empty() {
            return 1.0; // no data = assume stable
        }

        // For each current peak, find closest previous peak
        let mut total_drift = 0.0;
        for &cp in current_peaks {
            let min_dist = self.previous_peaks.iter()
                .map(|&pp| (cp - pp).abs())
                .fold(f64::MAX, f64::min);
            total_drift += min_dist;
        }

        let max_possible = bins as f64; // normalize
        let mean_drift = total_drift / current_peaks.len() as f64;
        (1.0 - mean_drift / max_possible).max(0.0)
    }

    /// Collect embedding samples with metadata.
    fn collect_samples(
        &self,
        storage: &GraphStorage,
        dim: usize,
        max_sample: usize,
    ) -> Vec<EmbeddingSample> {
        let mut samples = Vec::with_capacity(max_sample);

        for result in storage.iter_nodes_meta() {
            let meta = match result {
                Ok(m) => m,
                Err(_) => continue,
            };

            if meta.is_phantom || meta.energy <= 0.0 {
                continue;
            }

            let emb = match storage.get_embedding(&meta.id) {
                Ok(Some(e)) if e.dim == dim => e,
                _ => continue,
            };

            let coords_f64: Vec<f64> = emb.coords.iter().map(|&x| x as f64).collect();
            let norm: f64 = coords_f64.iter().map(|x| x * x).sum::<f64>().sqrt();

            // Skip degenerate embeddings
            if norm >= 1.0 || norm < 1e-12 {
                continue;
            }

            samples.push(EmbeddingSample {
                _id: meta.id,
                coords: coords_f64,
                norm,
                generation: meta.lsystem_generation,
            });

            if samples.len() >= max_sample {
                break;
            }
        }

        samples
    }
}

// ─────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────

struct EmbeddingSample {
    _id: uuid::Uuid,
    coords: Vec<f64>,
    norm: f64,
    generation: u32,
}

// ─────────────────────────────────────────────
// Statistics helpers
// ─────────────────────────────────────────────

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Build radial histogram: bin_i = count of norms in [i/bins, (i+1)/bins).
fn radial_histogram(norms: &[f64], bins: usize) -> Vec<usize> {
    let mut hist = vec![0usize; bins];
    for &r in norms {
        let idx = ((r * bins as f64).floor() as usize).min(bins - 1);
        hist[idx] += 1;
    }
    hist
}

/// Shannon entropy of histogram: H = -Σ p_i ln(p_i).
fn shannon_entropy(histogram: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    for &count in histogram {
        if count > 0 {
            let p = count as f64 / n;
            h -= p * p.ln();
        }
    }
    h
}

/// 3-point Gaussian smooth: [0.25, 0.5, 0.25].
fn smooth_histogram(histogram: &[usize]) -> Vec<f64> {
    let n = histogram.len();
    if n < 3 {
        return histogram.iter().map(|&x| x as f64).collect();
    }
    let mut out = vec![0.0; n];
    out[0] = histogram[0] as f64 * 0.75 + histogram[1] as f64 * 0.25;
    out[n - 1] = histogram[n - 2] as f64 * 0.25 + histogram[n - 1] as f64 * 0.75;
    for i in 1..n - 1 {
        out[i] = histogram[i - 1] as f64 * 0.25
            + histogram[i] as f64 * 0.5
            + histogram[i + 1] as f64 * 0.25;
    }
    out
}

/// Detect local maxima in smoothed histogram.
/// Returns peak positions as fractional bin indices (0..bins).
fn detect_peaks(smoothed: &[f64], total: usize) -> Vec<f64> {
    if smoothed.len() < 3 || total < 10 {
        return Vec::new();
    }
    let mean_val: f64 = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
    let threshold = mean_val * 0.5; // peak must be > 50% of mean

    let mut peaks = Vec::new();
    for i in 1..smoothed.len() - 1 {
        if smoothed[i] > smoothed[i - 1]
            && smoothed[i] > smoothed[i + 1]
            && smoothed[i] > threshold
        {
            peaks.push(i as f64);
        }
    }
    peaks
}

/// Angular variance from sampled pairs.
/// Samples up to `max_pairs` random pairs and computes Var(angle).
fn angular_variance(samples: &[EmbeddingSample], max_pairs: usize) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mut angles = Vec::with_capacity(max_pairs);
    let step = (samples.len() * samples.len() / max_pairs).max(1);
    let mut pair_idx = 0usize;

    'outer: for i in 0..samples.len() {
        for j in (i + 1)..samples.len() {
            pair_idx += 1;
            if pair_idx % step != 0 {
                continue;
            }

            let dot: f64 = samples[i].coords.iter()
                .zip(samples[j].coords.iter())
                .map(|(a, b)| a * b)
                .sum();
            let denom = samples[i].norm * samples[j].norm;
            if denom < 1e-12 {
                continue;
            }
            let cos_angle = (dot / denom).clamp(-1.0, 1.0);
            angles.push(cos_angle.acos());

            if angles.len() >= max_pairs {
                break 'outer;
            }
        }
    }

    if angles.is_empty() {
        return 0.0;
    }

    let (_, std) = mean_std(&angles);
    std * std // variance
}

/// Pearson correlation between lsystem_generation and ‖x‖.
///
/// Positive correlation = hierarchy intact (deeper generations → further from center).
/// Near zero or negative = hierarchy broken.
fn generation_norm_correlation(samples: &[EmbeddingSample]) -> f64 {
    if samples.len() < 5 {
        return 0.0;
    }

    let gens: Vec<f64> = samples.iter().map(|s| s.generation as f64).collect();
    let norms: Vec<f64> = samples.iter().map(|s| s.norm).collect();

    let (mean_g, _) = mean_std(&gens);
    let (mean_r, _) = mean_std(&norms);

    let n = samples.len() as f64;
    let mut cov = 0.0;
    let mut var_g = 0.0;
    let mut var_r = 0.0;

    for i in 0..samples.len() {
        let dg = gens[i] - mean_g;
        let dr = norms[i] - mean_r;
        cov += dg * dr;
        var_g += dg * dg;
        var_r += dr * dr;
    }

    let denom = (var_g * var_r).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    cov / denom // Pearson r ∈ [-1, 1]
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_basic() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (m, s) = mean_std(&vals);
        assert!((m - 5.0).abs() < 1e-10);
        assert!(s > 0.0);
    }

    #[test]
    fn empty_mean_std() {
        let (m, s) = mean_std(&[]);
        assert_eq!(m, 0.0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn histogram_correct_bins() {
        let norms = vec![0.1, 0.15, 0.5, 0.51, 0.9, 0.95];
        let hist = radial_histogram(&norms, 10);
        assert_eq!(hist[1], 2); // 0.1-0.2
        assert_eq!(hist[5], 2); // 0.5-0.6
        assert_eq!(hist[9], 2); // 0.9-1.0
    }

    #[test]
    fn entropy_uniform_is_maximal() {
        // Uniform distribution has maximum entropy
        let uniform = vec![10usize; 10];
        let h_uniform = shannon_entropy(&uniform, 100);

        // Concentrated distribution has low entropy
        let concentrated = vec![100, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let h_concentrated = shannon_entropy(&concentrated, 100);

        assert!(h_uniform > h_concentrated);
    }

    #[test]
    fn peak_detection_finds_two_peaks() {
        // Two clear peaks
        let hist = vec![1, 3, 8, 3, 1, 2, 7, 2, 1, 0];
        let smoothed = smooth_histogram(&hist);
        let peaks = detect_peaks(&smoothed, 28);
        assert!(peaks.len() >= 2, "expected 2 peaks, got {:?}", peaks);
    }

    #[test]
    fn peak_detection_uniform_no_peaks() {
        let hist = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
        let smoothed = smooth_histogram(&hist);
        let peaks = detect_peaks(&smoothed, 50);
        assert_eq!(peaks.len(), 0, "uniform should have no peaks");
    }

    #[test]
    fn ring_stability_perfect() {
        let mut monitor = HyperbolicHealthMonitor::new();
        monitor.previous_peaks = vec![5.0, 15.0, 25.0];
        let current = vec![5.0, 15.0, 25.0];
        let stability = monitor.compute_ring_stability(&current, 50);
        assert!((stability - 1.0).abs() < 1e-10, "identical peaks = stability 1.0");
    }

    #[test]
    fn ring_stability_with_drift() {
        let mut monitor = HyperbolicHealthMonitor::new();
        monitor.previous_peaks = vec![5.0, 15.0, 25.0];
        let current = vec![6.0, 16.0, 26.0]; // each drifted by 1
        let stability = monitor.compute_ring_stability(&current, 50);
        assert!(stability > 0.9 && stability < 1.0, "small drift = high stability, got {stability}");
    }

    #[test]
    fn diagnosis_normal() {
        let monitor = HyperbolicHealthMonitor::new();
        let config = AgencyConfig::default();
        let hist = vec![5; 50];
        let d = monitor.diagnose(3.0, 0.5, 0, &hist, 250, &config);
        assert_eq!(d, HyperbolicDiagnosis::Normal);
    }

    #[test]
    fn diagnosis_crowding() {
        let monitor = HyperbolicHealthMonitor::new();
        let mut config = AgencyConfig::default();
        config.hyp_health_crowding_r = 0.96;
        let hist = vec![5; 50];
        // mean_r > threshold, RPC ≤ 1, and extreme mean_r triggers even without history
        let d = monitor.diagnose(2.0, 0.99, 1, &hist, 250, &config);
        assert_eq!(d, HyperbolicDiagnosis::BoundaryCrowding);
    }

    #[test]
    fn diagnosis_stratification() {
        let monitor = HyperbolicHealthMonitor::new();
        let config = AgencyConfig::default();
        let hist = vec![5; 50];
        let d = monitor.diagnose(3.0, 0.7, 3, &hist, 250, &config);
        assert_eq!(d, HyperbolicDiagnosis::HealthyStratification);
    }

    #[test]
    fn diagnosis_attractor() {
        let monitor = HyperbolicHealthMonitor::new();
        let config = AgencyConfig::default();
        let mut hist = vec![5; 50];
        // High density in boundary bins (last 20% = bins 40-49)
        hist[45] = 100; // 100 >> mean(5) * 3 = 15
        let d = monitor.diagnose(3.0, 0.7, 0, &hist, 345, &config);
        assert_eq!(d, HyperbolicDiagnosis::SemanticAttractor);
    }

    #[test]
    fn generation_correlation_positive() {
        // Nodes with higher generation have higher norm
        let samples: Vec<EmbeddingSample> = (0..20).map(|i| {
            let r = 0.1 + (i as f64) * 0.04;
            EmbeddingSample {
                _id: uuid::Uuid::new_v4(),
                coords: vec![r, 0.0, 0.0],
                norm: r,
                generation: i,
            }
        }).collect();

        let corr = generation_norm_correlation(&samples);
        assert!(corr > 0.9, "perfect positive correlation expected, got {corr}");
    }

    #[test]
    fn angular_variance_orthogonal() {
        // All vectors point in same direction → variance should be very low
        let samples: Vec<EmbeddingSample> = (0..10).map(|i| {
            let r = 0.1 + (i as f64) * 0.05;
            EmbeddingSample {
                _id: uuid::Uuid::new_v4(),
                coords: vec![r, 0.0, 0.0],
                norm: r,
                generation: 0,
            }
        }).collect();

        let var = angular_variance(&samples, 100);
        assert!(var < 0.01, "same direction → low angular variance, got {var}");
    }
}
