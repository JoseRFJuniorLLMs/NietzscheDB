//! Vitality Variance Monitor — Cognitive Diversity Detector.
//!
//! From the Nezhmetdinov spec:
//! - Mean Alta + Variância Baixa = Esterilização (Monocultura)
//! - Mean Alta + Variância Saudável = Diversidade Cognitiva
//!
//! This module tracks Var(V) over time and alerts when the system
//! enters monoculture territory.

use serde::{Deserialize, Serialize};

/// Vitality variance snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VitalityVarianceSnapshot {
    pub cycle: u64,
    pub mean_vitality: f32,
    pub variance: f32,
    pub std_dev: f32,
    pub min_vitality: f32,
    pub max_vitality: f32,
    pub sample_size: usize,
    /// Classification of cognitive health.
    pub health_class: CognitiveHealthClass,
}

/// Classification of cognitive diversity based on mean + variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CognitiveHealthClass {
    /// Mean high + Variance healthy = diverse, creative system.
    Diverse,
    /// Mean high + Variance low = monoculture (esterilization).
    Monoculture,
    /// Mean low + Variance high = chaotic, unstable.
    Chaotic,
    /// Mean low + Variance low = dying, exhausted.
    Exhausted,
    /// Not enough data to classify.
    #[default]
    Unknown,
}

impl CognitiveHealthClass {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Diverse => "DIVERSE",
            Self::Monoculture => "MONOCULTURE",
            Self::Chaotic => "CHAOTIC",
            Self::Exhausted => "EXHAUSTED",
            Self::Unknown => "UNKNOWN",
        }
    }
}

/// Vitality Variance tracker.
#[derive(Debug, Clone)]
pub struct VitalityVarianceTracker {
    pub history: Vec<VitalityVarianceSnapshot>,
    /// Threshold: mean above this is "high".
    pub high_mean_threshold: f32,
    /// Threshold: variance below this is "low" (monoculture risk).
    pub low_variance_threshold: f32,
    /// Threshold: variance above this is "chaotic".
    pub high_variance_threshold: f32,
}

impl Default for VitalityVarianceTracker {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            high_mean_threshold: 0.6,
            low_variance_threshold: 0.01,
            high_variance_threshold: 0.15,
        }
    }
}

impl VitalityVarianceTracker {
    pub fn new(high_mean: f32, low_var: f32, high_var: f32) -> Self {
        Self {
            high_mean_threshold: high_mean,
            low_variance_threshold: low_var,
            high_variance_threshold: high_var,
            ..Self::default()
        }
    }

    /// Record vitality scores from a cycle and classify cognitive health.
    pub fn record(&mut self, cycle: u64, vitality_scores: &[f32]) -> VitalityVarianceSnapshot {
        if vitality_scores.is_empty() {
            let snap = VitalityVarianceSnapshot {
                cycle,
                health_class: CognitiveHealthClass::Unknown,
                ..Default::default()
            };
            self.history.push(snap.clone());
            return snap;
        }

        let n = vitality_scores.len() as f32;
        let sum: f32 = vitality_scores.iter().sum();
        let mean = sum / n;
        let variance = vitality_scores.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / n;
        let std_dev = variance.sqrt();
        let min = vitality_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vitality_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let health_class = self.classify(mean, variance);

        let snap = VitalityVarianceSnapshot {
            cycle,
            mean_vitality: mean,
            variance,
            std_dev,
            min_vitality: min,
            max_vitality: max,
            sample_size: vitality_scores.len(),
            health_class,
        };

        self.history.push(snap.clone());
        snap
    }

    /// Classify cognitive health based on mean and variance.
    fn classify(&self, mean: f32, variance: f32) -> CognitiveHealthClass {
        let high_mean = mean > self.high_mean_threshold;
        let low_var = variance < self.low_variance_threshold;
        let high_var = variance > self.high_variance_threshold;

        match (high_mean, low_var, high_var) {
            (true, true, false) => CognitiveHealthClass::Monoculture,
            (true, false, false) => CognitiveHealthClass::Diverse,
            (true, false, true) => CognitiveHealthClass::Diverse, // High mean + high var is OK
            (false, false, true) => CognitiveHealthClass::Chaotic,
            (false, true, false) => CognitiveHealthClass::Exhausted,
            (false, false, false) => CognitiveHealthClass::Exhausted,
            _ => CognitiveHealthClass::Unknown,
        }
    }

    /// Check if the system is in monoculture for the last N cycles.
    pub fn is_monoculture_streak(&self, n: usize) -> bool {
        if self.history.len() < n {
            return false;
        }
        self.history[self.history.len() - n..].iter()
            .all(|s| s.health_class == CognitiveHealthClass::Monoculture)
    }

    /// Latest snapshot.
    pub fn latest(&self) -> Option<&VitalityVarianceSnapshot> {
        self.history.last()
    }

    /// Trend: is variance decreasing over the last N cycles?
    pub fn variance_declining(&self, window: usize) -> bool {
        if self.history.len() < window {
            return false;
        }
        let recent = &self.history[self.history.len() - window..];
        let first_half_mean = recent[..window / 2].iter()
            .map(|s| s.variance).sum::<f32>() / (window / 2).max(1) as f32;
        let second_half_mean = recent[window / 2..].iter()
            .map(|s| s.variance).sum::<f32>() / (window - window / 2).max(1) as f32;
        second_half_mean < first_half_mean * 0.8 // 20% decline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diverse_system() {
        let mut tracker = VitalityVarianceTracker::default();
        let scores: Vec<f32> = (0..100).map(|i| 0.5 + (i as f32 / 200.0)).collect();
        let snap = tracker.record(1, &scores);
        assert_eq!(snap.health_class, CognitiveHealthClass::Diverse);
    }

    #[test]
    fn monoculture_detection() {
        let mut tracker = VitalityVarianceTracker::default();
        // All nodes have nearly identical high vitality
        let scores: Vec<f32> = (0..100).map(|_| 0.85).collect();
        let snap = tracker.record(1, &scores);
        assert_eq!(snap.health_class, CognitiveHealthClass::Monoculture);
    }

    #[test]
    fn exhausted_system() {
        let mut tracker = VitalityVarianceTracker::default();
        let scores: Vec<f32> = (0..100).map(|_| 0.1).collect();
        let snap = tracker.record(1, &scores);
        assert_eq!(snap.health_class, CognitiveHealthClass::Exhausted);
    }

    #[test]
    fn chaotic_system() {
        let mut tracker = VitalityVarianceTracker::default();
        // Low mean, high variance
        let mut scores = Vec::new();
        for i in 0..100 {
            scores.push(if i % 2 == 0 { 0.0 } else { 0.8 });
        }
        let snap = tracker.record(1, &scores);
        assert_eq!(snap.health_class, CognitiveHealthClass::Chaotic);
    }

    #[test]
    fn monoculture_streak() {
        let mut tracker = VitalityVarianceTracker::default();
        for i in 0..5 {
            let scores: Vec<f32> = (0..100).map(|_| 0.85).collect();
            tracker.record(i, &scores);
        }
        assert!(tracker.is_monoculture_streak(5));
        assert!(!tracker.is_monoculture_streak(6));
    }
}
