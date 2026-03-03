//! # Spectral — Graph Laplacian monitoring via λ₂ (Fiedler eigenvalue)
//!
//! Monitors the structural health of graph subregions by computing
//! the algebraic connectivity (second-smallest eigenvalue of the Laplacian).
//!
//! ## Why λ₂?
//!
//! The Fiedler eigenvalue λ₂ of the graph Laplacian L = D - A encodes
//! the "bottleneck" connectivity of the graph:
//!
//! - **λ₂ > 0.5**: Rigid — the subgraph is well-connected, structurally sound
//! - **0.1 < λ₂ ≤ 0.5**: Stable — connected but with some weak spots
//! - **0.0 < λ₂ ≤ 0.1**: Fragile — nearly disconnected, one removal could split it
//! - **λ₂ ≈ 0**: Disconnected — the subgraph has multiple components
//!
//! ## Algorithm
//!
//! For small subgraphs (n ≤ 300), uses the Jacobi eigenvalue algorithm
//! on the dense Laplacian matrix. This gives exact eigenvalues in O(n³).
//!
//! For larger subgraphs, uses power iteration with null-space deflation
//! to estimate λ₂ in O(n² · k) where k = iteration count.
//!
//! ## Design
//!
//! **Pure computation** — the caller provides the edge list and the
//! module returns a [`SpectralHealth`] report. No graph access.

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for spectral monitoring.
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// λ₂ threshold for "Rigid" classification.
    /// Default: 0.5
    pub rigid_threshold: f64,

    /// λ₂ threshold for "Stable" classification.
    /// Default: 0.1
    pub stable_threshold: f64,

    /// λ₂ below this is considered disconnected.
    /// Default: 1e-8
    pub disconnect_epsilon: f64,

    /// Maximum nodes for exact Jacobi eigenvalue computation.
    /// Above this, falls back to power iteration estimate.
    /// Default: 300
    pub jacobi_max_nodes: usize,

    /// Maximum sweeps for Jacobi algorithm.
    /// Default: 200
    pub jacobi_max_sweeps: usize,

    /// Convergence tolerance for eigenvalue computation.
    /// Default: 1e-10
    pub convergence_tolerance: f64,

    /// Maximum iterations for power iteration fallback.
    /// Default: 500
    pub power_iter_max: usize,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            rigid_threshold: 0.5,
            stable_threshold: 0.1,
            disconnect_epsilon: 1e-8,
            jacobi_max_nodes: 300,
            jacobi_max_sweeps: 200,
            convergence_tolerance: 1e-10,
            power_iter_max: 500,
        }
    }
}

// ─────────────────────────────────────────────
// ConnectivityClass
// ─────────────────────────────────────────────

/// Classification of graph connectivity based on λ₂.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectivityClass {
    /// λ₂ > rigid_threshold — well-connected, structurally sound
    Rigid,
    /// stable_threshold < λ₂ ≤ rigid_threshold — connected with weak spots
    Stable,
    /// disconnect_epsilon < λ₂ ≤ stable_threshold — nearly disconnected
    Fragile,
    /// λ₂ ≤ disconnect_epsilon — has multiple components
    Disconnected,
}

impl std::fmt::Display for ConnectivityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectivityClass::Rigid => write!(f, "Rigid"),
            ConnectivityClass::Stable => write!(f, "Stable"),
            ConnectivityClass::Fragile => write!(f, "Fragile"),
            ConnectivityClass::Disconnected => write!(f, "Disconnected"),
        }
    }
}

// ─────────────────────────────────────────────
// SpectralHealth — analysis result
// ─────────────────────────────────────────────

/// Spectral health report for a subgraph.
#[derive(Debug, Clone)]
pub struct SpectralHealth {
    /// Fiedler eigenvalue (λ₂) — algebraic connectivity.
    pub fiedler_eigenvalue: f64,

    /// Whether the graph is connected (λ₂ > epsilon).
    pub is_connected: bool,

    /// Connectivity classification.
    pub connectivity_class: ConnectivityClass,

    /// Largest eigenvalue (λ_max) — spectral radius.
    pub spectral_radius: f64,

    /// Spectral gap: λ_max - λ₂ (large gap = good expansion properties).
    pub spectral_gap: f64,

    /// Number of nodes in the subgraph.
    pub node_count: usize,

    /// Number of edges in the subgraph.
    pub edge_count: usize,

    /// Whether exact (Jacobi) or approximate (power iteration) was used.
    pub is_exact: bool,
}

impl std::fmt::Display for SpectralHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "λ₂={:.6} [{}] nodes={} edges={} gap={:.4}{}",
            self.fiedler_eigenvalue,
            self.connectivity_class,
            self.node_count,
            self.edge_count,
            self.spectral_gap,
            if self.is_exact { "" } else { " (approx)" },
        )
    }
}

// ─────────────────────────────────────────────
// SpectralMonitor
// ─────────────────────────────────────────────

/// Monitors the structural health of graph subregions via the Laplacian spectrum.
///
/// **Pure computation** — the caller provides the edge list.
pub struct SpectralMonitor {
    config: SpectralConfig,
}

impl SpectralMonitor {
    pub fn new(config: SpectralConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SpectralConfig::default())
    }

    /// Analyze the spectral health of a subgraph.
    ///
    /// # Arguments
    /// - `edges`: list of (from_index, to_index, weight) for the subgraph.
    ///   Node indices are 0-based and dense (0..n_nodes).
    /// - `n_nodes`: total number of nodes in the subgraph.
    ///
    /// # Returns
    /// A [`SpectralHealth`] report.
    pub fn analyze(&self, edges: &[(usize, usize, f64)], n_nodes: usize) -> SpectralHealth {
        if n_nodes <= 1 {
            return SpectralHealth {
                fiedler_eigenvalue: 0.0,
                is_connected: n_nodes == 1,
                connectivity_class: if n_nodes == 1 {
                    ConnectivityClass::Rigid
                } else {
                    ConnectivityClass::Disconnected
                },
                spectral_radius: 0.0,
                spectral_gap: 0.0,
                node_count: n_nodes,
                edge_count: edges.len(),
                is_exact: true,
            };
        }

        // Build the Laplacian matrix
        let mut laplacian = build_laplacian(edges, n_nodes);

        let (fiedler, spectral_radius, is_exact) = if n_nodes <= self.config.jacobi_max_nodes {
            // Exact: Jacobi eigenvalue algorithm
            let eigenvalues = jacobi_eigenvalues(
                &mut laplacian,
                self.config.jacobi_max_sweeps,
                self.config.convergence_tolerance,
            );
            let lambda2 = if eigenvalues.len() >= 2 {
                eigenvalues[1].max(0.0)
            } else {
                0.0
            };
            let lambda_max = eigenvalues.last().copied().unwrap_or(0.0);
            (lambda2, lambda_max, true)
        } else {
            // Approximate: power iteration with deflation
            let (lambda2, lambda_max) = power_iteration_fiedler(
                &laplacian,
                n_nodes,
                self.config.power_iter_max,
                self.config.convergence_tolerance,
            );
            (lambda2, lambda_max, false)
        };

        let is_connected = fiedler > self.config.disconnect_epsilon;
        let connectivity_class = self.classify(fiedler);

        SpectralHealth {
            fiedler_eigenvalue: fiedler,
            is_connected,
            connectivity_class,
            spectral_radius,
            spectral_gap: spectral_radius - fiedler,
            node_count: n_nodes,
            edge_count: edges.len(),
            is_exact,
        }
    }

    fn classify(&self, lambda2: f64) -> ConnectivityClass {
        if lambda2 > self.config.rigid_threshold {
            ConnectivityClass::Rigid
        } else if lambda2 > self.config.stable_threshold {
            ConnectivityClass::Stable
        } else if lambda2 > self.config.disconnect_epsilon {
            ConnectivityClass::Fragile
        } else {
            ConnectivityClass::Disconnected
        }
    }

    pub fn config(&self) -> &SpectralConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────
// Internal: Laplacian construction
// ─────────────────────────────────────────────

/// Build the weighted graph Laplacian L = D - A.
///
/// The Laplacian is symmetric positive semi-definite with smallest
/// eigenvalue 0 (eigenvector = constant vector).
fn build_laplacian(edges: &[(usize, usize, f64)], n: usize) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; n]; n];

    for &(i, j, w) in edges {
        if i >= n || j >= n {
            continue; // Skip invalid indices
        }
        let w = w.abs(); // Weights should be positive
        // Off-diagonal: -w
        l[i][j] -= w;
        l[j][i] -= w;
        // Diagonal: +w (both endpoints)
        l[i][i] += w;
        l[j][j] += w;
    }

    l
}

// ─────────────────────────────────────────────
// Internal: Jacobi eigenvalue algorithm
// ─────────────────────────────────────────────

/// Jacobi eigenvalue algorithm for symmetric matrices.
///
/// Returns all eigenvalues sorted in ascending order.
/// Uses the classical Jacobi method with the "tau trick" for numerical stability.
///
/// Complexity: O(n³) per sweep, typically converges in 5-10 sweeps.
fn jacobi_eigenvalues(
    a: &mut Vec<Vec<f64>>,
    max_sweeps: usize,
    tol: f64,
) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![a[0][0]];
    }

    let max_rotations = max_sweeps * n * (n - 1) / 2;

    for _rotation in 0..max_rotations {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break; // Converged
        }

        // Compute Jacobi rotation parameters
        let apq = a[p][q];
        let diff = a[p][p] - a[q][q];

        let t = if diff.abs() < 1e-30 {
            // θ = π/4
            if apq >= 0.0 { 1.0 } else { -1.0 }
        } else {
            let tau = diff / (2.0 * apq);
            // Choose the smaller root for numerical stability
            let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
            sign / (tau.abs() + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let tau_val = s / (1.0 + c);

        // Apply rotation
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        a[p][p] -= t * apq;
        a[q][q] += t * apq;

        for i in 0..n {
            if i != p && i != q {
                let aip = a[i][p];
                let aiq = a[i][q];
                a[i][p] = aip - s * (aiq + tau_val * aip);
                a[p][i] = a[i][p];
                a[i][q] = aiq + s * (aip - tau_val * aiq);
                a[q][i] = a[i][q];
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

// ─────────────────────────────────────────────
// Internal: Power iteration fallback
// ─────────────────────────────────────────────

/// Estimate λ₂ and λ_max using power iteration.
///
/// 1. Power iteration on L gives λ_max (largest eigenvalue)
/// 2. Deflation: L' = L + c·(1/n)·11^T shifts λ₁=0 to c
/// 3. Inverse iteration on L' (if feasible) gives λ₂
///
/// For the inverse iteration, we use the shifted power method:
/// M = λ_max·I - L has eigenvalues reversed, so power iteration
/// on M (projected to complement of its largest eigenvector) gives λ_max - λ₂.
fn power_iteration_fiedler(
    laplacian: &[Vec<f64>],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> (f64, f64) {
    if n <= 1 {
        return (0.0, 0.0);
    }

    // Step 1: Find λ_max via power iteration
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    // Perturb to avoid degenerate start
    for i in 0..n {
        x[i] += 0.01 * (i as f64 / n as f64 - 0.5);
    }
    normalize(&mut x);

    let mut lambda_max = 0.0;
    for _ in 0..max_iter {
        let y = mat_vec_mul(laplacian, &x);
        let new_lambda = dot(&x, &y);
        if (new_lambda - lambda_max).abs() < tol {
            lambda_max = new_lambda;
            break;
        }
        lambda_max = new_lambda;
        x = y;
        normalize(&mut x);
    }

    // Step 2: Find λ₂ via power iteration on M = λ_max·I - L
    // (projected onto complement of constant vector)
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    project_out_constant(&mut v);
    normalize(&mut v);

    let mut lambda2_complement = 0.0;
    for _ in 0..max_iter {
        // y = M·v = λ_max·v - L·v
        let lv = mat_vec_mul(laplacian, &v);
        let mut y: Vec<f64> = v.iter().zip(lv.iter()).map(|(&vi, &lvi)| lambda_max * vi - lvi).collect();
        project_out_constant(&mut y);

        let norm_y = vec_norm(&y);
        if norm_y < 1e-30 {
            break;
        }

        let new_lambda = dot(&v, &y);
        if (new_lambda - lambda2_complement).abs() < tol {
            lambda2_complement = new_lambda;
            break;
        }
        lambda2_complement = new_lambda;
        v = y;
        normalize(&mut v);
    }

    // λ₂ = λ_max - (largest eigenvalue of M in complement)
    let lambda2 = (lambda_max - lambda2_complement).max(0.0);

    (lambda2, lambda_max)
}

/// Project out the constant vector component: v = v - mean(v) · 1
fn project_out_constant(v: &mut [f64]) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize(v: &mut [f64]) {
    let n = vec_norm(v);
    if n > 1e-30 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

// ─────────────────────────────────────────────
// DriftTracker — λ₂ evolution monitoring (Option B)
// ─────────────────────────────────────────────

/// Tracks λ₂ evolution over time for the Evolutionary Model (Option B).
///
/// Instead of enforcing constant λ₂, allows slow controlled drift
/// that enables the graph to evolve while detecting destabilization.
///
/// ```text
/// Healthy drift:   |Δλ₂| ≤ max_rate per cycle
/// Destabilization: |Δλ₂| > max_rate for N cycles
/// ```
#[derive(Debug, Clone)]
pub struct DriftTracker {
    /// Historical λ₂ values (most recent last).
    pub history: Vec<f64>,

    /// Maximum entries to keep in history.
    pub max_history: usize,

    /// Maximum acceptable drift per cycle.
    pub max_drift_rate: f64,

    /// Number of consecutive violations before alarm.
    pub alarm_threshold: usize,
}

impl DriftTracker {
    pub fn new(max_drift_rate: f64, max_history: usize, alarm_threshold: usize) -> Self {
        Self {
            history: Vec::new(),
            max_history,
            max_drift_rate,
            alarm_threshold,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.05, 100, 3)
    }

    /// Record a new λ₂ measurement.
    pub fn record(&mut self, lambda2: f64) {
        self.history.push(lambda2);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get the latest drift (Δλ₂ from previous to current).
    pub fn latest_drift(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let n = self.history.len();
        Some(self.history[n - 1] - self.history[n - 2])
    }

    /// Count consecutive violations of the drift threshold.
    pub fn consecutive_violations(&self) -> usize {
        if self.history.len() < 2 {
            return 0;
        }
        self.history
            .windows(2)
            .rev()
            .take_while(|w| (w[1] - w[0]).abs() > self.max_drift_rate)
            .count()
    }

    /// Check if the drift is within acceptable bounds (healthy evolution).
    pub fn is_healthy(&self) -> bool {
        self.consecutive_violations() < self.alarm_threshold
    }

    /// Get the overall trend (positive = λ₂ increasing = stronger connectivity).
    pub fn trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let first = self.history[0];
        let last = *self.history.last().unwrap();
        (last - first) / self.history.len() as f64
    }

    /// Get the mean λ₂ over the tracked history.
    pub fn mean_lambda2(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history.iter().sum::<f64>() / self.history.len() as f64
    }

    /// Get the λ₂ variance over the tracked history.
    /// High variance = unstable regime. Near zero = steady state.
    pub fn variance(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_lambda2();
        self.history.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / (self.history.len() - 1) as f64
    }
}

// ─────────────────────────────────────────────
// AdaptiveBand — mobile target band for λ₂
// ─────────────────────────────────────────────

/// Mobile target band for λ₂ that evolves with the system.
///
/// The fundamental insight of the Evolutionary Model (Option B): instead of
/// forcing λ₂ to a fixed value, we allow it to drift slowly within a band
/// that itself adapts based on the system's energy state.
///
/// ## Band Evolution Rule
///
/// ```text
/// λ_target(t+1) = λ_target(t) + ε · ΔE_global
/// ```
///
/// Where:
/// - `ε` is the adaptation rate (small → conservative, large → aggressive)
/// - `ΔE_global` is the global energy change (positive = system improving)
/// - The band is clamped to `[min_target, max_target]`
///
/// ## Why Adaptive?
///
/// A growing knowledge graph **should** become more connected over time.
/// Forcing constant λ₂ would either:
/// - Prevent natural growth (if too high)
/// - Allow fragmentation (if too low)
///
/// The adaptive band allows the system to find its own equilibrium:
/// improving energy → tighter connectivity expected → target rises.
///
/// ## Edge of Chaos
///
/// For regime C (critically poised), ε should be small enough that the
/// band moves slowly relative to individual cycle fluctuations, but large
/// enough to track genuine structural evolution.
#[derive(Debug, Clone)]
pub struct AdaptiveBand {
    /// Current target λ₂ (center of the band).
    pub target: f64,

    /// Half-width of the acceptable band around the target.
    /// λ₂ is "healthy" if |λ₂ - target| ≤ half_width.
    pub half_width: f64,

    /// Adaptation rate ε: how fast the target responds to energy changes.
    /// Default: 0.01 (conservative, suitable for edge-of-chaos regime)
    pub epsilon: f64,

    /// Absolute minimum target. The band can never drift below this.
    /// Default: 0.05 (ensures minimal connectivity)
    pub min_target: f64,

    /// Absolute maximum target. The band can never drift above this.
    /// Default: 5.0 (prevents unrealistic rigidity demands)
    pub max_target: f64,

    /// History of target adjustments (for auditing).
    adjustments: Vec<BandAdjustment>,
    max_adjustments: usize,
}

/// Record of a single band adjustment.
#[derive(Debug, Clone)]
pub struct BandAdjustment {
    /// The energy delta that caused this adjustment.
    pub energy_delta: f64,
    /// The target before adjustment.
    pub old_target: f64,
    /// The target after adjustment.
    pub new_target: f64,
    /// The λ₂ at the time of adjustment.
    pub lambda2: f64,
}

impl AdaptiveBand {
    pub fn new(initial_target: f64, half_width: f64, epsilon: f64) -> Self {
        Self {
            target: initial_target,
            half_width,
            epsilon,
            min_target: 0.05,
            max_target: 5.0,
            adjustments: Vec::new(),
            max_adjustments: 50,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(0.5, 0.2, 0.01)
    }

    /// Adjust the target band based on global energy change.
    ///
    /// # Arguments
    /// - `energy_delta`: ΔE_global (positive = system improved this cycle)
    /// - `current_lambda2`: the measured λ₂ this cycle
    ///
    /// # Returns
    /// The new target value.
    pub fn adjust(&mut self, energy_delta: f64, current_lambda2: f64) -> f64 {
        let old_target = self.target;

        // λ_target(t+1) = λ_target(t) + ε · ΔE
        self.target += self.epsilon * energy_delta;
        self.target = self.target.clamp(self.min_target, self.max_target);

        // Record adjustment
        self.adjustments.push(BandAdjustment {
            energy_delta,
            old_target,
            new_target: self.target,
            lambda2: current_lambda2,
        });

        if self.adjustments.len() > self.max_adjustments {
            self.adjustments.remove(0);
        }

        self.target
    }

    /// Check if a λ₂ measurement is within the acceptable band.
    ///
    /// Returns true if |λ₂ - target| ≤ half_width.
    pub fn is_within_band(&self, lambda2: f64) -> bool {
        (lambda2 - self.target).abs() <= self.half_width
    }

    /// Get the lower bound of the acceptable band.
    pub fn lower(&self) -> f64 {
        (self.target - self.half_width).max(0.0)
    }

    /// Get the upper bound of the acceptable band.
    pub fn upper(&self) -> f64 {
        self.target + self.half_width
    }

    /// Get the distance from λ₂ to the band center (signed).
    /// Positive = above target, negative = below target.
    pub fn deviation(&self, lambda2: f64) -> f64 {
        lambda2 - self.target
    }

    /// Get the most recent adjustments for auditing.
    pub fn recent_adjustments(&self) -> &[BandAdjustment] {
        &self.adjustments
    }
}

impl std::fmt::Display for AdaptiveBand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Band[target={:.3}, range=[{:.3}, {:.3}], ε={:.4}]",
            self.target,
            self.lower(),
            self.upper(),
            self.epsilon,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_node() {
        let monitor = SpectralMonitor::with_defaults();
        let health = monitor.analyze(&[], 1);
        assert!(health.is_connected);
        assert_eq!(health.connectivity_class, ConnectivityClass::Rigid);
    }

    #[test]
    fn test_two_nodes_connected() {
        let monitor = SpectralMonitor::with_defaults();
        let edges = vec![(0, 1, 1.0)];
        let health = monitor.analyze(&edges, 2);
        assert!(health.is_connected);
        // For K₂: L = [[1,-1],[-1,1]], eigenvalues = {0, 2}
        assert!(
            health.fiedler_eigenvalue > 1.0,
            "λ₂ of K₂ should be ~2.0: {}",
            health.fiedler_eigenvalue
        );
    }

    #[test]
    fn test_two_nodes_disconnected() {
        let monitor = SpectralMonitor::with_defaults();
        let edges: Vec<(usize, usize, f64)> = vec![];
        let health = monitor.analyze(&edges, 2);
        assert!(!health.is_connected);
        assert_eq!(health.connectivity_class, ConnectivityClass::Disconnected);
        assert!(health.fiedler_eigenvalue.abs() < 1e-6);
    }

    #[test]
    fn test_complete_graph_k4() {
        // K₄: 4 nodes, all connected with weight 1.0
        let monitor = SpectralMonitor::with_defaults();
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (2, 3, 1.0),
        ];
        let health = monitor.analyze(&edges, 4);
        assert!(health.is_connected);
        // K₄: highly connected → λ₂ should be large (exact: 4.0)
        assert!(
            health.fiedler_eigenvalue > 1.0,
            "K₄ should have large λ₂: {}",
            health.fiedler_eigenvalue
        );
        assert_eq!(health.connectivity_class, ConnectivityClass::Rigid);
    }

    #[test]
    fn test_path_graph() {
        // Path: 0-1-2-3 (weakly connected)
        let monitor = SpectralMonitor::with_defaults();
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let health = monitor.analyze(&edges, 4);
        assert!(health.is_connected);
        // Path graph P₄: less connected than K₄
        assert!(
            health.fiedler_eigenvalue > 0.1,
            "Path should be connected: λ₂={}",
            health.fiedler_eigenvalue
        );
    }

    #[test]
    fn test_barbell_graph() {
        // Two triangles connected by a single edge (fragile bridge)
        let monitor = SpectralMonitor::with_defaults();
        let edges = vec![
            // Triangle 1: 0-1-2
            (0, 1, 1.0),
            (1, 2, 1.0),
            (0, 2, 1.0),
            // Bridge: 2-3
            (2, 3, 0.001), // Very weak bridge
            // Triangle 2: 3-4-5
            (3, 4, 1.0),
            (4, 5, 1.0),
            (3, 5, 1.0),
        ];
        let health = monitor.analyze(&edges, 6);
        assert!(health.is_connected);
        // With very weak bridge, λ₂ should be significantly less than
        // a fully connected graph. The exact value depends on bridge weight.
        assert!(
            health.fiedler_eigenvalue < 2.0,
            "Barbell should have reduced λ₂ vs complete: {}",
            health.fiedler_eigenvalue
        );
    }

    #[test]
    fn test_build_laplacian() {
        let edges = vec![(0, 1, 2.0), (1, 2, 3.0)];
        let l = build_laplacian(&edges, 3);
        // Node 0: degree = 2
        assert!((l[0][0] - 2.0).abs() < 1e-10);
        // Node 1: degree = 2 + 3 = 5
        assert!((l[1][1] - 5.0).abs() < 1e-10);
        // Node 2: degree = 3
        assert!((l[2][2] - 3.0).abs() < 1e-10);
        // Off-diagonal
        assert!((l[0][1] - (-2.0)).abs() < 1e-10);
        assert!((l[1][2] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_row_sum_zero() {
        // Row sums of a Laplacian should be 0
        let edges = vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0)];
        let l = build_laplacian(&edges, 3);
        for row in &l {
            let sum: f64 = row.iter().sum();
            assert!(
                sum.abs() < 1e-10,
                "Row sum should be 0: {}",
                sum
            );
        }
    }

    #[test]
    fn test_health_display() {
        let health = SpectralHealth {
            fiedler_eigenvalue: 0.42,
            is_connected: true,
            connectivity_class: ConnectivityClass::Stable,
            spectral_radius: 4.0,
            spectral_gap: 3.58,
            node_count: 10,
            edge_count: 15,
            is_exact: true,
        };
        let s = format!("{health}");
        assert!(s.contains("Stable"));
        assert!(s.contains("λ₂=0.42"));
    }

    #[test]
    fn test_empty_graph() {
        let monitor = SpectralMonitor::with_defaults();
        let health = monitor.analyze(&[], 0);
        assert!(!health.is_connected);
        assert_eq!(health.connectivity_class, ConnectivityClass::Disconnected);
    }

    #[test]
    fn test_connectivity_ordering() {
        // λ₂(K₄) > λ₂(Path) > λ₂(Disconnected)
        let monitor = SpectralMonitor::with_defaults();

        let k4_health = monitor.analyze(
            &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)],
            4,
        );
        let path_health = monitor.analyze(
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
            4,
        );
        let disc_health = monitor.analyze(&[], 4);

        assert!(
            k4_health.fiedler_eigenvalue > path_health.fiedler_eigenvalue,
            "K₄ ({}) should beat Path ({})",
            k4_health.fiedler_eigenvalue,
            path_health.fiedler_eigenvalue,
        );
        assert!(
            path_health.fiedler_eigenvalue > disc_health.fiedler_eigenvalue,
            "Path ({}) should beat Disconnected ({})",
            path_health.fiedler_eigenvalue,
            disc_health.fiedler_eigenvalue,
        );
    }

    // ── DriftTracker tests ──

    #[test]
    fn test_drift_tracker_healthy() {
        let mut tracker = DriftTracker::with_defaults();
        tracker.record(1.0);
        tracker.record(1.02);
        tracker.record(1.04);
        tracker.record(1.03);
        assert!(tracker.is_healthy());
        assert_eq!(tracker.consecutive_violations(), 0);
    }

    #[test]
    fn test_drift_tracker_violation() {
        let mut tracker = DriftTracker::new(0.05, 100, 2);
        tracker.record(1.0);
        tracker.record(0.8); // drift = -0.2 (violation)
        tracker.record(0.5); // drift = -0.3 (violation)
        assert!(!tracker.is_healthy());
        assert_eq!(tracker.consecutive_violations(), 2);
    }

    #[test]
    fn test_drift_tracker_trend() {
        let mut tracker = DriftTracker::with_defaults();
        tracker.record(1.0);
        tracker.record(1.1);
        tracker.record(1.2);
        let trend = tracker.trend();
        assert!(trend > 0.0, "Upward trend: {}", trend);
    }

    #[test]
    fn test_drift_tracker_latest() {
        let mut tracker = DriftTracker::with_defaults();
        tracker.record(1.0);
        tracker.record(1.05);
        assert!((tracker.latest_drift().unwrap() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_drift_tracker_empty() {
        let tracker = DriftTracker::with_defaults();
        assert!(tracker.latest_drift().is_none());
        assert!(tracker.is_healthy());
        assert_eq!(tracker.trend(), 0.0);
    }

    #[test]
    fn test_drift_tracker_variance() {
        let mut tracker = DriftTracker::with_defaults();
        tracker.record(1.0);
        tracker.record(1.0);
        tracker.record(1.0);
        assert!(tracker.variance() < 1e-10, "Constant series should have zero variance");

        let mut tracker2 = DriftTracker::with_defaults();
        tracker2.record(0.5);
        tracker2.record(1.5);
        tracker2.record(0.5);
        tracker2.record(1.5);
        assert!(tracker2.variance() > 0.2, "Oscillating series should have high variance");
    }

    // ── AdaptiveBand tests ──

    #[test]
    fn test_band_defaults() {
        let band = AdaptiveBand::with_defaults();
        assert!((band.target - 0.5).abs() < 1e-10);
        assert!(band.is_within_band(0.5));
        assert!(band.is_within_band(0.3)); // 0.5 - 0.2 = 0.3
        assert!(band.is_within_band(0.7)); // 0.5 + 0.2 = 0.7
        assert!(!band.is_within_band(0.05)); // Too low
        assert!(!band.is_within_band(1.0));  // Too high
    }

    #[test]
    fn test_band_adjust_positive_energy() {
        let mut band = AdaptiveBand::with_defaults();
        // System improving → target should rise
        let old = band.target;
        band.adjust(1.0, 0.5); // ΔE = +1.0
        assert!(
            band.target > old,
            "Target should rise with positive energy: {} → {}",
            old, band.target
        );
    }

    #[test]
    fn test_band_adjust_negative_energy() {
        let mut band = AdaptiveBand::with_defaults();
        let old = band.target;
        band.adjust(-1.0, 0.5); // ΔE = -1.0 → target drops
        assert!(
            band.target < old,
            "Target should drop with negative energy: {} → {}",
            old, band.target
        );
    }

    #[test]
    fn test_band_clamped_to_limits() {
        let mut band = AdaptiveBand::new(0.5, 0.2, 1.0); // Large ε
        band.adjust(100.0, 0.5); // Massive positive energy
        assert!(
            band.target <= band.max_target,
            "Target should be clamped: {}",
            band.target
        );

        let mut band2 = AdaptiveBand::new(0.5, 0.2, 1.0);
        band2.adjust(-100.0, 0.5); // Massive negative
        assert!(
            band2.target >= band2.min_target,
            "Target should be clamped: {}",
            band2.target
        );
    }

    #[test]
    fn test_band_deviation() {
        let band = AdaptiveBand::new(1.0, 0.3, 0.01);
        assert!((band.deviation(1.2) - 0.2).abs() < 1e-10);
        assert!((band.deviation(0.8) - (-0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_band_display() {
        let band = AdaptiveBand::with_defaults();
        let s = format!("{band}");
        assert!(s.contains("Band["));
        assert!(s.contains("target="));
    }

    #[test]
    fn test_band_audit_trail() {
        let mut band = AdaptiveBand::with_defaults();
        band.adjust(0.5, 0.4);
        band.adjust(-0.2, 0.5);
        let history = band.recent_adjustments();
        assert_eq!(history.len(), 2);
        assert!((history[0].energy_delta - 0.5).abs() < 1e-10);
    }
}
