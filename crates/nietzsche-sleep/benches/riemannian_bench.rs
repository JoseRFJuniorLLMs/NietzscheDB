//! Criterion benchmarks for Riemannian geometry primitives.
//!
//! Run with:
//! ```bash
//! cargo bench -p nietzsche-sleep
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nietzsche_sleep::{
    conformal_factor, exp_map, project_into_ball, random_tangent, riemannian_grad,
    AdamState, RiemannianAdam,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ── conformal_factor ─────────────────────────────────────────────────────────

fn bench_conformal_factor(c: &mut Criterion) {
    let x = vec![0.3, 0.1, 0.2, 0.0];
    c.bench_function("riemannian/conformal_factor_4d", |b| {
        b.iter(|| conformal_factor(&x))
    });
}

// ── project_into_ball ────────────────────────────────────────────────────────

fn bench_project_into_ball(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/project_into_ball");
    for &dim in &[4usize, 64, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let x = vec![0.5; dim];
            b.iter(|| project_into_ball(x.clone()));
        });
    }
    group.finish();
}

// ── exp_map ──────────────────────────────────────────────────────────────────

fn bench_exp_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/exp_map");

    for &dim in &[2usize, 16, 64, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let x = vec![0.1; dim];
            let v = vec![0.05; dim];
            b.iter(|| exp_map(&x, &v));
        });
    }

    group.finish();
}

// ── riemannian_grad ──────────────────────────────────────────────────────────

fn bench_riemannian_grad(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/riemannian_grad");

    for &dim in &[2usize, 64, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let x = vec![0.1; dim];
            let g = vec![0.01; dim];
            b.iter(|| riemannian_grad(&x, &g));
        });
    }

    group.finish();
}

// ── random_tangent ───────────────────────────────────────────────────────────

fn bench_random_tangent(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/random_tangent");

    for &dim in &[2usize, 64, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| random_tangent(dim, 0.02, &mut rng));
        });
    }

    group.finish();
}

// ── RiemannianAdam::step ─────────────────────────────────────────────────────

fn bench_adam_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/adam_step");

    for &dim in &[2usize, 64, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let adam  = RiemannianAdam::new(5e-3);
            let x     = vec![0.1; dim];
            let grad  = vec![0.01; dim];
            let mut state = AdamState::new(dim);
            b.iter(|| adam.step(&x, &grad, &mut state));
        });
    }

    group.finish();
}

// ── Adam convergence (10 steps) ──────────────────────────────────────────────

fn bench_adam_10_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("riemannian/adam_10_steps");

    for &dim in &[2usize, 64] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let adam   = RiemannianAdam::new(5e-3);
            let x      = vec![0.1; dim];
            let target = vec![0.0; dim];
            b.iter(|| {
                let mut pos   = x.clone();
                let mut state = AdamState::new(dim);
                for _ in 0..10 {
                    let grad: Vec<f64> = pos.iter().zip(target.iter())
                        .map(|(xi, ti)| 2.0 * (xi - ti))
                        .collect();
                    pos = adam.step(&pos, &grad, &mut state);
                }
                pos
            });
        });
    }

    group.finish();
}

// ── criterion wiring ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_conformal_factor,
    bench_project_into_ball,
    bench_exp_map,
    bench_riemannian_grad,
    bench_random_tangent,
    bench_adam_step,
    bench_adam_10_steps,
);
criterion_main!(benches);
