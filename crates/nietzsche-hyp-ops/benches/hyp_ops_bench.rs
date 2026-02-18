use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nietzsche_hyp_ops::*;

fn bench_exp_map_zero_64d(c: &mut Criterion) {
    let v: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01 - 0.32).collect();
    c.bench_function("exp_map_zero_64d", |b| {
        b.iter(|| exp_map_zero(black_box(&v)))
    });
}

fn bench_exp_map_zero_256d(c: &mut Criterion) {
    let v: Vec<f64> = (0..256).map(|i| (i as f64) * 0.003 - 0.384).collect();
    c.bench_function("exp_map_zero_256d", |b| {
        b.iter(|| exp_map_zero(black_box(&v)))
    });
}

fn bench_log_map_zero_64d(c: &mut Criterion) {
    let v: Vec<f64> = (0..64).map(|i| (i as f64) * 0.01 - 0.32).collect();
    let x = exp_map_zero(&v);
    c.bench_function("log_map_zero_64d", |b| {
        b.iter(|| log_map_zero(black_box(&x)))
    });
}

fn bench_mobius_add_64d(c: &mut Criterion) {
    let u = exp_map_zero(&(0..64).map(|i| (i as f64) * 0.01).collect::<Vec<_>>());
    let v = exp_map_zero(&(0..64).map(|i| (i as f64) * -0.01).collect::<Vec<_>>());
    c.bench_function("mobius_add_64d", |b| {
        b.iter(|| mobius_add(black_box(&u), black_box(&v)))
    });
}

fn bench_poincare_distance_64d(c: &mut Criterion) {
    let u = exp_map_zero(&(0..64).map(|i| (i as f64) * 0.01).collect::<Vec<_>>());
    let v = exp_map_zero(&(0..64).map(|i| (i as f64) * -0.01).collect::<Vec<_>>());
    c.bench_function("poincare_distance_64d", |b| {
        b.iter(|| poincare_distance(black_box(&u), black_box(&v)))
    });
}

fn bench_gyromidpoint_3x256d(c: &mut Criterion) {
    let a = exp_map_zero(&(0..256).map(|i| (i as f64) * 0.003 - 0.384).collect::<Vec<_>>());
    let b = exp_map_zero(&(0..256).map(|i| ((i * 7 % 256) as f64) * 0.003 - 0.384).collect::<Vec<_>>());
    let d = exp_map_zero(&(0..256).map(|i| ((i * 13 % 256) as f64) * 0.003 - 0.384).collect::<Vec<_>>());
    c.bench_function("gyromidpoint_3x256d", |b_iter| {
        b_iter.iter(|| gyromidpoint(black_box(&[&a[..], &b[..], &d[..]])))
    });
}

criterion_group!(
    benches,
    bench_exp_map_zero_64d,
    bench_exp_map_zero_256d,
    bench_log_map_zero_64d,
    bench_mobius_add_64d,
    bench_poincare_distance_64d,
    bench_gyromidpoint_3x256d,
);
criterion_main!(benches);
