//! Criterion benchmarks for Chebyshev heat-kernel diffusion.
//!
//! Run with:
//! ```bash
//! cargo bench -p nietzsche-pregel
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nietzsche_pregel::{
    apply_heat_kernel, chebyshev_coefficients, modified_bessel_i,
    HyperbolicLaplacian, NodeIndex,
};
use nietzsche_graph::{
    Edge, EdgeType, MockVectorStore, NietzscheDB, Node, PoincareVector,
};
use uuid::Uuid;

// ── helpers ──────────────────────────────────────────────────────────────────

fn open_db() -> (NietzscheDB<MockVectorStore>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db  = NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap();
    (db, dir)
}

fn ring_graph(n: usize) -> (NietzscheDB<MockVectorStore>, tempfile::TempDir, Vec<Uuid>) {
    let (mut db, dir) = open_db();
    let step = 0.8 / n as f64;
    let ids: Vec<Uuid> = (0..n)
        .map(|i| {
            let angle = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            let node  = Node::new(
                Uuid::new_v4(),
                PoincareVector::new(vec![angle.cos() * step * i as f64, angle.sin() * step * i as f64]),
                serde_json::json!({}),
            );
            let id = node.id;
            db.insert_node(node).unwrap();
            id
        })
        .collect();

    // Connect as a ring
    for i in 0..n {
        let e = Edge {
            id:        Uuid::new_v4(),
            from:      ids[i],
            to:        ids[(i + 1) % n],
            edge_type: EdgeType::Association,
            weight:    1.0,
            metadata:  Default::default(),
        };
        db.insert_edge(e).unwrap();
    }

    (db, dir, ids)
}

// ── Bessel functions ─────────────────────────────────────────────────────────

fn bench_modified_bessel_i(c: &mut Criterion) {
    let mut group = c.benchmark_group("chebyshev/modified_bessel_i");

    for &t in &[0.5f64, 1.0, 5.0, 10.0] {
        for &k in &[5usize, 10, 20] {
            group.bench_with_input(
                BenchmarkId::new(format!("t{t}_k"), k),
                &(t, k),
                |b, &(t, k)| b.iter(|| modified_bessel_i(t, k)),
            );
        }
    }

    group.finish();
}

// ── Chebyshev coefficients ───────────────────────────────────────────────────

fn bench_chebyshev_coefficients(c: &mut Criterion) {
    let mut group = c.benchmark_group("chebyshev/coefficients");

    for &k in &[5usize, 10, 20] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| chebyshev_coefficients(1.0, k))
        });
    }

    group.finish();
}

// ── apply_heat_kernel ────────────────────────────────────────────────────────

fn bench_apply_heat_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("chebyshev/apply_heat_kernel");

    for &n in &[10usize, 20, 40] {
        group.bench_with_input(BenchmarkId::new("ring_nodes", n), &n, |b, &n| {
            let (db, _dir, ids) = ring_graph(n);
            let nodes = db.storage().scan_nodes().unwrap();
            let lap   = HyperbolicLaplacian::from_graph(db.storage(), db.adjacency(), &nodes)
                .unwrap();

            // Unit impulse on first node
            let mut signal = vec![0.0; n];
            if !ids.is_empty() {
                let idx = lap.index().get_idx(&ids[0]).unwrap_or(0);
                signal[idx] = 1.0;
            }

            b.iter(|| apply_heat_kernel(&lap, &signal, 1.0, 10, 2.0));
        });
    }

    group.finish();
}

// ── Laplacian construction ────────────────────────────────────────────────────

fn bench_laplacian_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("chebyshev/laplacian_build");

    for &n in &[10usize, 20, 50] {
        group.bench_with_input(BenchmarkId::new("ring_nodes", n), &n, |b, &n| {
            let (db, _dir, _) = ring_graph(n);
            let nodes = db.storage().scan_nodes().unwrap();
            b.iter(|| {
                HyperbolicLaplacian::from_graph(db.storage(), db.adjacency(), &nodes).unwrap()
            });
        });
    }

    group.finish();
}

// ── criterion wiring ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_modified_bessel_i,
    bench_chebyshev_coefficients,
    bench_apply_heat_kernel,
    bench_laplacian_build,
);
criterion_main!(benches);
