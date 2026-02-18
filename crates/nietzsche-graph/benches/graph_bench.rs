//! Criterion benchmarks for nietzsche-graph core operations.
//!
//! Run with:
//! ```bash
//! cargo bench -p nietzsche-graph
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use nietzsche_graph::{
    bfs, dijkstra,
    traversal::{BfsConfig, DijkstraConfig},
    Edge, EdgeType, MockVectorStore, NietzscheDB, Node, PoincareVector,
};
use uuid::Uuid;

// ── helpers ─────────────────────────────────────────────────────────────────

fn open_db() -> (NietzscheDB<MockVectorStore>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db  = NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap();
    (db, dir)
}

fn mk_node(x: f64, y: f64) -> Node {
    Node::new(Uuid::new_v4(), PoincareVector::new(vec![x, y]), serde_json::json!({}))
}

fn populate(db: &mut NietzscheDB<MockVectorStore>, n: usize) -> Vec<Uuid> {
    let step = 0.9 / n as f64;
    (0..n)
        .map(|i| {
            let node = mk_node(i as f64 * step, 0.0);
            let id   = node.id;
            db.insert_node(node).unwrap();
            id
        })
        .collect()
}

fn populate_chain(db: &mut NietzscheDB<MockVectorStore>, n: usize) -> Vec<Uuid> {
    let ids = populate(db, n);
    for w in ids.windows(2) {
        let e = Edge {
            id:        Uuid::new_v4(),
            from:      w[0],
            to:        w[1],
            edge_type: EdgeType::Association,
            weight:    1.0,
            metadata:  Default::default(),
        };
        db.insert_edge(e).unwrap();
    }
    ids
}

// ── insert ───────────────────────────────────────────────────────────────────

fn bench_insert_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/insert");

    group.bench_function("single_node", |b| {
        let (mut db, _dir) = open_db();
        b.iter(|| {
            let node = mk_node(0.1, 0.0);
            db.insert_node(node).unwrap()
        });
    });

    for &n in &[10u32, 100, 500] {
        group.bench_with_input(BenchmarkId::new("batch", n), &n, |b, &n| {
            b.iter_batched(
                open_db,
                |(mut db, _dir)| {
                    for i in 0..n {
                        let x = (i as f64 * 0.001).min(0.9);
                        db.insert_node(mk_node(x, 0.0)).unwrap();
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ── scan ─────────────────────────────────────────────────────────────────────

fn bench_scan_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/scan");

    for &n in &[10usize, 100, 500] {
        group.bench_with_input(BenchmarkId::new("scan_nodes", n), &n, |b, &n| {
            let (mut db, _dir) = open_db();
            populate(&mut db, n);
            b.iter(|| db.storage().scan_nodes().unwrap());
        });
    }

    group.finish();
}

// ── Poincaré distance ────────────────────────────────────────────────────────

fn bench_poincare_distance(c: &mut Criterion) {
    c.bench_function("graph/poincare_distance", |b| {
        let u = PoincareVector::new(vec![0.3, 0.1, 0.2]);
        let v = PoincareVector::new(vec![0.1, 0.4, 0.0]);
        b.iter(|| u.distance(&v));
    });
}

// ── BFS ──────────────────────────────────────────────────────────────────────

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal/bfs");

    for &n in &[20usize, 50, 100] {
        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, &n| {
            let (mut db, _dir) = open_db();
            let ids = populate_chain(&mut db, n);
            let cfg = BfsConfig { max_depth: n, ..Default::default() };
            b.iter(|| bfs(db.storage(), db.adjacency(), ids[0], &cfg).unwrap());
        });
    }

    group.finish();
}

// ── Dijkstra ─────────────────────────────────────────────────────────────────

fn bench_dijkstra(c: &mut Criterion) {
    let mut group = c.benchmark_group("traversal/dijkstra");

    for &n in &[20usize, 50, 100] {
        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, &n| {
            let (mut db, _dir) = open_db();
            let ids = populate_chain(&mut db, n);
            let cfg = DijkstraConfig { max_nodes: n * 2, ..Default::default() };
            b.iter(|| dijkstra(db.storage(), db.adjacency(), ids[0], &cfg).unwrap());
        });
    }

    group.finish();
}

// ── get_node ─────────────────────────────────────────────────────────────────

fn bench_get_node(c: &mut Criterion) {
    let (mut db, _dir) = open_db();
    let ids = populate(&mut db, 100);
    let target = ids[50];

    c.bench_function("graph/get_node", |b| {
        b.iter(|| db.get_node(target).unwrap())
    });
}

// ── criterion wiring ─────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_insert_node,
    bench_scan_nodes,
    bench_poincare_distance,
    bench_bfs,
    bench_dijkstra,
    bench_get_node,
);
criterion_main!(benches);
