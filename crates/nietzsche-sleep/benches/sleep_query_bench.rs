//! Benchmark: query latency before vs. after a Sleep Cycle.
//!
//! Measures whether Riemannian reconsolidation degrades BFS / point-read
//! performance on the graph.
//!
//! Run with:
//! ```bash
//! cargo bench -p nietzsche-sleep --bench sleep_query_bench
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nietzsche_graph::{
    bfs, Edge, EdgeType, MockVectorStore, NietzscheDB, Node, PoincareVector,
    traversal::BfsConfig,
};
use nietzsche_sleep::{SleepConfig, SleepCycle};
use rand::{SeedableRng, rngs::StdRng};
use uuid::Uuid;

fn build_chain(n: usize) -> (NietzscheDB<MockVectorStore>, Vec<Uuid>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut db = NietzscheDB::open(dir.path(), MockVectorStore::default()).unwrap();
    let step = 0.9 / n as f64;
    let ids: Vec<Uuid> = (0..n)
        .map(|i| {
            let node = Node::new(
                Uuid::new_v4(),
                PoincareVector::new(vec![i as f32 * step as f32, 0.0]),
                serde_json::json!({}),
            );
            let id = node.id;
            db.insert_node(node).unwrap();
            id
        })
        .collect();
    for w in ids.windows(2) {
        db.insert_edge(Edge::new(w[0], w[1], EdgeType::Association, 1.0))
            .unwrap();
    }
    (db, ids, dir)
}

fn bench_sleep_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("sleep/query_latency");
    let n = 50usize;
    let cfg = BfsConfig { max_depth: n, ..Default::default() };

    // ── pre-sleep baseline ─────────────────────────────
    let (db_pre, ids_pre, _dir_pre) = build_chain(n);
    let root_pre = ids_pre[0];
    group.bench_with_input(BenchmarkId::new("bfs_pre_sleep", n), &n, |b, _| {
        b.iter(|| bfs(db_pre.storage(), db_pre.adjacency(), root_pre, &cfg).unwrap());
    });

    // ── post-sleep ─────────────────────────────────────
    let (mut db_post, ids_post, _dir_post) = build_chain(n);
    let root_post = ids_post[0];
    let sleep_cfg = SleepConfig::default();
    let mut rng = StdRng::seed_from_u64(42);
    let _report = SleepCycle::run(&sleep_cfg, &mut db_post, &mut rng).unwrap();
    group.bench_with_input(BenchmarkId::new("bfs_post_sleep", n), &n, |b, _| {
        b.iter(|| bfs(db_post.storage(), db_post.adjacency(), root_post, &cfg).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_sleep_query);
criterion_main!(benches);
