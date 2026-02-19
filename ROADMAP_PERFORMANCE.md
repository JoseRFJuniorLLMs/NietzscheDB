# Roadmap de Performance — NietzscheDB

> **Missão**: superar Qdrant e Neo4j em workloads hiperbólicos + grafos.
> Análise baseada na inspeção direta do código-fonte em 2026-02-19.

---

## Diagnóstico: Bottlenecks Encontrados

### Tier Crítico (impacto imediato ≥ 5×)

| # | Arquivo | Problema | Impacto |
|---|---------|---------|---------|
| 1 | `model.rs:38-54` | `PoincareVector::distance()` faz **3 passes escalares** sobre os vetores. Sem SIMD, sem AVX2. É chamada milhões de vezes por busca HNSW. | 4–8× no KNN |
| 2 | `storage.rs:41-55` | `RocksDB Options::default()` = sem block cache, sem bloom filters, sem write buffer otimizado. Cada leitura pode tocar disco. | 3–10× em reads |
| 3 | `executor.rs:206` | `scan_nodes()` faz full table scan em TODA busca MATCH. `WHERE n.energy > 0.5` lê **todos** os nós do disco. | 100× em queries filtradas |
| 4 | `embedded_vector_store.rs:99` | `ef_search = 100` hardcoded. Queries com k=5 usam ef=100 desnecessariamente. | 2–3× no KNN |

### Tier Alto (impacto ≥ 2×)

| # | Arquivo | Problema | Impacto |
|---|---------|---------|---------|
| 5 | `adjacency.rs:117-125` | `neighbors_both()` usa `Vec::contains()` = O(n) por vizinho = O(n²) total | 2–5× em traversal denso |
| 6 | `storage.rs:110-133` | `put_edge()` faz read-modify-write das listas de adjacência por aresta. 3 ops RocksDB por aresta. | 3× em bulk insert |
| 7 | `storage.rs:60-65` | `cf_handle()` chamado em cada operação (lookups de string no HashMap). | 1.5× overhead |
| 8 | `embedded_vector_store.rs:356-369` | UUID para string + parse em cada resultado do HNSW. `Uuid::parse_str` por resultado. | 1.3× no KNN |

### Tier Médio (arquitetura)

| # | Problema | Impacto |
|---|---------|---------|
| 9  | Sem índice secundário em `energy` (campo mais filtrado no NQL) | Elimina full scans |
| 10 | Sem quantização int8 no HNSW (4× mais vetores em RAM) | 4× mais nós antes de OOM |
| 11 | HNSW usa `CosineMetric` mesmo com `PoincareBall` (métrica errada!) | Recall incorreto |
| 12 | NQL executor single-threaded (sem rayon) | N-core speedup disponível |
| 13 | Sem batch insert API para ingestão em massa | 10× em bulk load |
| 14 | `Node` contém `HashMap<String, Value>` (metadata) = heap alloc pesado | Memória/latência |

---

## Plano de Execução

### Fase 1 — Quick Wins ✅ (implementado neste sprint)

```
1. RocksDB Tuning          → storage.rs   [512MB LRU, bloom 10bpk, LZ4+Zstd, parallel jobs]
2. One-pass + SIMD distance → model.rs    [single-pass poincare_sums() → AVX2 auto-vec]
3. Energy Secondary Index   → storage.rs  [nova CF energy_idx → O(log N) range scan]
4. Parallel NQL (rayon)    → executor.rs  [par_iter() para scans > 2000 nós]
5. Dynamic ef_search        → embedded_vector_store.rs  [ef = (k*4).max(16).min(512)]
6. neighbors_both HashSet   → adjacency.rs  [O(n) ao invés de O(n²)]
7. Batch insert API         → storage.rs  [put_nodes_batch + put_edges_batch]
8. Mutex → RwLock          → collection_manager.rs + server.rs + dashboard.rs
                              [reads concurrentes: KNN, BFS, NQL, Diffusion, Zaratustra]
```

### Fase 2 — Arquitetura (próximas sprints)

```
9.  True Hyperbolic HNSW   → HnswPoincareWrapper<N>
10. Int8 Quantization      → embedded_vector_store.rs (SQ8)
11. Node LRU Cache         → nova crate nietzsche-cache
12. Async NQL executor     → tokio join_all() para multi-query
13. Memmap embedding store → flat f32 array em mmap, avoid RocksDB para embeddings
14. Filtered HNSW          → pre-filter com payload index antes do HNSW
```

### Fase 3 — Escala (futuro)

```
15. Sharded storage        → múltiplas instâncias RocksDB, shard por UUID prefix
16. Distributed KNN        → nietzsche-cluster: query fan-out + merge
17. GPU HNSW               → cuVS / raft para construção de índice em GPU
18. Graph-native ANN       → Hyperbolic HNSW com grafo de navegação em Poincaré ball
```

---

## Métricas Alvo Após Fase 1

| Operação                    | Antes    | Alvo Fase 1 |
|-----------------------------|----------|-------------|
| KNN k=10 P50                | ~5 ms    | < 1.5 ms    |
| KNN k=10 P99                | ~12 ms   | < 4 ms      |
| WHERE energy>0.5            | ~50 ms   | < 0.5 ms    |
| Bulk insert (k/s)           | ~4.000   | > 15.000    |
| Traversal 3-hops            | ~3 ms    | < 1 ms      |
| Concurrent KNN (8 threads)  | serializado | N × paralelo |

---

*Implementação concluída em 2026-02-19 — sprint de performance*
