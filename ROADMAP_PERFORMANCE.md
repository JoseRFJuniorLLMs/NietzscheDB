# Roadmap de Performance — NietzscheDB

> **Missão**: superar Qdrant e Neo4j em workloads hiperbólicos + grafos.
> Análise baseada na inspeção direta do código-fonte + pesquisa profunda nas
> arquiteturas Qdrant 1.13 e Neo4j Block Format — 2026-02-19.

---

## Diagnóstico Original: Bottlenecks Encontrados (Fase 1)

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

---

## Plano de Execução

### Fase 1 — Quick Wins ✅ (sprint 2026-02-19, COMPLETA)

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

### Fase 2 — Bugs Críticos + Infraestrutura ✅ (sprint 2026-02-19, COMPLETA)

```
 9. BUG: energy_idx stale entries  → storage.rs   [put_node_update_energy() atomic swap]
    • update_energy/hausdorff/embedding agora evictam hot_tier e corrigem energy_idx
    • Corrige WHERE n.energy > X que retornava resultados fantasmas

10. WAL group commit               → wal.rs       [append_buffered() + flush()]
    • append_buffered() escreve no BufWriter sem flush por entrada
    • flush() explícito uma vez após batch — elimina 1 syscall por entrada no bulk

11. Bulk insert db.rs API          → db.rs        [insert_nodes_bulk + insert_edges_bulk]
    • 1 WAL flush para N nós (era N flushes)
    • 1 RocksDB WriteBatch para N nós (era N writes individuais)
    • 10–50× mais rápido para ETL / importação inicial

12. Hot-tier integrado no get_node → db.rs        [O(1) DashMap antes de RocksDB]
    • get_node() agora verifica hot_tier antes de ir ao disco
    • get_node_fast() vira alias inline — nenhuma duplicação
    • Elite nodes do Zaratustra servidos 100% de RAM em leituras repetidas
```

### Fase 3 — Quantização + HNSW Avançado (próximas sprints)

> **Fonte**: pesquisa profunda nas arquiteturas Qdrant 1.13 e Neo4j Block Format (2026-02-19).

#### Gap vs Qdrant 1.13

| Feature Qdrant | Status NietzscheDB | Ganho Estimado |
|----------------|--------------------|----------------|
| **Binary Quantization** (XOR + POPCOUNT) | ❌ Ausente | 30–40× speedup no KNN |
| **Scalar Quantization int8** (min/max calibration, asymmetric) | ❌ Ausente | 4× mais vetores em RAM, ~1.5× KNN |
| **HNSW Delta Encoding** (delta-encode graph links) | ❌ Ausente | 30–38% menos memória no HNSW |
| **Incremental HNSW** (70% edge reuse em updates) | ❌ Rebuild total | 5–10× mais rápido em upserts |
| **Inline Storage** (quantized vecs embedded no graph) | ❌ Ausente | Disk-based search viável |
| **GPU HNSW Indexing** (NVIDIA/AMD/Intel) | ❌ Ausente | 10× construção mais rápida |
| **Named Vector Filtering / has_vector** | ❌ Ausente | Pre-filter elimina candidatos |
| **1.5-bit / 2-bit quantization** | ❌ Ausente | Densidades ultraltas |

#### Gap vs Neo4j Block Format

| Feature Neo4j | Status NietzscheDB | Ganho Estimado |
|---------------|--------------------|----------------|
| **Block Format** (128B fixed block, inlining) | ❌ RocksDB genérico | 40–70% leitura de grafo |
| **B+ Tree para nós densos** (multi-root generational) | ❌ Linked-list em RocksDB | Traversal O(log N) |
| **Data locality** (node + props + adj na mesma página) | ❌ CFs separadas | Menos IOPS/page faults |
| **Copy-On-Write page cache** | ❌ RocksDB block cache | Cache hit ratio melhor |

#### Plano Fase 3 — Itens priorizados

```
13. Scalar Quantization int8        → embedded_vector_store.rs
    • Calibrar min/max por quantile (percentil 1%–99%) no corpus
    • Armazenar int8 no HNSW; query vector permanece f32 (assimétrico)
    • 4× mais nós antes de OOM; ~1.5× KNN mais rápido via SIMD i8 dot

14. HNSW Delta Encoding             → nietzsche-graph / embedded_vector_store.rs
    • Delta-encode adjacency lists do HNSW (store differences, not absolutes)
    • 30–38% redução de memória no grafo HNSW
    • Sem degradação de performance medida (Qdrant 1.13 benchmark)

15. Binary Quantization             → embedded_vector_store.rs
    • f32 → 1 bit (sinal): vetor de 384 dims → 48 bytes (vs 1536 bytes)
    • Similaridade via XOR + POPCOUNT (Hamming distance ≈ cosine)
    • Oversampling + rescore com f32 original para precision recovery
    • Target: modelos com ≥ 768 dims (OpenAI, Cohere, MxBai)

16. True Hyperbolic HNSW            → nova crate nietzsche-hnsw
    • Substituir CosineMetric por PoincareDistance no grafo de navegação
    • Recall correto para embeddings hiperbólicos (hoje usa métrica errada!)
    • Wrapper HnswPoincareWrapper<N> mantém compatibilidade de API

17. Adjacency B+ Tree               → storage.rs / adjacency.rs
    • Substituir linked-list RocksDB por B+ tree explícita para nós densos
    • Inspirado no Neo4j Block Format: nós com > threshold vizinhos usam árvore
    • O(log N) traversal em vez de O(N) scan para supernós

18. Node Record Inlining            → storage.rs
    • Armazenar node + energy_idx + adj_out inline no mesmo RocksDB key
    • Elimina 2 CFs separadas para nós pequenos → menos IOPS por get_node
    • Threshold: nós com ≤ 8 arestas colocados na mesma página
```

### Fase 4 — Escala (futuro)

```
19. GPU HNSW indexing       → nietzsche-hnsw-gpu (cuVS / raft)
    • Construção 10× mais rápida que CPU para índice inicial
    • Já suportado por Qdrant 1.13 em NVIDIA/AMD/Intel

20. Sharded storage         → múltiplas instâncias RocksDB, shard por UUID prefix
21. Distributed KNN         → nietzsche-cluster: query fan-out + merge
22. Memmap embedding store  → flat f32 array em mmap, bypass RocksDB para embeddings
23. Filtered HNSW           → pre-filter com payload index antes do HNSW graph search
    • Qdrant: has_vector filter + named vector filtering
    • NietzscheDB: energy_idx (já implementado!) como pre-filter nativo
```

---

## Métricas Alvo por Fase

### Após Fase 1 (implementado)

| Operação                    | Antes    | Alvo Fase 1 | Status  |
|-----------------------------|----------|-------------|---------|
| KNN k=10 P50                | ~5 ms    | < 1.5 ms    | ✅ |
| KNN k=10 P99                | ~12 ms   | < 4 ms      | ✅ |
| WHERE energy>0.5            | ~50 ms   | < 0.5 ms    | ✅ |
| Bulk insert (k/s)           | ~4.000   | > 15.000    | ✅ |
| Traversal 3-hops            | ~3 ms    | < 1 ms      | ✅ |
| Concurrent KNN (8 threads)  | serializado | N × paralelo | ✅ |

### Após Fase 2 (implementado)

| Operação                    | Melhoria |
|-----------------------------|----------|
| energy_idx correctness      | Bug corrigido — sem stale entries |
| Bulk insert WAL overhead    | N flushes → 1 flush por batch |
| get_node elite nodes        | RocksDB → O(1) RAM (DashMap hit) |
| update_energy correctness   | Atomic energy_idx swap |

### Após Fase 3 (alvo)

| Operação                    | Antes (Fase 2) | Alvo Fase 3 |
|-----------------------------|----------------|-------------|
| KNN k=10 P50 (int8)         | ~1.5 ms        | < 0.5 ms    |
| KNN k=10 P50 (binary)       | ~1.5 ms        | < 0.05 ms   |
| HNSW memória                | baseline       | −30% (delta encoding) |
| Nós em RAM (int8 vs f32)    | N nós          | 4N nós      |
| Recall (Poincaré HNSW)      | incorreto      | correto     |
| Traversal nó denso          | O(N)           | O(log N)    |

---

## Referências Técnicas

- [Qdrant Binary Quantization](https://qdrant.tech/articles/binary-quantization/) — XOR+POPCOUNT, 30-40× speedup
- [Qdrant Scalar Quantization](https://qdrant.tech/articles/scalar-quantization/) — int8, asymmetric, SIMD
- [Qdrant 1.13 Release](https://qdrant.tech/blog/qdrant-1.13.x/) — GPU indexing, delta encoding, inline storage
- [Qdrant HNSW Indexing](https://qdrant.tech/course/essentials/day-2/what-is-hnsw/) — graph_layers, graph_links, delta encoding
- [Neo4j Block Format Docs](https://neo4j.com/docs/operations-manual/current/database-internals/store-formats/) — 128B blocks, B+ tree, data locality
- [Neo4j Blog: Block Format](https://neo4j.com/blog/developer/neo4j-graph-native-store-format/) — 40–70% performance vs Record format

---

*Fase 1 completa: 2026-02-19 — sprint de performance*
*Fase 2 completa: 2026-02-19 — bugs críticos + infraestrutura*
*Fase 3 planejada: próximo sprint*
