# BUG A — Separar Embedding de CF_NODES (NodeMeta Architecture)

**Severidade: CRÍTICO | Impacto: 10–20× get_node() | Risco para Hiperbólico: NENHUM**

---

## Situação atual

O `Node` completo é serializado em bincode no `CF_NODES` do RocksDB:

```
Node serializado (dim=3072):
  embedding.coords:  24.576 bytes  ← Vec<f64> × 3072 — DUPLICADO no HNSW mmap!
  Resto do Node:      ~400 bytes
  TOTAL:             ~25 KB por nó (98% é o embedding)
```

O mesmo embedding já existe em `data_dir/hnsw/` (hyperspace-store mmap). É duplicação total.

---

## O problema

Toda operação `get_node()` — BFS, DFS, NQL filter, sleep, Zaratustra — desserializa 25 KB mesmo que só precise de `energy`, `depth`, `hausdorff_local` (~400 bytes).

**Para 1M nós:** ~25 GB no RocksDB, sendo 24 GB de embeddings redundantes.

---

## Solução proposta

```rust
// CF_NODES armazena apenas NodeMeta (~340 bytes = 73× menor):
struct NodeMeta {
    id:                  Uuid,
    depth:               f32,
    energy:              f32,
    node_type:           NodeType,
    hausdorff_local:     f32,
    lsystem_generation:  u32,
    created_at:          i64,
    content:             serde_json::Value,
    metadata:            HashMap<String, serde_json::Value>,
    // SEM embedding — fica somente no HNSW mmap store
}
```

---

## Impacto por operação

| Operação | Antes | Depois |
|----------|-------|--------|
| `get_node()` (BFS/traversal) | 25 KB desserializado | ~340 bytes |
| `hot_tier` RAM (1M nós elite) | 25 GB | ~340 MB |
| NQL `WHERE n.energy > X` | 25 KB por nó | 340 bytes × N |
| Sleep (precisa do embedding) | igual | igual (lê do HNSW mmap) |

---

## Risco hiperbólico

**NENHUM.** O embedding continua existindo, só muda de storage. A geometria Poincaré e o kernel de distância não são afetados.

---

## Arquivos afetados

| Arquivo | Mudança |
|---------|---------|
| `model.rs` | `Node` → `NodeMeta` (sem `embedding` field) |
| `storage.rs` | Serializar/deserializar `NodeMeta` em CF_NODES; migração de dados |
| `db.rs` | `get_node()` retorna `NodeMeta`; `get_embedding()` lê do HNSW mmap |
| `executor.rs` | Callers de `node.embedding` → novo `get_embedding(id)` |
| `sleep/riemannian.rs` | Adam optimizer: acessa embedding via nova API |
| `nietzsche-lsystem/engine.rs` | Gera coords a partir de embedding existente |
| `embedded_vector_store.rs` | Expor API de leitura de embedding por ID |
| Todos os callers de `node.embedding` | Mudar para API separada |

---

## Dependências antes de implementar

1. **API de leitura de embedding por ID** — `hyperspace-store` não expõe leitura direta sem busca. Precisamos de `fn get_embedding(hnsw_id: u32) -> &[f64]`.
2. **Migração de dados** — CF_NODES existentes têm `Node` completo em bincode. Breaking change no formato.
3. **Mapeamento UUID → HNSW id** — `EmbeddedVectorStore.uuid_to_hnsw: DashMap<Uuid, u32>` já existe mas precisa sobreviver a restarts (persistir o mapeamento).
4. **Implementar ANTES ou JUNTO com ITEM C** — para que a migração de dados aconteça uma única vez.

---

## Decisão de implementação

- **NÃO implementar isolado sem aprovação do comitê** — breaking API, migração de dados, risco de perda de dados se mal executado
- **Sprint dedicado**: estimativa 3–5 dias de engenharia + testes de regressão + plano de migração
- **Ver**: `risco_hiperbolico.md` Seção 1 para análise completa

---

*Registrado em 2026-02-19 — aguardando aprovação do comitê para implementação*

---

---

# ITEM C — f64 → f32 para armazenamento de embeddings

**Severidade: ALTO | Impacto: 2× memória, 2× SIMD | Risco para Hiperbólico: BAIXO**

---

## Situação atual

```rust
pub struct PoincareVector {
    pub coords: Vec<f64>,  // 8 bytes por coordenada
    pub dim: usize,
}
```

Para `dim=3072`: 24.576 bytes por embedding em f64.

---

## Análise para Poincaré ball

- Invariante ‖x‖ < 1.0 → coords ∈ (-1, 1)
- Modelos de ML produzem embeddings em f32 (7 dígitos decimais)
- Poincaré distance usa `acosh` — que precisa de f64 no final para precisão
- Mas os produtos internos (‖u-v‖², ‖u‖², ‖v‖²) podem ser computados em f32 acumulando em f64

---

## Risco específico — Nós na fronteira

Nós com ‖x‖ > 0.95 (nodes episódicos). A distância hiperbólica **explode** próximo à fronteira:

```
(1 - ‖u‖²) → 0  quando ‖u‖ → 1
```

Perda de precisão f32 pode causar distâncias erradas nesses nós.

**Exemplo numérico:**
```
‖x‖ = 0.99 → (1 - 0.99²) = 0.0199
Erro f32 em uma coord = 0.008 →
(1 - (0.99+0.008)²) = 0.0040   → diferença de 5× no denominador!
```

---

## Mitigação — f32 storage + f64 compute

Armazena em f32, promove para f64 antes do kernel de distância:

```rust
// Armazena Vec<f32>, computa em f64
fn distance_f32_coords(u: &[f32], v: &[f32]) -> f64 {
    let (diff_sq, nu, nv) = poincare_sums_f32_to_f64(u, v);
    // resto igual ao kernel atual
}

fn poincare_sums_f32_to_f64(u: &[f32], v: &[f32]) -> (f64, f64, f64) {
    let mut diff_sq   = 0.0f64;
    let mut norm_u_sq = 0.0f64;
    let mut norm_v_sq = 0.0f64;
    for i in 0..u.len() {
        let a = u[i] as f64;  // promoção f32 → f64 aqui
        let b = v[i] as f64;
        let d = a - b;
        diff_sq   += d * d;
        norm_u_sq += a * a;
        norm_v_sq += b * b;
    }
    (diff_sq, norm_u_sq, norm_v_sq)
}
```

AVX2: `_mm256_cvtps_pd` converte 4× f32 → 4× f64 por instrução. Overhead mínimo.

---

## Impacto

| Métrica | Antes (f64) | Depois (f32 storage + f64 compute) |
|---------|-------------|-------------------------------------|
| Memória por vetor (3072-dim) | 24.576 bytes | 12.288 bytes |
| Redução de memória | — | **2×** |
| SIMD na acumulação | 4 f64/ciclo (AVX2) | 8 f32/ciclo durante load/accumulate |
| Precisão da distância final | f64 nativa | f64 (promovido antes do kernel) |
| Risco boundary nodes (‖x‖ > 0.95) | — | Baixo com mitigação acima |

---

## Arquivos afetados

| Arquivo | Mudança |
|---------|---------|
| `model.rs` | `Vec<f64>` → `Vec<f32>` em `PoincareVector` |
| `model.rs` | Novo kernel `poincare_sums_f32_to_f64()` |
| `storage.rs` | Migração bincode (breaking: dados existentes) |
| `sleep/riemannian.rs` | Adam optimizer usa coords — revisar precisão |
| `nietzsche-lsystem/engine.rs` | Gera coordenadas novas — ajustar tipo |
| `embedded_vector_store.rs` | HNSW insert: `&vector.coords` muda para `&[f32]` |
| `nietzsche-hyp-ops` | Operações hiperbólicas (Möbius, exp map, log map) |

---

## Dependências antes de implementar

1. Verificar se `hyperspace-index::HnswIndex` aceita `&[f32]` (provavelmente sim, converte internamente)
2. Migração de dados: CF_NODES existentes têm `Vec<f64>` serializado em bincode
3. Sleep cycle / Adam optimizer: precisa acumular gradientes em f64 mas pode armazenar pesos em f32
4. Garantir que `project_into_ball()` e `is_valid()` funcionam com f32

---

## Decisão de implementação

- **PODE implementar isolado** — não depende do item A (separar embedding de CF_NODES)
- **Mas**: se item A for implementado depois, a migração de dados ocorre apenas uma vez
- **Recomendação**: implementar C depois de A para fazer apenas uma migração de dados

---

*Registrado em 2026-02-19 — aguardando aprovação para implementação*
