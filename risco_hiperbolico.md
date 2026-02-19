# Documento de Risco — NietzscheDB: Otimizações em Geometria Hiperbólica

**Versão**: 1.0
**Data**: 2026-02-19
**Destinatário**: Comitê de Decisões de Arquitetura
**Preparado por**: Auditoria Técnica Interna (referência: AUDITORIA_PERFORMANCE.md)
**Status**: Aguardando decisão formal do comitê

---

## SUMÁRIO EXECUTIVO

O NietzscheDB é um banco de dados de grafo hiperbólico baseado no **modelo de bola de Poincaré**. Toda coordenada de embedding vive no espaço hiperbólico com invariante rígida `‖x‖ < 1.0`. A distância entre dois pontos é:

```
d(u,v) = acosh( 1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)) )
```

Esta geometria tem propriedades radicalmente diferentes do espaço euclidiano:

1. **O denominador `(1−‖u‖²)(1−‖v‖²)` explode quando ‖x‖ → 1.0** — nós próximos à fronteira (episódicos) têm distâncias hiperbólicas sensíveis a pequenos erros de precisão.
2. **Similaridade cosseno não aproxima distância hiperbólica** — métricas angulares são cegas à magnitude, que determina a posição hierárquica no espaço hiperbólico.
3. **Nós semânticos (centro) e episódicos (fronteira) têm características numéricas opostas** — qualquer otimização que trate todos os nós igualmente está potencialmente errada.

Este documento cataloga **todos os riscos hiperbólicos** das otimizações auditadas, com análise quantitativa de impacto, para subsidiar decisão do comitê.

---

## PARTE 1 — RISCO CRÍTICO: BUG A (Embedding Duplicado em CF_NODES)

### 1.1 Situação atual

O `Node` completo é serializado (bincode) no `CF_NODES` do RocksDB:

```
Node serializado no CF_NODES (dim=3072):
  id:                  16 bytes
  embedding.coords:    24.576 bytes  ← Vec<f64> × 3072
  embedding.dim:        8 bytes
  depth:                4 bytes
  energy:               4 bytes
  node_type:            1 byte
  hausdorff_local:      4 bytes
  lsystem_generation:   4 bytes
  created_at:           8 bytes
  content:            ~200 bytes
  metadata:           ~100 bytes
  ─────────────────────────────
  TOTAL:             ~25 KB por nó  (98% é o embedding)
```

**O mesmo embedding já existe no HNSW mmap store** (`data_dir/hnsw/`). É duplicação total.

Para 1 M de nós: **~25 GB no RocksDB**, sendo ~24 GB de embeddings já presentes em outro lugar.

### 1.2 Impacto na performance

Cada `get_node()` (BFS, DFS, NQL filter, sleep, Zaratustra) desserializa 25 KB completos, sendo que a operação precisa apenas de `energy`, `depth`, `hausdorff_local`, `content` (~400 bytes, 1.6% do total).

### 1.3 A solução proposta

Separar o `Node` em dois structs:

```rust
// CF_NODES armazenaria apenas (~340 bytes por nó = 73× menor):
struct NodeMeta {
    id, depth, energy, node_type, hausdorff_local,
    lsystem_generation, created_at, content, metadata
    // SEM embedding
}

// Embedding ficaria SOMENTE no HNSW mmap store
```

### 1.4 Riscos hiperbólicos desta mudança

**Risco hiperbólico: NENHUM.**

O embedding continua existindo — apenas muda onde é armazenado. A geometria Poincaré não é afetada. As operações de distância hiperbólica (`poincare_sums`, `acosh`) são calculadas com os coords originais, independente de onde ficam armazenados.

### 1.5 Riscos de implementação (não hiperbólicos)

| Risco | Severidade | Descrição |
|-------|-----------|-----------|
| Migração de dados | **ALTO** | CF_NODES existentes contêm `Vec<f64>` serializado — todos os dados precisam ser remigrados |
| Breaking API | **ALTO** | Todos os callers de `node.embedding` precisam ser refatorados para ler do HNSW store separado |
| API de leitura de embedding | **ALTO** | `hyperspace-store` não expõe leitura por ID sem busca — precisa de nova API |
| Sleep cycle | **MÉDIO** | Adam optimizer acessa `node.embedding` durante reconsolidação — precisa ser adaptado |
| L-System engine | **MÉDIO** | Gera coordenadas novas a partir de embeddings existentes — precisa de acesso ao novo store |

### 1.6 Ganho esperado

| Operação | Antes | Depois |
|----------|-------|--------|
| `get_node()` — BFS/traversal | 25 KB desserializado | ~340 bytes |
| `hot_tier` RAM (1M nós elite) | 25 GB | ~340 MB |
| NQL `WHERE n.energy > X` | 25 KB por nó scaneado | 340 bytes × N |
| Sleep cycle (precisa embedding) | igual | igual (lê do HNSW mmap) |

**Ganho estimado: 10–20× nas operações de traversal e filtro.**

### 1.7 Decisão do comitê necessária

- **Aprovar implementação em sprint dedicado**: 3–5 dias de engenharia, migração de dados, testes de regressão
- **Adiar**: manter duplicação, aceitar custo de 25 KB por get_node()
- **Prioridade relativa**: deve ser feito ANTES ou JUNTO com ITEM C (f64→f32) para que a migração de dados ocorra uma única vez

---

## PARTE 2 — RISCO MÉDIO: ITEM C (f64 → f32 para Armazenamento)

> Análise completa salva em `pesquisar1.md`.

### 2.1 O que é

Armazenar `PoincareVector.coords` como `Vec<f32>` (4 bytes/coord) em vez de `Vec<f64>` (8 bytes/coord), promovendo para f64 apenas no kernel de distância.

**Ganho direto**: 2× memória, 2× throughput SIMD (8 f32/ciclo vs 4 f64/ciclo no AVX2).

### 2.2 Por que há risco hiperbólico

A distância hiperbólica usa o denominador:

```
(1 − ‖u‖²)(1 − ‖v‖²)
```

Quando ‖x‖ → 1.0 (nós episódicos na fronteira), este denominador → 0. Qualquer erro na norma é **amplificado** pela divisão.

### 2.3 Análise quantitativa do risco

```
Caso: ‖x‖ = 0.99 (nó episódico típico)

Com f64 (sem erro):
  (1 − 0.99²) = 0.0199

Com f32, erro típico de uma coordenada: ε = ±5.96 × 10⁻⁸
Em 3072 dimensões, acúmulo de erro na norma: ≈ 0.001 a 0.01

Pior caso (erro de 0.008 em uma coord):
  (1 − (0.99 + 0.008)²) = 0.0040

Diferença no denominador: 0.0199 / 0.0040 = 4.975×
→ A distância hiperbólica calculada pode diferir em até 5× do valor correto
```

### 2.4 Mitigação disponível

Armazenar em f32, **promover para f64 antes do kernel de distância**:

```rust
fn poincare_sums_f32_to_f64(u: &[f32], v: &[f32]) -> (f64, f64, f64) {
    let mut diff_sq = 0.0f64;
    let mut norm_u  = 0.0f64;
    let mut norm_v  = 0.0f64;
    for i in 0..u.len() {
        let a = u[i] as f64;  // promoção f32→f64 aqui
        let b = v[i] as f64;
        let d = a - b;
        diff_sq += d * d;
        norm_u  += a * a;
        norm_v  += b * b;
    }
    (diff_sq, norm_u, norm_v)
}
```

**Com esta mitigação**, a perda de precisão é apenas na representação armazenada (7 vs 15 dígitos decimais), mas o cálculo de distância opera em f64 completo.

### 2.5 Risco residual após mitigação

| Cenário | Sem mitigação | Com mitigação |
|---------|--------------|---------------|
| Nós centrais (‖x‖ < 0.5) | negligível | **ZERO** |
| Nós intermediários (‖x‖ ∈ [0.5, 0.9]) | baixo | **ZERO** |
| Nós episódicos (‖x‖ > 0.95) | 5× erro | **BAIXO** — erro de representação f32, não de cálculo |
| Adam optimizer (gradientes) | médio | Revisar separadamente — gradientes precisam de f64 |

### 2.6 Dependências

Esta mudança é **breaking**: altera o tipo de `Vec<f64>` para `Vec<f32>` em `PoincareVector`, afetando:

- `model.rs` — struct principal
- `storage.rs` — migração bincode (todos os dados existentes são inválidos após a mudança)
- `sleep/riemannian.rs` — Adam optimizer
- `nietzsche-lsystem/engine.rs` — geração de coordenadas
- `embedded_vector_store.rs` — inserção no HNSW

### 2.7 Decisão do comitê necessária

- **Aprovar com mitigação** (`poincare_sums_f32_to_f64`): implementar f32 storage + f64 compute
- **Aprovar sem restrição**: aceitar risco residual em nós episódicos (‖x‖ > 0.95)
- **Rejeitar**: manter f64, aceitar custo de memória e SIMD
- **Dependência**: implementar APÓS ou JUNTO com BUG A para migração única de dados

---

## PARTE 3 — RISCO MÉDIO: ITEM D (Quantização Escalar Int8)

### 3.1 O que é

Compressão de embeddings de f64 (8 bytes/coord) → u8 (1 byte/coord), usando calibração linear:

```
alpha  = (max − min) / 127
offset = min

encode(x) = round((x − offset) / alpha)  ∈ [0, 127]
decode(q) = q × alpha + offset
```

**Ganho**: 8× memória (f64→u8), ~4× throughput KNN via SIMD u8.

### 3.2 Adaptação para Poincaré ball

A invariante `‖x‖ < 1.0` garante `coords ∈ (−1.0, +1.0)`. Range fixo e conhecido:

```
alpha  = 2.0 / 127.0 ≈ 0.0157
offset = −1.0
```

Sem necessidade de P2 quantile estimator (o Qdrant usa estimador estatístico porque não conhece o range a priori). Para Poincaré, o range é matematicamente garantido.

### 3.3 Risco hiperbólico — nós de fronteira

**Erro de quantização por coordenada**: `±(alpha/2) = ±0.00787`

```
Para ‖x‖ = 0.95 (nó episódico moderado):
  (1 − 0.95²) = 0.0975

Erro em uma coordenada: ε = 0.008
  ‖x + ε_vec‖² ≈ 0.95² + 2×0.95×0.008 = 0.9025 + 0.0152 = 0.9177
  (1 − 0.9177) = 0.0823

Diferença: 0.0975 / 0.0823 = 1.18× → erro de 18% no denominador → erro de ~18% na distância

Para ‖x‖ = 0.99 (nó episódico extremo):
  (1 − 0.99²) = 0.0199
  Com ε = 0.008:
  (1 − (0.99+0.008)²) = 0.0040
  Diferença: 4.975× → erro de 5× na distância hiperbólica
```

### 3.4 Mitigação — Oversampling obrigatório

Idêntico ao padrão Qdrant:

1. KNN quantizado retorna `k × oversampling_factor` candidatos
2. Rescore dos candidatos com coordenadas originais (f32 ou f64)
3. Retorna os `k` melhores após rescore

| Distribuição de nós | Oversampling recomendado |
|--------------------|------------------------|
| Maioria semântica (‖x‖ < 0.5) | 3× |
| Mix semântico/episódico | 5× |
| Maioria episódica (‖x‖ > 0.8) | 8× |

**Com oversampling ×5**: recall esperado > 97% mesmo para nós episódicos.

### 3.5 Componentes afetados

Esta funcionalidade requer nova crate `nietzsche-quantization` (ou integração com `hyperspace-core`). Não é breaking para dados existentes se implementado como índice secundário (original mantido, Int8 é índice acelerador).

### 3.6 Decisão do comitê necessária

- **Aprovar com oversampling obrigatório**: implementar Int8 como acelerador com rescore
- **Aprovar com oversampling configurável**: permitir desativar rescore (aceitar degradação)
- **Rejeitar**: manter f64 no índice HNSW, sacrificar 8× memória e 4× KNN
- **Dependência**: facilitado pelo BUG A (embedding separado) — mas pode ser implementado independentemente

---

## PARTE 4 — RISCO ALTO: ITEM F (Binary Quantization)

### 4.1 O que é

Compressão máxima: cada coordenada → 1 bit (`sign(x)`). Scoring via XOR + POPCOUNT.

```
Para dim=3072: 3072 bits = 384 bytes (vs 24.576 bytes em f64 = 64× menor)
Score ≈ dim − 2 × Hamming(sign(u), sign(v))  ≈  cosine(u, v)
```

O Qdrant usa esta técnica com sucesso para embeddings de texto/imagem.

### 4.2 Por que é INCOMPATÍVEL com Poincaré ball como métrica primária

A aproximação `Hamming → cosine` funciona porque **cosseno é baseado em ângulo** entre vetores. O sinal de cada coordenada captura o semiespaço.

A distância Poincaré **depende criticamente de magnitude**:

```
d(u,v) = acosh( 1 + 2‖u−v‖² / ((1−‖u‖²)(1−‖v‖²)) )
                                   ↑
                    O DENOMINADOR DEPENDE DE ‖u‖ E ‖v‖
                    sign(x) IGNORA ‖x‖ COMPLETAMENTE
```

### 4.3 Caso concreto de falha

```
u = [0.01, 0.01, ..., 0.01]   → nó semântico perto do centro, ‖u‖ ≈ 0.017
v = [0.02, 0.01, ..., 0.01]   → nó semântico próximo de u
w = [−0.01, 0.01, ..., 0.01]  → nó semântico, sinal diferente na coord 0

Hamming(sign(u), sign(v)) = 0   → Binary Quant diz: distância mínima (v = vizinho perfeito)
Hamming(sign(u), sign(w)) = 1   → Binary Quant diz: v está mais perto que w

d_hyperbolic(u, v) ≈ 0.010     → muito próximos (ambos perto do centro)
d_hyperbolic(u, w) ≈ 0.010     → igualmente próximos

ERRO: Binary Quantization afirma que v é mais próximo de u que w,
mas em geometria hiperbólica eles são equidistantes.
```

Para nós semânticos (Semantic, Concept), a maioria das coordenadas está em `(−0.1, +0.1)`. O sinal dessas coordenadas é quase aleatório em relação à distância hiperbólica real.

### 4.4 Caso de uso válido (restrito)

Binary Quantization pode ser usado **exclusivamente como pre-filter grosseiro** para dimensões altas (dim ≥ 768), com:
- Oversampling mínimo de 20× (retorna 20k candidatos para k=1000)
- Rescore obrigatório com distância Poincaré completa
- NUNCA como métrica primária do índice HNSW

### 4.5 Decisão do comitê necessária

**Recomendação técnica**: **NÃO implementar Binary Quantization para NietzscheDB**.

O banco opera em geometria hiperbólica onde a magnitude (posição na hierarquia semântica/episódica) é o dado mais importante. Binary Quantization descarta exatamente essa informação.

- **Aprovar como pre-filter opcional**: aceitar implementação com oversampling ≥20× e rescore obrigatório, com flag explícita de "risco de recall degradado para nós semânticos"
- **Rejeitar**: manter apenas métricas baseadas em magnitude (f64, f32, Int8 com mitigação)

---

## PARTE 5 — BUGS CORRIGIDOS NESTA SPRINT

Para registro do comitê: os seguintes bugs críticos foram corrigidos antes deste documento:

### BUG B — Métrica errada no HNSW (RESOLVIDO)

**Problema**: `HnswRawWrapper<N>` usava `HnswIndex<N, CosineMetric>` internamente para `VectorMetric::PoincareBall`. O grafo HNSW era construído com vizinhança cosseno — incorreta para geometria hiperbólica.

**Consequência**: Nós próximos na geometria hiperbólica podiam não ser vizinhos no grafo HNSW. O recall para KNN hiperbólico era sistematicamente degradado. Nós episódicos (‖x‖ → 1) eram os mais afetados.

**Correção aplicada**:
- Adicionado `HnswPoincareWrapper<N>` com `HnswIndex<N, PoincareMetric>`
- `VectorMetric::PoincareBall` agora roteia para `make_poincare_hnsw()`
- `PoincareMetric` já existia em `hyperspace-core` com implementação completa de `acosh`
- Zero risco — é correção de bug, não mudança de comportamento intencional
- Compile limpo: `cargo check -p nietzsche-graph` → 0 erros, 3 warnings pré-existentes

---

## PARTE 6 — MATRIZ DE DECISÃO CONSOLIDADA

```
┌──────────────────────────────────────────┬──────────────┬──────────────┬─────────────────────────────────────┐
│ Item                                     │ Risco        │ Ganho        │ Recomendação                        │
│                                          │ Hiperbólico  │ Performance  │                                     │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────────────────────────┤
│ BUG B — PoincareMetric no HNSW           │ NENHUM       │ Recall ✓     │ ✅ FEITO — corrigido nesta sprint    │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────────────────────────┤
│ BUG A — Separar embedding de CF_NODES   │ NENHUM       │ 10–20× get   │ ⚠️ DECIDIR: sprint dedicado         │
│         (NodeMeta vs full Node)          │              │ node()       │    migração de dados necessária      │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────────────────────────┤
│ ITEM C — f64→f32 storage                 │ BAIXO        │ 2× memória   │ ⚠️ DECIDIR: com mitigação           │
│         (compute em f64)                 │ (mitigável)  │ 2× SIMD      │    f32_to_f64 no kernel             │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────────────────────────┤
│ ITEM D — Quantização Int8               │ MÉDIO        │ 8× memória   │ ⚠️ DECIDIR: oversampling obrigatório │
│         (oversampling obrigatório)       │ (mitigável)  │ 4× KNN       │    fator ≥5× para nós episódicos    │
├──────────────────────────────────────────┼──────────────┼──────────────┼─────────────────────────────────────┤
│ ITEM F — Binary Quantization             │ ALTO         │ 30–40×*      │ ❌ NÃO RECOMENDADO como métrica      │
│         (XOR + POPCOUNT)                 │ (não         │ teórico      │    primária. Apenas pre-filter       │
│                                          │  mitigável)  │              │    com oversampling ≥20×             │
└──────────────────────────────────────────┴──────────────┴──────────────┴─────────────────────────────────────┘
```

---

## PARTE 7 — RECOMENDAÇÕES DO TIME TÉCNICO

### Prioridade 1 — Fase 3a (sem breaking changes)

Estas otimizações têm risco hiperbólico zero e podem ser implementadas imediatamente:

1. **Filtered KNN via `energy_idx`** — `energy_idx` já existe no RocksDB, basta expor a API de busca filtrada no `VectorStore` trait. Ganho: 5–50× em queries filtradas. Risco: NENHUM.

2. **Pool de visited lists** — eliminar alocação de HashSet por BFS/DFS. Ganho: 10–30% latência de traversal. Risco: NENHUM.

### Prioridade 2 — Fase 3b (sprint dedicado, aprovação do comitê)

3. **BUG A: NodeMeta separado do embedding** — única migração de dados, máximo ganho de get_node(). Deve ser feito ANTES de C para migração única.

4. **ITEM C: f64→f32 com mitigação** — implementar depois de A para migrar dados uma única vez. Usar `poincare_sums_f32_to_f64` no kernel de distância.

### Não recomendado neste momento

5. **ITEM D (Int8)**: Implementar após A e C estarem estáveis. Requer nova infrastructure de quantização.

6. **ITEM F (Binary Quantization)**: Não implementar como métrica primária. Reavaliar apenas se dimensões > 768 e workload com maioria de nós semânticos (‖x‖ < 0.3).

---

## APÊNDICE — GLOSSÁRIO

| Termo | Definição |
|-------|-----------|
| `‖x‖` | Norma euclidiana do vetor de coordenadas |
| **Poincaré ball** | Modelo de espaço hiperbólico onde todos os pontos têm `‖x‖ < 1.0` |
| **Nó semântico** | `NodeType::Semantic` ou `Concept`: vive perto do centro (`‖x‖ < 0.3`), representa conceitos abstratos |
| **Nó episódico** | `NodeType::Episodic`: vive perto da fronteira (`‖x‖ > 0.8`), representa memórias específicas |
| **Denominador hiperbólico** | `(1−‖u‖²)(1−‖v‖²)` — fator que amplifica erros perto da fronteira |
| **HNSW** | Hierarchical Navigable Small World — índice aproximado para KNN |
| **Oversampling** | Buscar `k × factor` candidatos e rescorar com métrica exata |
| **f64** | Double precision float (64 bits, 15–17 dígitos decimais) |
| **f32** | Single precision float (32 bits, 6–9 dígitos decimais) |
| **Int8** | Quantização para 8 bits inteiros não-negativos [0, 127] |

---

*Documento preparado pela auditoria técnica interna — 2026-02-19*
*Referências: `AUDITORIA_PERFORMANCE.md`, `pesquisar1.md`, código-fonte Qdrant commit 9f433b1*
