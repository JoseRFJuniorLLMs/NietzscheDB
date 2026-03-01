# NietzscheDB Dashboard — Análise de Cobertura

**Data:** 2026-02-21
**Escopo:** Cruzamento dos 38 crates vs. Dashboard React 19 + Cosmograph 2.1

---

## 1. Resumo Executivo

| Métrica                              | Valor   |
|--------------------------------------|---------|
| Total de crates no workspace         | 38      |
| Crates **visíveis** no dashboard     | 8       |
| Crates **parcialmente** visíveis     | 4       |
| Crates **invisíveis** no dashboard   | 26      |
| Endpoints REST disponíveis no backend| 40+     |
| Endpoints consumidos pelo dashboard  | 12      |
| **Cobertura atual do dashboard**     | **~21%**|

---

## 2. O que o Dashboard MOSTRA Hoje

### 6 Páginas ativas:

| Página           | Rota        | O que mostra                                                              |
|------------------|-------------|---------------------------------------------------------------------------|
| **Overview**     | `/`         | Status online/offline, Total Vectors, RAM, Disk, CPU, Collections, Uptime, Version, Metric Space, Quantization |
| **Collections**  | `/collections` | Tabela de coleções (nome, dimensão, métrica, vetores, fila), CRUD         |
| **Data Explorer**| `/explorer` | Raw data (últimos 50 vetores), Search Playground (k-NN top-5)            |
| **Graph Explorer**| `/graph`   | Cosmograph com nós/arestas, filtros por tipo, energy, depth, hausdorff, timeline |
| **Nodes**        | `/nodes`    | Topologia Leader/Follower, Lamport clock, peers conectados                |
| **Settings**     | `/settings` | Snippets de integração (Python/cURL/Node.js), Logs ao vivo, WAL download |

---

## 3. Cruzamento: 38 Crates vs. Dashboard

### ✅ VISÍVEL — Crate refletido no dashboard

| # | Crate                    | Onde aparece no dashboard                             | Cobertura |
|---|--------------------------|-------------------------------------------------------|-----------|
| 1 | `nietzsche-graph`        | Graph Explorer (nós, arestas, tipos, energy, depth)   | Alta      |
| 2 | `nietzsche-hyp-ops`      | Graph Explorer (depth = posição Poincaré, hausdorff)  | Parcial   |
| 3 | `nietzsche-api`          | Todos endpoints REST consumidos                       | Parcial   |
| 4 | `nietzsche-filtered-knn` | Data Explorer → Search Playground (k-NN)              | Parcial   |
| 5 | `nietzsche-named-vectors`| Collections Page (dimensão, métrica)                  | Parcial   |
| 6 | `nietzsche-pq`           | Overview → campo "Quantization: I8"                   | Mínima    |
| 7 | `nietzsche-secondary-idx`| Data Explorer → metadata display                      | Mínima    |
| 8 | `nietzsche-*` (9 crates)| Base storage refletido em stats/health                | Indireta  |

### ⚠️ PARCIALMENTE VISÍVEL — Backend tem endpoint, dashboard NÃO consome

| # | Crate                    | Endpoint REST existe?       | Dashboard mostra? | O que falta                          |
|---|--------------------------|-----------------------------|-------------------|--------------------------------------|
| 9 | `nietzsche-algo`         | `/api/algo/*` (10 algoritmos) | **NÃO**          | Página inteira de algoritmos         |
| 10| `nietzsche-sleep`        | `POST /api/sleep`           | **NÃO**          | Trigger + visualização do ciclo      |
| 11| `nietzsche-agency`       | `/api/agency/*` (8 endpoints) | **NÃO**         | Página de agência autônoma           |
| 12| `nietzsche-query`        | `POST /api/query` (NQL)     | **NÃO**          | Console NQL interativo               |

### ❌ INVISÍVEL — Sem endpoint REST e sem UI no dashboard

| #  | Crate                    | Funcionalidade                                    | Tem gRPC? |
|----|--------------------------|---------------------------------------------------|-----------|
| 13 | `nietzsche-lsystem`      | Crescimento fractal L-System                      | Não       |
| 14 | `nietzsche-pregel`       | Difusão heat kernel hiperbólico (Chebyshev)       | Via gRPC  |
| 15 | `nietzsche-zaratustra`   | Evolução autônoma (Vontade de Poder, Übermensch)  | Via gRPC  |
| 16 | `nietzsche-sensory`      | Compressão multi-modal (f32→f16→int8→PQ→gone)     | Via gRPC  |
| 17 | `nietzsche-dream`        | Queries especulativas com ruído estocástico       | Não       |
| 18 | `nietzsche-wiederkehr`   | Agentes DAEMON de patrulha                        | Não       |
| 19 | `nietzsche-narrative`    | Detecção de arcos narrativos                      | Via REST  |
| 20 | `nietzsche-hnsw-gpu`     | NVIDIA cuVS CAGRA (GPU)                           | Não       |
| 21 | `nietzsche-tpu`          | Google PJRT (TPU v5e/v6e/v7)                      | Não       |
| 22 | `nietzsche-cugraph`      | Travessia de grafo em GPU                         | Não       |
| 23 | `nietzsche-mcp`          | 19 tools MCP para AI assistants                   | N/A       |
| 24 | `nietzsche-kafka`        | CDC via Kafka Connect                             | Via gRPC  |
| 25 | `nietzsche-table`        | Camada relacional SQLite                          | Não       |
| 26 | `nietzsche-media`        | Armazenamento de mídia via OpenDAL                | Não       |
| 27 | `nietzsche-sdk`          | SDKs (Go, Python, TypeScript, C++)                | N/A       |

---

## 4. Endpoints Backend Não Consumidos pelo Dashboard

```
REST Endpoints SEM UI:
──────────────────────────────────────────────────────────────
POST   /api/node              ← CRUD individual de nós
GET    /api/node/:id
DELETE /api/node/:id
POST   /api/edge              ← CRUD individual de arestas
DELETE /api/edge/:id
POST   /api/batch/nodes       ← Bulk insert
POST   /api/batch/edges
POST   /api/query             ← NQL (19 tipos de query!)
GET    /api/search            ← Full-text search
POST   /api/sleep             ← Trigger ciclo de sono
GET    /api/algo/pagerank     ← PageRank
GET    /api/algo/louvain      ← Detecção de comunidades
GET    /api/algo/labelprop    ← Propagação de labels
GET    /api/algo/betweenness  ← Centralidade betweenness
GET    /api/algo/closeness    ← Centralidade closeness
GET    /api/algo/degree       ← Centralidade de grau
GET    /api/algo/wcc          ← Componentes fracamente conexos
GET    /api/algo/scc          ← Componentes fortemente conexos
GET    /api/algo/triangles    ← Contagem de triângulos
GET    /api/algo/jaccard      ← Similaridade Jaccard
POST   /api/backup            ← Criar backup
GET    /api/backup            ← Listar backups
GET    /api/export/nodes      ← Exportar nós (CSV/JSONL)
GET    /api/export/edges      ← Exportar arestas (CSV/JSONL)
GET    /api/agency/health     ← Relatórios de saúde autônoma
GET    /api/agency/health/latest
GET    /api/agency/counterfactual/remove/:id  ← Motor contrafactual
POST   /api/agency/counterfactual/add
GET    /api/agency/desires    ← Sinais de desejo pendentes
POST   /api/agency/desires/:id/fulfill
GET    /api/agency/observer   ← Meta-nó Observer Identity
GET    /api/agency/evolution  ← Estado evolutivo
GET    /api/agency/narrative  ← Arcos narrativos
POST   /api/agency/quantum/map      ← Poincaré → Bloch states
POST   /api/agency/quantum/fidelity ← Fidelidade quântica
GET    /api/cluster/ring      ← Hash ring consistente
GET    /metrics               ← Prometheus metrics
```

**Total: 30+ endpoints REST não usados pelo dashboard**

---

## 5. Funcionalidades Placeholders no Dashboard

| Feature              | Página       | Estado         |
|----------------------|-------------|----------------|
| Export Snapshot       | Collections | Botão disabled |
| Restore Snapshot      | Settings    | Botão disabled |
| Recharts (dep)        | —           | Instalado, não usado |

---

## 6. Mapa de Lacunas por Categoria

### 🔴 Crítico — Features core sem visibilidade

| Lacuna                               | Crates envolvidos                     | Impacto |
|--------------------------------------|---------------------------------------|---------|
| **Console NQL**                      | `nietzsche-query`                     | Os 19 tipos de query (MATCH, CREATE, DREAM, DIFFUSE, NARRATE...) não têm UI |
| **Algoritmos de Grafo**              | `nietzsche-algo`                      | 10 algoritmos prontos no backend sem página |
| **Agência Autônoma**                 | `nietzsche-agency`, `nietzsche-wiederkehr` | Daemons, MetaObserver, Motor Contrafactual invisíveis |
| **Ciclo de Sono**                    | `nietzsche-sleep`                     | RiemannianAdam reconsolidation sem trigger/monitor |

### 🟡 Importante — Features avançadas sem visibilidade

| Lacuna                               | Crates envolvidos                     | Impacto |
|--------------------------------------|---------------------------------------|---------|
| **Evolução Zaratustra**              | `nietzsche-zaratustra`                | Vontade de Poder, Eterno Retorno, Übermensch não monitoráveis |
| **Crescimento L-System**             | `nietzsche-lsystem`                   | Regras fractais não visualizáveis |
| **Dream Queries**                    | `nietzsche-dream`                     | Queries especulativas não acessíveis |
| **Narrativas**                       | `nietzsche-narrative`                 | Arcos narrativos endpoint existe, sem UI |
| **Compressão Sensory**               | `nietzsche-sensory`                   | Degradação progressiva (f32→gone) não monitorável |

### 🟢 Secundário — Infra/aceleração sem visibilidade

| Lacuna                               | Crates envolvidos                     | Impacto |
|--------------------------------------|---------------------------------------|---------|
| **GPU/TPU Status**                   | `nietzsche-hnsw-gpu`, `nietzsche-tpu`, `nietzsche-cugraph` | Sem indicadores de aceleração HW |
| **Kafka CDC**                        | `nietzsche-kafka`                     | Sem monitor de streaming CDC |
| **Camada SQL**                       | `nietzsche-table`                     | SQLite tables não navegáveis |
| **Media Storage**                    | `nietzsche-media`                     | Sem browser de mídia (S3/GCS) |
| **Prometheus Metrics**               | `nietzsche-api` (`/metrics`)          | Endpoint existe, sem painel Grafana embeddido |

---

## 7. Dados do Grafo Disponíveis mas Sub-utilizados

O endpoint `GET /api/graph` retorna estes campos por nó:

```json
{
  "id": "uuid",
  "node_type": "Semantic|Episodic|Concept|DreamSnapshot|Somatic|Linguistic|Composite",
  "energy": 0.85,
  "depth": 0.42,
  "hausdorff": 0.13,
  "created_at": 1740000000,
  "content": { "label": "...", "title": "..." }
}
```

O **Cosmograph Graph Explorer** já visualiza TODOS esses campos:
- ✅ `node_type` → cor dos nós
- ✅ `energy` → tamanho dos nós + histograma
- ✅ `depth` → histograma Poincaré
- ✅ `hausdorff` → histograma
- ✅ `created_at` → timeline temporal
- ✅ `edge_type` → barras categóricas
- ✅ `weight` → espessura das arestas

**Porém**: o embedding hiperbólico completo (coordenadas Poincaré) NÃO é enviado via REST — só via gRPC.

---

## 8. Resumo Visual — Cobertura por Grupo

```
Motor Principal          [██████████░░░░░] 65%  — graph+hyp-ops visíveis, query invisível
Crescimento & Evolução   [░░░░░░░░░░░░░░░]  0%  — lsystem, pregel, zaratustra ZERO UI
Sono & Memória           [░░░░░░░░░░░░░░░]  0%  — sleep, sensory, dream ZERO UI
Agência Autônoma         [░░░░░░░░░░░░░░░]  0%  — agency, wiederkehr, narrative ZERO UI
Busca & Indexação         [████████░░░░░░░] 50%  — knn+pq visíveis, algo+secondary parcial
Aceleração GPU/TPU       [░░░░░░░░░░░░░░░]  0%  — hnsw-gpu, tpu, cugraph ZERO UI
API & Integração         [██████░░░░░░░░░] 40%  — REST parcial, mcp/kafka/table/media ZERO UI
```

---

## 9. Recomendação: Top 5 Páginas para Aumentar Cobertura

| Prioridade | Nova Página Proposta        | Crates que cobriria                              | Endpoints já prontos |
|------------|---------------------------|--------------------------------------------------|----------------------|
| **P0**     | **NQL Console**            | `nietzsche-query`                                | `POST /api/query`    |
| **P1**     | **Graph Algorithms**       | `nietzsche-algo`                                 | 10x `GET /api/algo/*`|
| **P2**     | **Agency Monitor**         | `nietzsche-agency`, `nietzsche-wiederkehr`, `nietzsche-narrative` | 8x `GET/POST /api/agency/*` |
| **P3**     | **Sleep & Dream**          | `nietzsche-sleep`, `nietzsche-dream`, `nietzsche-sensory` | `POST /api/sleep`    |
| **P4**     | **Evolution (Zaratustra)** | `nietzsche-zaratustra`, `nietzsche-lsystem`      | Via gRPC (precisa REST) |

Com essas 5 páginas, a cobertura salta de **~21%** para **~65%**.
