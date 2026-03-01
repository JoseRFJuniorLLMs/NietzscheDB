# Arquitetura Completa: EVA Core <-> NietzscheDB

## Visao Geral do Ecossistema

```
                            +-----------------------+
                            |     USUARIOS/CLIENTES |
                            +-----------+-----------+
                                        |
          +-----------------------------+-----------------------------+
          |                             |                             |
          v                             v                             v
+-------------------+     +-------------------+     +-------------------+
|   EVA-Front       |     |   EVA-Mobile      |     |   EVA-Kids        |
|   React/Vite      |     |   Flutter/Dart    |     |   Angular         |
|   :3000 (dev)     |     |   136.113.25.218  |     |   :4200 (dev)     |
+--------+----------+     +--------+----------+     +--------+----------+
         |                          |                          |
         |  HTTPS/WSS              |  HTTPS/REST              |  HTTPS
         |                          |                          |
         +----------+---------------+--------------------------+
                    |
                    v
         +---------------------+
         |  Nginx Reverse      |
         |  Proxy :8090        |
         |  SSL Termination    |
         +----------+----------+
                    |
         +----------+----------+
         |                     |
         v                     v
+------------------+  +------------------+
| EVA-Back         |  | EVA (Go)         |
| FastAPI :8001    |  | Backend :8091    |
| Python           |  | Go               |
| NietzscheDB       |  | Cerebro da EVA   |
+--------+---------+  +--------+---------+
         |                      |
         | REST API             | gRPC (principal)
         |                      |
         +----------+-----------+
                    |
                    v
    +=======================================+
    ||                                     ||
    ||   N I E T Z S C H E D B            ||
    ||   Multi-Manifold Graph DB           ||
    ||                                     ||
    +=======================================+
```

---

## Fluxo Detalhado: EVA (Go) <-> NietzscheDB

```
+===========================================================================+
||                        EVA BACKEND (Go) :8091                           ||
||                                                                         ||
||  +-------------------+  +-------------------+  +-------------------+    ||
||  | Cortex            |  | Hippocampus       |  | Tools             |    ||
||  | (Raciocinio)      |  | (Memoria)         |  | (Acoes)           |    ||
||  +--------+----------+  +--------+----------+  +--------+----------+    ||
||           |                      |                      |               ||
||           +----------+-----------+----------------------+               ||
||                      |                                                  ||
||                      v                                                  ||
||  +------------------------------------------------------------------+  ||
||  |              brainstem/infrastructure/nietzsche/                  |  ||
||  |                                                                  |  ||
||  |  +------------------+  +------------------+  +----------------+  |  ||
||  |  | client.go        |  | graph_adapter.go |  | vector_adapter |  |  ||
||  |  | NietzscheClient  |  | GraphAdapter     |  | VectorAdapter  |  |  ||
||  |  |                  |  | (substitui       |  | (substitui     |  |  ||
||  |  | 40+ metodos gRPC |  |  NietzscheDB)          |  |  NietzscheDB)       |  |  ||
||  |  +--------+---------+  +--------+---------+  +-------+--------+  |  ||
||  |           |                      |                    |           |  ||
||  +------------------------------------------------------------------+  ||
||              |                      |                    |              ||
+===========================================================================+
               |                      |                    |
               |     gRPC :50051      |                    |
               |     (protobuf)       |                    |
               v                      v                    v
+===========================================================================+
||                    NIETZSCHEDB SERVER (Rust)                             ||
||                                                                         ||
||  +===================================================================+  ||
||  ||                   CAMADA gRPC :50051                             ||  ||
||  ||                   (tonic + prost)                                ||  ||
||  ||                                                                 ||  ||
||  ||  Servico: nietzsche.NietzscheDB                                 ||  ||
||  ||                                                                 ||  ||
||  ||  CRUD Nos:                    Grafos:                           ||  ||
||  ||   - InsertNode                - Bfs (busca em largura)          ||  ||
||  ||   - GetNode                   - Dijkstra (caminho minimo)       ||  ||
||  ||   - DeleteNode                - Diffuse (difusao termica)       ||  ||
||  ||   - MergeNode (MERGE NietzscheDB)   - ShortestPath (A* via gRPC)     ||  ||
||  ||                                                                 ||  ||
||  ||  CRUD Arestas:                Busca Vetorial:                   ||  ||
||  ||   - InsertEdge                - KnnSearch (HNSW hiperbolico)    ||  ||
||  ||   - DeleteEdge                - HybridSearch (vetor + BM25)     ||  ||
||  ||   - MergeEdge                 - FullTextSearch (BM25)           ||  ||
||  ||                                                                 ||  ||
||  ||  Queries:                     Autonomia:                        ||  ||
||  ||   - Query (NQL)               - TriggerSleep (reconsolidacao)   ||  ||
||  ||   - SubscribeCDC (stream)     - InvokeZaratustra (evolucao)     ||  ||
||  ||                                                                 ||  ||
||  ||  Sensorial:                   Sistema:                          ||  ||
||  ||   - InsertSensory             - HealthCheck                     ||  ||
||  ||   - GetSensory                - GetStats                        ||  ||
||  ||   - Reconstruct               - CreateCollection                ||  ||
||  +===================================================================+  ||
||                                                                         ||
||  +===================================================================+  ||
||  ||                   CAMADA HTTP/REST :8080                         ||  ||
||  ||                   (axum)                                        ||  ||
||  ||                                                                 ||  ||
||  ||  Dashboard SPA:               API REST:                         ||  ||
||  ||   GET /                        GET  /api/stats                  ||  ||
||  ||   (React embeddido)            GET  /api/collections            ||  ||
||  ||                                GET  /api/graph?collection=X     ||  ||
||  ||  CRUD REST:                    GET  /api/node/:id               ||  ||
||  ||   POST   /api/node             POST /api/query (NQL)            ||  ||
||  ||   DELETE /api/node/:id         POST /api/sleep                  ||  ||
||  ||   POST   /api/edge             POST /api/backup                 ||  ||
||  ||   DELETE /api/edge/:id         GET  /api/backup                 ||  ||
||  ||                                                                 ||  ||
||  ||  Batch:                        Algoritmos:                      ||  ||
||  ||   POST /api/batch/nodes        GET /api/algo/pagerank           ||  ||
||  ||   POST /api/batch/edges        GET /api/algo/louvain            ||  ||
||  ||                                GET /api/algo/betweenness        ||  ||
||  ||  Exportacao:                   GET /api/algo/closeness          ||  ||
||  ||   GET /api/export/nodes        GET /api/algo/degree             ||  ||
||  ||   GET /api/export/edges        GET /api/algo/wcc                ||  ||
||  ||                                GET /api/algo/scc                ||  ||
||  ||  Monitoramento:                GET /api/algo/triangles          ||  ||
||  ||   GET /metrics (Prometheus)    GET /api/algo/jaccard            ||  ||
||  ||   GET /api/cluster/ring        GET /api/algo/labelprop          ||  ||
||  +===================================================================+  ||
||                                                                         ||
+===========================================================================+
```

---

## Arquitetura Interna do NietzscheDB (38 Crates)

```
+===========================================================================+
||                     NIETZSCHEDB - MOTOR INTERNO                         ||
||                                                                         ||
||  +---------------------------+  +------------------------------------+  ||
||  | nietzsche-server          |  | nietzsche-api                      |  ||
||  | (binario principal)       |  | (proto/nietzsche.proto)            |  ||
||  | gRPC + HTTP + Background  |  | Definicoes protobuf para todos    |  ||
||  +------------+--------------+  | os RPCs e mensagens               |  ||
||               |                 +------------------------------------+  ||
||               v                                                         ||
||  +------------------------------------------------------------------+  ||
||  |                    CAMADA DE QUERY                                |  ||
||  |                                                                  |  ||
||  |  +-------------------+  +-------------------+                    |  ||
||  |  | nietzsche-query   |  | nietzsche-algo    |                    |  ||
||  |  | Parser NQL        |  | 10 Algoritmos:    |                    |  ||
||  |  | (pest grammar)    |  |  PageRank         |                    |  ||
||  |  |                   |  |  Louvain          |                    |  ||
||  |  | MATCH, WHERE,     |  |  Label Propagation|                    |  ||
||  |  | RETURN, CREATE,   |  |  Betweenness      |                    |  ||
||  |  | DELETE, SET,      |  |  Closeness        |                    |  ||
||  |  | MERGE, SYNTHESIS  |  |  Degree           |                    |  ||
||  |  | + 13 funcoes      |  |  WCC, SCC         |                    |  ||
||  |  |   matematicas     |  |  Triangles        |                    |  ||
||  |  +-------------------+  |  Jaccard          |                    |  ||
||  |                         |  A* (via gRPC)    |                    |  ||
||  |                         +-------------------+                    |  ||
||  +------------------------------------------------------------------+  ||
||               |                                                         ||
||               v                                                         ||
||  +------------------------------------------------------------------+  ||
||  |                    CAMADA DE INTELIGENCIA                         |  ||
||  |                                                                  |  ||
||  |  +----------------+ +----------------+ +----------------------+  |  ||
||  |  | nietzsche-     | | nietzsche-     | | nietzsche-           |  |  ||
||  |  | agency         | | zaratustra     | | wiederkehr           |  |  ||
||  |  |                | |                | |                      |  |  ||
||  |  | MetaObserver   | | Vontade de     | | Eterno Retorno      |  |  ||
||  |  | Health Reports | | Potencia       | | Ciclos autonomos    |  |  ||
||  |  | Desejos        | | Evolucao       | | Periodicidade       |  |  ||
||  |  | Reactor        | | Estrategias    | |                      |  |  ||
||  |  +----------------+ +----------------+ +----------------------+  |  ||
||  |                                                                  |  ||
||  |  +----------------+ +----------------+ +----------------------+  |  ||
||  |  | nietzsche-     | | nietzsche-     | | nietzsche-           |  |  ||
||  |  | sleep          | | dream          | | narrative            |  |  ||
||  |  |                | |                | |                      |  |  ||
||  |  | Reconsolidacao | | Geracao de     | | Sintese narrativa    |  |  ||
||  |  | Riemanniana    | | sonhos via     | | dos eventos do       |  |  ||
||  |  | (Adam optimizer| | exploracao     | | grafo em texto       |  |  ||
||  |  |  no manifold)  | | do grafo       | | natural              |  |  ||
||  |  +----------------+ +----------------+ +----------------------+  |  ||
||  |                                                                  |  ||
||  |  +----------------+ +----------------+                           |  ||
||  |  | nietzsche-     | | nietzsche-     |                           |  ||
||  |  | lsystem        | | sensory        |                           |  ||
||  |  |                | |                |                           |  ||
||  |  | Crescimento    | | Compressao     |                           |  ||
||  |  | organico       | | audio/imagem   |                           |  ||
||  |  | (regras L)     | | latentes       |                           |  ||
||  |  +----------------+ +----------------+                           |  ||
||  +------------------------------------------------------------------+  ||
||               |                                                         ||
||               v                                                         ||
||  +------------------------------------------------------------------+  ||
||  |                    CAMADA DE DADOS                                |  ||
||  |                                                                  |  ||
||  |  +----------------------------+  +----------------------------+  |  ||
||  |  | nietzsche-graph            |  | nietzsche-hyp-ops          |  |  ||
||  |  |                            |  |                            |  |  ||
||  |  | GraphStorage (RocksDB)     |  | Operacoes Hiperbolicas:    |  |  ||
||  |  |  CF_NODES     (NodeMeta)   |  |  Distancia Poincare       |  |  ||
||  |  |  CF_EMBEDDINGS (Vec<f32>)  |  |  Mobius Addition          |  |  ||
||  |  |  CF_EDGES     (Edge)       |  |  Exp Map / Log Map        |  |  ||
||  |  |  CF_ADJ_OUT   (adjacencia) |  |  Projecao Klein           |  |  ||
||  |  |  CF_ADJ_IN    (adjacencia) |  |  Frechet Mean             |  |  ||
||  |  |  CF_META      (metadados)  |  |                            |  |  ||
||  |  |  CF_SENSORY   (sensorial)  |  +----------------------------+  |  ||
||  |  |  CF_ENERGY_IDX (indice)    |                                  |  ||
||  |  |  CF_META_IDX  (indice)     |  +----------------------------+  |  ||
||  |  |  CF_LISTS     (listas)     |  | embedded_vector_store     |  |  ||
||  |  |                            |  |                            |  |  ||
||  |  | CollectionManager          |  | HNSW Index (CPU)           |  |  ||
||  |  |  23 collections ativas     |  |  Distancia hiperbolica    |  |  ||
||  |  |  RwLock por collection     |  |  Poincare ball ‖x‖ < 1   |  |  ||
||  |  +----------------------------+  +----------------------------+  |  ||
||  |                                                                  |  ||
||  |  +----------------------------+  +----------------------------+  |  ||
||  |  | nietzsche-hnsw-gpu        |  | nietzsche-tpu              |  |  ||
||  |  | (opcional)                 |  | (opcional)                 |  |  ||
||  |  |                            |  |                            |  |  ||
||  |  | NVIDIA cuVS CAGRA         |  | Google PJRT                |  |  ||
||  |  | Aceleracao GPU             |  | TPU v5e/v6e/v7            |  |  ||
||  |  +----------------------------+  +----------------------------+  |  ||
||  +------------------------------------------------------------------+  ||
||               |                                                         ||
||               v                                                         ||
||  +------------------------------------------------------------------+  ||
||  |              PERSISTENCIA (RocksDB Embeddido)                    |  ||
||  |                                                                  |  ||
||  |   /var/lib/nietzsche/collections/                                |  ||
||  |     +-- eva_core/rocksdb/       (70 MB, 14.612 nos, 2.344 ar.)  |  ||
||  |     +-- patient_graph/rocksdb/  (15 MB, 129 nos, 2 arestas)     |  ||
||  |     +-- memories/rocksdb/       (15 MB)                          |  ||
||  |     +-- eva_self_knowledge/     (15 MB)                          |  ||
||  |     +-- eva_learnings/          (15 MB)                          |  ||
||  |     +-- eva_codebase/           (15 MB)                          |  ||
||  |     +-- eva_docs/               (15 MB)                          |  ||
||  |     +-- speaker_embeddings/     (15 MB)                          |  ||
||  |     +-- stories/                (15 MB)                          |  ||
||  |     +-- signifier_chains/       (15 MB)                          |  ||
||  |     +-- default/                (15 MB)                          |  ||
||  |     +-- ... (23 collections total, 552 MB)                       |  ||
||  +------------------------------------------------------------------+  ||
||                                                                         ||
+===========================================================================+
```

---

## Fluxo de uma Operacao Tipica (Memoria da EVA)

```
  Usuario fala com EVA
         |
         v
  +------------------+
  | Gemini 2.5 Flash |  <-- Modelo de audio nativo (INTOCAVEL)
  | Native Audio     |      gemini-2.5-flash-native-audio-preview-12-2025
  | (WebSocket)      |
  +--------+---------+
           |
           | Transcricao + Embedding (3072D)
           v
  +------------------+
  | EVA Backend (Go) |
  | :8091            |
  +--------+---------+
           |
           |  1. Gera embedding via Gemini
           |  2. Processa intencao do usuario
           |  3. Chama NietzscheDB para armazenar/buscar
           |
           v
  +==================================+
  |  client.go                       |
  |                                  |
  |  // Armazenar memoria            |
  |  vectorAdapter.Upsert(           |
  |    ctx,                          |
  |    "memories",        // colecao |
  |    uuid,              // id      |
  |    embedding,         // 3072D   |
  |    metadata,          // JSON    |
  |  )                               |
  |                                  |
  |  // Buscar memorias similares    |
  |  vectorAdapter.Search(           |
  |    ctx,                          |
  |    "memories",        // colecao |
  |    queryVector,       // 3072D   |
  |    k=10,              // top-10  |
  |    userID,            // filtro  |
  |  )                               |
  |                                  |
  |  // Criar relacao no grafo       |
  |  graphAdapter.MergeNode(         |
  |    ctx,                          |
  |    MergeNodeOpts{                |
  |      Label: "Memory",           |
  |      Props: map[string]any{...}, |
  |      Coords: []float64{...},    |  <-- Coordenadas Poincare
  |      Energy: 0.8,               |  <-- Energia hiperbolica
  |    },                            |
  |  )                               |
  |                                  |
  |  // Query NQL                    |
  |  graphAdapter.ExecuteNQL(        |
  |    ctx,                          |
  |    "MATCH (m:Memory)             |
  |     WHERE m.type = 'episodic'    |
  |     AND m.energy > 0.5           |
  |     RETURN m                     |
  |     ORDER BY m.energy DESC       |
  |     LIMIT 20",                   |
  |    nil,              // params   |
  |    "eva_core",       // colecao  |
  |  )                               |
  +==============+===================+
                 |
                 | gRPC (protobuf, binario)
                 | localhost:50051
                 |
                 v
  +==================================+
  |  NietzscheDB Server              |
  |                                  |
  |  1. Deserializa protobuf         |
  |  2. Roteia para collection       |
  |  3. Executa operacao:            |
  |     - InsertNode -> RocksDB      |
  |     - KnnSearch -> HNSW index    |
  |     - Query -> NQL parser        |
  |  4. Serializa resposta           |
  |  5. Retorna via gRPC stream      |
  +==================================+
```

---

## Processos Autonomos em Background

```
  +===========================================================================+
  ||  NIETZSCHEDB - PROCESSOS AUTONOMOS (tokio::spawn)                       ||
  ||                                                                         ||
  ||  +-------------------+     +-------------------+     +--------------+   ||
  ||  | Agency Engine     |     | Sleep Scheduler   |     | TTL Reaper   |   ||
  ||  | (AGENCY_TICK=60s) |     | (SLEEP_INTERVAL)  |     | (TTL_REAPER) |   ||
  ||  |                   |     |                   |     |              |   ||
  ||  | Para cada colecao:|     | Ciclo de sono:    |     | Remove nos   |   ||
  ||  |  1. Daemons       |     |  - Perturbacao    |     | expirados    |   ||
  ||  |  2. Health report |     |    (ruido Adam)   |     | (expires_at) |   ||
  ||  |  3. Gap detection |     |  - Otimizacao     |     +--------------+   ||
  ||  |  4. Reactor       |     |    Riemanniana    |                        ||
  ||  |  5. Desire engine |     |  - Hausdorff      |     +--------------+   ||
  ||  |  6. L-System tick |     |    check          |     | Backup       |   ||
  ||  |  7. Evolution     |     |  - Commit/Reject  |     | Scheduler    |   ||
  ||  |  8. Dream trigger |     +-------------------+     | (BACKUP_INT) |   ||
  ||  |  9. Narrative     |                               +--------------+   ||
  ||  +-------------------+     +-------------------+                        ||
  ||                            | Zaratustra        |     +--------------+   ||
  ||                            | (ZARATUSTRA_INT)  |     | Wiederkehr   |   ||
  ||                            |                   |     | Daemon       |   ||
  ||                            | Evolucao          |     | (Eterno      |   ||
  ||                            | autonoma:         |     |  Retorno)    |   ||
  ||                            |  - Fitness eval   |     +--------------+   ||
  ||                            |  - Estrategia     |                        ||
  ||                            |  - Mutacao        |                        ||
  ||                            +-------------------+                        ||
  ||                                                                         ||
  +===========================================================================+
```

---

## SDKs Disponiveis (Multi-Linguagem)

```
  +------------------------------------------------------------------+
  |                    CLIENTES / SDKs                                |
  |                                                                  |
  |  +------------------+  +------------------+  +----------------+  |
  |  | Rust SDK         |  | Python SDK       |  | Go SDK         |  |
  |  | nietzsche-sdk    |  | nietzsche_db.py  |  | sdks/go/       |  |
  |  |                  |  |                  |  |                |  |
  |  | NietzscheClient  |  | NietzscheClient  |  | Usado pela EVA |  |
  |  | .connect(addr)   |  | ("host:50051")   |  | (producao)     |  |
  |  | .insert_node()   |  | .insert_node()   |  |                |  |
  |  | .knn_search()    |  | .knn_search()    |  |                |  |
  |  | .query(nql)      |  | .query(nql)      |  |                |  |
  |  +------------------+  +------------------+  +----------------+  |
  |                                                                  |
  |  +------------------+  +--------------------------------------+  |
  |  | TypeScript SDK   |  | HTTP REST (qualquer linguagem)       |  |
  |  | nietzsche_db.ts  |  |                                      |  |
  |  |                  |  | curl http://host:8080/api/stats      |  |
  |  | NietzscheClient  |  | curl -X POST http://host:8080/api/  |  |
  |  | (Node.js/Deno)   |  |   query -d '{"nql":"MATCH..."}'     |  |
  |  +------------------+  +--------------------------------------+  |
  |                                                                  |
  |  Todos os SDKs usam gRPC (protobuf) na porta 50051              |
  |  REST API disponivel na porta 8080 para integracao leve          |
  +------------------------------------------------------------------+
```

---

## Tabela de Portas

| Porta  | Servico              | Protocolo     | Descricao                         |
|--------|----------------------|---------------|-----------------------------------|
| 50051  | NietzscheDB gRPC     | gRPC/protobuf | API principal (grafo + vetores)   |
| 8080   | NietzscheDB REST     | HTTP/JSON     | Dashboard + API REST              |
| 8091   | EVA Backend          | HTTP/JSON     | Backend Go da EVA                 |
| 8090   | Nginx Proxy          | HTTPS/WSS     | Proxy reverso SSL                 |
| 8001   | EVA-Back FastAPI     | HTTP/JSON     | Backend Python (NietzscheDB)       |
| 3000   | EVA-Front (dev)      | HTTP          | Frontend React (desenvolvimento)  |
| 11434  | Ollama               | HTTP          | LLM local (embeddings/traducao)   |
| 9090   | Prometheus           | HTTP          | Metricas de monitoramento         |

---

## Formatos de Dados no Fio (Wire Format)

### gRPC (porta 50051)
- **Serializacao**: Protocol Buffers (binario, compacto)
- **Vetores**: `repeated double coords` (f64) no proto, f32 no storage
- **IDs**: UUID como `string` no proto, 16 bytes no RocksDB
- **Conteudo**: JSON como `string` (campo generico)
- **Streaming**: `SubscribeCDC` usa server-side streaming

### REST (porta 8080)
- **Serializacao**: JSON (texto, legivel)
- **Vetores**: nao expostos na REST API (apenas metadados)
- **IDs**: UUID como string JSON

### Storage Interno (RocksDB)
- **Serializacao**: bincode (Rust, binario posicional)
- **4 formatos coexistem**: V0 / V1 / V1.5 / V2 (migracao automatica)
- **Compressao**: LZ4 (memtable) + Zstd (bottommost level)

---

## Substituicoes Realizadas pela EVA

| Antes (Legacy)       | Depois (NietzscheDB)           | Protocolo    |
|----------------------|--------------------------------|--------------|
| NietzscheDB (Cypher)       | NietzscheDB (NQL)              | gRPC :50051  |
| NietzscheDB (REST)        | NietzscheDB (KnnSearch)        | gRPC :50051  |
| NietzscheDB (Cache)        | NietzscheDB (CF_META + TTL)    | gRPC :50051  |
| 3 bancos separados   | 1 banco unificado              | 1 protocolo  |

---

*Documento gerado em 2026-02-22*
*NietzscheDB v0.1.0 — 38 crates, 23 collections, 14.825 nos ativos*
