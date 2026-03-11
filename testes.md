# NietzscheDB - Testes de Integracao EVA Memory

**Ficheiro**: `tests/test_eva_memory.py`
**Total**: 73 testes, 21 categorias
**Status**: ALL PASSING

## Como executar

```bash
# Todos os testes
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v

# Uma categoria especifica
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -k "TestNodeCRUD"

# Um teste especifico
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -k "test_insert_episodic_node"
```

**Dependencias**: `pip install grpcio grpcio-tools pytest`

---

## Categorias

| # | Classe | Testes | Cobertura |
|---|--------|--------|-----------|
| 1 | TestCollectionManagement | 4 | create, list, info, drop |
| 2 | TestNodeCRUD | 8 | Episodic/Semantic/Concept, custom ID, get/delete, energy |
| 3 | TestEdgeCRUD | 4 | CONTAINS, TEMPORAL_NEXT, HEBBIAN_VISUAL, delete |
| 4 | TestMergeUpsert | 3 | merge create, match, edge |
| 5 | TestKnnSearch | 2 | busca vetorial Poincare, filtros |
| 6 | TestNQLQueries | 5 | MATCH, WHERE, COUNT, error handling |
| 7 | TestSensoryLayer | 5 | insert/get/reconstruct/degrade, multimodal |
| 8 | TestBatchOperations | 2 | batch nodes (10), batch edges |
| 9 | TestCacheTTL | 5 | set/get/del/overwrite, reap expired |
| 10 | TestGraphTraversal | 2 | BFS, Dijkstra |
| 11 | TestGraphAlgorithms | 6 | PageRank, Louvain, WCC, Degree, Betweenness, Triangles |
| 12 | TestSynthesis | 2 | 2-node, multi-node |
| 13 | TestFullTextSearch | 1 | busca textual |
| 14 | TestListStore | 2 | rpush/lrange, len |
| 15 | TestPerceptionPipeline | 2 | Scene2D+Objects+edges, temporal sequence |
| 16 | TestEpisodicMemoryPipeline | 2 | conversacao EVA-Mind, topic extraction |
| 17 | TestPoincareConstraints | 3 | hierarquia magnitude, zero vec, dim mismatch |
| 18 | TestAdminStats | 3 | health, stats, node count |
| 19 | TestEVAProductionCollections | 1 | verifica colecoes EVA em producao (read-only) |
| 20 | TestStress | 2 | 100 nodes batch, KNN rapido |
| 21 | TestEVACognitiveIntrospection | 9 | aprendeu/esqueceu/associou/deseja (read-only, producao) |

---

## Lista completa de testes

### 1. TestCollectionManagement
- `test_create_collection` - Cria colecao temporaria (128D, poincare)
- `test_list_collections` - Lista colecoes e verifica que a temporaria aparece
- `test_collection_info` - Obtem info detalhada (dim, metrica, contagens)
- `test_drop_collection` - Remove colecao temporaria

### 2. TestNodeCRUD
- `test_insert_episodic_node` - Insere no tipo Episodic com coords Poincare
- `test_insert_semantic_node` - Insere no tipo Semantic
- `test_insert_concept_node` - Insere no tipo Concept
- `test_insert_with_custom_id` - Insere com UUID customizado
- `test_get_nonexistent_node` - Verifica erro ao buscar no inexistente
- `test_delete_node` - Deleta no e confirma remocao
- `test_update_energy` - Atualiza energia de um no
- `test_energy_decay_simulation` - Simula decaimento de energia

### 3. TestEdgeCRUD
- `test_insert_contains_edge` - Cria edge tipo CONTAINS entre dois nos
- `test_insert_temporal_edge` - Cria edge tipo TEMPORAL_NEXT
- `test_insert_hebbian_edge` - Cria edge tipo HEBBIAN_VISUAL
- `test_delete_edge` - Deleta edge e confirma remocao

### 4. TestMergeUpsert
- `test_merge_node_create` - MergeNode cria no quando nao existe
- `test_merge_node_match` - MergeNode encontra no existente (match por content)
- `test_merge_edge` - MergeEdge cria edge (sempre retorna created=True)

### 5. TestKnnSearch
- `test_knn_basic` - Busca KNN basica no espaco Poincare
- `test_knn_with_filters` - Busca KNN com filtros por tipo de no

### 6. TestNQLQueries
- `test_match_episodic` - `MATCH (n:Episodic) RETURN n`
- `test_match_semantic` - `MATCH (n:Semantic) RETURN n`
- `test_match_with_where` - `MATCH` com clausula `WHERE` em campos do content
- `test_query_count` - `MATCH ... RETURN COUNT(n)`
- `test_query_error_handling` - Verifica erro em queries NQL invalidas

### 7. TestSensoryLayer
- `test_insert_and_get_sensory` - Insere e recupera dado sensorial (Image)
- `test_reconstruct_sensory` - Reconstroi dado sensorial (com perda de precisao)
- `test_degrade_sensory` - Degrada dado sensorial progressivamente
- `test_sensory_nonexistent` - Verifica erro ao buscar sensorial inexistente
- `test_multimodal_sensory` - Insere dados Audio e Text

### 8. TestBatchOperations
- `test_batch_insert_nodes` - Insere 10 nos em batch
- `test_batch_insert_edges` - Insere edges em batch

### 9. TestCacheTTL
- `test_cache_set_get` - Set e Get no cache
- `test_cache_del` - Delete no cache
- `test_cache_nonexistent` - Get de chave inexistente
- `test_cache_overwrite` - Overwrite de valor existente
- `test_reap_expired` - Reap de entradas expiradas por TTL

### 10. TestGraphTraversal
- `test_bfs` - Breadth-First Search a partir de um no
- `test_dijkstra` - Caminho mais curto (Dijkstra) entre dois nos

### 11. TestGraphAlgorithms
- `test_pagerank` - PageRank (nos mais influentes)
- `test_louvain` - Deteccao de comunidades (Louvain)
- `test_wcc` - Weakly Connected Components
- `test_degree_centrality` - Centralidade por grau
- `test_betweenness` - Centralidade de intermediacao
- `test_triangle_count` - Contagem de triangulos

### 12. TestSynthesis
- `test_synthesis_two_nodes` - Sintese Riemanniana entre 2 nos
- `test_synthesis_multi` - Sintese com multiplos nos

### 13. TestFullTextSearch
- `test_full_text_basic` - Busca full-text por palavras no content

### 14. TestListStore
- `test_list_rpush_and_lrange` - RPUSH e LRANGE (lista ordenada)
- `test_list_len` - Comprimento da lista

### 15. TestPerceptionPipeline
- `test_full_perception_cycle` - Ciclo completo: Scene2D + Objects + edges CONTAINS/SPATIAL_NEAR
- `test_temporal_sequence` - Sequencia temporal de cenas com edges TEMPORAL_NEXT

### 16. TestEpisodicMemoryPipeline
- `test_save_conversation_memory` - Salva memoria de conversacao EVA-Mind
- `test_topic_extraction` - Extrai topicos de uma conversacao

### 17. TestPoincareConstraints
- `test_magnitude_hierarchy` - Verifica hierarquia por magnitude (pai < filho no Poincare ball)
- `test_zero_vector_node` - Insere no com vetor zero (raiz da hierarquia)
- `test_dim_mismatch_handling` - Verifica comportamento com dimensao errada

### 18. TestAdminStats
- `test_health_check` - Health check do servidor
- `test_get_stats` - Estatisticas globais (total nos, edges, colecoes)
- `test_collection_node_count` - Contagem de nos por colecao

### 19. TestEVAProductionCollections
- `test_eva_collections_exist` - Verifica que colecoes EVA existem em producao (read-only)

### 20. TestStress
- `test_rapid_insert_100_nodes` - Insere 100 nos rapidamente via batch
- `test_rapid_search_after_insert` - KNN rapido apos insercao massiva

### 21. TestEVACognitiveIntrospection
- `test_learned_eva_mind_contents` - Mostra conteudo real de eva_mind (habitos, assessments, sessoes)
- `test_learned_eva_core_sample` - Top 15 nos mais influentes via PageRank
- `test_learned_collection_sizes` - Tamanho de cada colecao cognitiva EVA
- `test_forgotten_low_energy_nodes` - Nos com energia < 0.05 (memorias esquecidas)
- `test_forgotten_temporal_decay_stats` - Termodinamica e self-healing do Agency
- `test_associated_edge_types` - Conta edges por tipo (LSystemGenerated, Association, Hebbian...)
- `test_associated_communities` - Clusters de conhecimento (Louvain)
- `test_desires_knowledge_gaps` - Gaps no espaco Poincare que a EVA quer preencher
- `test_cognitive_summary` - Panorama geral: memorias, associacoes, fase, interpretacao

**Execucao**:
```bash
NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -s -k "TestEVACognitiveIntrospection"
```
**Dependencia extra**: `pip install requests` (usa HTTP API para endpoints do Agency)

---

## Gotchas descobertos nos testes

- Server requer **UUID valido** nos IDs (nao aceita strings arbitrarias)
- `OriginalShape` no InsertSensory e tagged enum JSON: `{"Image": {"width": W, "height": H, "channels": C}}`
- NQL `MATCH (n:Episodic) RETURN n` pode retornar vazio em colecoes novas (HNSW index delay)
- Reconstruct sensory tem **perda de precisao** significativa (quantizacao interna)
- MergeEdge sempre retorna `created=True` (server nao faz dedup por from+to+type)
- `grpcio>=1.78.0` obrigatorio (stubs gerados com essa versao)

---

## Colecoes de producao (ultima verificacao)

| Colecao | Nos | Edges | Dim | Metrica |
|---------|-----|-------|-----|---------|
| eva_codebase | 147,561 | 69,413 | 3072 | cosine |
| gene_ontology | 98,453 | 145,695 | 128 | poincare |
| patient_graph | 93,714 | 46,790 | 3072 | poincare |
| eva_core | 50,108 | 52,092 | 3072 | poincare |
| eva_docs | 48,251 | 47,207 | 3072 | cosine |
| malaria | 32,070 | 6,976 | 128 | cosine |
| default | 20,867 | 20,902 | 3072 | cosine |
| science_galaxies | 5,208 | 5,126 | 128 | poincare |
| malaria-vision | 3,565 | 0 | 1408 | cosine |
| tech_galaxies | 3,191 | 5,584 | 128 | poincare |
| culture_galaxies | 1,977 | 7,064 | 128 | poincare |
| knowledge_galaxies | 1,941 | 7,028 | 128 | poincare |
| **Total (41 colecoes)** | **~512K** | **~418K** | — | — |
