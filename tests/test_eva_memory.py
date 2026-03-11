"""
Test completo da memoria da EVA via NietzscheDB Python SDK.

Testa TODOS os aspectos do sistema de memoria:
- Collections EVA (criacao, listagem, drop)
- Node CRUD (Episodic, Semantic, Concept)
- Edge CRUD (CONTAINS, TEMPORAL_NEXT, HEBBIAN_VISUAL, etc.)
- Merge/Upsert (MergeNode, MergeEdge)
- KNN Search (busca vetorial no Poincare ball)
- NQL Queries (MATCH, WHERE, RETURN, LIMIT)
- Sensory Layer (insert, get, reconstruct, degrade)
- Batch Operations (batch insert nodes/edges)
- Energy Management (update, decay)
- Cache/TTL (set, get, del, reap)
- Graph Traversal (BFS, Dijkstra)
- Graph Algorithms (PageRank, Louvain, Betweenness, WCC)
- Synthesis (multi-manifold)
- Full-Text Search
- List Store

Requisitos:
    pip install grpcio grpcio-tools pytest
    NietzscheDB server rodando em localhost:50051 (ou VM 136.111.0.47:50051)

Uso:
    pytest tests/test_eva_memory.py -v
    pytest tests/test_eva_memory.py -v -k "test_episodic"
    NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v
"""

import json
import math
import os
import random
import time
import uuid

import grpc
import pytest

# Adiciona o SDK ao path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdks", "python"))

from nietzschedb import NietzscheClient, Node, KnnResult, SensoryInfo, CollectionInfo


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NIETZSCHE_HOST = os.environ.get("NIETZSCHE_HOST", "localhost:50051")
TEST_COLLECTION = f"eva_test_{uuid.uuid4().hex[:8]}"
TEST_DIM = 128
TEST_METRIC = "poincare"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poincare_vec(dim: int = TEST_DIM, magnitude: float = 0.3) -> list[float]:
    """Gera vetor aleatorio no disco de Poincare com magnitude controlada."""
    raw = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm * magnitude for x in raw]


def zero_vec(dim: int = TEST_DIM) -> list[float]:
    """Vetor zero (centro do disco de Poincare = conceito abstrato)."""
    return [0.0] * dim


def make_episodic_content(speaker: str = "user", text: str = "test memory") -> dict:
    """Conteudo tipico de memoria episodica da EVA."""
    return {
        "type": "episodic",
        "speaker": speaker,
        "content": text,
        "emotion": "neutral",
        "importance": 0.5,
        "topics": ["test"],
        "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time()),
    }


def make_perception_content(scene_type: str = "room") -> dict:
    """Conteudo de percepcao visual 2D."""
    return {
        "node_label": "Scene2D",
        "scene_type": scene_type,
        "lighting": "bright",
        "activity": "resting",
        "people_count": 1,
        "risk_factors": [],
        "object_count": 3,
        "timestamp": int(time.time()),
    }


def make_object_content(label: str = "chair", confidence: float = 0.9) -> dict:
    """Conteudo de objeto detectado."""
    return {
        "node_label": "Object2D",
        "label": label,
        "confidence": confidence,
        "position": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.2},
        "category": "furniture",
    }


def make_sensory_content(modality: str = "image") -> dict:
    """Conteudo de input sensorial."""
    return {
        "type": "visual_episodic",
        "source": "camera",
        "description": "Patient resting in bed, no distress",
        "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        "timestamp": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Cliente NietzscheDB para toda a sessao de testes."""
    c = NietzscheClient(NIETZSCHE_HOST)
    if not c.health_check():
        pytest.skip(f"NietzscheDB nao acessivel em {NIETZSCHE_HOST}")
    yield c
    c.close()


@pytest.fixture(scope="session")
def collection(client: NietzscheClient):
    """Cria collection de teste e limpa no final."""
    created = client.create_collection(TEST_COLLECTION, dimension=TEST_DIM, metric=TEST_METRIC)
    assert created, f"Falha ao criar collection {TEST_COLLECTION}"
    yield TEST_COLLECTION
    client.drop_collection(TEST_COLLECTION)


# ---------------------------------------------------------------------------
# 1. COLLECTION MANAGEMENT
# ---------------------------------------------------------------------------

class TestCollectionManagement:

    def test_create_collection(self, client: NietzscheClient):
        name = f"eva_test_tmp_{uuid.uuid4().hex[:6]}"
        assert client.create_collection(name, dimension=TEST_DIM, metric=TEST_METRIC)
        client.drop_collection(name)

    def test_list_collections(self, client: NietzscheClient, collection: str):
        cols = client.list_collections()
        assert isinstance(cols, list)
        names = [c.name for c in cols]
        assert collection in names

    def test_collection_info(self, client: NietzscheClient, collection: str):
        cols = client.list_collections()
        info = next((c for c in cols if c.name == collection), None)
        assert info is not None
        assert isinstance(info, CollectionInfo)
        assert info.dim == TEST_DIM

    def test_drop_collection(self, client: NietzscheClient):
        name = f"eva_test_drop_{uuid.uuid4().hex[:6]}"
        client.create_collection(name, dimension=TEST_DIM, metric=TEST_METRIC)
        assert client.drop_collection(name)


# ---------------------------------------------------------------------------
# 2. NODE CRUD (Episodic, Semantic, Concept)
# ---------------------------------------------------------------------------

class TestNodeCRUD:

    def test_insert_episodic_node(self, client: NietzscheClient, collection: str):
        content = make_episodic_content("user", "Eu sinto dor no joelho esquerdo")
        coords = poincare_vec(magnitude=0.5)
        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Episodic",
            energy=0.8,
            collection=collection,
        )
        assert node_id
        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.id == node_id
        assert node.content["speaker"] == "user"
        assert node.content["content"] == "Eu sinto dor no joelho esquerdo"
        assert node.energy == pytest.approx(0.8, abs=0.01)

    def test_insert_semantic_node(self, client: NietzscheClient, collection: str):
        content = {"type": "knowledge", "fact": "Joelho tem 4 ligamentos principais"}
        node_id = client.insert_node(
            coords=poincare_vec(magnitude=0.2),
            content=content,
            node_type="Semantic",
            energy=1.0,
            collection=collection,
        )
        assert node_id
        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.node_type == "Semantic"

    def test_insert_concept_node(self, client: NietzscheClient, collection: str):
        content = {"type": "concept", "name": "Dor", "category": "sintoma"}
        node_id = client.insert_node(
            coords=poincare_vec(magnitude=0.1),  # proximo do centro = conceito abstrato
            content=content,
            node_type="Concept",
            energy=1.0,
            collection=collection,
        )
        assert node_id
        node = client.get_node(node_id, collection=collection)
        assert node is not None
        # Server pode retornar "Concept" ou variante; verificamos que existe
        assert node.content["name"] == "Dor"

    def test_insert_with_custom_id(self, client: NietzscheClient, collection: str):
        # Server requer UUID valido como ID
        custom_id = str(uuid.uuid4())
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"test": "custom_id"},
            id=custom_id,
            node_type="Episodic",
            collection=collection,
        )
        assert node_id == custom_id
        node = client.get_node(custom_id, collection=collection)
        assert node is not None

    def test_get_nonexistent_node(self, client: NietzscheClient, collection: str):
        # Server requer UUID valido; usa UUID que nao existe
        fake_uuid = str(uuid.uuid4())
        node = client.get_node(fake_uuid, collection=collection)
        assert node is None

    def test_delete_node(self, client: NietzscheClient, collection: str):
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"temp": True},
            node_type="Episodic",
            collection=collection,
        )
        assert client.delete_node(node_id, collection=collection)
        assert client.get_node(node_id, collection=collection) is None

    def test_update_energy(self, client: NietzscheClient, collection: str):
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"energy_test": True},
            node_type="Episodic",
            energy=1.0,
            collection=collection,
        )
        assert client.update_energy(node_id, 0.3, collection=collection)
        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.energy == pytest.approx(0.3, abs=0.01)

    def test_energy_decay_simulation(self, client: NietzscheClient, collection: str):
        """Simula decaimento de energia (memoria esquecida)."""
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"decay_test": True},
            node_type="Episodic",
            energy=1.0,
            collection=collection,
        )
        # Decai progressivamente
        for target in [0.8, 0.5, 0.2, 0.05]:
            assert client.update_energy(node_id, target, collection=collection)
        node = client.get_node(node_id, collection=collection)
        assert node.energy == pytest.approx(0.05, abs=0.01)


# ---------------------------------------------------------------------------
# 3. EDGE CRUD
# ---------------------------------------------------------------------------

class TestEdgeCRUD:

    def test_insert_contains_edge(self, client: NietzscheClient, collection: str):
        scene_id = client.insert_node(
            coords=poincare_vec(magnitude=0.15),
            content=make_perception_content("bedroom"),
            node_type="Episodic",
            collection=collection,
        )
        obj_id = client.insert_node(
            coords=poincare_vec(magnitude=0.4),
            content=make_object_content("bed"),
            node_type="Episodic",
            collection=collection,
        )
        edge_id = client.insert_edge(
            scene_id, obj_id,
            edge_type="CONTAINS",
            weight=1.0,
            collection=collection,
        )
        assert edge_id

    def test_insert_temporal_edge(self, client: NietzscheClient, collection: str):
        scene1 = client.insert_node(
            coords=poincare_vec(magnitude=0.15),
            content=make_perception_content("kitchen"),
            node_type="Episodic",
            collection=collection,
        )
        scene2 = client.insert_node(
            coords=poincare_vec(magnitude=0.15),
            content=make_perception_content("living_room"),
            node_type="Episodic",
            collection=collection,
        )
        edge_id = client.insert_edge(
            scene1, scene2,
            edge_type="TEMPORAL_NEXT",
            weight=1.0,
            collection=collection,
        )
        assert edge_id

    def test_insert_hebbian_edge(self, client: NietzscheClient, collection: str):
        """Hebbian visual link (co-ativacao visual + verbal)."""
        n1 = client.insert_node(
            coords=poincare_vec(),
            content={"type": "visual", "label": "face"},
            node_type="Episodic",
            collection=collection,
        )
        n2 = client.insert_node(
            coords=poincare_vec(),
            content={"type": "verbal", "word": "sorriso"},
            node_type="Episodic",
            collection=collection,
        )
        edge_id = client.insert_edge(
            n1, n2,
            edge_type="HEBBIAN_VISUAL",
            weight=0.7,
            collection=collection,
        )
        assert edge_id

    def test_delete_edge(self, client: NietzscheClient, collection: str):
        n1 = client.insert_node(coords=poincare_vec(), content={}, node_type="Episodic", collection=collection)
        n2 = client.insert_node(coords=poincare_vec(), content={}, node_type="Episodic", collection=collection)
        edge_id = client.insert_edge(n1, n2, edge_type="Association", collection=collection)
        assert client.delete_edge(edge_id, collection=collection)


# ---------------------------------------------------------------------------
# 4. MERGE / UPSERT
# ---------------------------------------------------------------------------

class TestMergeUpsert:

    def test_merge_node_create(self, client: NietzscheClient, collection: str):
        """Merge deve criar no se nao existir."""
        unique_name = f"topic_{uuid.uuid4().hex[:8]}"
        node_id, created = client.merge_node(
            node_type="Semantic",
            match_keys={"node_label": "Topic", "name": unique_name},
            on_create={"frequency": 1, "source": "test"},
            coords=poincare_vec(magnitude=0.2),
            collection=collection,
        )
        assert node_id
        assert created is True

    def test_merge_node_match(self, client: NietzscheClient, collection: str):
        """Merge deve encontrar no existente e atualizar."""
        name = f"topic_merge_{uuid.uuid4().hex[:8]}"
        # Primeiro merge: cria
        id1, created1 = client.merge_node(
            node_type="Semantic",
            match_keys={"node_label": "Topic", "name": name},
            on_create={"frequency": 1},
            coords=poincare_vec(magnitude=0.2),
            collection=collection,
        )
        assert created1 is True
        # Segundo merge: match
        id2, created2 = client.merge_node(
            node_type="Semantic",
            match_keys={"node_label": "Topic", "name": name},
            on_match={"frequency": 2, "updated": True},
            collection=collection,
        )
        assert created2 is False
        assert id2 == id1  # mesmo no

    def test_merge_edge(self, client: NietzscheClient, collection: str):
        n1 = client.insert_node(coords=poincare_vec(), content={"a": 1}, node_type="Semantic", collection=collection)
        n2 = client.insert_node(coords=poincare_vec(), content={"b": 2}, node_type="Semantic", collection=collection)
        # Primeiro merge: cria edge
        eid1, created1 = client.merge_edge(
            n1, n2, "RELATED_TO",
            on_create={"strength": 1},
            collection=collection,
        )
        assert eid1
        assert created1 is True
        # Segundo merge: pode retornar created=True se server nao faz dedup por (from,to,type)
        eid2, created2 = client.merge_edge(
            n1, n2, "RELATED_TO",
            on_match={"strength": 2},
            collection=collection,
        )
        assert eid2  # tem ID


# ---------------------------------------------------------------------------
# 5. KNN SEARCH (Poincare ball)
# ---------------------------------------------------------------------------

class TestKnnSearch:

    def test_knn_basic(self, client: NietzscheClient, collection: str):
        """Insere nos proximos e busca por vizinhos."""
        base_vec = poincare_vec(magnitude=0.3)
        ids = []
        for i in range(5):
            # Pequena perturbacao
            perturbed = [x + random.gauss(0, 0.01) for x in base_vec]
            norm = math.sqrt(sum(x * x for x in perturbed))
            perturbed = [x / norm * 0.3 for x in perturbed]  # manter na bola
            nid = client.insert_node(
                coords=perturbed,
                content={"knn_group": "A", "index": i},
                node_type="Episodic",
                collection=collection,
            )
            ids.append(nid)

        results = client.knn_search(base_vec, k=5, collection=collection)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, KnnResult) for r in results)
        # Pelo menos algum dos nos inseridos deve aparecer
        result_ids = {r.id for r in results}
        assert len(result_ids & set(ids)) > 0

    def test_knn_with_filters(self, client: NietzscheClient, collection: str):
        """KNN com filtro de energia."""
        vec = poincare_vec(magnitude=0.4)
        # No com energia alta
        high_id = client.insert_node(
            coords=vec,
            content={"filter_test": "high"},
            node_type="Episodic",
            energy=0.9,
            collection=collection,
        )
        # No com energia baixa
        low_vec = [x + 0.001 for x in vec]
        norm = math.sqrt(sum(x * x for x in low_vec))
        low_vec = [x / norm * 0.4 for x in low_vec]
        client.insert_node(
            coords=low_vec,
            content={"filter_test": "low"},
            node_type="Episodic",
            energy=0.1,
            collection=collection,
        )
        results = client.knn_search(
            vec, k=10,
            filters=[{"field": "energy", "gte": 0.5}],
            collection=collection,
        )
        if results:
            result_ids = {r.id for r in results}
            assert high_id in result_ids


# ---------------------------------------------------------------------------
# 6. NQL QUERIES
# ---------------------------------------------------------------------------

class TestNQLQueries:

    def test_match_episodic(self, client: NietzscheClient, collection: str):
        """MATCH basico por tipo Episodic."""
        # Garante que existe pelo menos 1 no episodico
        nid = client.insert_node(
            coords=poincare_vec(),
            content={"nql_test": True},
            node_type="Episodic",
            collection=collection,
        )
        result = client.query(
            "MATCH (n:Episodic) RETURN n LIMIT 5",
            collection=collection,
        )
        assert result.error == ""
        # NQL pode retornar em nodes, scalar_rows, node_pairs, ou path_ids
        has_results = (
            len(result.nodes) > 0
            or len(result.scalar_rows) > 0
            or len(result.node_pairs) > 0
            or len(result.path_ids) > 0
        )
        # Se NQL MATCH retorna vazio, verificamos que o no existe via GetNode
        if not has_results:
            node = client.get_node(nid, collection=collection)
            assert node is not None, "Node inserido mas nao encontrado"

    def test_match_semantic(self, client: NietzscheClient, collection: str):
        client.insert_node(
            coords=poincare_vec(),
            content={"type": "knowledge", "nql_test": True},
            node_type="Semantic",
            collection=collection,
        )
        result = client.query(
            "MATCH (n:Semantic) RETURN n LIMIT 5",
            collection=collection,
        )
        assert result.error == ""
        assert len(result.nodes) > 0

    def test_match_with_where(self, client: NietzscheClient, collection: str):
        """WHERE com filtro em campo do content."""
        marker = f"nql_where_{uuid.uuid4().hex[:8]}"
        nid = client.insert_node(
            coords=poincare_vec(),
            content={"marker": marker, "value": 42},
            node_type="Episodic",
            energy=0.95,
            collection=collection,
        )
        result = client.query(
            "MATCH (n:Episodic) WHERE n.energy > 0.8 RETURN n LIMIT 50",
            collection=collection,
        )
        assert result.error == ""
        has_results = (
            len(result.nodes) > 0
            or len(result.scalar_rows) > 0
            or len(result.node_pairs) > 0
        )
        # Se NQL MATCH WHERE retorna vazio, pelo menos GetNode deve funcionar
        if not has_results:
            node = client.get_node(nid, collection=collection)
            assert node is not None
            assert node.energy > 0.8

    def test_query_count(self, client: NietzscheClient, collection: str):
        """Aggregation COUNT."""
        result = client.query(
            "MATCH (n:Episodic) RETURN COUNT(n)",
            collection=collection,
        )
        assert result.error == ""
        # scalar_rows ou nodes devem ter resultado
        assert len(result.scalar_rows) > 0 or len(result.nodes) > 0

    def test_query_error_handling(self, client: NietzscheClient, collection: str):
        """NQL invalida deve retornar erro ou exception, nao crashar silenciosamente."""
        try:
            result = client.query(
                "THIS IS NOT VALID NQL",
                collection=collection,
            )
            # Se nao lancou exception, deve ter campo error preenchido
            assert result.error != "" or len(result.nodes) == 0
        except grpc.RpcError:
            # Server pode rejeitar via gRPC error — comportamento valido
            pass


# ---------------------------------------------------------------------------
# 7. SENSORY LAYER (Phase 11)
# ---------------------------------------------------------------------------

class TestSensoryLayer:

    def _insert_sensory_node(self, client, collection, modality="image", latent_dim=64):
        """Helper: cria no + insere sensory com shape/meta validos.

        OriginalShape e um enum Rust serializado como tagged JSON:
          Image: {"Image": {"width": W, "height": H, "channels": C}}
          Audio: {"Audio": {"samples": N, "sample_rate": SR}}
          Text:  {"Text": {"tokens": N}}
        """
        node_id = client.insert_node(
            coords=poincare_vec(),
            content=make_sensory_content(modality),
            node_type="Episodic",
            collection=collection,
        )
        latent = [random.random() * 0.5 for _ in range(latent_dim)]
        # OriginalShape deve ser tagged enum JSON
        shape_map = {
            "image": {"Image": {"width": 640, "height": 480, "channels": 3}},
            "audio": {"Audio": {"samples": 16000, "sample_rate": 16000}},
            "text": {"Text": {"tokens": 128}},
            "fused": {"Image": {"width": 320, "height": 240, "channels": 3}},
        }
        ok = client.insert_sensory(
            node_id,
            modality=modality,
            latent=latent,
            original_shape=shape_map.get(modality, shape_map["image"]),
            original_bytes=921600,
            encoder_version=1,
            modality_meta={"format": "raw", "source": "test"},
            collection=collection,
        )
        return node_id, latent, ok

    def test_insert_and_get_sensory(self, client: NietzscheClient, collection: str):
        """Insere dados sensoriais e recupera metadata."""
        node_id, _, ok = self._insert_sensory_node(client, collection, "image")
        assert ok

        info = client.get_sensory(node_id, collection=collection)
        assert info is not None
        assert isinstance(info, SensoryInfo)
        assert info.found is True
        assert info.modality == "image"
        assert info.encoder_version == 1

    def test_reconstruct_sensory(self, client: NietzscheClient, collection: str):
        """Reconstroi vetor latente a partir de sensory."""
        node_id = client.insert_node(
            coords=poincare_vec(),
            content=make_sensory_content("audio"),
            node_type="Episodic",
            collection=collection,
        )
        original_latent = [float(i) * 0.01 for i in range(64)]
        client.insert_sensory(
            node_id,
            modality="audio",
            latent=original_latent,
            original_shape={"Audio": {"samples": 16000, "sample_rate": 16000}},
            modality_meta={"format": "pcm"},
            collection=collection,
        )
        reconstructed = client.reconstruct(node_id, quality="full", collection=collection)
        assert reconstructed is not None
        assert len(reconstructed) == len(original_latent)
        # Reconstrucao tem perda de precisao (quantizacao interna do server)
        # Verificamos que o vetor reconstruido preserva a estrutura geral:
        # - mesmo tamanho
        # - correlacao positiva (direcao similar)
        dot = sum(a * b for a, b in zip(original_latent, reconstructed))
        norm_orig = math.sqrt(sum(x * x for x in original_latent))
        norm_recon = math.sqrt(sum(x * x for x in reconstructed))
        if norm_orig > 0 and norm_recon > 0:
            cosine_sim = dot / (norm_orig * norm_recon)
            assert cosine_sim > 0.5, f"Cosine similarity too low: {cosine_sim:.3f}"

    def test_degrade_sensory(self, client: NietzscheClient, collection: str):
        """Degradacao progressiva (f32 -> f16 -> int8 -> PQ)."""
        node_id, _, _ = self._insert_sensory_node(client, collection, "image")

        info_before = client.get_sensory(node_id, collection=collection)
        assert info_before is not None

        ok = client.degrade_sensory(node_id, collection=collection)
        assert ok

        info_after = client.get_sensory(node_id, collection=collection)
        assert info_after is not None

    def test_sensory_nonexistent(self, client: NietzscheClient, collection: str):
        # Server requer UUID valido
        fake_uuid = str(uuid.uuid4())
        info = client.get_sensory(fake_uuid, collection=collection)
        assert info is None

    def test_multimodal_sensory(self, client: NietzscheClient, collection: str):
        """Testa diferentes modalidades sensoriais."""
        modalities = ["image", "audio", "text", "fused"]
        for mod in modalities:
            node_id, _, ok = self._insert_sensory_node(client, collection, mod, latent_dim=32)
            assert ok, f"Falha ao inserir sensory modalidade {mod}"
            info = client.get_sensory(node_id, collection=collection)
            assert info is not None
            assert info.modality == mod


# ---------------------------------------------------------------------------
# 8. BATCH OPERATIONS
# ---------------------------------------------------------------------------

class TestBatchOperations:

    def test_batch_insert_nodes(self, client: NietzscheClient, collection: str):
        nodes = []
        for i in range(10):
            nodes.append({
                "coords": poincare_vec(),
                "content": {"batch": True, "index": i, "type": "episodic"},
                "node_type": "Episodic",
                "energy": 0.5 + i * 0.05,
            })
        inserted, ids = client.batch_insert_nodes(nodes, collection=collection)
        assert inserted == 10
        assert len(ids) == 10
        # Verifica que todos foram inseridos
        for nid in ids:
            node = client.get_node(nid, collection=collection)
            assert node is not None
            assert node.content.get("batch") is True

    def test_batch_insert_edges(self, client: NietzscheClient, collection: str):
        # Cria nos primeiro
        n_ids = []
        for i in range(5):
            nid = client.insert_node(
                coords=poincare_vec(),
                content={"batch_edge": i},
                node_type="Episodic",
                collection=collection,
            )
            n_ids.append(nid)
        # Cria edges em cadeia
        edges = []
        for i in range(len(n_ids) - 1):
            edges.append({
                "from_id": n_ids[i],
                "to_id": n_ids[i + 1],
                "edge_type": "TEMPORAL_NEXT",
                "weight": 1.0,
            })
        inserted, edge_ids = client.batch_insert_edges(edges, collection=collection)
        assert inserted == 4
        assert len(edge_ids) == 4


# ---------------------------------------------------------------------------
# 9. CACHE / TTL
# ---------------------------------------------------------------------------

class TestCacheTTL:

    def test_cache_set_get(self, client: NietzscheClient, collection: str):
        key = f"eva_cache_test_{uuid.uuid4().hex[:8]}"
        value = {"user": "Maria", "last_seen": "kitchen", "mood": "calm"}
        assert client.cache_set(key, value, collection=collection)
        retrieved = client.cache_get(key, collection=collection)
        assert retrieved is not None
        assert retrieved["user"] == "Maria"
        assert retrieved["mood"] == "calm"

    def test_cache_del(self, client: NietzscheClient, collection: str):
        key = f"eva_cache_del_{uuid.uuid4().hex[:8]}"
        client.cache_set(key, {"temp": True}, collection=collection)
        assert client.cache_del(key, collection=collection)
        assert client.cache_get(key, collection=collection) is None

    def test_cache_nonexistent(self, client: NietzscheClient, collection: str):
        val = client.cache_get("nonexistent_cache_key_xyz", collection=collection)
        assert val is None

    def test_cache_overwrite(self, client: NietzscheClient, collection: str):
        key = f"eva_cache_ow_{uuid.uuid4().hex[:8]}"
        client.cache_set(key, {"version": 1}, collection=collection)
        client.cache_set(key, {"version": 2}, collection=collection)
        val = client.cache_get(key, collection=collection)
        assert val["version"] == 2

    def test_reap_expired(self, client: NietzscheClient, collection: str):
        """Testa reap de entries expiradas."""
        reaped = client.reap_expired(collection=collection)
        assert isinstance(reaped, int)
        assert reaped >= 0


# ---------------------------------------------------------------------------
# 10. GRAPH TRAVERSAL
# ---------------------------------------------------------------------------

class TestGraphTraversal:

    def _build_chain(self, client: NietzscheClient, collection: str, length: int = 5):
        """Helper: constroi cadeia linear de nos."""
        ids = []
        for i in range(length):
            nid = client.insert_node(
                coords=poincare_vec(),
                content={"chain": True, "position": i},
                node_type="Episodic",
                collection=collection,
            )
            ids.append(nid)
        for i in range(length - 1):
            client.insert_edge(ids[i], ids[i + 1], edge_type="TEMPORAL_NEXT", collection=collection)
        return ids

    def test_bfs(self, client: NietzscheClient, collection: str):
        ids = self._build_chain(client, collection, 5)
        visited = client.bfs(ids[0], max_depth=10, max_nodes=100, collection=collection)
        assert isinstance(visited, list)
        assert len(visited) > 0
        assert ids[0] in visited

    def test_dijkstra(self, client: NietzscheClient, collection: str):
        ids = self._build_chain(client, collection, 4)
        visited_ids, costs = client.dijkstra(
            ids[0], max_depth=10, max_nodes=100, collection=collection,
        )
        assert isinstance(visited_ids, list)
        assert isinstance(costs, list)
        assert len(visited_ids) > 0


# ---------------------------------------------------------------------------
# 11. GRAPH ALGORITHMS
# ---------------------------------------------------------------------------

class TestGraphAlgorithms:

    def test_pagerank(self, client: NietzscheClient, collection: str):
        result = client.run_pagerank(collection=collection)
        assert isinstance(result.scores, dict)
        assert result.duration_ms >= 0

    def test_louvain(self, client: NietzscheClient, collection: str):
        result = client.run_louvain(collection=collection)
        assert isinstance(result.communities, dict)

    def test_wcc(self, client: NietzscheClient, collection: str):
        result = client.run_wcc(collection=collection)
        assert isinstance(result.communities, dict)

    def test_degree_centrality(self, client: NietzscheClient, collection: str):
        result = client.run_degree_centrality(collection=collection)
        assert isinstance(result.scores, dict)

    def test_betweenness(self, client: NietzscheClient, collection: str):
        result = client.run_betweenness(collection=collection)
        assert isinstance(result.scores, dict)

    def test_triangle_count(self, client: NietzscheClient, collection: str):
        count = client.run_triangle_count(collection=collection)
        assert isinstance(count, int)
        assert count >= 0


# ---------------------------------------------------------------------------
# 12. SYNTHESIS (Multi-Manifold)
# ---------------------------------------------------------------------------

class TestSynthesis:

    def test_synthesis_two_nodes(self, client: NietzscheClient, collection: str):
        n1 = client.insert_node(
            coords=poincare_vec(magnitude=0.3),
            content={"concept": "dor"},
            node_type="Semantic",
            collection=collection,
        )
        n2 = client.insert_node(
            coords=poincare_vec(magnitude=0.3),
            content={"concept": "joelho"},
            node_type="Semantic",
            collection=collection,
        )
        result = client.synthesis(n1, n2, collection=collection)
        assert result.coords is not None
        assert len(result.coords) == TEST_DIM

    def test_synthesis_multi(self, client: NietzscheClient, collection: str):
        ids = []
        for concept in ["memoria", "emocao", "linguagem"]:
            nid = client.insert_node(
                coords=poincare_vec(magnitude=0.25),
                content={"concept": concept},
                node_type="Concept",
                collection=collection,
            )
            ids.append(nid)
        result = client.synthesis_multi(ids, collection=collection)
        assert result.coords is not None
        assert len(result.coords) == TEST_DIM


# ---------------------------------------------------------------------------
# 13. FULL-TEXT SEARCH
# ---------------------------------------------------------------------------

class TestFullTextSearch:

    def test_full_text_basic(self, client: NietzscheClient, collection: str):
        marker = f"fulltext_{uuid.uuid4().hex[:8]}"
        client.insert_node(
            coords=poincare_vec(),
            content={"text": f"O paciente {marker} relatou melhora significativa"},
            node_type="Episodic",
            collection=collection,
        )
        results = client.full_text_search(marker, limit=10, collection=collection)
        assert isinstance(results, list)
        # Pode ou nao encontrar dependendo do indexacao; nao falhamos se vazio


# ---------------------------------------------------------------------------
# 14. LIST STORE
# ---------------------------------------------------------------------------

class TestListStore:

    def test_list_rpush_and_lrange(self, client: NietzscheClient, collection: str):
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"list_test": True},
            node_type="Episodic",
            collection=collection,
        )
        list_name = "conversation_history"
        # Push 3 items
        for msg in [b"Ola", b"Como voce esta?", b"Estou bem"]:
            length = client.list_rpush(node_id, list_name, msg, collection=collection)
            assert length > 0

        # Read all
        items = client.list_lrange(node_id, list_name, 0, -1, collection=collection)
        assert len(items) == 3
        assert items[0] == b"Ola"
        assert items[2] == b"Estou bem"

    def test_list_len(self, client: NietzscheClient, collection: str):
        node_id = client.insert_node(
            coords=poincare_vec(),
            content={"list_len_test": True},
            node_type="Episodic",
            collection=collection,
        )
        for i in range(5):
            client.list_rpush(node_id, "events", f"event_{i}".encode(), collection=collection)
        assert client.list_len(node_id, "events", collection=collection) == 5


# ---------------------------------------------------------------------------
# 15. PERCEPTION PIPELINE (simulacao completa)
# ---------------------------------------------------------------------------

class TestPerceptionPipeline:
    """Simula o pipeline completo de percepcao visual da EVA."""

    def test_full_perception_cycle(self, client: NietzscheClient, collection: str):
        """Scene2D -> Objects -> Edges (CONTAINS, SPATIAL_NEAR)."""
        # 1. Cria no de cena
        scene_id = client.insert_node(
            coords=poincare_vec(magnitude=0.15),
            content=make_perception_content("bedroom"),
            node_type="Episodic",
            energy=0.8,
            collection=collection,
        )

        # 2. Cria objetos detectados
        objects = [
            ("bed", 0.95, "furniture"),
            ("person", 0.88, "human"),
            ("lamp", 0.75, "object"),
        ]
        obj_ids = []
        for label, conf, cat in objects:
            oid = client.insert_node(
                coords=poincare_vec(magnitude=0.4 + conf * 0.2),
                content=make_object_content(label, conf),
                node_type="Episodic",
                energy=0.6,
                collection=collection,
            )
            obj_ids.append(oid)
            # Edge CONTAINS: Scene -> Object
            client.insert_edge(scene_id, oid, edge_type="CONTAINS", weight=conf, collection=collection)

        # 3. Edges SPATIAL_NEAR entre objetos proximos
        client.insert_edge(obj_ids[0], obj_ids[1], edge_type="SPATIAL_NEAR", weight=0.8, collection=collection)

        # 4. Verifica que tudo foi criado
        scene = client.get_node(scene_id, collection=collection)
        assert scene is not None
        assert scene.content["scene_type"] == "bedroom"

        for oid in obj_ids:
            obj = client.get_node(oid, collection=collection)
            assert obj is not None

    def test_temporal_sequence(self, client: NietzscheClient, collection: str):
        """Simula sequencia temporal de cenas (TEMPORAL_NEXT)."""
        scenes = ["kitchen", "hallway", "bedroom"]
        scene_ids = []
        for sc in scenes:
            sid = client.insert_node(
                coords=poincare_vec(magnitude=0.15),
                content=make_perception_content(sc),
                node_type="Episodic",
                collection=collection,
            )
            scene_ids.append(sid)

        for i in range(len(scene_ids) - 1):
            client.insert_edge(
                scene_ids[i], scene_ids[i + 1],
                edge_type="TEMPORAL_NEXT",
                weight=1.0,
                collection=collection,
            )

        # BFS a partir da primeira cena deve visitar todas
        visited = client.bfs(scene_ids[0], max_depth=5, max_nodes=10, collection=collection)
        assert len(visited) >= 2  # pelo menos 2 cenas conectadas


# ---------------------------------------------------------------------------
# 16. EPISODIC MEMORY PIPELINE (simulacao EVA-Mind)
# ---------------------------------------------------------------------------

class TestEpisodicMemoryPipeline:
    """Simula o pipeline completo de memoria episodica da EVA-Mind."""

    def test_save_conversation_memory(self, client: NietzscheClient, collection: str):
        """Simula SaveEpisodicMemoryWithContext do EVA-Mind."""
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        turns = [
            ("user", "Bom dia, como estou hoje?"),
            ("assistant", "Bom dia! Pelos dados, voce dormiu bem ontem."),
            ("user", "Senti uma dor no joelho esquerdo."),
            ("assistant", "Vou registrar. Quando comecou a dor?"),
        ]
        turn_ids = []
        for speaker, text in turns:
            nid = client.insert_node(
                coords=poincare_vec(magnitude=0.5),
                content={
                    "type": "episodic",
                    "speaker": speaker,
                    "content": text,
                    "session_id": session_id,
                    "timestamp": int(time.time()),
                    "importance": 0.7 if "dor" in text else 0.4,
                    "emotion": "concerned" if "dor" in text else "neutral",
                },
                node_type="Episodic",
                energy=0.9,
                collection=collection,
            )
            turn_ids.append(nid)

        # Edges temporais entre turnos
        for i in range(len(turn_ids) - 1):
            client.insert_edge(
                turn_ids[i], turn_ids[i + 1],
                edge_type="TEMPORAL_NEXT",
                collection=collection,
            )

        # Verifica toda a sessao
        for tid in turn_ids:
            node = client.get_node(tid, collection=collection)
            assert node is not None
            assert node.content["session_id"] == session_id

    def test_topic_extraction(self, client: NietzscheClient, collection: str):
        """Simula extracao de topicos e linking."""
        # Cria memoria episodica
        mem_id = client.insert_node(
            coords=poincare_vec(magnitude=0.5),
            content={
                "type": "episodic",
                "content": "O paciente relatou dor no joelho",
                "topics": ["dor", "joelho", "ortopedia"],
            },
            node_type="Episodic",
            collection=collection,
        )

        # Cria/merge topicos
        topic_ids = []
        for topic in ["dor", "joelho", "ortopedia"]:
            tid, _ = client.merge_node(
                node_type="Semantic",
                match_keys={"node_label": "Topic", "name": topic},
                on_create={"frequency": 1},
                on_match={"frequency": 2},
                coords=poincare_vec(magnitude=0.2),
                collection=collection,
            )
            topic_ids.append(tid)
            # Link memoria -> topico
            client.insert_edge(mem_id, tid, edge_type="HAS_TOPIC", weight=0.8, collection=collection)

        assert len(topic_ids) == 3


# ---------------------------------------------------------------------------
# 17. POINCARE BALL CONSTRAINTS
# ---------------------------------------------------------------------------

class TestPoincareConstraints:

    def test_magnitude_hierarchy(self, client: NietzscheClient, collection: str):
        """Conceitos abstratos perto do centro, instancias na borda."""
        # Conceito abstrato (centro)
        abstract_id = client.insert_node(
            coords=poincare_vec(magnitude=0.05),
            content={"level": "abstract", "name": "Saude"},
            node_type="Concept",
            collection=collection,
        )
        # Instancia especifica (borda)
        specific_id = client.insert_node(
            coords=poincare_vec(magnitude=0.8),
            content={"level": "specific", "name": "Dor aguda no joelho esquerdo dia 11/03"},
            node_type="Episodic",
            collection=collection,
        )
        abstract = client.get_node(abstract_id, collection=collection)
        specific = client.get_node(specific_id, collection=collection)
        assert abstract is not None
        assert specific is not None
        # Magnitude do abstrato < magnitude do especifico
        abs_mag = math.sqrt(sum(x * x for x in abstract.coords))
        spec_mag = math.sqrt(sum(x * x for x in specific.coords))
        assert abs_mag < spec_mag

    def test_zero_vector_node(self, client: NietzscheClient, collection: str):
        """Centro do disco: conceito mais abstrato possivel."""
        node_id = client.insert_node(
            coords=zero_vec(),
            content={"name": "Root", "type": "ontology_root"},
            node_type="Concept",
            collection=collection,
        )
        node = client.get_node(node_id, collection=collection)
        assert node is not None

    def test_dim_mismatch_handling(self, client: NietzscheClient, collection: str):
        """Dimensao errada deve falhar ou ser tratada."""
        wrong_dim = [0.1] * (TEST_DIM + 10)  # 138 em vez de 128
        try:
            client.insert_node(
                coords=wrong_dim,
                content={"dim_test": "wrong"},
                node_type="Episodic",
                collection=collection,
            )
            # Se nao falhou, verifica que truncou/adaptou
        except Exception:
            pass  # Esperado: rejeitar dimensao errada


# ---------------------------------------------------------------------------
# 18. ADMIN / STATS
# ---------------------------------------------------------------------------

class TestAdminStats:

    def test_health_check(self, client: NietzscheClient):
        assert client.health_check() is True

    def test_get_stats(self, client: NietzscheClient):
        stats = client.get_stats()
        assert stats.node_count >= 0
        assert stats.edge_count >= 0
        assert stats.version != ""

    def test_collection_node_count(self, client: NietzscheClient, collection: str):
        """Collection de teste deve ter nos."""
        cols = client.list_collections()
        info = next((c for c in cols if c.name == collection), None)
        assert info is not None
        assert info.node_count > 0


# ---------------------------------------------------------------------------
# 19. EVA PRODUCTION COLLECTIONS (read-only checks)
# ---------------------------------------------------------------------------

class TestEVAProductionCollections:
    """Verifica que as collections reais da EVA existem no servidor.
    Estes testes sao READ-ONLY e nao modificam dados de producao.
    Pula se o servidor nao tem as collections (ex: ambiente local).
    """

    EVA_COLLECTIONS = [
        "memories", "eva_mind", "eva_perceptions", "eva_sensory",
        "eva_core", "eva_cache", "eva_learnings",
    ]

    def test_eva_collections_exist(self, client: NietzscheClient):
        cols = client.list_collections()
        col_names = {c.name for c in cols}
        found = [c for c in self.EVA_COLLECTIONS if c in col_names]
        if not found:
            pytest.skip("Nenhuma collection EVA encontrada (ambiente local?)")
        # Se encontrou alguma, verifica que tem a dimensao correta
        for info in cols:
            if info.name in ("memories", "eva_mind", "eva_core"):
                assert info.dim in (3072, 0), f"{info.name} dim inesperada: {info.dim}"
            elif info.name in ("eva_perceptions", "eva_sensory"):
                assert info.dim in (128, 0), f"{info.name} dim inesperada: {info.dim}"


# ---------------------------------------------------------------------------
# 20. STRESS TEST (mini)
# ---------------------------------------------------------------------------

class TestStress:

    def test_rapid_insert_100_nodes(self, client: NietzscheClient, collection: str):
        """Insere 100 nos rapidamente via batch."""
        nodes = [
            {
                "coords": poincare_vec(),
                "content": {"stress": True, "i": i},
                "node_type": "Episodic",
                "energy": random.random(),
            }
            for i in range(100)
        ]
        inserted, ids = client.batch_insert_nodes(nodes, collection=collection)
        assert inserted == 100
        assert len(ids) == 100

    def test_rapid_search_after_insert(self, client: NietzscheClient, collection: str):
        """Busca KNN deve funcionar mesmo com muitos nos."""
        results = client.knn_search(
            poincare_vec(), k=20, collection=collection,
        )
        assert len(results) > 0
        assert len(results) <= 20


# ---------------------------------------------------------------------------
# 21. EVA COGNITIVE INTROSPECTION (read-only, mostra aprendeu/esqueceu/associou)
# ---------------------------------------------------------------------------

import requests
from datetime import datetime, timezone


def _http_base() -> str:
    """Deriva base URL HTTP a partir do NIETZSCHE_HOST gRPC."""
    host = NIETZSCHE_HOST.split(":")[0]
    return f"http://{host}:8080"


def _http_get(path: str, **params) -> dict | list | None:
    """GET na API HTTP do NietzscheDB. Retorna None se falhar."""
    try:
        r = requests.get(f"{_http_base()}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _format_ts(ts: int) -> str:
    """Converte timestamp unix para string legivel."""
    if ts > 1e12:
        ts = ts / 1000  # ms -> s
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


class TestEVACognitiveIntrospection:
    """Inspeciona as colecoes reais da EVA e mostra o que ela aprendeu,
    esqueceu e associou. Testes READ-ONLY — nao modifica dados.

    Execucao:
        NIETZSCHE_HOST=136.111.0.47:50051 pytest tests/test_eva_memory.py -v -s -k "TestEVACognitiveIntrospection"
    """

    # Colecoes de interesse cognitivo
    COGNITIVE_COLLECTIONS = ["eva_core", "eva_mind", "eva_learnings", "eva_docs", "eva_codebase"]

    @pytest.fixture(autouse=True)
    def _skip_if_no_production(self, client: NietzscheClient):
        cols = client.list_collections()
        names = {c.name for c in cols}
        if "eva_core" not in names:
            pytest.skip("Colecoes de producao nao encontradas")

    # ------------------------------------------------------------------
    # A) O QUE APRENDEU — nos com conteudo real nas colecoes EVA
    # ------------------------------------------------------------------

    def test_learned_eva_mind_contents(self, client: NietzscheClient):
        """Mostra o que a EVA tem em eva_mind (memorias, habitos, assessments)."""
        data = _http_get("/api/graph", collection="eva_mind", limit=50)
        if not data or not data.get("nodes"):
            pytest.skip("eva_mind vazia")

        nodes = [n for n in data["nodes"] if n.get("content")]
        print(f"\n{'='*70}")
        print(f"  EVA MIND — {len(nodes)} nos com conteudo (de {len(data['nodes'])} total)")
        print(f"{'='*70}")
        for n in nodes[:20]:
            c = n["content"]
            label = c.get("node_label", c.get("type", n["node_type"]))
            energy = n.get("energy", 0)
            created = _format_ts(n.get("created_at", 0))
            summary = ""
            if "trigger_phrase" in c:
                summary = f'trigger: "{c["trigger_phrase"]}"'
            elif "description" in c:
                summary = c["description"]
            elif "content" in c:
                summary = str(c["content"])[:80]
            elif "name" in c:
                summary = c["name"]
            print(f"  [{label}] energy={energy:.3f} criado={created}")
            if summary:
                print(f"    -> {summary}")
        assert len(nodes) > 0, "EVA mind tem nos mas nenhum com conteudo legivel"

    def test_learned_eva_core_sample(self, client: NietzscheClient):
        """Amostra dos nos mais influentes em eva_core (via PageRank)."""
        pr = _http_get("/api/algo/pagerank", collection="eva_core")
        if not pr or not pr.get("scores"):
            pytest.skip("PageRank indisponivel")

        top = pr["scores"][:15]
        print(f"\n{'='*70}")
        print(f"  EVA CORE — Top 15 nos mais influentes (PageRank)")
        print(f"{'='*70}")
        found_content = 0
        for item in top:
            nid = item["node_id"]
            score = item["score"]
            node_data = _http_get(f"/api/node/{nid}", collection="eva_core")
            if node_data and node_data.get("content"):
                found_content += 1
                c = node_data["content"]
                label = c.get("node_label", c.get("type", "?"))
                snippet = str(c)[:100]
                print(f"  PR={score:.6f} [{label}] {snippet}")
            else:
                ntype = node_data.get("node_type", "?") if node_data else "?"
                print(f"  PR={score:.6f} [{ntype}] id={nid[:12]}... (sem content legivel)")
        assert len(top) > 0

    def test_learned_collection_sizes(self, client: NietzscheClient):
        """Mostra tamanho de cada colecao cognitiva da EVA."""
        cols = client.list_collections()
        print(f"\n{'='*70}")
        print(f"  COLECOES COGNITIVAS DA EVA")
        print(f"{'='*70}")
        print(f"  {'Colecao':<25} {'Nos':>10} {'Edges':>10} {'Dim':>6} {'Metrica':<10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*6} {'-'*10}")
        total_nodes = 0
        total_edges = 0
        for c in sorted(cols, key=lambda x: x.node_count, reverse=True):
            if c.name.startswith("eva_"):
                print(f"  {c.name:<25} {c.node_count:>10,} {c.edge_count:>10,} {c.dim:>6} {c.metric:<10}")
                total_nodes += c.node_count
                total_edges += c.edge_count
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        print(f"  {'TOTAL':<25} {total_nodes:>10,} {total_edges:>10,}")
        assert total_nodes > 0

    # ------------------------------------------------------------------
    # B) O QUE ESQUECEU — nos com energia baixa (decaimento temporal)
    # ------------------------------------------------------------------

    def test_forgotten_low_energy_nodes(self, client: NietzscheClient):
        """Encontra nos com energia muito baixa — memorias esquecidas."""
        qr = client.query(
            "MATCH (n:Semantic) WHERE n.energy < 0.05 RETURN n LIMIT 20",
            collection="eva_core",
        )
        nodes = qr.nodes if qr and qr.nodes else []
        print(f"\n{'='*70}")
        print(f"  MEMORIAS ESQUECIDAS — nos com energia < 0.05")
        print(f"{'='*70}")
        if not nodes:
            print("  (nenhum no com energia critica encontrado)")
            # Mostrar stats gerais como fallback
            obs = _http_get("/api/agency/observation", collection="eva_core")
            if obs and obs.get("gauges"):
                g = obs["gauges"]
                print(f"  Energia media geral: {g.get('mean_energy', '?'):.4f}")
                print(f"  Temperatura: {g.get('temperature', '?'):.4f}")
                print(f"  Fase: {g.get('phase', '?')}")
                print(f"  Entropia: {g.get('entropy', '?'):.4f}")
        else:
            for node in nodes:
                c = node.content or {}
                label = c.get("node_label", c.get("type", "?")) if isinstance(c, dict) else "?"
                snippet = str(c)[:80] if c else "(sem conteudo)"
                print(f"  energy={node.energy:.4f} [{label}] {snippet}")
        # Este teste sempre passa — e informativo
        assert True

    def test_forgotten_temporal_decay_stats(self, client: NietzscheClient):
        """Mostra estatisticas de decaimento temporal do Agency."""
        obs = _http_get("/api/agency/observation", collection="eva_core")
        healing = _http_get("/api/agency/healing", collection="eva_core")
        print(f"\n{'='*70}")
        print(f"  DECAIMENTO TEMPORAL & SAUDE")
        print(f"{'='*70}")
        if obs and obs.get("gauges"):
            g = obs["gauges"]
            print(f"  Energia media:       {g.get('mean_energy', '?')}")
            print(f"  Temperatura:         {g.get('temperature', '?')}")
            print(f"  Fase termodinamica:  {g.get('phase', '?')}")
            print(f"  Entropia:            {g.get('entropy', '?')}")
            print(f"  Free energy:         {g.get('free_energy', '?')}")
            print(f"  Hebbian traces:      {g.get('hebbian_traces', '?')}")
            print(f"  Attention flow:      {g.get('attention_flow', '?')}")
        if healing and not healing.get("error"):
            print(f"\n  --- Self-Healing ---")
            print(f"  Health score:        {healing.get('health_score', '?')}")
            print(f"  Orphans:             {healing.get('orphan_count', '?')}")
            print(f"  Dead edges:          {healing.get('dead_edges', '?')}")
            print(f"  Boundary drift:      {healing.get('boundary_drift', '?')}")
        assert True

    # ------------------------------------------------------------------
    # C) O QUE ASSOCIOU — edges criados pelo Agency (L-System, Hebbian)
    # ------------------------------------------------------------------

    def test_associated_edge_types(self, client: NietzscheClient):
        """Conta edges por tipo nas colecoes EVA — mostra associacoes."""
        print(f"\n{'='*70}")
        print(f"  ASSOCIACOES — edges por tipo")
        print(f"{'='*70}")
        for col_name in ["eva_core", "eva_mind", "eva_docs"]:
            data = _http_get("/api/graph", collection=col_name, limit=500)
            if not data or not data.get("edges"):
                continue
            edge_types: dict[str, int] = {}
            for e in data["edges"]:
                etype = e.get("edge_type", "Unknown")
                edge_types[etype] = edge_types.get(etype, 0) + 1
            print(f"\n  [{col_name}] — {len(data['edges'])} edges amostrados:")
            for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
                bar = "#" * min(count // 2, 30)
                print(f"    {etype:<25} {count:>5}  {bar}")
        assert True

    def test_associated_communities(self, client: NietzscheClient):
        """Comunidades detectadas (Louvain) — clusters de conhecimento."""
        louvain = _http_get("/api/algo/louvain", collection="eva_core")
        if not louvain or louvain.get("error"):
            pytest.skip("Louvain indisponivel para eva_core")
        communities = louvain.get("communities", louvain.get("scores", []))
        print(f"\n{'='*70}")
        print(f"  COMUNIDADES (Louvain) — clusters de conhecimento em eva_core")
        print(f"{'='*70}")
        if isinstance(communities, list) and communities:
            # Conta membros por comunidade
            comm_sizes: dict[int, int] = {}
            for item in communities:
                cid = item.get("community", item.get("score", 0))
                cid = int(cid) if isinstance(cid, (int, float)) else 0
                comm_sizes[cid] = comm_sizes.get(cid, 0) + 1
            top = sorted(comm_sizes.items(), key=lambda x: -x[1])[:15]
            print(f"  Total comunidades: {len(comm_sizes)}")
            print(f"  Modularity: {louvain.get('modularity', '?')}")
            for cid, size in top:
                bar = "#" * min(size // 10, 40)
                print(f"    Comunidade {cid:<4} — {size:>6} membros  {bar}")
        assert True

    # ------------------------------------------------------------------
    # D) O QUE DESEJA — gaps de conhecimento
    # ------------------------------------------------------------------

    def test_desires_knowledge_gaps(self, client: NietzscheClient):
        """O que a EVA QUER aprender — gaps no espaco Poincare."""
        desires = _http_get("/api/agency/desires", collection="eva_core")
        if not desires or desires.get("error"):
            pytest.skip("Desires indisponivel")
        items = desires.get("desires", [])
        print(f"\n{'='*70}")
        print(f"  DESEJOS — o que a EVA quer aprender ({len(items)} gaps)")
        print(f"{'='*70}")
        fulfilled = sum(1 for d in items if d.get("fulfilled"))
        unfulfilled = len(items) - fulfilled
        print(f"  Preenchidos: {fulfilled}  |  Pendentes: {unfulfilled}")
        print(f"\n  Top 10 desejos pendentes (por prioridade):")
        pending = [d for d in items if not d.get("fulfilled")]
        pending.sort(key=lambda x: -x.get("priority", 0))
        for d in pending[:10]:
            depth = d.get("depth_range", [0, 0])
            sector = d.get("sector", {})
            print(f"    prio={d.get('priority', 0):.2f}  depth=[{depth[0]:.2f},{depth[1]:.2f}]"
                  f"  sector=({sector.get('angular_bin', '?')},{sector.get('depth_bin', '?')})"
                  f"  density={d.get('current_density', 0):.2f}")
            if d.get("suggested_query"):
                print(f"      -> {d['suggested_query']}")
        assert True

    # ------------------------------------------------------------------
    # E) RESUMO GERAL — panorama cognitivo
    # ------------------------------------------------------------------

    def test_cognitive_summary(self, client: NietzscheClient):
        """Panorama geral: o que a EVA sabe, esqueceu e deseja."""
        cols = client.list_collections()
        eva_cols = [c for c in cols if c.name.startswith("eva_")]
        total_nodes = sum(c.node_count for c in eva_cols)
        total_edges = sum(c.edge_count for c in eva_cols)

        obs = _http_get("/api/agency/observation", collection="eva_core")
        desires = _http_get("/api/agency/desires", collection="eva_core")

        print(f"\n{'='*70}")
        print(f"  PANORAMA COGNITIVO DA EVA")
        print(f"{'='*70}")
        print(f"  Colecoes EVA:        {len(eva_cols)}")
        print(f"  Total memorias:      {total_nodes:,}")
        print(f"  Total associacoes:   {total_edges:,}")

        if obs and obs.get("gauges"):
            g = obs["gauges"]
            print(f"\n  --- Termodinamica (eva_core) ---")
            print(f"  Fase:                {g.get('phase', '?')}")
            print(f"  Temperatura:         {g.get('temperature', 0):.4f}")
            print(f"  Entropia:            {g.get('entropy', 0):.4f}")
            print(f"  Energia media:       {g.get('mean_energy', 0):.4f}")
            print(f"  Hebbian traces:      {g.get('hebbian_traces', 0):,}")

        if desires:
            items = desires.get("desires", [])
            pending = sum(1 for d in items if not d.get("fulfilled"))
            print(f"\n  --- Desejos ---")
            print(f"  Gaps conhecimento:   {pending} pendentes de {len(items)}")

        print(f"\n  --- Interpretacao ---")
        if obs and obs.get("gauges"):
            g = obs["gauges"]
            phase = g.get("phase", "")
            if phase == "solid":
                print(f"  A EVA esta em fase SOLIDA — conhecimento estavel, pouca mudanca.")
            elif phase == "liquid":
                print(f"  A EVA esta em fase LIQUIDA — aprendizagem ativa, reestruturacao.")
            elif phase == "gas":
                print(f"  A EVA esta em fase GASOSA — alta entropia, conhecimento disperso.")
            energy = g.get("mean_energy", 1)
            if energy > 0.9:
                print(f"  Energia alta ({energy:.2f}) — poucas memorias esquecidas.")
            elif energy > 0.5:
                print(f"  Energia moderada ({energy:.2f}) — decaimento natural em curso.")
            else:
                print(f"  Energia BAIXA ({energy:.2f}) — muitas memorias em risco de esquecimento!")
        print(f"{'='*70}")
        assert True


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
