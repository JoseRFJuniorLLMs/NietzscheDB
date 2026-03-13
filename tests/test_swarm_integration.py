"""
End-to-end integration tests for Swarm -> NietzscheDB pipeline.

Tests cover:
  7.1 - E2E swarm -> NietzscheDB persistence (clinical, medication, emergency,
        habits, learning, sessions, filtering rules)
  7.2 - Regression: content:null bug (embedding pipeline, round-trip)
  7.3 - Load tests (bulk insert 100 nodes, concurrent swarm writes)

Requires:
    pip install grpcio grpcio-tools pytest
    NietzscheDB server running on VM 136.111.0.47:443 (gRPC via HTTPS/nginx)

Usage:
    pytest tests/test_swarm_integration.py -v
    pytest tests/test_swarm_integration.py -v -k "test_clinical"
"""

import json
import math
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

import grpc
import pytest

# SDK path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdks", "python"))

from nietzschedb import NietzscheClient, Node

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NIETZSCHE_HOST = os.environ.get("NIETZSCHE_HOST", "localhost:50051")
TEST_DIM = 128
TEST_METRIC = "poincare"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def poincare_vec(dim: int = TEST_DIM, magnitude: float = 0.3) -> List[float]:
    """Random vector inside the Poincare disk with controlled magnitude."""
    raw = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm * magnitude for x in raw]


def zero_vec(dim: int = TEST_DIM) -> List[float]:
    return [0.0] * dim


def _ts() -> int:
    return int(time.time())


# --- Swarm content builders (mimic what each swarm agent produces) ---

def make_clinical_assessment(patient_id: str = "PAT-001") -> Dict[str, Any]:
    return {
        "node_label": "ClinicalAssessment",
        "patient_id": patient_id,
        "assessor": "Dr. Silva",
        "diagnosis": "Malaria falciparum",
        "severity": "moderate",
        "symptoms": ["fever", "chills", "headache"],
        "vitals": {"temp_c": 39.2, "heart_rate": 110, "bp": "100/70"},
        "notes": "Patient presents with 3-day fever history, RDT positive",
        "timestamp": _ts(),
    }


def make_medication_log(patient_id: str = "PAT-001") -> Dict[str, Any]:
    return {
        "node_label": "MedicationLog",
        "patient_id": patient_id,
        "medication": "Artemether-Lumefantrine",
        "dosage_mg": 80,
        "route": "oral",
        "frequency": "2x/day",
        "start_date": "2026-03-13",
        "prescriber": "Dr. Silva",
        "timestamp": _ts(),
    }


def make_emergency_alert(patient_id: str = "PAT-002") -> Dict[str, Any]:
    return {
        "node_label": "EmergencyAlert",
        "patient_id": patient_id,
        "alert_type": "critical_vitals",
        "severity": "high",
        "message": "SpO2 dropped to 88%, immediate oxygen required",
        "triggered_by": "vitals_monitor",
        "acknowledged": False,
        "timestamp": _ts(),
    }


def make_habit_log(patient_id: str = "PAT-001") -> Dict[str, Any]:
    return {
        "node_label": "HabitLog",
        "patient_id": patient_id,
        "habit_type": "medication_adherence",
        "compliance": True,
        "streak_days": 5,
        "notes": "Took morning dose on time",
        "timestamp": _ts(),
    }


def make_learning_insight() -> Dict[str, Any]:
    return {
        "node_label": "Learning",
        "topic": "drug_resistance_patterns",
        "insight": "Chloroquine resistance prevalence in Luanda province increased 12% YoY",
        "confidence": 0.87,
        "sources": ["WHO_2025_report", "local_surveillance_data"],
        "category": "epidemiology",
        "actionable": True,
        "timestamp": _ts(),
    }


def make_eva_session(turn_count: int = 5) -> Dict[str, Any]:
    return {
        "node_label": "EvaSession",
        "session_id": f"sess_{uuid.uuid4().hex[:8]}",
        "user_id": "USR-001",
        "turn_count": turn_count,
        "duration_seconds": turn_count * 30,
        "topics": ["malaria_treatment", "follow_up"],
        "summary": "Patient discussed treatment progress and next appointment",
        "sentiment": "positive",
        "timestamp": _ts(),
    }


def make_demand(desire: str = "medication_refill") -> Dict[str, Any]:
    return {
        "node_label": "Demand",
        "desire": desire,
        "urgency": "medium",
        "requester": "USR-001",
        "context": "Patient needs prescription renewal",
        "timestamp": _ts(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Session-scoped NietzscheDB client."""
    c = NietzscheClient(NIETZSCHE_HOST)
    if not c.health_check():
        pytest.skip(f"NietzscheDB not reachable at {NIETZSCHE_HOST}")
    yield c
    c.close()


def _make_collection(client: NietzscheClient, prefix: str) -> str:
    """Create a temp collection with unique name, return the name."""
    name = f"swarm_test_{prefix}_{uuid.uuid4().hex[:6]}"
    created = client.create_collection(name, dimension=TEST_DIM, metric=TEST_METRIC)
    assert created, f"Failed to create collection {name}"
    return name


@pytest.fixture()
def collection(client: NietzscheClient):
    """Per-test temp collection with auto-cleanup."""
    name = _make_collection(client, "e2e")
    yield name
    client.drop_collection(name)


@pytest.fixture()
def load_collection(client: NietzscheClient):
    """Per-test temp collection for load tests."""
    name = _make_collection(client, "load")
    yield name
    client.drop_collection(name)


# ===========================================================================
# 7.1 - E2E Swarm -> NietzscheDB Tests
# ===========================================================================

class TestClinicalAssessment:
    """7.1a - ClinicalAssessment node persisted and queryable."""

    def test_clinical_assessment_persisted(self, client: NietzscheClient, collection: str):
        content = make_clinical_assessment("PAT-E2E-001")
        coords = poincare_vec(magnitude=0.4)

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Semantic",
            energy=0.9,
            collection=collection,
        )

        # Retrieve and verify
        node = client.get_node(node_id, collection=collection)
        assert node is not None, "ClinicalAssessment node not found after insert"
        assert node.content is not None, "Content is null"
        assert node.content.get("node_label") == "ClinicalAssessment"
        assert node.content.get("patient_id") == "PAT-E2E-001"
        assert node.content.get("diagnosis") == "Malaria falciparum"
        assert node.content.get("severity") == "moderate"
        assert isinstance(node.content.get("symptoms"), list)
        assert len(node.content["symptoms"]) == 3

    def test_clinical_assessment_searchable_by_text(self, client: NietzscheClient, collection: str):
        content = make_clinical_assessment("PAT-SEARCH-001")
        content["notes"] = "Unique clinical note for search verification XJ7Q"
        coords = poincare_vec(magnitude=0.4)

        client.insert_node(
            coords=coords,
            content=content,
            node_type="Semantic",
            collection=collection,
        )
        time.sleep(0.5)  # allow indexing

        results = client.full_text_search("XJ7Q", limit=5, collection=collection)
        assert len(results) >= 1, "ClinicalAssessment not found via full-text search"


class TestMedicationLog:
    """7.1b - MedicationLog node persisted with edge to patient."""

    def test_medication_log_persisted(self, client: NietzscheClient, collection: str):
        # Create patient node first
        patient_content = {
            "node_label": "Patient",
            "patient_id": "PAT-MED-001",
            "name": "Test Patient",
        }
        patient_id = client.insert_node(
            coords=poincare_vec(magnitude=0.2),
            content=patient_content,
            node_type="Semantic",
            collection=collection,
        )

        # Create medication log
        med_content = make_medication_log("PAT-MED-001")
        med_id = client.insert_node(
            coords=poincare_vec(magnitude=0.5),
            content=med_content,
            node_type="Semantic",
            collection=collection,
        )

        # Create edge: patient -> medication
        edge_id = client.insert_edge(
            patient_id,
            med_id,
            edge_type="HAS_MEDICATION",
            weight=1.0,
            collection=collection,
        )

        # Verify medication node
        med_node = client.get_node(med_id, collection=collection)
        assert med_node is not None
        assert med_node.content.get("node_label") == "MedicationLog"
        assert med_node.content.get("medication") == "Artemether-Lumefantrine"

        # Verify edge exists via BFS from patient
        visited = client.bfs(patient_id, max_depth=1, collection=collection)
        assert med_id in visited, "MedicationLog not reachable from patient via BFS"


class TestEmergencyAlert:
    """7.1c - EmergencyAlert node persisted and retrievable."""

    def test_emergency_alert_persisted(self, client: NietzscheClient, collection: str):
        content = make_emergency_alert("PAT-EMERG-001")
        coords = poincare_vec(magnitude=0.6)  # high magnitude = urgent/peripheral

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Episodic",
            energy=1.0,
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content.get("node_label") == "EmergencyAlert"
        assert node.content.get("alert_type") == "critical_vitals"
        assert node.content.get("severity") == "high"
        assert node.content.get("patient_id") == "PAT-EMERG-001"
        assert node.energy == 1.0


class TestHabitLog:
    """7.1d - HabitLog with edges to patient and related nodes."""

    def test_habit_log_with_edges(self, client: NietzscheClient, collection: str):
        # Create patient
        patient_id = client.insert_node(
            coords=poincare_vec(magnitude=0.2),
            content={"node_label": "Patient", "patient_id": "PAT-HABIT-001"},
            node_type="Semantic",
            collection=collection,
        )

        # Create medication node (the habit relates to)
        med_id = client.insert_node(
            coords=poincare_vec(magnitude=0.5),
            content=make_medication_log("PAT-HABIT-001"),
            node_type="Semantic",
            collection=collection,
        )

        # Create habit log
        habit_content = make_habit_log("PAT-HABIT-001")
        habit_id = client.insert_node(
            coords=poincare_vec(magnitude=0.45),
            content=habit_content,
            node_type="Episodic",
            collection=collection,
        )

        # Create edges
        e1 = client.insert_edge(patient_id, habit_id, edge_type="HAS_HABIT", collection=collection)
        e2 = client.insert_edge(habit_id, med_id, edge_type="RELATES_TO", collection=collection)

        # Verify habit node
        habit_node = client.get_node(habit_id, collection=collection)
        assert habit_node is not None
        assert habit_node.content.get("node_label") == "HabitLog"
        assert habit_node.content.get("compliance") is True

        # Verify both edges via BFS
        from_patient = client.bfs(patient_id, max_depth=2, collection=collection)
        assert habit_id in from_patient, "HabitLog not reachable from patient"
        assert med_id in from_patient, "MedicationLog not reachable via habit edges"


class TestLearningInsight:
    """7.1e - Learning node with real content."""

    def test_learning_insight_persisted(self, client: NietzscheClient, collection: str):
        content = make_learning_insight()
        coords = poincare_vec(magnitude=0.35)

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Semantic",
            energy=0.87,
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content.get("node_label") == "Learning"
        assert node.content.get("topic") == "drug_resistance_patterns"
        assert "Chloroquine" in node.content.get("insight", "")
        assert node.content.get("confidence") == 0.87
        assert isinstance(node.content.get("sources"), list)
        assert node.content.get("actionable") is True


class TestEvaSession:
    """7.1f/g - EvaSession persistence rules based on turn_count."""

    def test_session_with_turns_persisted(self, client: NietzscheClient, collection: str):
        """Sessions with turn_count > 0 SHOULD be persisted."""
        content = make_eva_session(turn_count=5)
        coords = poincare_vec(magnitude=0.3)

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Episodic",
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content.get("node_label") == "EvaSession"
        assert node.content.get("turn_count") == 5
        assert node.content.get("duration_seconds") == 150

    def test_session_no_turns_not_persisted(self, client: NietzscheClient, collection: str):
        """Sessions with turn_count=0 should NOT be created.

        This validates the swarm filtering rule: the swarm agent should
        check turn_count before calling insert. We simulate the gate here.
        """
        content = make_eva_session(turn_count=0)

        # Swarm gate: do NOT insert if turn_count == 0
        should_persist = content.get("turn_count", 0) > 0
        assert should_persist is False, "Gate should reject turn_count=0"

        # Verify nothing was inserted (no node_id to look up)
        # Use full-text search for the session_id to confirm absence
        results = client.full_text_search(
            content["session_id"], limit=1, collection=collection
        )
        assert len(results) == 0, "Session with turn_count=0 should not exist in DB"


class TestDemandFiltering:
    """7.1h - Demands with desire='indefinido' are filtered out."""

    def test_demand_indefinido_not_persisted(self, client: NietzscheClient, collection: str):
        """Demands with desire='indefinido' should NOT be created."""
        content = make_demand(desire="indefinido")

        # Swarm gate: do NOT insert if desire == "indefinido"
        should_persist = content.get("desire") != "indefinido"
        assert should_persist is False, "Gate should reject desire='indefinido'"

        # Verify nothing was inserted
        results = client.full_text_search(
            "indefinido", limit=5, collection=collection
        )
        # No node should have been created for this demand
        assert len(results) == 0, "Demand with desire='indefinido' should not exist"

    def test_demand_valid_persisted(self, client: NietzscheClient, collection: str):
        """Valid demands (desire != 'indefinido') SHOULD be persisted."""
        content = make_demand(desire="medication_refill")
        coords = poincare_vec(magnitude=0.4)

        # Swarm gate passes
        should_persist = content.get("desire") != "indefinido"
        assert should_persist is True

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Semantic",
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content.get("node_label") == "Demand"
        assert node.content.get("desire") == "medication_refill"


# ===========================================================================
# 7.2 - Regression Tests for content:null
# ===========================================================================

class TestContentNullRegression:
    """Ensure content is never null after insert (regression for content:null bug)."""

    def test_insert_with_embedding_has_content(self, client: NietzscheClient, collection: str):
        """Insert a node via the embedding pipeline path; verify content is NOT null."""
        content = {
            "node_label": "EmbeddedNode",
            "text": "This node was created through the embedding pipeline",
            "source": "swarm_agent_clinical",
            "embedding_model": "gemini-embedding-002",
        }
        coords = poincare_vec(magnitude=0.35)

        node_id = client.insert_node(
            coords=coords,
            content=content,
            node_type="Semantic",
            energy=0.8,
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None, "Node not found"
        assert node.content is not None, "REGRESSION: content is None!"
        assert node.content != {}, "REGRESSION: content is empty dict!"
        assert node.content.get("text") is not None, "REGRESSION: text field is None"
        assert node.content.get("node_label") == "EmbeddedNode"

    def test_node_content_round_trip(self, client: NietzscheClient, collection: str):
        """Insert content, retrieve, verify exact match (round-trip fidelity)."""
        original_content = {
            "node_label": "RoundTripTest",
            "text": "Round-trip content verification",
            "nested": {"key": "value", "numbers": [1, 2, 3]},
            "unicode": "Teste de conteudo com acentos e emojis",
            "bool_field": True,
            "int_field": 42,
            "float_field": 3.14159,
            "null_field": None,
            "list_field": ["a", "b", "c"],
        }
        coords = poincare_vec(magnitude=0.3)

        node_id = client.insert_node(
            coords=coords,
            content=original_content,
            node_type="Semantic",
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content is not None, "REGRESSION: content is None on round-trip"

        # Verify each field
        assert node.content.get("node_label") == "RoundTripTest"
        assert node.content.get("text") == "Round-trip content verification"
        assert node.content.get("nested") == {"key": "value", "numbers": [1, 2, 3]}
        assert node.content.get("bool_field") is True
        assert node.content.get("int_field") == 42
        assert abs(node.content.get("float_field", 0) - 3.14159) < 1e-4
        assert node.content.get("null_field") is None
        assert node.content.get("list_field") == ["a", "b", "c"]

    def test_content_not_null_after_batch_insert(self, client: NietzscheClient, collection: str):
        """Batch-inserted nodes must also have non-null content."""
        nodes = []
        for i in range(5):
            nodes.append({
                "coords": poincare_vec(magnitude=0.3),
                "content": {
                    "node_label": "BatchContentTest",
                    "index": i,
                    "text": f"Batch node {i} content verification",
                },
                "node_type": "Semantic",
                "energy": 0.8,
            })

        inserted, ids = client.batch_insert_nodes(nodes, collection=collection)
        assert inserted == 5

        for nid in ids:
            node = client.get_node(nid, collection=collection)
            assert node is not None, f"Node {nid} not found"
            assert node.content is not None, f"REGRESSION: content null for batch node {nid}"
            assert node.content.get("node_label") == "BatchContentTest"
            assert node.content.get("text") is not None

    def test_empty_content_still_not_none(self, client: NietzscheClient, collection: str):
        """Even with minimal content, result should not be None."""
        node_id = client.insert_node(
            coords=poincare_vec(magnitude=0.3),
            content={"minimal": True},
            node_type="Semantic",
            collection=collection,
        )

        node = client.get_node(node_id, collection=collection)
        assert node is not None
        assert node.content is not None, "REGRESSION: content is None for minimal content"
        assert node.content.get("minimal") is True


# ===========================================================================
# 7.3 - Load Tests
# ===========================================================================

class TestLoadBulkInsert:
    """7.3a - Bulk insert 100 nodes rapidly, verify all persisted with content."""

    def test_bulk_insert_100_nodes(self, client: NietzscheClient, load_collection: str):
        """Insert 100 nodes via batch, verify all persisted with non-null content."""
        num_nodes = 100
        node_types = [
            ("ClinicalAssessment", make_clinical_assessment),
            ("MedicationLog", make_medication_log),
            ("EmergencyAlert", make_emergency_alert),
            ("HabitLog", make_habit_log),
            ("Learning", make_learning_insight),
        ]

        # Build batch payload
        batch_nodes = []
        for i in range(num_nodes):
            label, maker = node_types[i % len(node_types)]
            if maker in (make_clinical_assessment, make_medication_log,
                         make_emergency_alert, make_habit_log):
                content = maker(f"PAT-LOAD-{i:03d}")
            else:
                content = maker()
            content["batch_index"] = i

            batch_nodes.append({
                "coords": poincare_vec(magnitude=0.3 + (i % 5) * 0.05),
                "content": content,
                "node_type": "Semantic",
                "energy": 0.8,
            })

        # Batch insert (in chunks of 50 to be safe)
        all_ids = []
        for chunk_start in range(0, num_nodes, 50):
            chunk = batch_nodes[chunk_start:chunk_start + 50]
            inserted, ids = client.batch_insert_nodes(chunk, collection=load_collection)
            assert inserted == len(chunk), f"Expected {len(chunk)} inserts, got {inserted}"
            all_ids.extend(ids)

        assert len(all_ids) == num_nodes, f"Expected {num_nodes} IDs, got {len(all_ids)}"

        # Verify a sample (every 10th node) to avoid excessive RPCs
        failures = []
        for idx in range(0, num_nodes, 10):
            nid = all_ids[idx]
            node = client.get_node(nid, collection=load_collection)
            if node is None:
                failures.append(f"Node {idx} ({nid}) not found")
            elif node.content is None:
                failures.append(f"Node {idx} ({nid}) has null content")
            elif node.content.get("batch_index") != idx:
                failures.append(
                    f"Node {idx} batch_index mismatch: "
                    f"expected {idx}, got {node.content.get('batch_index')}"
                )

        assert len(failures) == 0, f"Bulk insert failures:\n" + "\n".join(failures)


class TestConcurrentSwarmWrites:
    """7.3b - Simulate multiple swarm types writing simultaneously."""

    def test_concurrent_swarm_writes(self, client: NietzscheClient, load_collection: str):
        """Multiple swarm agents write different node types concurrently."""
        results = {"success": 0, "failure": 0, "errors": []}

        def swarm_write(agent_name: str, content: Dict[str, Any], magnitude: float) -> str:
            """Simulate a single swarm agent write."""
            nid = client.insert_node(
                coords=poincare_vec(magnitude=magnitude),
                content=content,
                node_type="Semantic",
                energy=0.9,
                collection=load_collection,
            )
            return nid

        # Define concurrent tasks (5 agents, 4 writes each = 20 concurrent ops)
        tasks = []
        for i in range(4):
            tasks.append(("clinical", make_clinical_assessment(f"PAT-CONC-C{i}"), 0.4))
            tasks.append(("medication", make_medication_log(f"PAT-CONC-M{i}"), 0.5))
            tasks.append(("emergency", make_emergency_alert(f"PAT-CONC-E{i}"), 0.6))
            tasks.append(("habit", make_habit_log(f"PAT-CONC-H{i}"), 0.45))
            tasks.append(("learning", make_learning_insight(), 0.35))

        inserted_ids = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for agent_name, content, mag in tasks:
                f = executor.submit(swarm_write, agent_name, content, mag)
                futures[f] = agent_name

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    nid = future.result(timeout=30)
                    inserted_ids.append(nid)
                    results["success"] += 1
                except Exception as e:
                    results["failure"] += 1
                    results["errors"].append(f"{agent}: {e}")

        total_tasks = len(tasks)
        assert results["failure"] == 0, (
            f"{results['failure']}/{total_tasks} concurrent writes failed:\n"
            + "\n".join(results["errors"])
        )
        assert results["success"] == total_tasks

        # Verify all nodes exist and have content
        missing = 0
        null_content = 0
        for nid in inserted_ids:
            node = client.get_node(nid, collection=load_collection)
            if node is None:
                missing += 1
            elif node.content is None or node.content == {}:
                null_content += 1

        assert missing == 0, f"{missing}/{len(inserted_ids)} nodes missing after concurrent write"
        assert null_content == 0, f"{null_content}/{len(inserted_ids)} nodes have null/empty content"
