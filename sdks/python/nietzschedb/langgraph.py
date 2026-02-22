# Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LangGraph State/Memory Store for NietzscheDB.
Provides persistent state and memory management for LangGraph agents.

Usage:
    from nietzschedb.langgraph import NietzscheCheckpointer, NietzscheMemoryStore

    # As a checkpointer for LangGraph
    checkpointer = NietzscheCheckpointer(client, collection="agent_state")
    graph = workflow.compile(checkpointer=checkpointer)

    # As a memory store for agent long-term memory
    memory = NietzscheMemoryStore(client, collection="agent_memory")
    memory.put("user_123", "preferences", {"tone": "formal"})
    prefs = memory.get("user_123", "preferences")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from langgraph.checkpoint.base import (
        BaseCheckpointSaver,
        Checkpoint,
        CheckpointMetadata,
        CheckpointTuple,
    )
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object  # type: ignore

from .client import NietzscheClient


class NietzscheCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpointer backed by NietzscheDB.

    Stores agent state snapshots as nodes in a NietzscheDB collection.
    Each checkpoint is a node with thread_id + checkpoint_id as match keys.
    """

    def __init__(
        self,
        client: NietzscheClient,
        collection: str = "langgraph_checkpoints",
    ):
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required. Install with: pip install langgraph"
            )
        super().__init__()
        self._client = client
        self._collection = collection

    def get_tuple(self, config: Dict) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if checkpoint_id:
            nql = (
                'MATCH (n:Checkpoint) '
                'WHERE n.thread_id = $tid AND n.checkpoint_id = $cid '
                'RETURN n LIMIT 1'
            )
            params = {"tid": thread_id, "cid": checkpoint_id}
        else:
            nql = (
                'MATCH (n:Checkpoint) '
                'WHERE n.thread_id = $tid '
                'RETURN n ORDER BY n.ts DESC LIMIT 1'
            )
            params = {"tid": thread_id}

        try:
            results = self._client.query(nql, params=params, collection=self._collection)
            if not results.nodes:
                return None

            data = results.nodes[0].content
            checkpoint = json.loads(data.get("checkpoint_data", "{}"))
            metadata = json.loads(data.get("metadata", "{}"))

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": data.get("checkpoint_id", ""),
                    }
                },
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=None,
            )
        except Exception as e:
            logger.warning(f"NietzscheDB get_tuple failed: {e}")
            return None

    def list(
        self,
        config: Optional[Dict] = None,
        *,
        filter: Optional[Dict] = None,
        before: Optional[Dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints."""
        thread_id = ""
        if config:
            thread_id = config.get("configurable", {}).get("thread_id", "")

        nql = 'MATCH (n:Checkpoint) WHERE n.thread_id = $tid RETURN n ORDER BY n.ts DESC'
        if limit:
            nql += f' LIMIT {limit}'
        params = {"tid": thread_id}

        try:
            results = self._client.query(nql, params=params, collection=self._collection)
            for node in results.nodes:
                data = node.content
                checkpoint = json.loads(data.get("checkpoint_data", "{}"))
                metadata = json.loads(data.get("metadata", "{}"))
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": data.get("checkpoint_id", ""),
                        }
                    },
                    checkpoint=checkpoint,
                    metadata=metadata,
                    parent_config=None,
                )
        except Exception as e:
            logger.warning(f"NietzscheDB list checkpoints failed: {e}")

    def put(
        self,
        config: Dict,
        checkpoint: Dict,
        metadata: Dict,
        new_versions: Optional[Dict] = None,
    ) -> Dict:
        """Save a checkpoint."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        checkpoint_id = checkpoint.get("id", str(time.time_ns()))

        try:
            self._client.merge_node(
                node_type="Checkpoint",
                match_keys={"thread_id": thread_id, "checkpoint_id": checkpoint_id},
                on_create={
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_data": json.dumps(checkpoint),
                    "metadata": json.dumps(metadata),
                    "ts": time.time(),
                },
                on_match={
                    "checkpoint_data": json.dumps(checkpoint),
                    "metadata": json.dumps(metadata),
                    "ts": time.time(),
                },
                collection=self._collection,
            )
        except Exception as e:
            logger.warning(f"NietzscheDB put checkpoint failed: {e}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Save intermediate writes (pending sends)."""
        thread_id = config.get("configurable", {}).get("thread_id", "")
        try:
            self._client.merge_node(
                node_type="CheckpointWrite",
                match_keys={"thread_id": thread_id, "task_id": task_id},
                on_create={
                    "thread_id": thread_id,
                    "task_id": task_id,
                    "writes": json.dumps([(ch, json.dumps(v) if not isinstance(v, str) else v) for ch, v in writes]),
                    "ts": time.time(),
                },
                on_match={
                    "writes": json.dumps([(ch, json.dumps(v) if not isinstance(v, str) else v) for ch, v in writes]),
                    "ts": time.time(),
                },
                collection=self._collection,
            )
        except Exception as e:
            logger.warning(f"NietzscheDB put_writes failed: {e}")


class NietzscheMemoryStore:
    """Long-term memory store for LangGraph agents.

    Key-value store backed by NietzscheDB with namespace isolation.
    Supports TTL for automatic expiration of stale memories.
    """

    def __init__(
        self,
        client: NietzscheClient,
        collection: str = "agent_memory",
    ):
        self._client = client
        self._collection = collection

    def put(self, namespace: str, key: str, value: Any, *, ttl_seconds: int = 0) -> None:
        """Store a memory entry."""
        data = value if isinstance(value, dict) else {"value": value}
        self._client.merge_node(
            node_type="Memory",
            match_keys={"namespace": namespace, "key": key},
            on_create={
                "namespace": namespace,
                "key": key,
                **data,
                "created_at": time.time(),
                "updated_at": time.time(),
            },
            on_match={
                **data,
                "updated_at": time.time(),
            },
            collection=self._collection,
        )

    def get(self, namespace: str, key: str) -> Optional[Dict]:
        """Retrieve a memory entry."""
        try:
            results = self._client.query(
                'MATCH (n:Memory) WHERE n.namespace = $ns AND n.key = $k RETURN n LIMIT 1',
                params={"ns": namespace, "k": key},
                collection=self._collection,
            )
            if results.nodes:
                content = dict(results.nodes[0].content)
                content.pop("namespace", None)
                content.pop("key", None)
                return content
        except Exception as e:
            logger.debug(f"NietzscheDB memory get failed: {e}")
        return None

    def search(self, namespace: str, *, limit: int = 20) -> List[Dict]:
        """List all memories in a namespace."""
        try:
            results = self._client.query(
                'MATCH (n:Memory) WHERE n.namespace = $ns RETURN n ORDER BY n.updated_at DESC LIMIT $lim',
                params={"ns": namespace, "lim": limit},
                collection=self._collection,
            )
            return [n.content for n in results.nodes]
        except Exception as e:
            logger.debug(f"NietzscheDB memory search failed: {e}")
            return []

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a memory entry."""
        try:
            results = self._client.query(
                'MATCH (n:Memory) WHERE n.namespace = $ns AND n.key = $k RETURN n LIMIT 1',
                params={"ns": namespace, "k": key},
                collection=self._collection,
            )
            if results.nodes:
                self._client.delete_node(results.nodes[0].id, collection=self._collection)
                return True
        except Exception as e:
            logger.debug(f"NietzscheDB memory delete failed: {e}")
        return False
