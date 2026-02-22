# Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
DSPy Retriever Module for NietzscheDB.
Implements the DSPy Retrieve interface so NietzscheDB can be used as a
retrieval module in DSPy programs.

Usage:
    import dspy
    from nietzschedb.dspy import NietzscheRM

    retriever = NietzscheRM(
        addr="localhost:50051",
        collection="my_docs",
        k=5,
    )
    dspy.settings.configure(rm=retriever)
    results = retriever("What is NietzscheDB?")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False
    dspy = None  # type: ignore

from .client import NietzscheClient


class NietzscheRM:
    """DSPy Retrieval Module backed by NietzscheDB.

    Uses hybrid search (BM25 + vector KNN) when embedding_fn is provided,
    otherwise falls back to full-text BM25 search.
    """

    def __init__(
        self,
        collection: str,
        *,
        addr: str = "localhost:50051",
        client: Optional[NietzscheClient] = None,
        k: int = 5,
        text_key: str = "text",
        embedding_fn: Optional[Any] = None,
    ):
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "dspy is required for NietzscheRM. "
                "Install with: pip install dspy-ai"
            )

        self._client = client or NietzscheClient(addr=addr)
        self._collection = collection
        self._k = k
        self._text_key = text_key
        self._embedding_fn = embedding_fn

    def __call__(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[List[Dict], List[List[Dict]]]:
        """Retrieve passages for one or more queries."""
        k = k or self._k

        if isinstance(query_or_queries, str):
            return self._retrieve_single(query_or_queries, k)

        return [self._retrieve_single(q, k) for q in query_or_queries]

    def _retrieve_single(self, query: str, k: int) -> List[Dict]:
        """Retrieve passages for a single query."""
        passages = []

        if self._embedding_fn is not None:
            # Hybrid search: BM25 + vector KNN
            query_vector = self._embedding_fn(query)
            if isinstance(query_vector, list) and len(query_vector) > 0:
                if isinstance(query_vector[0], list):
                    query_vector = query_vector[0]  # batch embed returned list of lists
                results = self._client.hybrid_search(
                    query_coords=query_vector,
                    text_query=query,
                    k=k,
                    collection=self._collection,
                )
            else:
                results = self._client.full_text_search(
                    query_text=query, limit=k, collection=self._collection
                )
        else:
            # Full-text search only (BM25)
            results = self._client.full_text_search(
                query_text=query, limit=k, collection=self._collection
            )

        for knn_result in results:
            node = self._client.get_node(knn_result.id, collection=self._collection)
            if node is None:
                continue

            text = node.content.get(self._text_key, "")
            passage = {
                "long_text": text,
                "score": 1.0 / (1.0 + knn_result.distance) if knn_result.distance else 0.0,
                "nietzsche_id": node.id,
                **{k: v for k, v in node.content.items() if k != self._text_key},
            }
            passages.append(passage)

        # Sort by score descending
        passages.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Convert to dspy.Prediction format if available
        if _DSPY_AVAILABLE and hasattr(dspy, "Prediction"):
            return [
                dspy.Prediction(long_text=p["long_text"], score=p["score"])
                for p in passages
            ]

        return passages

    def forward(self, query: str, k: Optional[int] = None) -> Any:
        """DSPy Module forward method."""
        results = self._retrieve_single(query, k or self._k)
        if _DSPY_AVAILABLE and hasattr(dspy, "Prediction"):
            return dspy.Prediction(passages=results)
        return results
