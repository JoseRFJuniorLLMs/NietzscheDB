# Copyright (C) 2025-2026 Jose R F Junior <web2ajax@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LangChain VectorStore integration for NietzscheDB.
Implements the LangChain VectorStore interface so NietzscheDB can be used
as a drop-in replacement for Chroma, Pinecone, Qdrant, etc.

Usage:
    from nietzschedb.langchain import NietzscheVectorStore
    from langchain_core.embeddings import Embeddings

    store = NietzscheVectorStore(
        client=NietzscheClient("localhost:50051"),
        collection="my_docs",
        embedding=my_embedding_model,
    )
    store.add_texts(["hello world", "foo bar"])
    results = store.similarity_search("hello", k=5)
"""

from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

logger = logging.getLogger(__name__)

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Stubs so the module can be imported without langchain
    Document = Any  # type: ignore
    Embeddings = Any  # type: ignore
    VectorStore = object  # type: ignore

from .client import NietzscheClient


class NietzscheVectorStore(VectorStore):
    """LangChain VectorStore backed by NietzscheDB's hyperbolic KNN search."""

    def __init__(
        self,
        client: NietzscheClient,
        collection: str,
        embedding: Embeddings,
        *,
        text_key: str = "text",
        node_type: str = "Document",
    ):
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for NietzscheVectorStore. "
                "Install with: pip install langchain-core"
            )
        self._client = client
        self._collection = collection
        self._embedding = embedding
        self._text_key = text_key
        self._node_type = node_type

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts with embeddings to NietzscheDB."""
        texts_list = list(texts)
        vectors = self._embedding.embed_documents(texts_list)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        nodes = []
        for i, (text, vector, meta) in enumerate(zip(texts_list, vectors, metadatas)):
            content = {self._text_key: text, **meta}
            nodes.append({
                "id": ids[i],
                "coords": vector,
                "content": content,
                "node_type": self._node_type,
            })

        inserted, node_ids = self._client.batch_insert_nodes(
            nodes, collection=self._collection
        )
        logger.debug(f"Inserted {inserted} documents into {self._collection}")
        return node_ids if node_ids else ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to query."""
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return documents with similarity scores."""
        query_vector = self._embedding.embed_query(query)
        results = self._client.knn_search(
            query_coords=query_vector,
            k=k,
            collection=self._collection,
        )

        docs_and_scores = []
        for knn_result in results:
            node = self._client.get_node(knn_result.id, collection=self._collection)
            if node is None:
                continue

            text = node.content.get(self._text_key, "")
            metadata = {k: v for k, v in node.content.items() if k != self._text_key}
            metadata["_nietzsche_id"] = node.id
            metadata["_distance"] = knn_result.distance

            doc = Document(page_content=text, metadata=metadata)
            # Convert distance to similarity score (lower distance = higher similarity)
            score = 1.0 / (1.0 + knn_result.distance)
            docs_and_scores.append((doc, score))

        return docs_and_scores

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to embedding vector."""
        results = self._client.knn_search(
            query_coords=embedding,
            k=k,
            collection=self._collection,
        )

        docs = []
        for knn_result in results:
            node = self._client.get_node(knn_result.id, collection=self._collection)
            if node is None:
                continue
            text = node.content.get(self._text_key, "")
            metadata = {k: v for k, v in node.content.items() if k != self._text_key}
            metadata["_nietzsche_id"] = node.id
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by ID."""
        if ids is None:
            return False
        for doc_id in ids:
            self._client.delete_node(doc_id, collection=self._collection)
        return True

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Optional[NietzscheClient] = None,
        collection: str = "langchain_docs",
        **kwargs: Any,
    ) -> "NietzscheVectorStore":
        """Create a NietzscheVectorStore from a list of texts."""
        if client is None:
            addr = kwargs.pop("addr", "localhost:50051")
            client = NietzscheClient(addr=addr)

        store = cls(client=client, collection=collection, embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def as_retriever(self, **kwargs: Any):
        """Return a LangChain retriever wrapping this store."""
        from langchain_core.vectorstores import VectorStoreRetriever
        return VectorStoreRetriever(vectorstore=self, **kwargs)
