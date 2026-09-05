"""ChromaDB adapter for the VectorStore interface — the default backend.

chromadb is already a core dependency (see pyproject.toml), so this adapter
needs no optional extra and no lazy import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import chromadb

from kb_arena.vectorstores.base import VectorMatch, VectorStore


def _where_clause(where: Mapping[str, str] | None) -> dict[str, Any] | None:
    if not where:
        return None
    if len(where) == 1:
        ((key, value),) = where.items()
        return {key: value}
    return {"$and": [{key: value} for key, value in where.items()]}


class ChromaVectorStore(VectorStore):
    """Wraps one ChromaDB collection behind the VectorStore interface."""

    def __init__(
        self,
        *,
        client: Any = None,
        collection_name: str = "kb_arena",
        path: str = ".chroma",
    ) -> None:
        self._client = client if client is not None else chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        self._collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[dict(metadata) for metadata in metadatas],
        )

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=_where_clause(where),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            VectorMatch(
                id=ids[i],
                document=documents[i],
                metadata=dict(metadatas[i]),
                distance=float(distances[i]),
            )
            for i in range(len(ids))
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self._collection.delete(ids=list(ids))

    def count(self, where: Mapping[str, str] | None = None) -> int:
        if where is None:
            return self._collection.count()
        result = self._collection.get(where=_where_clause(where), include=[])
        return len(result["ids"])
