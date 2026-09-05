"""Qdrant adapter for the VectorStore interface.

Optional — requires `pip install 'kb-arena[qdrant]'`. qdrant_client is
lazy-imported inside the methods that need it, so a core install never pulls
it in and `import kb_arena.vectorstores` stays free of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kb_arena.vectorstores.base import VectorMatch, VectorStore


def _install_hint() -> str:
    return (
        "qdrant-client is required for the Qdrant vector store. "
        "Install with: pip install 'kb-arena[qdrant]'"
    )


def _import_qdrant_client():
    try:
        import qdrant_client
    except ImportError as exc:
        raise ImportError(_install_hint()) from exc
    return qdrant_client


def _filter(where: Mapping[str, str] | None):
    if not where:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value)) for key, value in where.items()
        ]
    )


class QdrantVectorStore(VectorStore):
    """Wraps one Qdrant collection behind the VectorStore interface."""

    def __init__(
        self,
        *,
        client: Any = None,
        collection_name: str = "kb_arena",
        url: str = "http://localhost:6333",
    ) -> None:
        if client is not None:
            self._client = client
        else:
            qdrant_client = _import_qdrant_client()
            self._client = qdrant_client.QdrantClient(url=url)
        self._collection_name = collection_name

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=ids[i],
                vector=list(embeddings[i]),
                payload={"document": documents[i], **dict(metadatas[i])},
            )
            for i in range(len(ids))
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(embedding),
            query_filter=_filter(where),
            limit=top_k,
            with_payload=True,
        )
        return [
            VectorMatch(
                id=str(point.id),
                document=str(point.payload.get("document", "")),
                metadata={key: value for key, value in point.payload.items() if key != "document"},
                distance=float(point.score),
            )
            for point in response.points
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self._client.delete(collection_name=self._collection_name, points_selector=list(ids))

    def count(self, where: Mapping[str, str] | None = None) -> int:
        result = self._client.count(
            collection_name=self._collection_name, count_filter=_filter(where)
        )
        return int(result.count)
