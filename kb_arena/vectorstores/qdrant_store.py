"""Qdrant adapter for the VectorStore interface.

Optional — requires `pip install 'kb-arena[qdrant]'`. qdrant_client is
lazy-imported inside the methods that need it, so a core install never pulls
it in and `import kb_arena.vectorstores` stays free of it.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from kb_arena.vectorstores.base import VectorMatch, VectorStore

# Qdrant takes an unsigned integer or a UUID as a point id, and a chunk id is
# neither. Hashing the chunk id into a UUID keeps the mapping deterministic
# across processes, and the payload carries the original so a caller reads back
# the id it wrote. Passing the chunk id straight through raised
# "Point id c1 is not a valid UUID" against a real server.
_POINT_NAMESPACE = uuid.UUID("6f0f4d8e-9c4a-5a2b-9a2e-4b6b8f2c1d70")
_CHUNK_ID_FIELD = "chunk_id"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


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
                id=_point_id(ids[i]),
                vector=list(embeddings[i]),
                # Metadata expands FIRST. Expanding it last let a chunk carry
                # `{"chunk_id": "c2", "document": "spoofed"}` and take another
                # record's identity, so a query answered with the wrong id and
                # a delete of that id left the real point in place.
                payload={
                    **dict(metadatas[i]),
                    "document": documents[i],
                    _CHUNK_ID_FIELD: ids[i],
                },
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
                id=str(point.payload.get(_CHUNK_ID_FIELD) or point.id),
                document=str(point.payload.get("document", "")),
                metadata={
                    key: value
                    for key, value in point.payload.items()
                    if key not in ("document", _CHUNK_ID_FIELD)
                },
                distance=float(point.score),
            )
            for point in response.points
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=[_point_id(chunk_id) for chunk_id in ids],
        )

    def count(self, where: Mapping[str, str] | None = None) -> int:
        result = self._client.count(
            collection_name=self._collection_name, count_filter=_filter(where)
        )
        return int(result.count)
