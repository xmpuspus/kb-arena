"""Abstract vector-store interface every adapter implements.

ChromaDB is the default (chroma_store.ChromaVectorStore) and stays a core
dependency. Qdrant, pgvector, and LanceDB live behind optional extras in
qdrant_store.py, pgvector_store.py, and lancedb_store.py.

`where` is a plain equality filter (field -> value, ANDed together) on
purpose: Chroma's own `$and`/`$or` filter DSL has no equivalent in the other
stores, so it does not belong on this interface. Adapters translate the plain
filter into their own native form.

Adapters own no index-activation or staged-generation machinery; that stays
in kb_arena.strategies.chroma_index for the strategies that need atomic
rebuilds. This interface is record-level: upsert, query, delete, count.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field


def validate_top_k(top_k: int) -> int:
    """One bound every adapter applies before a value reaches its backend.

    A caller-controlled `top_k` reached Chroma's `n_results`, Qdrant's `limit`,
    a SQL `LIMIT` and LanceDB's `limit()` unchecked, so one request could ask a
    remote store for a billion rows. The strategies bound the same parameter
    with `strategies.base.validate_top_k`, and this interface is a second door
    into the same stores.
    """
    from kb_arena.strategies.base import MAX_RETRIEVAL_CANDIDATES

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer")
    if not 1 <= top_k <= MAX_RETRIEVAL_CANDIDATES:
        raise ValueError(f"top_k must be between 1 and {MAX_RETRIEVAL_CANDIDATES}")
    return top_k


class VectorMatch(BaseModel):
    """One result from VectorStore.query(), in the adapter's own rank order."""

    id: str
    document: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: float


class VectorStore(ABC):
    """Record-level interface every vector-store adapter implements."""

    @abstractmethod
    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Insert or overwrite records by id."""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        """Return up to top_k matches, best match first, for one embedding."""

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove records by id."""

    @abstractmethod
    def count(self, where: Mapping[str, str] | None = None) -> int:
        """Return the number of records, optionally scoped by an equality filter."""
