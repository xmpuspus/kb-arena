"""LanceDB adapter for the VectorStore interface.

Optional — requires `pip install 'kb-arena[lancedb]'`. LanceDB's own API is
already dict- and string-based (no request/model classes to construct), so
this adapter needs no SDK types beyond the table object itself.

lancedb is lazy-imported inside __init__, and only on the path that connects
for the caller. An injected `client` (an open LanceDB table) skips the
import entirely, so `import kb_arena.vectorstores` never pulls it in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kb_arena.vectorstores.base import VectorMatch, VectorStore


def _quote(value: str) -> str:
    """Escape a value for LanceDB's SQL-like filter string.

    LanceDB parses `where`/`filter` as a predicate, so a value is never bound
    as a parameter the way a SQL driver would bind one. Doubling an embedded
    quote keeps a value from closing its string literal early.
    """
    return value.replace("'", "''")


def _where_sql(where: Mapping[str, str] | None) -> str | None:
    if not where:
        return None
    return " AND ".join(
        f"metadata['{_quote(key)}'] = '{_quote(value)}'" for key, value in where.items()
    )


class LanceDBVectorStore(VectorStore):
    """Wraps one LanceDB table behind the VectorStore interface."""

    def __init__(
        self,
        *,
        client: Any = None,
        table_name: str = "kb_arena",
        uri: str = ".lancedb",
    ) -> None:
        if client is not None:
            self._table = client
            return
        try:
            import lancedb
        except ImportError as exc:
            raise ImportError(
                "lancedb is required for the LanceDB vector store. "
                "Install with: pip install 'kb-arena[lancedb]'"
            ) from exc
        db = lancedb.connect(uri)
        self._table = db.open_table(table_name)

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        rows = [
            {
                "id": ids[i],
                "document": documents[i],
                "vector": list(embeddings[i]),
                "metadata": dict(metadatas[i]),
            }
            for i in range(len(ids))
        ]
        merge = self._table.merge_insert("id")
        merge = merge.when_matched_update_all().when_not_matched_insert_all()
        merge.execute(rows)

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        search = self._table.search(list(embedding)).limit(top_k)
        clause = _where_sql(where)
        if clause:
            search = search.where(clause)
        rows = search.to_list()
        return [
            VectorMatch(
                id=str(row["id"]),
                document=str(row["document"]),
                metadata=dict(row.get("metadata") or {}),
                distance=float(row.get("_distance", 0.0)),
            )
            for row in rows
        ]

    def delete(self, ids: Sequence[str]) -> None:
        id_list = ", ".join(f"'{_quote(item)}'" for item in ids)
        self._table.delete(f"id IN ({id_list})")

    def count(self, where: Mapping[str, str] | None = None) -> int:
        clause = _where_sql(where)
        if clause is None:
            return self._table.count_rows()
        return self._table.count_rows(clause)
