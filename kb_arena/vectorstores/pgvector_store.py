"""pgvector adapter for the VectorStore interface.

Optional — requires `pip install 'kb-arena[pgvector]'` (psycopg only; the
Postgres server needs the `vector` extension, not a Python package). An
embedding is formatted as a `[a,b,c]` literal and bound as a plain string
parameter, cast to `vector` in SQL, so this needs no SDK model classes.

psycopg is lazy-imported inside __init__, and only on the path that connects
for the caller. An injected `client` (an open psycopg connection) skips the
import entirely, so `import kb_arena.vectorstores` never pulls it in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from kb_arena.vectorstores.base import VectorMatch, VectorStore


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


def _where_sql(where: Mapping[str, str] | None) -> tuple[str, list[str]]:
    """Build a `WHERE metadata ->> %s = %s AND ...` clause with every value bound.

    `->>` takes its key as an ordinary text argument, so both the filter key
    and the filter value bind as parameters. Nothing from the caller's filter
    dict is ever written into the SQL text itself.
    """
    if not where:
        return "", []
    clauses = ["metadata ->> %s = %s"] * len(where)
    params: list[str] = []
    for key, value in where.items():
        params.extend((key, value))
    return " WHERE " + " AND ".join(clauses), params


class PgVectorStore(VectorStore):
    """Wraps one Postgres table with a `vector` column behind the interface."""

    def __init__(
        self,
        *,
        client: Any = None,
        table_name: str = "kb_arena",
        dsn: str = "",
    ) -> None:
        if client is not None:
            self._conn = client
        else:
            try:
                import psycopg
            except ImportError as exc:
                raise ImportError(
                    "psycopg is required for the pgvector store. "
                    "Install with: pip install 'kb-arena[pgvector]'"
                ) from exc
            self._conn = psycopg.connect(dsn)
        self._table = table_name

    def upsert(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        import json

        with self._conn.cursor() as cur:
            for i in range(len(ids)):
                cur.execute(
                    f"INSERT INTO {self._table} (id, document, embedding, metadata) "
                    "VALUES (%s, %s, %s::vector, %s::jsonb) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "document = EXCLUDED.document, embedding = EXCLUDED.embedding, "
                    "metadata = EXCLUDED.metadata",
                    (
                        ids[i],
                        documents[i],
                        _vector_literal(embeddings[i]),
                        json.dumps(dict(metadatas[i])),
                    ),
                )
        self._conn.commit()

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        where: Mapping[str, str] | None = None,
    ) -> list[VectorMatch]:
        clause, params = _where_sql(where)
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, document, metadata, embedding <-> %s::vector AS distance "
                f"FROM {self._table}{clause} ORDER BY distance LIMIT %s",
                [_vector_literal(embedding), *params, top_k],
            )
            rows = cur.fetchall()
        return [
            VectorMatch(
                id=str(row[0]),
                document=str(row[1]),
                metadata=dict(row[2] or {}),
                distance=float(row[3]),
            )
            for row in rows
        ]

    def delete(self, ids: Sequence[str]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE id = ANY(%s)", (list(ids),))
        self._conn.commit()

    def count(self, where: Mapping[str, str] | None = None) -> int:
        clause, params = _where_sql(where)
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}{clause}", params)
            return int(cur.fetchone()[0])
