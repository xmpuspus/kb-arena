"""Interface tests: every VectorStore adapter, exercised with its client mocked.

No test makes a real network or database call. qdrant-client, psycopg, and
lancedb are optional extras (pyproject.toml) and are not installed here, so
the Qdrant adapter's SDK-specific model classes (PointStruct, Filter, ...)
are stood in for with a fake `qdrant_client` module registered in
sys.modules. pgvector and LanceDB need no SDK classes for a mocked client, so
they take a plain MagicMock/fake table.
"""

from __future__ import annotations

import subprocess
import sys
import types
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from kb_arena.vectorstores.base import VectorMatch
from kb_arena.vectorstores.chroma_store import ChromaVectorStore
from kb_arena.vectorstores.lancedb_store import LanceDBVectorStore
from kb_arena.vectorstores.pgvector_store import PgVectorStore
from kb_arena.vectorstores.qdrant_store import QdrantVectorStore

# --- Chroma ---


@pytest.fixture
def chroma_store(mock_chroma_client):
    collection = mock_chroma_client.get_or_create_collection.return_value
    collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["doc one"]],
        "metadatas": [[{"source_doc_id": "d1"}]],
        "distances": [[0.1]],
    }
    collection.count.return_value = 2
    return ChromaVectorStore(client=mock_chroma_client), collection


# --- Qdrant (SDK stubbed via sys.modules, client itself mocked) ---


@dataclass
class _FakePointStruct:
    id: str
    vector: list
    payload: dict


@dataclass
class _FakeMatchValue:
    value: Any


@dataclass
class _FakeFieldCondition:
    key: str
    match: _FakeMatchValue


@dataclass
class _FakeFilter:
    must: list = field(default_factory=list)


@pytest.fixture
def qdrant_store(monkeypatch):
    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.PointStruct = _FakePointStruct
    fake_models.Filter = _FakeFilter
    fake_models.FieldCondition = _FakeFieldCondition
    fake_models.MatchValue = _FakeMatchValue
    fake_qdrant_client.models = fake_models
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)

    client = MagicMock()
    point = types.SimpleNamespace(
        id="c1", payload={"document": "doc one", "source_doc_id": "d1"}, score=0.9
    )
    client.query_points.return_value = types.SimpleNamespace(points=[point])
    client.count.return_value = types.SimpleNamespace(count=2)
    # Cosine is Qdrant's default and the metric this project configures, so the
    # fixture reports it. A test that needs another metric overrides it.
    client.get_collection.return_value = types.SimpleNamespace(
        config=types.SimpleNamespace(
            params=types.SimpleNamespace(vectors=types.SimpleNamespace(distance="Cosine"))
        )
    )
    return QdrantVectorStore(client=client), client


# --- pgvector (plain SQL, no SDK model classes needed) ---


@pytest.fixture
def pgvector_store():
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [("c1", "doc one", {"source_doc_id": "d1"}, 0.1)]
    cursor.fetchone.return_value = (2,)
    return PgVectorStore(client=conn), conn


# --- LanceDB (dict-based API, no SDK model classes needed) ---


@pytest.fixture
def lancedb_store():
    table = MagicMock()
    search_builder = table.search.return_value
    search_builder.limit.return_value = search_builder
    search_builder.where.return_value = search_builder
    search_builder.to_list.return_value = [
        {"id": "c1", "document": "doc one", "metadata": {"source_doc_id": "d1"}, "_distance": 0.1}
    ]
    table.count_rows.return_value = 2
    merge_builder = table.merge_insert.return_value
    merge_builder.when_matched_update_all.return_value = merge_builder
    merge_builder.when_not_matched_insert_all.return_value = merge_builder
    return LanceDBVectorStore(client=table), table


ADAPTER_FIXTURES = ["chroma_store", "qdrant_store", "pgvector_store", "lancedb_store"]


@pytest.mark.parametrize("fixture_name", ADAPTER_FIXTURES)
def test_query_returns_a_ranked_list_of_vector_matches(fixture_name, request):
    store, _ = request.getfixturevalue(fixture_name)

    matches = store.query([0.1, 0.2, 0.3], top_k=1)

    assert isinstance(matches, list)
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, VectorMatch)
    assert match.id == "c1"
    assert match.document == "doc one"
    assert match.metadata.get("source_doc_id") == "d1"
    assert isinstance(match.distance, float)


@pytest.mark.parametrize("fixture_name", ADAPTER_FIXTURES)
def test_count_returns_an_int(fixture_name, request):
    store, _ = request.getfixturevalue(fixture_name)

    assert store.count() == 2


def test_chroma_upsert_and_delete_forward_the_ids(chroma_store):
    store, collection = chroma_store

    store.upsert(["c1"], ["doc one"], [[0.1, 0.2, 0.3]], [{"source_doc_id": "d1"}])
    assert collection.upsert.call_args.kwargs["ids"] == ["c1"]

    store.delete(["c1", "c2"])
    assert collection.delete.call_args.kwargs["ids"] == ["c1", "c2"]


def test_qdrant_maps_a_chunk_id_to_a_uuid_and_keeps_the_original(qdrant_store):
    """Qdrant takes an unsigned integer or a UUID, and a chunk id is neither.

    Forwarding the chunk id raised "Point id c1 is not a valid UUID" against a
    real server. The mapping is deterministic, so two processes agree, and the
    payload carries the original so a caller reads back the id it wrote.
    """
    from kb_arena.vectorstores.qdrant_store import _CHUNK_ID_FIELD, _point_id

    store, client = qdrant_store

    store.upsert(["c1"], ["doc one"], [[0.1, 0.2, 0.3]], [{"source_doc_id": "d1"}])
    [point] = client.upsert.call_args.kwargs["points"]

    assert point.id == _point_id("c1") != "c1"
    assert uuid.UUID(point.id).version == 5
    assert point.payload[_CHUNK_ID_FIELD] == "c1"

    store.delete(["c1", "c2"])
    assert client.delete.call_args.kwargs["points_selector"] == [
        _point_id("c1"),
        _point_id("c2"),
    ]


def test_a_qdrant_match_reads_back_the_chunk_id_the_caller_wrote(qdrant_store):
    """The uuid is storage, and the caller never sees it."""
    from kb_arena.vectorstores.qdrant_store import _CHUNK_ID_FIELD, _point_id

    store, client = qdrant_store
    point = SimpleNamespace(
        id=_point_id("c1"),
        payload={"document": "doc one", _CHUNK_ID_FIELD: "c1", "source_doc_id": "d1"},
        score=0.25,
    )
    client.query_points.return_value = SimpleNamespace(points=[point])

    [match] = store.query([0.1, 0.2, 0.3], top_k=1)

    assert match.id == "c1"
    assert match.metadata == {"source_doc_id": "d1"}


def test_pgvector_upsert_and_delete_forward_the_ids(pgvector_store):
    store, conn = pgvector_store
    cursor = conn.cursor.return_value.__enter__.return_value

    store.upsert(["c1"], ["doc one"], [[0.1, 0.2, 0.3]], [{"source_doc_id": "d1"}])
    upsert_args = cursor.execute.call_args_list[0].args[1]
    assert upsert_args[0] == "c1"

    store.delete(["c1", "c2"])
    delete_args = cursor.execute.call_args.args[1]
    assert delete_args == (["c1", "c2"],)


def test_lancedb_upsert_and_delete_forward_the_ids(lancedb_store):
    store, table = lancedb_store
    merge_builder = table.merge_insert.return_value

    store.upsert(["c1"], ["doc one"], [[0.1, 0.2, 0.3]], [{"source_doc_id": "d1"}])
    rows = merge_builder.execute.call_args.args[0]
    assert rows[0]["id"] == "c1"

    store.delete(["c1", "c2"])
    where_clause = table.delete.call_args.args[0]
    assert "'c1'" in where_clause and "'c2'" in where_clause


def test_vectorstores_import_pulls_no_optional_sdk():
    """Every optional adapter's SDK import is lazy — confirms I-02's own rule.

    Runs in a fresh interpreter. This test file's own module-level imports
    already put kb_arena.vectorstores in this process's sys.modules, so a
    reload here would hit import caches and prove nothing.
    """
    code = (
        "import kb_arena.vectorstores, sys; "
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'qdrant_client', 'psycopg', 'lancedb'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_lancedb_filters_before_the_vector_search(lancedb_store):
    """LanceDB post-filters by default, so `top_k` rows become fewer than `top_k`.

    The search would take the nearest `top_k` and the filter would then cut
    them, which reads as a store holding less than it does. Every other adapter
    filters first, and one interface promises one behaviour.
    """
    store, table = lancedb_store

    store.query([0.1, 0.2, 0.3], top_k=5, where={"source_doc_id": "d1"})

    assert table.search.return_value.where.call_args.kwargs == {"prefilter": True}


def test_qdrant_metadata_cannot_take_another_record_s_identity(qdrant_store):
    """Metadata expanded last let a chunk name itself `c2` and spoof the document.

    A query then answered with the wrong id, and deleting that id left the real
    point in place under the UUID for `c1`.
    """
    from kb_arena.vectorstores.qdrant_store import _CHUNK_ID_FIELD

    store, client = qdrant_store

    store.upsert(
        ["c1"],
        ["trusted"],
        [[0.1, 0.2, 0.3]],
        [{"chunk_id": "c2", "document": "spoofed", "source_doc_id": "d1"}],
    )
    [point] = client.upsert.call_args.kwargs["points"]

    assert point.payload[_CHUNK_ID_FIELD] == "c1"
    assert point.payload["document"] == "trusted"


def test_a_failed_pgvector_statement_leaves_the_connection_usable(pgvector_store):
    """A cursor context manager does not roll back a failed transaction.

    One bad statement left the shared connection inside a failed transaction,
    and every later call raised `InFailedSqlTransaction`, including reads that
    had nothing to do with the failure.
    """
    store, conn = pgvector_store
    conn.cursor.return_value.__enter__.return_value.execute.side_effect = RuntimeError("bad vector")

    with pytest.raises(RuntimeError):
        store.upsert(["c1"], ["doc"], [[0.1, 0.2]], [{}])

    conn.rollback.assert_called_once()


def test_a_qdrant_score_becomes_a_distance(qdrant_store):
    """Qdrant answers a similarity, and the interface promises a distance.

    Passing the score through inverted the order, so a consumer sorting
    `VectorMatch.distance` ascending promoted the worse match.
    """
    from kb_arena.vectorstores.qdrant_store import _CHUNK_ID_FIELD

    store, client = qdrant_store
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="a", payload={"document": "near", _CHUNK_ID_FIELD: "c1"}, score=1.0),
            SimpleNamespace(id="b", payload={"document": "far", _CHUNK_ID_FIELD: "c2"}, score=0.0),
        ]
    )

    near, far = store.query([0.1, 0.2, 0.3], top_k=2)

    assert near.distance == 0.0
    assert far.distance == 1.0
    assert sorted([near, far], key=lambda m: m.distance)[0].document == "near"


def test_a_euclid_collection_keeps_its_scores_as_distances(qdrant_store):
    """Qdrant's score means different things per collection metric.

    Cosine and dot answer a similarity, higher for a better match. Euclid
    answers a distance already. Subtracting every score from one reversed the
    order on a Euclid collection, so sorting ascending picked the farther point.
    """
    from kb_arena.vectorstores.qdrant_store import _CHUNK_ID_FIELD

    store, client = qdrant_store
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(distance="Euclid")))
    )
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="a", payload={"document": "near", _CHUNK_ID_FIELD: "c1"}, score=0.1),
            SimpleNamespace(id="b", payload={"document": "far", _CHUNK_ID_FIELD: "c2"}, score=0.9),
        ]
    )

    near, far = store.query([0.1, 0.2, 0.3], top_k=2)

    assert near.distance == pytest.approx(0.1)
    assert far.distance == pytest.approx(0.9)


def test_an_unreadable_metric_reads_as_cosine(qdrant_store):
    """Cosine is Qdrant's default and the one this project configures."""
    store, client = qdrant_store
    client.get_collection.side_effect = RuntimeError("no such collection")

    assert store._as_distance(1.0) == pytest.approx(0.0)


def test_a_metric_lookup_that_fails_is_not_cached(qdrant_store):
    """Caching a failed lookup fixed the wrong conversion in place for good.

    One timeout would have made every later distance wrong on a Euclid
    collection, even after the lookup started working again.
    """
    store, client = qdrant_store
    client.get_collection.side_effect = RuntimeError("timed out")

    store._as_distance(0.5)
    assert store._similarity_metric is None

    client.get_collection.side_effect = None
    client.get_collection.return_value = SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(distance="Euclid")))
    )

    assert store._as_distance(0.1) == pytest.approx(0.1)


def test_a_table_name_that_is_not_an_identifier_is_refused():
    """A table cannot be a bound parameter, so the name reaches the statement whole.

    A configured name carrying `; DROP TABLE ...` would change the statement.
    """
    from unittest.mock import MagicMock

    from kb_arena.vectorstores.pgvector_store import PgVectorStore

    with pytest.raises(ValueError, match="plain SQL identifier"):
        PgVectorStore(client=MagicMock(), table_name="kb_arena; DROP TABLE audit_log; --")
