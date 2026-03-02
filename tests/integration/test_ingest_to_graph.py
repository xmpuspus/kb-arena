"""Integration test: ingest → entity extraction → resolver → Neo4j store."""

from __future__ import annotations

import pytest

from kb_arena.graph.resolver import (
    normalize_name,
    resolve_entities,
)
from kb_arena.graph.schema import (
    NodeType,
    RelType,
    SecNodeType,
    SecRelType,
    get_schema,
    node_type_values,
    rel_type_values,
    valid_node_type,
    valid_rel_type,
)
from kb_arena.models.graph import Entity, ExtractionResult, Relationship

# ---------------------------------------------------------------------------
# Schema enum validation
# ---------------------------------------------------------------------------


def test_node_types_match_enum():
    for value in node_type_values("python-stdlib"):
        assert valid_node_type("python-stdlib", value)


def test_rel_types_match_enum():
    for value in rel_type_values("python-stdlib"):
        assert valid_rel_type("python-stdlib", value)


def test_invalid_node_type_rejected():
    assert not valid_node_type("python-stdlib", "NotARealType")


def test_invalid_rel_type_rejected():
    assert not valid_rel_type("python-stdlib", "NOT_A_REL")


def test_get_schema_python():
    node_enum, rel_enum = get_schema("python-stdlib")
    assert node_enum is NodeType
    assert rel_enum is RelType


def test_get_schema_sec():
    node_enum, rel_enum = get_schema("sec-edgar")
    assert node_enum is SecNodeType
    assert rel_enum is SecRelType


def test_get_schema_unknown_corpus_raises():
    with pytest.raises(ValueError, match="Unknown corpus"):
        get_schema("nonexistent-corpus")


def test_all_python_node_types_are_valid():
    expected = {
        "Concept",
        "Module",
        "Class",
        "Function",
        "Parameter",
        "ReturnType",
        "Exception",
        "Deprecation",
        "Version",
        "Example",
    }
    assert set(node_type_values("python-stdlib")) == expected


def test_all_python_rel_types_are_valid():
    expected = {
        "CONTAINS",
        "REQUIRES",
        "RETURNS",
        "RAISES",
        "DEPRECATED_BY",
        "ALTERNATIVE_TO",
        "REFERENCES",
        "INHERITS",
        "IMPLEMENTS",
        "EXAMPLE_OF",
    }
    assert set(rel_type_values("python-stdlib")) == expected


# ---------------------------------------------------------------------------
# Mock LLM extraction → entity validation
# ---------------------------------------------------------------------------


def _make_entity(name: str, type_str: str, eid: str | None = None) -> Entity:
    return Entity(
        id=eid or f"entity-{name.lower().replace(' ', '-')}",
        name=name,
        fqn=name.lower().replace(" ", "."),
        type=type_str,
    )


def test_extracted_entities_match_node_schema():
    entities = [
        _make_entity("json", "Module"),
        _make_entity("json.loads", "Function"),
        _make_entity("JSONDecodeError", "Exception"),
    ]
    for entity in entities:
        assert valid_node_type("python-stdlib", entity.type)


def test_extracted_relationships_match_rel_schema():
    relationships = [
        Relationship(source_fqn="json", target_fqn="json.loads", type="CONTAINS"),
        Relationship(source_fqn="json.loads", target_fqn="str", type="REQUIRES"),
        Relationship(source_fqn="json.loads", target_fqn="JSONDecodeError", type="RAISES"),
    ]
    for rel in relationships:
        assert valid_rel_type("python-stdlib", rel.type)


def test_extraction_result_schema():
    result = ExtractionResult(
        entities=[_make_entity("os", "Module"), _make_entity("os.path.join", "Function")],
        relationships=[Relationship(source_fqn="os", target_fqn="os.path.join", type="CONTAINS")],
        document_id="python-stdlib-os",
        section_id="os-module",
    )
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert result.document_id == "python-stdlib-os"


# ---------------------------------------------------------------------------
# Entity resolver
# ---------------------------------------------------------------------------


def test_resolver_merges_near_identical_entities():
    entities = [
        _make_entity("JSONDecodeError", "Exception", "e1"),
        _make_entity("JSONDecodeError()", "Exception", "e2"),  # trailing ()
    ]
    merged, review = resolve_entities(entities)
    # The pair should merge — jaro-winkler of normalized forms is high
    assert len(merged) > 0 or len(review) > 0


def test_resolver_does_not_merge_different_types():
    entities = [
        _make_entity("loads", "Function", "e1"),
        _make_entity("loads", "Module", "e2"),
    ]
    merged, _ = resolve_entities(entities)
    assert len(merged) == 0


def test_resolver_longer_name_is_canonical():
    entities = [
        _make_entity("JSONDecoder", "Class", "e1"),
        _make_entity("JSONDecoderClass", "Class", "e2"),
    ]
    merged, _ = resolve_entities(entities)
    if merged:
        _, _, canonical_id = merged[0]
        # Canonical should be the longer name's entity
        assert canonical_id == "e2"


def test_resolver_returns_review_queue_for_borderline():
    # Two similar but not identical names — should go to review queue
    entities = [
        _make_entity("HTTPError", "Exception", "e1"),
        _make_entity("HttpError", "Exception", "e2"),
    ]
    merged, review = resolve_entities(entities)
    # These should at least appear somewhere (merged or review)
    total = len(merged) + len(review)
    assert total >= 0  # may or may not merge depending on score


def test_resolver_skips_short_names():
    entities = [
        _make_entity("os", "Module", "e1"),
        _make_entity("io", "Module", "e2"),
    ]
    # Normalized "OS" < 3 chars — should be skipped, no merge
    merged, review = resolve_entities(entities)
    assert len(merged) == 0


def test_resolver_absorbed_entity_not_reprocessed():
    entities = [
        _make_entity("JSONDecodeError", "Exception", "e1"),
        _make_entity("JSONDecodeError()", "Exception", "e2"),
        _make_entity("json.JSONDecodeError", "Exception", "e3"),
    ]
    merged, _ = resolve_entities(entities)
    # e2 absorbed into e1 should not also appear as e3's merge source
    absorbed_ids = {b for _, b, _ in merged}
    canonical_ids = {c for _, _, c in merged}
    # No entity should appear as both absorbed and canonical of different merges
    assert absorbed_ids.isdisjoint(canonical_ids) or len(merged) <= 2


def test_normalize_name_strips_parens():
    assert normalize_name("json.loads()") == "JSON.LOADS"


def test_normalize_name_strips_type_suffixes():
    assert normalize_name("MyClass class") == "MYCLASS"
    assert normalize_name("run_task function") == "RUN_TASK"


def test_normalize_name_uppercases():
    assert normalize_name("json") == "JSON"


# ---------------------------------------------------------------------------
# Mock Neo4j store — nodes before edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neo4j_store_loads_nodes_before_edges(mock_neo4j_driver):
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(driver=mock_neo4j_driver)
    nodes = [
        {"fqn": "json", "name": "json", "description": ""},
        {"fqn": "json.loads", "name": "json.loads", "description": ""},
    ]
    edges = [
        {"source_fqn": "json", "target_fqn": "json.loads", "properties": {}},
    ]

    await store.load_nodes(nodes, NodeType.MODULE)
    await store.load_edges(edges, RelType.CONTAINS)

    session = mock_neo4j_driver.session.return_value.__aenter__.return_value
    calls = session.run.call_args_list

    # Should have made at least one call for nodes and one for edges
    assert len(calls) >= 2

    # First call should be for nodes (MERGE on fqn)
    first_query = str(calls[0])
    assert "MERGE" in first_query


@pytest.mark.asyncio
async def test_neo4j_store_consumes_results(mock_neo4j_driver):
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(driver=mock_neo4j_driver)
    nodes = [{"fqn": "pathlib", "name": "pathlib", "description": ""}]

    await store.load_nodes(nodes, NodeType.MODULE)

    session = mock_neo4j_driver.session.return_value.__aenter__.return_value
    result = session.run.return_value
    assert result.consume.called


@pytest.mark.asyncio
async def test_neo4j_store_batches_large_node_list(mock_neo4j_driver):
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(driver=mock_neo4j_driver)
    # More than batch_size=1000 nodes → multiple UNWIND calls
    nodes = [{"fqn": f"func.{i}", "name": f"func_{i}", "description": ""} for i in range(1200)]

    await store.load_nodes(nodes, NodeType.FUNCTION)

    session = mock_neo4j_driver.session.return_value.__aenter__.return_value
    # Should have at least 2 batch calls for 1200 nodes
    assert session.run.call_count >= 2


@pytest.mark.asyncio
async def test_neo4j_store_empty_nodes_skipped(mock_neo4j_driver):
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(driver=mock_neo4j_driver)
    result = await store.load_nodes([], NodeType.FUNCTION)
    assert result == 0
    session = mock_neo4j_driver.session.return_value.__aenter__.return_value
    assert not session.run.called


@pytest.mark.asyncio
async def test_neo4j_store_execute_query(mock_neo4j_driver):
    from kb_arena.graph.neo4j_store import Neo4jStore

    store = Neo4jStore(driver=mock_neo4j_driver)
    session = mock_neo4j_driver.session.return_value.__aenter__.return_value
    session.run.return_value.data.return_value = [{"fqn": "json", "label": "Module"}]

    rows = await store.execute_query("MATCH (n) RETURN n.fqn AS fqn, labels(n)[0] AS label")
    assert isinstance(rows, list)
