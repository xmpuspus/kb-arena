"""Tests for Cypher template parameter presence and CypherGenerator fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.graph import cypher_templates
from kb_arena.graph.cypher_generator import CypherGenerator, _pick_template, _validate_cypher

# ── Template parameter coverage ────────────────────────────────────────────────


def test_single_entity_lookup_has_fqn_param():
    assert "$fqn" in cypher_templates.SINGLE_ENTITY_LOOKUP


def test_multi_hop_has_target_and_depth():
    assert "$target" in cypher_templates.MULTI_HOP_QUERY
    assert "$depth" in cypher_templates.MULTI_HOP_QUERY
    assert "$allowed_rel_types" in cypher_templates.MULTI_HOP_QUERY


def test_comparison_query_has_entity_params():
    assert "$entity_a" in cypher_templates.COMPARISON_QUERY
    assert "$entity_b" in cypher_templates.COMPARISON_QUERY


def test_dependency_chain_has_start():
    assert "$start" in cypher_templates.DEPENDENCY_CHAIN
    # Must use valid schema rel types only
    assert "DEPENDS_ON" in cypher_templates.DEPENDENCY_CHAIN
    assert "REQUIRES" not in cypher_templates.DEPENDENCY_CHAIN
    assert "INHERITS" not in cypher_templates.DEPENDENCY_CHAIN


def test_cross_reference_has_fqn():
    assert "$fqn" in cypher_templates.CROSS_REFERENCE
    # Uses generic relationship match (any type)
    assert "REFERENCES" not in cypher_templates.CROSS_REFERENCE


def test_type_hierarchy_has_fqn():
    assert "$fqn" in cypher_templates.TYPE_HIERARCHY
    # Must use EXTENDS (valid schema), not INHERITS
    assert "EXTENDS" in cypher_templates.TYPE_HIERARCHY
    assert "INHERITS" not in cypher_templates.TYPE_HIERARCHY


def test_fulltext_search_has_query_and_limit():
    assert "$query" in cypher_templates.FULLTEXT_ENTITY_SEARCH
    assert "$limit" in cypher_templates.FULLTEXT_ENTITY_SEARCH


# ── Template selection ─────────────────────────────────────────────────────────


def test_pick_template_hierarchy():
    assert _pick_template("show me the inheritance hierarchy for Exception") == "TYPE_HIERARCHY"


def test_pick_template_extends():
    assert _pick_template("what extends this component?") == "TYPE_HIERARCHY"


def test_pick_template_dependency():
    assert _pick_template("what does Lambda depend on?") == "DEPENDENCY_CHAIN"


def test_pick_template_comparison():
    assert _pick_template("compare Lambda vs EC2") == "COMPARISON_QUERY"


def test_pick_template_returns_none_for_unknown():
    assert _pick_template("xxxxxxx qqqqqq") is None


# ── Cypher validation ──────────────────────────────────────────────────────────


def test_validate_cypher_accepts_allowlisted_template():
    assert _validate_cypher(cypher_templates.SINGLE_ENTITY_LOOKUP) is True


def test_validate_cypher_accepts_allowlisted_fulltext():
    assert _validate_cypher(cypher_templates.FULLTEXT_ENTITY_SEARCH) is True


def test_validate_cypher_rejects_unowned_match():
    assert _validate_cypher("MATCH (n) RETURN n") is False


def test_validate_cypher_rejects_mixed_owned_and_unowned_matches():
    cypher = "MATCH (owned:KBArenaEntity) MATCH (secret:OtherAppSecret) RETURN secret.value AS name"
    assert _validate_cypher(cypher) is False


def test_validate_cypher_rejects_plain_text():
    assert _validate_cypher("Here is your answer: Lambda runs code without servers.") is False


def test_validate_cypher_rejects_create():
    assert _validate_cypher("MATCH (n) CREATE (m) RETURN m") is False


def test_validate_cypher_rejects_delete():
    assert _validate_cypher("MATCH (n) DELETE n") is False


def test_validate_cypher_rejects_detach_delete():
    assert _validate_cypher("MATCH (n) DETACH DELETE n") is False


def test_validate_cypher_rejects_set():
    assert _validate_cypher("MATCH (n) SET n.name = 'evil' RETURN n") is False


def test_validate_cypher_rejects_merge():
    assert _validate_cypher("MERGE (n:Node {name: 'test'}) RETURN n") is False


def test_validate_cypher_rejects_drop():
    assert _validate_cypher("MATCH (n) DROP INDEX my_index") is False


# ── CypherGenerator fallback ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generator_uses_template_when_required_params_are_present():
    mock_llm = AsyncMock()

    gen = CypherGenerator(mock_llm, "aws-compute")
    cypher, params = await gen.generate("find Lambda", {"fqn": "lambda"})

    assert cypher == cypher_templates.SINGLE_ENTITY_LOOKUP
    assert params == {"fqn": "lambda"}
    mock_llm.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_generator_uses_fulltext_when_template_params_are_missing():
    mock_llm = AsyncMock()

    gen = CypherGenerator(mock_llm, "aws-compute")
    cypher, params = await gen.generate("what does Lambda depend on?")

    assert cypher == cypher_templates.FULLTEXT_ENTITY_SEARCH
    assert params["query"] == "what does Lambda depend on?"
    mock_llm.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_generator_never_executes_llm_supplied_cypher():
    mock_llm = AsyncMock()
    mock_llm.extract.return_value.text = (
        "MATCH (owned:KBArenaEntity) MATCH (secret:OtherAppSecret) RETURN secret.value AS name"
    )

    gen = CypherGenerator(mock_llm, "aws-compute")
    cypher, params = await gen.generate("some unknown query with no keywords xyz123")

    assert cypher == cypher_templates.FULLTEXT_ENTITY_SEARCH
    assert "query" in params
    mock_llm.extract.assert_not_awaited()
