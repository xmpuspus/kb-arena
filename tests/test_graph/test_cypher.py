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


def test_deprecation_chain_has_start_params():
    assert "$start_fqn" in cypher_templates.DEPRECATION_CHAIN
    assert "$start_name" in cypher_templates.DEPRECATION_CHAIN


def test_cross_reference_has_fqn():
    assert "$fqn" in cypher_templates.CROSS_REFERENCE


def test_type_hierarchy_has_fqn():
    assert "$fqn" in cypher_templates.TYPE_HIERARCHY


def test_usage_examples_has_fqn_and_name():
    assert "$fqn" in cypher_templates.USAGE_EXAMPLES
    assert "$name" in cypher_templates.USAGE_EXAMPLES


def test_fulltext_search_has_query_and_limit():
    assert "$query" in cypher_templates.FULLTEXT_ENTITY_SEARCH
    assert "$limit" in cypher_templates.FULLTEXT_ENTITY_SEARCH


# ── Template selection ─────────────────────────────────────────────────────────


def test_pick_template_deprecation():
    assert _pick_template("what deprecated this function?") == "DEPRECATION_CHAIN"


def test_pick_template_hierarchy():
    assert _pick_template("show me the inheritance hierarchy for Exception") == "TYPE_HIERARCHY"


def test_pick_template_examples():
    assert _pick_template("find usage examples for Lambda InvokeFunction") == "USAGE_EXAMPLES"


def test_pick_template_dependency():
    assert _pick_template("what does Lambda depend on?") == "DEPENDENCY_CHAIN"


def test_pick_template_comparison():
    assert _pick_template("compare Lambda vs EC2") == "COMPARISON_QUERY"


def test_pick_template_returns_none_for_unknown():
    assert _pick_template("xxxxxxx qqqqqq") is None


# ── Cypher validation ──────────────────────────────────────────────────────────


def test_validate_cypher_accepts_match():
    assert _validate_cypher("MATCH (n) RETURN n") is True


def test_validate_cypher_accepts_call():
    cypher = "CALL db.index.fulltext.queryNodes('x', 'q') YIELD node RETURN node"
    assert _validate_cypher(cypher) is True


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
async def test_generator_uses_llm_when_valid():
    mock_llm = AsyncMock()
    from kb_arena.llm.client import LLMResponse

    mock_llm.extract.return_value = LLMResponse(text="MATCH (n {fqn: $fqn}) RETURN n")

    gen = CypherGenerator(mock_llm, "aws-compute")
    cypher, params = await gen.generate("find Lambda", {"fqn": "lambda"})

    assert "MATCH" in cypher
    assert params == {"fqn": "lambda"}


@pytest.mark.asyncio
async def test_generator_falls_back_on_invalid_llm_output():
    mock_llm = AsyncMock()
    from kb_arena.llm.client import LLMResponse

    mock_llm.extract.return_value = LLMResponse(text="This is not Cypher at all.")

    gen = CypherGenerator(mock_llm, "aws-compute")
    # "deprecat" keyword → DEPRECATION_CHAIN template
    cypher, _ = await gen.generate("what deprecated this api?")

    assert "$start_fqn" in cypher  # DEPRECATION_CHAIN template


@pytest.mark.asyncio
async def test_generator_falls_back_on_llm_exception():
    mock_llm = AsyncMock()
    mock_llm.extract.side_effect = RuntimeError("API down")

    gen = CypherGenerator(mock_llm, "aws-compute")
    cypher, params = await gen.generate("some unknown query with no keywords xyz123")

    # Last resort: fulltext search
    assert "$query" in cypher
    assert "query" in params
