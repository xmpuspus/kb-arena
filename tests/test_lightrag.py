"""LightRAG strategy — local neighborhood and global community retrieval.

Regression coverage for the one rule S-08 cannot drop: every chunk, local or
global, keeps its source_doc_id and source_section_id through the graph and
back, in the same "graph:{doc}::{section}" shape knowledge_graph established.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.strategies.lightrag import (
    LightRAGStrategy,
    _largest_component,
)


def _local_records():
    return [
        {
            "name": "AWS Lambda",
            "fqn": "aws.lambda",
            "type": "Service",
            "description": "Serverless compute",
            "source_doc_id": "lambda-overview",
            "source_section_id": "aws-lambda",
        },
        {
            "name": "Lambda Layers",
            "fqn": "aws.lambda.layers",
            "type": "Component",
            "description": "Shared code and libraries",
            "source_doc_id": "lambda-overview",
            "source_section_id": "layers",
        },
    ]


def _global_records():
    return [
        {
            "name": "AWS Lambda",
            "fqn": "aws.lambda",
            "type": "Service",
            "description": "Serverless compute",
            "source_doc_id": "lambda-overview",
            "source_section_id": "aws-lambda",
        },
        {
            "name": "API Gateway",
            "fqn": "aws.apigateway",
            "type": "Service",
            "description": "Managed API front door",
            "source_doc_id": "apigateway-overview",
            "source_section_id": "apigateway",
        },
    ]


@pytest.mark.asyncio
async def test_local_retrieval_walks_the_neighborhood_of_a_matched_entity():
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._run_cypher = AsyncMock(return_value=_local_records())

    records = await strategy._local_retrieval("What is aws.lambda?", "all")

    assert records == _local_records()
    strategy._run_cypher.assert_awaited_once()
    cypher, params = strategy._run_cypher.await_args.args
    assert params["fqn"] == "aws.lambda"


@pytest.mark.asyncio
async def test_local_retrieval_with_no_matched_entity_runs_no_query():
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._run_cypher = AsyncMock(return_value=_local_records())

    records = await strategy._local_retrieval("tell me about it", "all")

    assert records == []
    strategy._run_cypher.assert_not_awaited()


def test_global_retrieval_groups_candidates_into_the_largest_connected_component():
    records = [
        {"entity_id": "c::a", "name": "A", "neighbor_ids": ["c::b"]},
        {"entity_id": "c::b", "name": "B", "neighbor_ids": ["c::a"]},
        {"entity_id": "c::c", "name": "C", "neighbor_ids": []},  # isolated, smaller
    ]

    component = _largest_component(records)

    assert {r["entity_id"] for r in component} == {"c::a", "c::b"}


@pytest.mark.asyncio
async def test_global_retrieval_reads_the_community_summary_and_drops_the_adjacency_ids():
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._run_cypher = AsyncMock(
        return_value=[
            {**_global_records()[0], "entity_id": "c::aws.lambda", "neighbor_ids": ["c::x"]},
            {**_global_records()[1], "entity_id": "c::x", "neighbor_ids": ["c::aws.lambda"]},
        ]
    )

    records = await strategy._global_retrieval("serverless compute", "all")

    assert len(records) == 2
    assert all("entity_id" not in r and "neighbor_ids" not in r for r in records)
    assert {r["source_doc_id"] for r in records} == {"lambda-overview", "apigateway-overview"}


@pytest.mark.asyncio
async def test_query_answers_with_both_and_labels_which_path_produced_each_chunk():
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._local_retrieval = AsyncMock(return_value=_local_records())
    strategy._global_retrieval = AsyncMock(return_value=_global_records())
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(
        return_value=LLMResponse(text="answer", input_tokens=10, output_tokens=5, cost_usd=0.001)
    )

    result = await strategy.query("What is aws.lambda?")

    assert result.mock is False
    modes = {c.chunk_id: c.metadata["retrieval_mode"] for c in result.retrieval.retrieved}
    assert modes["graph:lambda-overview::aws-lambda"] == "local"
    assert modes["graph:apigateway-overview::apigateway"] == "global"


@pytest.mark.asyncio
async def test_source_doc_and_section_ids_survive_local_and_global_retrieval_unchanged():
    """S-08's one non-negotiable rule: the id passes through the graph and back."""
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._local_retrieval = AsyncMock(return_value=_local_records())
    strategy._global_retrieval = AsyncMock(return_value=_global_records())
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(return_value=LLMResponse(text="answer"))

    result = await strategy.query("What is aws.lambda?")

    by_id = {c.chunk_id: c for c in result.retrieval.retrieved}
    local_chunk = by_id["graph:lambda-overview::aws-lambda"]
    assert local_chunk.doc_id == "lambda-overview"
    assert local_chunk.metadata["retrieval_mode"] == "local"

    global_chunk = by_id["graph:apigateway-overview::apigateway"]
    assert global_chunk.doc_id == "apigateway-overview"
    assert global_chunk.metadata["retrieval_mode"] == "global"


@pytest.mark.asyncio
async def test_query_without_a_driver_returns_mock_data_not_a_fabricated_answer():
    strategy = LightRAGStrategy(neo4j_driver=None)

    result = await strategy.query("What is aws.lambda?")

    assert result.mock is True
    assert "not connected" in result.answer.lower()


@pytest.mark.asyncio
async def test_the_prompt_reads_only_the_chunks_the_trace_reports():
    """The prompt used to read both branch lists, which hold more than `top_k`.

    A model answering from a chunk the trace never reported makes a claim
    nobody can check against the sources the run recorded.
    """
    strategy = LightRAGStrategy(neo4j_driver=object())
    strategy._local_retrieval = AsyncMock(return_value=_local_records())
    strategy._global_retrieval = AsyncMock(return_value=_global_records())
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(
        return_value=LLMResponse(text="answer", input_tokens=10, output_tokens=5, cost_usd=0.001)
    )

    result = await strategy.query("What is aws.lambda?", top_k=1)

    reported = [chunk.content for chunk in result.retrieval.retrieved]
    context = strategy._llm.generate.call_args.kwargs["context"]

    assert len(reported) == 1
    assert reported[0] in context
    strategy_all = LightRAGStrategy(neo4j_driver=object())
    strategy_all._local_retrieval = AsyncMock(return_value=_local_records())
    strategy_all._global_retrieval = AsyncMock(return_value=_global_records())
    strategy_all._llm = AsyncMock()
    strategy_all._llm.generate = AsyncMock(
        return_value=LLMResponse(text="answer", input_tokens=10, output_tokens=5, cost_usd=0.001)
    )
    everything = await strategy_all.query("What is aws.lambda?", top_k=10)
    all_chunks = [chunk.content for chunk in everything.retrieval.retrieved]

    dropped = [text for text in all_chunks if text not in reported]
    assert dropped, "the fixture must hold more chunks than top_k, or this proves nothing"
    for text in dropped:
        assert text not in context
