"""Tests for Strategy 13: Multi-Query — sub-query decomposition fused with RRF.

Mocked LLM and Chroma clients throughout; no real model or database call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import MAX_RETRIEVAL_CANDIDATES, AnswerResult
from kb_arena.strategies.multi_query import MultiQueryStrategy


def _trace_chunks(pairs):
    """pairs: list of (chunk_id, doc_id, content, rank)."""
    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=doc,
            content=content,
            score=0.5,
            rank=rank,
            source_strategy="naive_vector",
        )
        for cid, doc, content, rank in pairs
    ]


class _FakeLLM:
    """Returns one queued response per call and records what each call asked for."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def generate(self, query, context="", system_prompt="", **kwargs):
        self.calls.append({"query": query, "context": context, "system_prompt": system_prompt})
        return self._responses.pop(0)


class _FailingLLM:
    async def generate(self, *args, **kwargs):
        raise RuntimeError("provider outage")


@pytest.mark.asyncio
async def test_multi_query_retrieves_once_per_subquery_and_fuses_with_rrf(
    monkeypatch, mock_chroma_client
):
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "multi_query_n", 2)
    strategy = MultiQueryStrategy(chroma_client=mock_chroma_client)
    strategy._llm = _FakeLLM(
        [
            LLMResponse(
                text="sub one\nsub two", input_tokens=20, output_tokens=10, cost_usd=0.0004
            ),
            LLMResponse(text="final answer", input_tokens=50, output_tokens=20, cost_usd=0.0009),
        ]
    )

    result_one = AnswerResult(
        answer="discarded one",
        sources=["docA"],
        retrieval=RetrievalTrace(
            query="sub one",
            retrieved=_trace_chunks(
                [("a::1", "docA", "shared chunk", 1), ("b::1", "docB", "only in one", 2)]
            ),
            top_k=5,
        ),
        strategy="naive_vector",
        tokens_used=15,
        cost_usd=0.0002,
    )
    result_two = AnswerResult(
        answer="discarded two",
        sources=["docA"],
        retrieval=RetrievalTrace(
            query="sub two",
            retrieved=_trace_chunks([("a::1", "docA", "shared chunk", 1)]),
            top_k=5,
        ),
        strategy="naive_vector",
        tokens_used=12,
        cost_usd=0.0001,
    )

    async def fake_query(question, top_k=5, corpus="all"):
        return {"sub one": result_one, "sub two": result_two}[question]

    strategy._base.query = AsyncMock(side_effect=fake_query)

    result = await strategy.query("What sets the connection timeout?", top_k=5)

    assert strategy._base.query.await_count == 2
    # The chunk both sub-queries surfaced ranks ahead of the one only one found.
    assert [c.chunk_id for c in result.retrieval.retrieved][0] == "a::1"
    assert {c.chunk_id for c in result.retrieval.retrieved} == {"a::1", "b::1"}
    assert result.strategy == "multi_query"
    assert result.answer == "final answer"
    assert result.retrieval.query == "What sets the connection timeout?"
    assert result.sources == ["docA", "docB"]
    # rewrite (30) + sub one (15) + sub two (12) + final generation (70)
    assert result.tokens_used == 30 + 15 + 12 + 70
    assert result.cost_usd == pytest.approx(0.0004 + 0.0002 + 0.0001 + 0.0009)


@pytest.mark.asyncio
async def test_multi_query_rejects_top_k_above_the_candidate_ceiling(mock_chroma_client):
    strategy = MultiQueryStrategy(chroma_client=mock_chroma_client)

    with pytest.raises(ValueError, match="top_k must be between"):
        await strategy.query("Q", top_k=MAX_RETRIEVAL_CANDIDATES + 1)


@pytest.mark.asyncio
async def test_multi_query_raises_when_the_decomposition_call_fails(mock_chroma_client):
    strategy = MultiQueryStrategy(chroma_client=mock_chroma_client)
    strategy._llm = _FailingLLM()

    with pytest.raises(RuntimeError, match="provider outage"):
        await strategy.query("What sets the connection timeout?")


@pytest.mark.asyncio
async def test_multi_query_raises_when_no_subquery_survives_filtering(mock_chroma_client):
    strategy = MultiQueryStrategy(chroma_client=mock_chroma_client)
    # Every line is 3 characters or fewer, so none survives the length filter.
    strategy._llm = _FakeLLM([LLMResponse(text="a\nb\n", input_tokens=5, output_tokens=1)])

    with pytest.raises(ValueError, match="no usable sub-queries"):
        await strategy.query("What sets the connection timeout?")


@pytest.mark.asyncio
async def test_multi_query_build_index_delegates_to_naive_vector(
    mock_chroma_client, sample_documents
):
    strategy = MultiQueryStrategy(chroma_client=mock_chroma_client)
    strategy._base.build_index = AsyncMock()

    await strategy.build_index(sample_documents)

    strategy._base.build_index.assert_awaited_once_with(sample_documents)


def test_the_sub_query_count_has_a_ceiling(monkeypatch):
    """The setting IS the fan-out: one retrieval and one generation per sub-query.

    Only a floor was enforced, so `KB_ARENA_MULTI_QUERY_N=10000` bought ten
    thousand concurrent generations from one question.
    """
    from kb_arena.settings import settings
    from kb_arena.strategies.multi_query import MAX_SUB_QUERIES

    monkeypatch.setattr(settings, "multi_query_n", 10000)

    assert min(max(int(settings.multi_query_n), 1), MAX_SUB_QUERIES) == MAX_SUB_QUERIES
