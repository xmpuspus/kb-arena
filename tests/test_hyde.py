"""Tests for Strategy 12: HyDE — Hypothetical Document Embeddings.

Mocked LLM and Chroma clients throughout; no real model or database call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import MAX_RETRIEVAL_CANDIDATES, AnswerResult
from kb_arena.strategies.hyde import HYDE_SYSTEM_PROMPT, HydeStrategy


def _trace_chunks(contents):
    return [
        RetrievedChunk(
            chunk_id=f"doc::sec::{i}",
            doc_id="doc",
            content=c,
            score=0.5,
            rank=i + 1,
            source_strategy="naive_vector",
        )
        for i, c in enumerate(contents)
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
async def test_hyde_retrieves_with_the_hypothetical_answer_not_the_question(mock_chroma_client):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)
    chunks = _trace_chunks(["c0", "c1"])
    base_result = AnswerResult(
        answer="discarded",
        sources=["doc"],
        retrieval=RetrievalTrace(query="a plausible answer", retrieved=chunks, top_k=5),
        strategy="naive_vector",
        tokens_used=40,
        cost_usd=0.001,
    )
    strategy._base.query = AsyncMock(return_value=base_result)
    strategy._llm = _FakeLLM(
        [
            LLMResponse(
                text="a plausible answer", input_tokens=20, output_tokens=10, cost_usd=0.0004
            ),
            LLMResponse(text="final answer", input_tokens=50, output_tokens=20, cost_usd=0.0009),
        ]
    )

    result = await strategy.query("What is the timeout default?", top_k=5)

    # The rewrite call asks the LLM about the real question ...
    assert strategy._llm.calls[0]["query"] == "What is the timeout default?"
    assert strategy._llm.calls[0]["system_prompt"] == HYDE_SYSTEM_PROMPT
    # ... but naive_vector is queried with the rewrite's output, not the question.
    strategy._base.query.assert_awaited_once_with("a plausible answer", top_k=5, corpus="all")
    # The final generation answers the real question again, over the retrieved context.
    assert strategy._llm.calls[1]["query"] == "What is the timeout default?"
    assert strategy._llm.calls[1]["context"] == "c0\n\n---\n\nc1"

    assert result.strategy == "hyde"
    assert result.answer == "final answer"
    assert result.retrieval.query == "What is the timeout default?"
    assert [c.content for c in result.retrieval.retrieved] == ["c0", "c1"]
    assert result.sources == ["doc"]
    # rewrite (30) + naive_vector's own discarded call (40) + final generation (70)
    assert result.tokens_used == 30 + 40 + 70
    assert result.cost_usd == pytest.approx(0.0004 + 0.001 + 0.0009)


@pytest.mark.asyncio
async def test_hyde_query_empty_candidates_passes_through(mock_chroma_client):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)
    empty = AnswerResult(
        answer="",
        retrieval=RetrievalTrace(query="a plausible answer", retrieved=[], top_k=5),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=empty)
    strategy._llm = _FakeLLM(
        [
            LLMResponse(text="a plausible answer", input_tokens=10, output_tokens=5),
            LLMResponse(text="final answer", input_tokens=10, output_tokens=5),
        ]
    )

    result = await strategy.query("What is the timeout default?", top_k=5)

    assert result.retrieval.retrieved == []
    assert result.sources == []
    assert result.answer == "final answer"


@pytest.mark.asyncio
async def test_hyde_rejects_top_k_above_the_candidate_ceiling(mock_chroma_client):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)

    with pytest.raises(ValueError, match="top_k must be between"):
        await strategy.query("Q", top_k=MAX_RETRIEVAL_CANDIDATES + 1)


@pytest.mark.asyncio
async def test_hyde_raises_when_the_rewrite_call_fails(mock_chroma_client):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)
    strategy._llm = _FailingLLM()

    with pytest.raises(RuntimeError, match="provider outage"):
        await strategy.query("What is the timeout default?")


@pytest.mark.asyncio
async def test_hyde_raises_on_an_empty_rewrite(mock_chroma_client):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)
    strategy._llm = _FakeLLM([LLMResponse(text="   ", input_tokens=5, output_tokens=0)])

    with pytest.raises(ValueError, match="empty hypothetical document"):
        await strategy.query("What is the timeout default?")


@pytest.mark.asyncio
async def test_hyde_build_index_delegates_to_naive_vector(mock_chroma_client, sample_documents):
    strategy = HydeStrategy(chroma_client=mock_chroma_client)
    strategy._base.build_index = AsyncMock()

    await strategy.build_index(sample_documents)

    strategy._base.build_index.assert_awaited_once_with(sample_documents)
