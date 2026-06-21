"""Regression + robustness tests for the swallowed-empty-trace bug class.

retriever-lab's `_retrieve_only` catches any strategy exception and returns an
empty trace (so one broken strategy doesn't abort the whole run). That swallow
hid a real bug: `rerank_vector` accessed `c.source` (no such field), crashed on
every query, and scored a silent 0 in retriever-lab. These tests lock in the fix
and the new dead-strategy guard that flags the crash signature.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.benchmark.retriever_lab import _empty_trace_failures
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult


def _candidates(n):
    return [
        RetrievedChunk(
            chunk_id=f"doc::sec::{i}",
            doc_id="doc",
            content=f"c{i}",
            score=0.5,
            rank=i + 1,
            source_strategy="naive_vector",
        )
        for i in range(n)
    ]


# --- Regression: rerank_vector must not crash on the c.source/c.doc_id path ---


@pytest.mark.asyncio
async def test_rerank_vector_returns_nonempty_trace(mock_chroma_client, mock_llm_client):
    from kb_arena.strategies.rerank_vector import RerankVectorStrategy

    strategy = RerankVectorStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)
    base = AnswerResult(
        answer="",
        sources=["doc"],
        retrieval=RetrievalTrace(query="Q", retrieved=_candidates(3), latency_ms=1.0, top_k=12),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=base)

    class _FakeReranker:
        def score(self, query, passages):
            return [float(len(passages) - i) for i in range(len(passages))]

    strategy._reranker = _FakeReranker()

    result = await strategy.query("Q", top_k=2)
    assert result.strategy == "rerank_vector"
    # The bug made this 0 (AttributeError on c.source swallowed into empty trace).
    assert len(result.retrieval.retrieved) == 2
    assert result.sources == ["doc"]  # exercises the c.doc_id source-building path


# --- Dead-strategy guard: empty trace on most questions != low recall ---


def test_empty_trace_failures_flags_crash_signature():
    by_strategy = {
        "good": {"questions": 10, "empty_retrieval": 0, "mean_recall_at_k": 0.40},
        "crashed": {"questions": 10, "empty_retrieval": 10, "mean_recall_at_k": 0.0},
        "half_empty": {"questions": 10, "empty_retrieval": 5, "mean_recall_at_k": 0.0},
        "low_recall": {"questions": 10, "empty_retrieval": 0, "mean_recall_at_k": 0.05},
    }
    failed = set(_empty_trace_failures(by_strategy))
    assert failed == {"crashed", "half_empty"}
    assert "low_recall" not in failed  # genuinely-retrieving-but-irrelevant is not a crash


def test_empty_trace_failures_ignores_zero_questions():
    assert _empty_trace_failures({"x": {"questions": 0, "empty_retrieval": 0}}) == []


def test_empty_trace_failures_threshold():
    by = {"s": {"questions": 10, "empty_retrieval": 4}}
    assert _empty_trace_failures(by, threshold=0.5) == []  # 4/10 < 50%
    assert _empty_trace_failures(by, threshold=0.4) == ["s"]  # 4/10 >= 40%
