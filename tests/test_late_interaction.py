"""Tests for Strategy 12: Late Interaction, a ColBERT-style MaxSim reranker.

The MaxSim math is pure NumPy (no transformers, no torch), so these run in
core CI. `[late-interaction]` is exercised only through a mocked encoder.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.catalog import STRATEGY_CATALOG, public_catalog
from kb_arena.strategies.late_interaction import (
    LateInteractionStrategy,
    late_interaction_scores,
    maxsim,
)


def test_maxsim_averages_the_best_match_per_query_token():
    # Two query tokens: one matches doc token 0 exactly, the other matches
    # nothing well. MaxSim is the average of each query token's best match.
    query_tokens = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc_tokens = np.array([[1.0, 0.0], [1.0, 0.0]])
    score = maxsim(query_tokens, doc_tokens)
    assert score == pytest.approx((1.0 + 0.0) / 2, abs=1e-9)


def test_maxsim_empty_tokens_score_zero():
    assert maxsim(np.zeros((0, 4)), np.array([[1.0, 0.0, 0.0, 0.0]])) == pytest.approx(0.0)
    assert maxsim(np.array([[1.0, 0.0, 0.0, 0.0]]), np.zeros((0, 4))) == pytest.approx(0.0)


def test_late_interaction_scores_ranks_the_closer_document_first():
    query_tokens = np.array([[1.0, 0.0]])
    docs = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    scores = late_interaction_scores(query_tokens, docs)
    assert scores[0] > scores[1]


def test_late_interaction_missing_extra_is_explicit(monkeypatch):
    import importlib.util

    from kb_arena.strategies import get_strategy

    real_find_spec = importlib.util.find_spec

    def without_transformers(name: str, *args, **kwargs):
        if name in ("transformers", "torch"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_transformers)

    with pytest.raises(ImportError, match=r"kb-arena\[late-interaction\]"):
        get_strategy("late_interaction")

    spec = next(s for s in STRATEGY_CATALOG if s.name == "late_interaction")
    assert spec.optional_extra == "late-interaction"
    row = next(r for r in public_catalog([]) if r["name"] == "late_interaction")
    assert row["status"] == "unavailable"
    assert "kb-arena[late-interaction]" in row["unavailable_reason"]


@pytest.mark.asyncio
async def test_late_interaction_query_reranks_by_maxsim(mock_chroma_client, mock_llm_client):
    strategy = LateInteractionStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)

    chunks = [
        RetrievedChunk(
            chunk_id=f"doc::sec::{i}",
            doc_id="doc",
            content=c,
            score=0.5,
            rank=i + 1,
            source_strategy="naive_vector",
        )
        for i, c in enumerate(["c0", "c1"])
    ]
    base = AnswerResult(
        answer="",
        sources=["doc"],
        retrieval=RetrievalTrace(query="Q", retrieved=chunks, latency_ms=1.0, top_k=12),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=base)

    # c1's tokens exactly match Q's token; c0's do not.
    table = {
        "Q": np.array([[1.0, 0.0]]),
        "c0": np.array([[0.0, 1.0]]),
        "c1": np.array([[1.0, 0.0]]),
    }

    class _FakeEncoder:
        def encode(self, texts):
            return [table[t] for t in texts]

    strategy._encoder = _FakeEncoder()

    result = await strategy.query("Q", top_k=2)
    assert result.strategy == "late_interaction"
    kept = result.retrieval.retrieved
    assert kept[0].content == "c1"
    assert kept[0].chunk_id == "doc::sec::1"
    assert kept[0].doc_id == "doc"
    assert kept[0].metadata["late_interaction_maxsim"] == pytest.approx(1.0, abs=1e-9)
    assert kept[0].score >= kept[1].score
