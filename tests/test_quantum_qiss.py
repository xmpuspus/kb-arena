"""Tests for Strategy 10: QISS — Quantum-Inspired Semantic Similarity.

The fidelity math is pure NumPy (no qiskit, no API), so these run in core CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.quantum.qiss import (
    QISSStrategy,
    classical_mixture_scores,
    density_matrix,
    fidelity,
    fidelity_scores,
    superposition_state,
    unit,
)


def _cos(a, b):
    a, b = unit(a), unit(b)
    return float(np.dot(a, b))


# --- The contract: single-query QISS score == cos² of naive_vector similarity ---


def test_fidelity_equals_cosine_squared():
    q = np.array([1.0, 2.0, 3.0, 0.5])
    d = np.array([0.4, 1.1, 2.7, 0.0])
    assert fidelity(q, d) == pytest.approx(_cos(q, d) ** 2, abs=1e-12)


def test_fidelity_self_is_one():
    v = np.array([0.3, 0.7, 0.1, 0.6])
    assert fidelity(v, v) == pytest.approx(1.0, abs=1e-12)


def test_fidelity_antiparallel_is_one():
    # Fidelity is |⟨q|d⟩|² — sign-insensitive, so -v matches v with fidelity 1.
    v = np.array([0.3, 0.7, 0.1, 0.6])
    assert fidelity(v, -v) == pytest.approx(1.0, abs=1e-12)


def test_fidelity_orthogonal_is_zero():
    assert fidelity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-12)


def test_density_matrix_is_rank_one_unit_trace():
    rho = density_matrix([3.0, 4.0])  # unit -> [0.6, 0.8]
    assert np.trace(rho) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.matrix_rank(rho) == 1


def test_fidelity_scores_match_per_pair_and_bounds():
    q = np.array([1.0, 2.0, 3.0, 0.5])
    docs = np.array([[0.4, 1.1, 2.7, 0.0], [1.0, 2.0, 3.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
    scores = fidelity_scores(q, docs)
    assert scores.shape == (3,)
    assert scores[1] == pytest.approx(1.0, abs=1e-9)  # q vs q
    for i in range(3):
        assert scores[i] == pytest.approx(_cos(q, docs[i]) ** 2, abs=1e-9)
        assert 0.0 <= scores[i] <= 1.0


# --- Scientific contribution: superposition interference != classical mixture ---


def test_superposition_state_is_unit():
    subs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    psi = superposition_state(subs)
    assert np.linalg.norm(psi) == pytest.approx(1.0, abs=1e-12)


def test_superposition_interference_differs_from_classical():
    # Two sub-queries and a doc aligned with their sum: the coherent score must
    # exceed the diagonal-only classical mixture (constructive interference).
    subs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    doc = np.array([[1.0, 1.0, 0.0]])  # equally aligned with both sub-queries
    q_state = superposition_state(subs)
    coherent = fidelity_scores(q_state, doc)[0]
    classical = classical_mixture_scores(subs, doc)[0]
    assert coherent != pytest.approx(classical, abs=1e-6)
    assert coherent > classical  # constructive on a sum-aligned doc


# --- Query path (mocked base + embeddings, no API) ---


def _trace_chunks(contents):
    chunks = [
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
    return chunks


@pytest.mark.asyncio
async def test_qiss_query_reranks_by_fidelity(mock_chroma_client, mock_llm_client):
    from unittest.mock import AsyncMock

    strategy = QISSStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)

    contents = ["c0", "c1", "c2"]
    chunks = _trace_chunks(contents)
    base_result = AnswerResult(
        answer="",
        sources=["doc"],
        retrieval=RetrievalTrace(query="Q", retrieved=chunks, latency_ms=1.0, top_k=12),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=base_result)

    table = {
        "Q": [1.0, 0.0, 0.0, 0.0],
        "c0": [0.2, 0.9, 0.0, 0.0],  # low fidelity with Q
        "c1": [1.0, 0.0, 0.0, 0.0],  # fidelity 1 with Q -> should rank first
        "c2": [0.6, 0.6, 0.0, 0.0],  # mid fidelity
    }
    strategy._embed_fn = lambda texts: [table[t] for t in texts]

    result = await strategy.query("Q", top_k=2)
    assert isinstance(result, AnswerResult)
    assert result.strategy == "qiss"
    kept = result.retrieval.retrieved
    assert len(kept) == 2
    assert kept[0].content == "c1"  # highest fidelity
    assert kept[0].score == pytest.approx(1.0, abs=1e-6)
    assert kept[0].metadata["qiss_fidelity"] == pytest.approx(1.0, abs=1e-6)
    # Monotonic non-increasing fidelity ordering.
    assert kept[0].score >= kept[1].score


@pytest.mark.asyncio
async def test_qiss_query_empty_candidates_passes_through(mock_chroma_client, mock_llm_client):
    from unittest.mock import AsyncMock

    strategy = QISSStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)
    empty = AnswerResult(
        answer="nothing",
        retrieval=RetrievalTrace(query="Q", retrieved=[], latency_ms=0.0, top_k=5),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=empty)
    result = await strategy.query("Q", top_k=5)
    assert result.retrieval.retrieved == []
