"""Tests for the retrieval-ceiling diagnostic and per-query cost efficiency.

Both are pure aggregations over already-measured numbers — no pricing or recall
assumptions are baked in, only arithmetic on real run output.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kb_arena.benchmark.reporter import cost_efficiency
from kb_arena.benchmark.retriever_lab import _summarize_ceiling

# --- Retrieval ceiling (ranking headroom) ---


def test_summarize_ceiling_headroom():
    s = _summarize_ceiling([0.2, 0.4, 0.6], [0.8, 0.9, 1.0], top_k=5, ceiling_k=20)
    assert s["recall_at_top_k"] == pytest.approx(0.4)
    assert s["recall_at_ceiling_k"] == pytest.approx(0.9)
    assert s["ranking_headroom"] == pytest.approx(0.5)  # 0.9 - 0.4
    assert s["top_k"] == 5 and s["ceiling_k"] == 20
    assert s["questions"] == 3


def test_summarize_ceiling_clamps_nonnegative():
    # ceiling recall below top-k recall can't happen for nested cutoffs, but the
    # headroom is clamped so the reported number is never negative.
    s = _summarize_ceiling([0.9], [0.5], top_k=5, ceiling_k=20)
    assert s["ranking_headroom"] == 0.0


def test_summarize_ceiling_empty():
    s = _summarize_ceiling([], [], top_k=5, ceiling_k=20)
    assert s["questions"] == 0
    assert s["recall_at_top_k"] == 0.0


# --- Cost efficiency (per query) ---


def _result(tokens, n, cost, ndcg):
    return SimpleNamespace(
        total_questions=n,
        total_cost_usd=cost,
        mean_ndcg_at_k=ndcg,
        records=[SimpleNamespace(tokens_used=t) for t in tokens],
    )


def test_cost_efficiency_per_query():
    ce = cost_efficiency(_result([100, 200], n=2, cost=0.02, ndcg=0.5))
    assert ce["tokens_per_query"] == pytest.approx(150.0)
    assert ce["cost_per_query_usd"] == pytest.approx(0.01)
    assert ce["ndcg_per_1k_tokens"] == pytest.approx(0.5 / (150 / 1000))


def test_cost_efficiency_zero_tokens_no_divzero():
    # A pure vector strategy under the retrieval-only path spends no tokens; the
    # efficiency ratio must not divide by zero.
    ce = cost_efficiency(_result([], n=1, cost=0.0, ndcg=0.3))
    assert ce["tokens_per_query"] == 0.0
    assert ce["cost_per_query_usd"] == 0.0
    assert ce["ndcg_per_1k_tokens"] == 0.0
