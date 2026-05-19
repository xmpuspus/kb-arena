"""v0.8.0 ranking metrics — MAP, R-Precision, bpref, graded NDCG.

Worked examples chosen so the answer can be checked by hand.
"""

from __future__ import annotations

import math

import pytest

from kb_arena.benchmark.ir_metrics import (
    average_precision,
    bpref,
    compute_all,
    ndcg_at_k,
    r_precision,
)
from kb_arena.models.retrieval import RetrievedChunk


def _chunk(cid: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id="d", content="", score=1.0, rank=rank, source_strategy="t"
    )


# ── MAP / Average Precision ───────────────────────────────────────────────────


def test_average_precision_perfect_ranking_is_one():
    # All 3 relevant items in top 3 → AP = (1 + 1 + 1) / 3 = 1.0
    retrieved = ["a", "b", "c", "x", "y"]
    expected = {"a", "b", "c"}
    assert average_precision(retrieved, expected) == 1.0


def test_average_precision_handles_misses_and_partial():
    # Ranks: a@1 (rel), x@2 (irrel), b@3 (rel), y@4 (irrel), c@5 (rel)
    # AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.6667 + 0.6) / 3 = 0.7556
    retrieved = ["a", "x", "b", "y", "c"]
    expected = {"a", "b", "c"}
    assert average_precision(retrieved, expected) == pytest.approx(0.7556, rel=1e-3)


def test_average_precision_zero_when_no_relevant_retrieved():
    assert average_precision(["x", "y", "z"], {"a", "b"}) == 0.0


def test_average_precision_empty_expected_returns_zero():
    assert average_precision(["a", "b"], set()) == 0.0


def test_average_precision_strips_strategy_prefix():
    # graph:doc::sec should hierarchically match expected doc::sec.
    retrieved = ["graph:doc1::sec_a", "graph:doc1::sec_b"]
    expected = {"doc1::sec_a", "doc1::sec_b"}
    assert average_precision(retrieved, expected) == 1.0


# ── R-Precision ───────────────────────────────────────────────────────────────


def test_r_precision_equals_precision_at_r():
    # 4 relevant; R-Prec = precision at top-4 = 3/4 (one miss in top-4)
    retrieved = ["a", "b", "x", "c", "d", "y"]
    expected = {"a", "b", "c", "d"}
    assert r_precision(retrieved, expected) == pytest.approx(0.75)


def test_r_precision_zero_for_empty_expected():
    assert r_precision(["a", "b"], set()) == 0.0


def test_r_precision_truncates_to_r_even_if_more_retrieved():
    retrieved = ["a", "b", "c", "a", "b", "c"]  # R=2 → top 2 = ["a","b"] → 2/2
    expected = {"a", "b"}
    assert r_precision(retrieved, expected) == 1.0


# ── bpref ─────────────────────────────────────────────────────────────────────


def test_bpref_perfect_retrieval_is_one():
    # No non-relevant ranked above any relevant → bpref = 1.0
    retrieved = ["a", "b", "c"]
    expected_rel = {"a", "b", "c"}
    judged_nonrel = {"x", "y", "z"}  # all 3 non-rel judged
    assert bpref(retrieved, expected_rel, judged_nonrel) == 1.0


def test_bpref_penalizes_non_relevant_ranked_above_relevant():
    # ranking: x(nonrel), a(rel), y(nonrel), b(rel), c(rel)
    # |R|=3, |N|=2, min(R,N)=2
    # for a: 1 nonrel above -> 1 - 1/2 = 0.5
    # for b: 2 nonrel above -> 1 - 2/2 = 0.0
    # for c: 2 nonrel above -> 1 - 2/2 = 0.0
    # bpref = (0.5+0+0)/3 = 0.1667
    retrieved = ["x", "a", "y", "b", "c"]
    expected_rel = {"a", "b", "c"}
    judged_nonrel = {"x", "y"}
    assert bpref(retrieved, expected_rel, judged_nonrel) == pytest.approx(1 / 6, rel=1e-3)


def test_bpref_ignores_unjudged_chunks():
    # 'u' is unjudged (neither relevant nor in judged_nonrel) — must NOT count.
    # ranking: u, a, u, b -> for a:0 nonrel above; for b:0 nonrel above -> bpref=1.0
    retrieved = ["u", "a", "u", "b"]
    expected_rel = {"a", "b"}
    judged_nonrel: set[str] = set()
    assert bpref(retrieved, expected_rel, judged_nonrel) == 1.0


def test_bpref_zero_when_no_relevant_retrieved():
    assert bpref(["x", "y"], {"a"}, {"x", "y"}) == 0.0


def test_bpref_clamps_nonrelevant_above_to_denominator():
    # Pathological case found by real production run: 1 relevant, many
    # non-relevant ranked above it. Without clamping the inner term goes
    # negative (1 - 3/1 = -2). TREC's canonical bpref clamps to [0, 1].
    retrieved = ["x", "y", "z", "a"]
    expected = {"a"}
    judged_nonrel = {"x", "y", "z"}  # R=1, N=3, denom=1; n_above=3 unclamped→-2
    out = bpref(retrieved, expected, judged_nonrel)
    assert 0.0 <= out <= 1.0
    assert out == pytest.approx(0.0)


# ── Graded NDCG with exponential gain ─────────────────────────────────────────


def test_ndcg_exponential_gain_matches_linear_for_binary():
    # With binary grades {a:1, b:1, c:1} both formulas should agree at NDCG=1.0.
    retrieved = ["a", "b", "c"]
    relevance = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert ndcg_at_k(retrieved, relevance, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(retrieved, relevance, k=3, exponential_gain=True) == pytest.approx(1.0)


def test_ndcg_exponential_gain_boosts_highly_relevant():
    # Grades 1 and 3. Linear: gain ratio 3:1. Exponential 2^rel-1: 7:1.
    # Swapping their positions hurts exponential NDCG more than linear.
    retrieved_good = ["high", "low"]
    retrieved_bad = ["low", "high"]
    relevance = {"high": 3.0, "low": 1.0}
    lin_good = ndcg_at_k(retrieved_good, relevance, k=2)
    lin_bad = ndcg_at_k(retrieved_bad, relevance, k=2)
    exp_good = ndcg_at_k(retrieved_good, relevance, k=2, exponential_gain=True)
    exp_bad = ndcg_at_k(retrieved_bad, relevance, k=2, exponential_gain=True)
    # Both penalize swapping; exponential penalizes more sharply.
    assert lin_good > lin_bad
    assert exp_good > exp_bad
    assert (lin_good - lin_bad) < (exp_good - exp_bad)


def test_ndcg_exponential_gain_perfect_at_k1_when_top_item_highest_grade():
    relevance = {"x": 3.0, "y": 1.0}
    assert ndcg_at_k(["x"], relevance, k=1, exponential_gain=True) == pytest.approx(1.0)


# ── compute_all wires the new fields ──────────────────────────────────────────


def test_compute_all_populates_new_fields():
    retrieved = [_chunk("a", 1), _chunk("x", 2), _chunk("b", 3)]
    m = compute_all(retrieved=retrieved, expected_ids={"a", "b"}, k=5, judged_nonrelevant={"x"})
    # AP = (1/1 + 2/3) / 2 = 0.8333
    assert m.average_precision == pytest.approx((1.0 + 2 / 3) / 2, rel=1e-3)
    # R=2 → top-2 = [a, x] → 1/2
    assert m.r_precision == pytest.approx(0.5)
    # bpref with 1 nonrel above b, |R|=2, min(R,N)=1 → for a:1.0, for b:1-1/1=0 → 0.5
    assert m.bpref == pytest.approx(0.5)


def test_compute_all_back_compat_without_judged_nonrel():
    # Pre-v0.8 callers pass no judged_nonrel; bpref still computable using the
    # retrieved-but-unjudged items as a proxy (TREC bpref-10 style fallback).
    retrieved = [_chunk("a", 1), _chunk("x", 2), _chunk("b", 3)]
    m = compute_all(retrieved=retrieved, expected_ids={"a", "b"}, k=5)
    # bpref must be in [0,1] and finite, not NaN.
    assert 0.0 <= m.bpref <= 1.0
    assert math.isfinite(m.bpref)
