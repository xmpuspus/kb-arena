"""Unit tests for kb_arena/benchmark/ir_metrics.py — pure functions, no I/O."""

from __future__ import annotations

import math

import pytest

from kb_arena.benchmark.ir_metrics import (
    compute_all,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from kb_arena.models.retrieval import RetrievedChunk


def _chunk(chunk_id: str, doc_id: str = "doc", rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content="",
        rank=rank,
        source_strategy="test",
    )


def test_recall_at_k_perfect():
    retrieved = ["a", "b", "c"]
    assert recall_at_k(retrieved, {"a", "b", "c"}, k=3) == 1.0


def test_recall_at_k_partial():
    retrieved = ["a", "x", "y"]
    assert recall_at_k(retrieved, {"a", "b"}, k=3) == 0.5


def test_recall_at_k_empty_expected_returns_zero():
    assert recall_at_k(["a"], set(), k=5) == 0.0


def test_precision_at_k_perfect():
    retrieved = ["a", "b", "c"]
    assert precision_at_k(retrieved, {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_zero_k():
    assert precision_at_k(["a"], {"a"}, k=0) == 0.0


def test_precision_at_k_k_larger_than_retrieved():
    retrieved = ["a", "b"]
    # 2 hits out of 2 actually retrieved (denominator = min(k, len))
    assert precision_at_k(retrieved, {"a", "b"}, k=10) == 1.0


def test_hit_at_k_present():
    assert hit_at_k(["x", "y", "a"], {"a"}, k=3) == 1


def test_hit_at_k_absent():
    assert hit_at_k(["x", "y", "z"], {"a"}, k=3) == 0


def test_hit_at_k_only_outside_top_k():
    # expected item exists at rank 5 but k=3 — miss
    assert hit_at_k(["x", "y", "z", "w", "a"], {"a"}, k=3) == 0


def test_mrr_first_position():
    assert mrr(["a", "x"], {"a"}) == 1.0


def test_mrr_third_position():
    assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)


def test_mrr_no_match():
    assert mrr(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect():
    retrieved = ["a", "b", "c"]
    rel = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert ndcg_at_k(retrieved, rel, k=3) == pytest.approx(1.0)


def test_ndcg_ranking_sensitive():
    # Same set, worse order — NDCG should drop
    rel = {"a": 1.0, "b": 1.0}
    perfect = ndcg_at_k(["a", "b", "x"], rel, k=3)
    swapped = ndcg_at_k(["x", "a", "b"], rel, k=3)
    assert perfect == pytest.approx(1.0)
    assert swapped < perfect
    # And recall is the same (both retrieve a and b within top-3)
    assert recall_at_k(["a", "b", "x"], {"a", "b"}, k=3) == recall_at_k(
        ["x", "a", "b"], {"a", "b"}, k=3
    )


def test_ndcg_idcg_zero_returns_zero():
    assert ndcg_at_k(["a"], {}, k=3) == 0.0


def test_ndcg_partial_match_known_value():
    # Single expected at rank 2 -> dcg = 1/log2(3); idcg = 1/log2(2) = 1
    val = ndcg_at_k(["x", "a"], {"a": 1.0}, k=2)
    assert val == pytest.approx(1.0 / math.log2(3))


def test_compute_all_perfect_chunk_level():
    retrieved = [_chunk(cid, "d", rank=i + 1) for i, cid in enumerate(["a", "b", "c"])]
    metrics = compute_all(retrieved, expected_ids={"a", "b", "c"}, k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 1.0
    assert metrics.hit_at_k == 1
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert metrics.expected_count == 3
    assert metrics.retrieved_count == 3
    assert set(metrics.hits) == {"a", "b", "c"}
    assert metrics.fallback_doc_level is False


def test_compute_all_doc_level_fallback():
    retrieved = [
        _chunk("c1", "doc1", rank=1),
        _chunk("c2", "doc2", rank=2),
        _chunk("c3", "doc1", rank=3),
    ]
    # No chunk-level labels; use doc-level expected
    metrics = compute_all(
        retrieved,
        expected_ids=set(),
        k=3,
        expected_doc_ids={"doc1"},
    )
    assert metrics.fallback_doc_level is True
    # doc1 appears at ranks 1 and 3 -> recall = 1/1
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_at_k == 1
    assert metrics.mrr == 1.0
    assert "doc1" in metrics.hits


def test_compute_all_empty_retrieval():
    metrics = compute_all([], expected_ids={"a"}, k=5)
    assert metrics.recall_at_k == 0.0
    assert metrics.precision_at_k == 0.0
    assert metrics.hit_at_k == 0
    assert metrics.mrr == 0.0
    assert metrics.ndcg_at_k == 0.0
    assert metrics.retrieved_count == 0


def test_compute_all_carries_k():
    retrieved = [_chunk("a", rank=1)]
    metrics = compute_all(retrieved, expected_ids={"a"}, k=10)
    assert metrics.k == 10


def test_hierarchical_match_section_label_matches_subchunk():
    """Expected section ID 'doc::sec' should match retrieved sub-chunk 'doc::sec::0'."""
    retrieved = [_chunk("doc::sec::0", rank=1), _chunk("doc::sec::1", rank=2)]
    metrics = compute_all(retrieved, expected_ids={"doc::sec"}, k=5)
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_at_k == 1
    assert metrics.mrr == 1.0
    # Both subchunks count as the same canonical hit for hits list
    assert metrics.hits == ["doc::sec"]


def test_hierarchical_match_does_not_match_unrelated_prefix():
    retrieved = [_chunk("doc::other::0", rank=1)]
    metrics = compute_all(retrieved, expected_ids={"doc::sec"}, k=5)
    assert metrics.recall_at_k == 0.0
    assert metrics.hit_at_k == 0


def test_hierarchical_match_no_partial_segment():
    """Prefix matching must respect '::' delimiters (no partial-segment match)."""
    retrieved = [_chunk("documentation::0", rank=1)]
    metrics = compute_all(retrieved, expected_ids={"doc"}, k=5)
    assert metrics.recall_at_k == 0.0


def test_strategy_namespace_prefix_stripping():
    """RAPTOR L0 chunks should match section-level expected labels."""
    retrieved = [
        _chunk("L0:lambda-overview::aws-lambda::0", rank=1),
        _chunk("L1:cluster-summary-1", rank=2),
    ]
    metrics = compute_all(
        retrieved,
        expected_ids={"lambda-overview::aws-lambda"},
        k=5,
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.hit_at_k == 1
    assert metrics.mrr == 1.0


def test_strategy_prefix_stripping_only_known_prefixes():
    """An unknown 'foo:' prefix must NOT match a section label."""
    retrieved = [_chunk("foo:lambda-overview::aws-lambda::0", rank=1)]
    metrics = compute_all(retrieved, expected_ids={"lambda-overview::aws-lambda"}, k=5)
    assert metrics.recall_at_k == 0.0


def test_hierarchical_ndcg_dedupes_per_expected():
    """Two sub-chunks of the same section should not double-count NDCG."""
    retrieved = [
        _chunk("doc::sec::0", rank=1),
        _chunk("doc::sec::1", rank=2),
        _chunk("doc::sec::2", rank=3),
    ]
    metrics = compute_all(retrieved, expected_ids={"doc::sec"}, k=3)
    # NDCG should be 1.0 — single expected matched at rank 1 with IDCG also rank 1
    assert metrics.ndcg_at_k == pytest.approx(1.0)
