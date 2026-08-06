"""Per-tier IR aggregation in retriever_lab — does the win hold on hard queries?"""

from __future__ import annotations

import pytest

from kb_arena.benchmark.retriever_lab import _bootstrap_ci, _summarize_with_tiers
from kb_arena.models.benchmark import RetrievalMetrics


def _m(recall=0.5, ndcg=0.5, mrr=0.5):
    return RetrievalMetrics(
        k=5,
        recall_at_k=recall,
        precision_at_k=recall * 0.5,
        hit_at_k=1 if recall > 0 else 0,
        mrr=mrr,
        ndcg_at_k=ndcg,
        average_precision=recall,
        r_precision=recall,
        bpref=recall,
        expected_count=2,
        retrieved_count=5,
    )


def test_summarize_with_tiers_breaks_down_per_tier():
    rows = [
        (1, _m(recall=0.9, ndcg=0.9)),
        (1, _m(recall=0.8, ndcg=0.85)),
        (5, _m(recall=0.2, ndcg=0.2)),
        (5, _m(recall=0.1, ndcg=0.15)),
    ]
    summary = _summarize_with_tiers(rows)
    # Overall keeps the existing aggregate shape.
    assert "mean_recall_at_k" in summary
    assert summary["questions"] == 4
    # Per-tier breakdown is added.
    assert summary["by_tier"][1]["mean_recall_at_k"] > summary["by_tier"][5]["mean_recall_at_k"]
    assert summary["by_tier"][1]["questions"] == 2
    assert summary["by_tier"][5]["questions"] == 2


def test_summarize_with_tiers_handles_empty():
    summary = _summarize_with_tiers([])
    assert summary["questions"] == 0
    assert summary["by_tier"] == {}


def test_summarize_with_tiers_includes_new_v08_metrics():
    rows = [(1, _m(recall=0.5, ndcg=0.5))]
    summary = _summarize_with_tiers(rows)
    assert "mean_average_precision" in summary
    assert "mean_r_precision" in summary
    assert "mean_bpref" in summary


def test_bootstrap_runtime_failure_is_not_reported_as_zero_uncertainty(monkeypatch):
    import scipy.stats

    def fail_bootstrap(*args, **kwargs):
        raise OSError("bootstrap worker failed")

    monkeypatch.setattr(scipy.stats, "bootstrap", fail_bootstrap)

    with pytest.raises(OSError, match="bootstrap worker failed"):
        _bootstrap_ci([0.0, 1.0])
