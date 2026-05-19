"""v0.8.0 optimizer statistics — bootstrap CIs, Wilcoxon, win-rate, efficiency, Pareto."""

from __future__ import annotations

import math

import pytest

from kb_arena.benchmark.optimizer import (
    OptimizeResult,
    TrialConfig,
    TrialResult,
    pareto_optimal_strategies,
    summarize_optimization,
)


def _cfg(strategy: str, top_k: int = 5) -> TrialConfig:
    return TrialConfig(
        strategy=strategy,
        top_k=top_k,
        chunk_tokens=512,
        embedding_provider="openai",
        reranker_backend="bge",
    )


def _trial(strategy: str, top_k: int, scores: list[float], latencies: list[float]) -> TrialResult:
    return TrialResult(
        cfg=_cfg(strategy, top_k=top_k),
        per_question_scores=scores,
        per_question_latency_ms=latencies,
    )


def test_trial_result_mean_helpers():
    t = _trial("x", 5, [0.4] * 10, [100.0] * 10)
    assert t.mean_score == pytest.approx(0.4)
    assert t.mean_latency_ms == pytest.approx(100.0)


def test_summarize_picks_best_and_computes_bootstrap_ci():
    base = _cfg("naive_vector", top_k=5)
    baseline = _trial("naive_vector", 5, [0.40] * 30, [50.0] * 30)
    better = _trial("naive_vector", 10, [0.55] * 30, [60.0] * 30)
    worse = _trial("naive_vector", 3, [0.30] * 30, [40.0] * 30)
    res = summarize_optimization("naive_vector", [baseline, better, worse], baseline=base)

    assert isinstance(res, OptimizeResult)
    assert res.best_config.top_k == 10
    assert res.best_score == pytest.approx(0.55)
    assert res.baseline_score == pytest.approx(0.40)

    # CI is a tight band around the mean for a constant series.
    lo, hi = res.best_score_ci
    assert 0.0 <= lo <= res.best_score <= hi <= 1.0
    assert hi - lo < 1e-6  # zero variance → degenerate CI


def test_summarize_wilcoxon_significant_when_clear_lift():
    base = _cfg("bm25")
    baseline = _trial("bm25", 5, [0.30 + 0.01 * i for i in range(30)], [10.0] * 30)
    better = _trial(
        "bm25", 10, [0.45 + 0.01 * i for i in range(30)], [10.0] * 30
    )  # uniformly +0.15
    res = summarize_optimization("bm25", [baseline, better], baseline=base)

    assert res.p_value is not None
    assert res.p_value < 0.05
    assert res.significant is True
    assert res.win_rate_vs_baseline == pytest.approx(1.0)


def test_summarize_wilcoxon_not_significant_for_noise():
    base = _cfg("naive_vector")
    # Same scores everywhere → no improvement to detect.
    flat = [0.4, 0.5, 0.3, 0.6, 0.45, 0.55, 0.35, 0.5, 0.4, 0.6]
    baseline = _trial("naive_vector", 5, flat, [10.0] * 10)
    # Identical (a no-op trial) — must not be flagged as significant.
    same = _trial("naive_vector", 10, flat, [10.0] * 10)
    res = summarize_optimization("naive_vector", [baseline, same], baseline=base)
    # Either Wilcoxon errors (all-zero diffs) and we report None, or it returns 1.0.
    assert res.significant is False


def test_summarize_win_rate_reflects_per_question_outcome():
    base = _cfg("bm25")
    baseline = _trial("bm25", 5, [0.4, 0.4, 0.4, 0.4, 0.4], [10.0] * 5)
    # New trial wins 3 of 5 (questions 0,1,2 > baseline).
    new = _trial("bm25", 10, [0.5, 0.5, 0.5, 0.3, 0.3], [10.0] * 5)
    res = summarize_optimization("bm25", [baseline, new], baseline=base)
    assert res.win_rate_vs_baseline == pytest.approx(0.6)


def test_summarize_efficiency_metric_per_ms():
    base = _cfg("contextual_vector")
    baseline = _trial("contextual_vector", 5, [0.4] * 10, [200.0] * 10)
    fast = _trial("contextual_vector", 10, [0.5] * 10, [100.0] * 10)
    res = summarize_optimization("contextual_vector", [baseline, fast], baseline=base)

    # NDCG per ms = 0.5 / 100 = 0.005, baseline 0.4 / 200 = 0.002
    assert res.best_metric_per_ms == pytest.approx(0.5 / 100.0, rel=1e-3)
    assert res.baseline_metric_per_ms == pytest.approx(0.4 / 200.0, rel=1e-3)
    assert math.isfinite(res.best_metric_per_ms)


def test_pareto_optimal_strategies_filters_dominated():
    # A point (score, latency) is dominated when another has score>=this AND latency<=this
    # with at least one strict inequality. (Higher score better; lower latency better.)
    base = _cfg("x")

    def _opt(name, score, latency_ms):
        return OptimizeResult(
            strategy=name,
            best_config=_cfg(name),
            best_score=score,
            baseline_config=base,
            baseline_score=score,
            n_trials=1,
            best_metric_per_ms=score / latency_ms,
            baseline_metric_per_ms=score / latency_ms,
        )

    results = [
        _opt("a", 0.5, 100.0),  # cheap, mid score
        _opt("b", 0.7, 200.0),  # higher score, higher latency
        _opt("c", 0.4, 300.0),  # strictly dominated by a
        _opt("d", 0.7, 200.0),  # tie with b (same score & latency) — both kept
    ]
    pareto = pareto_optimal_strategies(results)
    names = {r.strategy for r in pareto}
    assert "a" in names  # cheap mid-score: on the frontier
    assert "b" in names
    assert "d" in names  # tie-with-b stays
    assert "c" not in names  # strictly dominated by a


def test_pareto_handles_zero_latency_gracefully():
    base = _cfg("x")

    def _opt(name, score, latency_ms):
        return OptimizeResult(
            strategy=name,
            best_config=_cfg(name),
            best_score=score,
            baseline_config=base,
            baseline_score=score,
            n_trials=1,
            best_metric_per_ms=0.0,
            baseline_metric_per_ms=0.0,
        )

    # No crash even with zero latency reported.
    out = pareto_optimal_strategies([_opt("a", 0.5, 0.0), _opt("b", 0.6, 0.0)])
    assert len(out) >= 1
