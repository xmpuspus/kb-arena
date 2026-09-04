"""A sweep's winner is the best of many, so its p-value is Holm-adjusted and marked exploratory."""

from __future__ import annotations

import pytest

from kb_arena.benchmark.optimizer import (
    OptimizeResult,
    TrialConfig,
    TrialResult,
    apply_run_wide_holm,
    holm_adjust,
    strategy_report,
    summarize_optimization,
)


def test_holm_adjust_matches_the_hand_computed_step_down():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.2, None, 0.01]) == [pytest.approx(0.2), None, pytest.approx(0.02)]
    assert holm_adjust([0.9, 0.8]) == [1.0, 1.0]
    assert holm_adjust([]) == []
    assert holm_adjust([None]) == [None]


def _trial(top_k: int, scores: list[float], latency: float = 10.0) -> TrialResult:
    return TrialResult(
        cfg=TrialConfig(strategy="bm25", top_k=top_k),
        per_question_scores=scores,
        per_question_latency_ms=[latency] * len(scores),
    )


def test_a_sweep_adjusts_the_winner_and_marks_it_exploratory():
    base = [0.5] * 20
    trials = [
        _trial(5, base),
        _trial(10, [0.5 + 0.01 * (i % 3) for i in range(20)]),  # noise
        _trial(20, [0.9] * 20),  # a clear lift
    ]

    result = summarize_optimization("bm25", trials, TrialConfig(strategy="bm25", top_k=5))

    assert result.best_config.top_k == 20
    assert result.n_comparisons == 2
    assert result.correction == "holm"
    assert result.p_value_raw is not None
    assert result.p_value == pytest.approx(min(1.0, 2 * result.p_value_raw))
    assert result.significant is True
    assert result.exploratory is True
    assert result.publishable is False
    assert result.trial_p_values[0] is None


def test_one_trial_against_the_baseline_needs_no_correction():
    trials = [_trial(5, [0.5] * 20), _trial(10, [0.9] * 20)]

    result = summarize_optimization("bm25", trials, TrialConfig(strategy="bm25", top_k=5))

    assert result.n_comparisons == 1
    assert result.correction == "none"
    assert result.p_value == result.p_value_raw
    assert result.exploratory is False
    assert result.publishable is result.significant is True


def test_the_baseline_as_winner_has_no_p_value():
    trials = [_trial(5, [0.9] * 20), _trial(10, [0.5] * 20)]

    result = summarize_optimization("bm25", trials, TrialConfig(strategy="bm25", top_k=5))

    assert result.p_value is None
    assert result.significant is False
    assert result.publishable is False


def test_run_wide_holm_counts_every_test_in_the_run():
    base = TrialConfig(strategy="bm25", top_k=5)
    r1 = summarize_optimization("bm25", [_trial(5, [0.5] * 20), _trial(10, [0.9] * 20)], base)
    r2 = summarize_optimization(
        "naive", [_trial(5, [0.5] * 20), _trial(10, [0.55] * 20), _trial(20, [0.6] * 20)], base
    )
    results = {"bm25": r1, "naive": r2}

    family = apply_run_wide_holm(results)

    assert family == 3
    assert r1.n_comparisons == 3 and r2.n_comparisons == 3
    assert r1.correction == "holm"
    assert r1.p_value >= r1.p_value_raw
    assert r1.exploratory is True and r1.publishable is False
    report = strategy_report(r1)
    assert report["n_comparisons"] == 3 and report["correction"] == "holm"
    assert report["p_value_raw"] == r1.p_value_raw


def test_the_report_carries_the_flags_for_a_bare_result():
    r = OptimizeResult(
        strategy="bm25",
        best_config=TrialConfig(strategy="bm25", top_k=5),
        best_score=0.5,
        baseline_config=TrialConfig(strategy="bm25", top_k=5),
        baseline_score=0.5,
    )
    report = strategy_report(r)
    assert report["exploratory"] is False
    assert report["publishable"] is False
    assert report["correction"] == "none"
