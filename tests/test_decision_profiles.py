"""The report ranks under a named profile, one table per experiment, never across keys."""

from __future__ import annotations

import json

import pytest

from kb_arena.benchmark import reporter
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, LatencyStats, Score
from kb_arena.settings import settings


def _result(strategy, accuracy, p50_ms, cost, manifest=None, success=1.0):
    bench = BenchmarkResult(
        corpus="c",
        strategy=strategy,
        run_id="r1",
        records=[
            AnswerRecord(
                question_id="q1", strategy=strategy, answer="a", score=Score(accuracy=accuracy)
            )
        ],
        accuracy_by_tier={1: accuracy},
        latency=LatencyStats(
            avg_ms=p50_ms, p50_ms=p50_ms, p95_ms=p50_ms, p99_ms=p50_ms, min_ms=p50_ms, max_ms=p50_ms
        ),
        total_cost_usd=cost,
        total_questions=1,
        manifest=manifest or {},
    )
    bench.reliability.success_rate = success
    return bench


def test_profiles_reorder_the_same_strategies():
    accurate_slow = _result("graph", 0.9, 6000.0, 0.05)
    fast_rough = _result("bm25", 0.6, 100.0, 0.0)

    first = [
        row["strategy"]
        for row in reporter.rank_strategies([accurate_slow, fast_rough], "accuracy-first")
    ]
    latency = [
        row["strategy"]
        for row in reporter.rank_strategies([accurate_slow, fast_rough], "latency-bound")
    ]
    cost = [
        row["strategy"]
        for row in reporter.rank_strategies([accurate_slow, fast_rough], "cost-bound")
    ]

    assert first[0] == "graph"
    assert latency[0] == "bm25"
    assert cost[0] == "bm25"


def test_every_profile_names_its_weights_and_they_sum_to_one():
    for name, weights in reporter.PROFILES.items():
        assert set(weights) == {"accuracy", "reliability", "latency", "cost"}, name
        assert sum(weights.values()) == pytest.approx(1.0), name


def test_results_from_two_experiments_never_share_a_table():
    key_a = {"question_set_fingerprint": "f1", "judge": {"model": "judge-a"}, "top_k": 5}
    key_b = {"question_set_fingerprint": "f1", "judge": {"model": "judge-b"}, "top_k": 5}
    results = [
        _result("bm25", 0.5, 100.0, 0.0, key_a),
        _result("graph", 0.9, 100.0, 0.0, key_a),
        _result("bm25", 0.99, 100.0, 0.0, key_b),  # a kinder judge
    ]

    groups = reporter.group_by_experiment(results)
    assert len(groups) == 2
    lines: list[str] = []
    reporter._add_ranking_section(lines, results, "accuracy-first")
    text = "\n".join(lines)

    assert "2 experiments in these results" in text
    assert text.count("| Rank |") == 2
    assert "judge judge-a" in text and "judge judge-b" in text
    assert "Profile `accuracy-first`" in text
    summary = reporter._build_summary(results)
    assert set(summary["rankings"]) == set(groups)
    assert summary["profiles"] == reporter.PROFILES


def test_a_legacy_file_groups_under_legacy():
    results = [_result("bm25", 0.5, 100.0, 0.0), _result("graph", 0.6, 100.0, 0.0)]
    assert list(reporter.group_by_experiment(results)) == ["legacy"]
    lines: list[str] = []
    reporter._add_ranking_section(lines, results)
    assert "| 1 | graph" in "\n".join(lines)


def test_the_report_command_takes_a_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    for bench in (_result("bm25", 0.6, 100.0, 0.0), _result("naive_vector", 0.9, 6000.0, 0.05)):
        (tmp_path / f"c_{bench.strategy}.json").write_text(bench.model_dump_json())

    reporter.generate_report(
        corpus="c", output=str(tmp_path / "report.md"), profile="latency-bound"
    )

    text = (tmp_path / "report.md").read_text()
    assert "Profile `latency-bound`" in text
    assert text.index("| 1 | bm25") < text.index("| 2 | naive_vector")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["rankings"]["legacy"]["by_profile"]["accuracy-first"][0] == "naive_vector"
    with pytest.raises(ValueError, match="unknown profile"):
        reporter.generate_report(corpus="c", output=str(tmp_path / "x.md"), profile="vibes")
