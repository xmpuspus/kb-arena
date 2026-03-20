"""Tests for CSV and HTML export."""

from kb_arena.benchmark.reporter import _build_csv, _build_html
from kb_arena.models.benchmark import BenchmarkResult, LatencyStats, ReliabilityStats


def _make_result(strategy="naive_vector", accuracy=0.75):
    return BenchmarkResult(
        corpus="test",
        strategy=strategy,
        run_id="abc123",
        accuracy_by_tier={"1": accuracy, "2": accuracy},
        completeness_by_tier={"1": 0.8, "2": 0.8},
        faithfulness_by_tier={"1": 0.9, "2": 0.9},
        latency=LatencyStats(avg_ms=500, p50_ms=450, p95_ms=900, p99_ms=1200),
        reliability=ReliabilityStats(
            success_rate=0.95,
            error_rate=0.05,
            error_count=1,
            empty_count=0,
            timeout_count=0,
            empty_rate=0.0,
        ),
        total_cost_usd=0.15,
    )


def test_csv_has_header_and_rows():
    results = [_make_result("naive_vector"), _make_result("bm25", 0.6)]
    csv = _build_csv(results)
    lines = csv.strip().split("\n")
    assert len(lines) == 3  # header + 2 data rows
    assert "corpus" in lines[0]
    assert "naive_vector" in lines[1]
    assert "bm25" in lines[2]


def test_html_is_valid():
    results = [_make_result()]
    html = _build_html(results, "aws-compute")
    assert "<!DOCTYPE html>" in html
    assert "naive_vector" in html
    assert "aws-compute" in html
    assert "</table>" in html
