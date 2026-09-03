"""The reporter reads every strategy from the catalog instead of a stale list."""

from __future__ import annotations

import json

from kb_arena.benchmark import reporter
from kb_arena.models.benchmark import BenchmarkResult
from kb_arena.settings import settings
from kb_arena.strategies.catalog import STRATEGY_CATALOG


def test_reporter_strategy_names_match_the_catalog():
    assert list(reporter.STRATEGY_NAMES) == [spec.name for spec in STRATEGY_CATALOG]
    for name in ("bm25", "rerank_vector", "qiss", "sqr"):
        assert name in reporter.STRATEGY_NAMES


def _write_result(results_dir, corpus: str, strategy: str) -> None:
    result = BenchmarkResult(corpus=corpus, strategy=strategy, run_id="r1")
    (results_dir / f"{corpus}_{strategy}.json").write_text(
        json.dumps(result.model_dump(mode="json"))
    )


def test_reporter_loads_results_for_every_catalog_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    for strategy in ("bm25", "rerank_vector", "qiss", "sqr", "naive_vector"):
        _write_result(tmp_path, "corp", strategy)

    loaded = reporter._load_results("corp")

    assert sorted(r.strategy for r in loaded) == sorted(
        ["bm25", "rerank_vector", "qiss", "sqr", "naive_vector"]
    )


def test_reporter_discovers_a_corpus_that_only_has_bm25(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_result(tmp_path, "lexical-only", "bm25")

    assert reporter._discover_result_corpora() == ["lexical-only"]
