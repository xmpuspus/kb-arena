"""Optimizer search-space logic — pure, deterministic, no I/O.

These pin the behaviour that makes `kb-arena optimize` meaningful: only sweep
dimensions a strategy actually consumes, always include the baseline so the
reported delta is honest, and pick the genuine best.
"""

from __future__ import annotations

import pytest

from kb_arena.benchmark.optimizer import (
    TrialConfig,
    applicable_dims,
    build_trials,
    select_best,
)

BASE = TrialConfig(
    strategy="x",
    top_k=5,
    chunk_tokens=512,
    embedding_provider="openai",
    reranker_backend="bge",
)


def _base(strategy: str) -> TrialConfig:
    return BASE.model_copy(update={"strategy": strategy})


def test_applicable_dims_per_strategy():
    assert applicable_dims("bm25") == {"top_k"}
    assert applicable_dims("naive_vector") == {"top_k", "chunk_tokens", "embedding_provider"}
    assert "reranker_backend" in applicable_dims("rerank_vector")
    assert "reranker_backend" not in applicable_dims("naive_vector")
    assert "chunk_tokens" not in applicable_dims("qna_pairs")  # QnA isn't token-chunked


def test_bm25_only_sweeps_top_k():
    trials = build_trials(
        "bm25",
        top_ks=[3, 5, 10],
        chunk_sizes=[256, 512],
        embedding_providers=["openai", "bge"],
        reranker_backends=["bge", "cohere"],
        baseline=_base("bm25"),
    )
    # 3 top_k values only — other dims collapse to baseline.
    assert len(trials) == 3
    assert {t.top_k for t in trials} == {3, 5, 10}
    assert all(t.chunk_tokens == 512 for t in trials)
    assert all(t.embedding_provider == "openai" for t in trials)


def test_grid_expansion_is_cartesian_over_applicable_dims():
    trials = build_trials(
        "naive_vector",
        top_ks=[3, 5],
        chunk_sizes=[256, 512],
        embedding_providers=["openai", "bge"],
        reranker_backends=["voyage"],  # ignored for naive_vector
        baseline=_base("naive_vector"),
    )
    assert len(trials) == 2 * 2 * 2  # top_k x chunk x embedding


def test_baseline_is_always_first_trial():
    trials = build_trials(
        "naive_vector",
        top_ks=[7, 5],
        chunk_sizes=[1024, 512],
        embedding_providers=["bge", "openai"],
        reranker_backends=[],
        baseline=_base("naive_vector"),
    )
    assert trials[0] == _base("naive_vector")


def test_random_method_is_seed_deterministic_and_capped():
    kw = dict(
        top_ks=[3, 5, 10, 20],
        chunk_sizes=[128, 256, 512, 1024],
        embedding_providers=["openai", "bge", "voyage"],
        reranker_backends=[],
        baseline=_base("naive_vector"),
        method="random",
        max_trials=5,
    )
    a = build_trials("naive_vector", seed=42, **kw)
    b = build_trials("naive_vector", seed=42, **kw)
    c = build_trials("naive_vector", seed=99, **kw)
    assert a == b
    assert a != c
    assert len(a) == 5
    assert a[0] == _base("naive_vector")  # baseline still first


def test_select_best_picks_max_and_reports_delta():
    base = _base("naive_vector")
    better = base.model_copy(update={"top_k": 10})
    scored = [(base, 0.40), (better, 0.55), (base.model_copy(update={"top_k": 3}), 0.30)]
    res = select_best("naive_vector", scored, baseline=base)
    assert res.best_config == better
    assert res.best_score == 0.55
    assert res.baseline_score == 0.40
    assert round(res.delta, 4) == 0.15
    assert res.improved is True


def test_select_best_no_improvement_reports_zero_delta():
    base = _base("bm25")
    scored = [(base, 0.50), (base.model_copy(update={"top_k": 20}), 0.42)]
    res = select_best("bm25", scored, baseline=base)
    assert res.best_config == base
    assert res.delta == 0.0
    assert res.improved is False


def test_build_trials_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_trials(
            "bm25",
            top_ks=[5],
            chunk_sizes=[],
            embedding_providers=[],
            reranker_backends=[],
            baseline=_base("bm25"),
            method="genetic",
        )
