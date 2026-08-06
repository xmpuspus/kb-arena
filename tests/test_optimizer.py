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
    assert applicable_dims("qna_pairs") == {"top_k"}
    assert applicable_dims("raptor") == {"top_k"}


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


def test_unbounded_search_rejects_oversized_cartesian_product(monkeypatch):
    from kb_arena.benchmark import optimizer

    monkeypatch.setattr(optimizer, "MAX_UNBOUNDED_TRIALS", 3)
    with pytest.raises(ValueError, match="Search space has"):
        build_trials(
            "naive_vector",
            top_ks=[1, 2, 3, 4],
            chunk_sizes=[],
            embedding_providers=[],
            reranker_backends=[],
            baseline=_base("naive_vector"),
        )


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


@pytest.mark.parametrize("method", ["grid", "random"])
def test_max_trials_bounds_search_space_construction(method, monkeypatch):
    def product_must_not_run(*args, **kwargs):
        raise AssertionError("cartesian product was consumed past max_trials")

    monkeypatch.setattr("kb_arena.benchmark.optimizer.itertools.product", product_must_not_run)
    values = list(range(1_000))

    trials = build_trials(
        "naive_vector",
        top_ks=values,
        chunk_sizes=values,
        embedding_providers=[f"provider-{i}" for i in values],
        reranker_backends=[],
        baseline=_base("naive_vector"),
        method=method,
        max_trials=1,
    )

    assert trials == [_base("naive_vector")]


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


def test_needs_rebuild_is_relative_to_baseline_not_prev():
    """A trial whose rebuild dims match the persistent-index (baseline) config
    must not trigger a rebuild — even if it is the first trial. The legacy
    prev=None signalled 'first trial → rebuild' which re-embedded the whole
    corpus for the baseline itself."""
    from kb_arena.benchmark.optimizer import needs_rebuild

    base = TrialConfig(
        strategy="naive_vector",
        top_k=5,
        chunk_tokens=512,
        embedding_provider="openai",
        reranker_backend="bge",
    )
    same_dims_diff_topk = base.model_copy(update={"top_k": 10})
    diff_chunk = base.model_copy(update={"chunk_tokens": 256})
    diff_emb = base.model_copy(update={"embedding_provider": "bge"})

    assert needs_rebuild(base, base) is False
    assert needs_rebuild(same_dims_diff_topk, base) is False  # top_k is query-time
    assert needs_rebuild(diff_chunk, base) is True
    assert needs_rebuild(diff_emb, base) is True


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


@pytest.mark.asyncio
async def test_score_trial_stubs_llm_during_index_rebuild(monkeypatch):
    from kb_arena import strategies
    from kb_arena.benchmark.optimizer import _score_trial
    from kb_arena.llm.client import LLMClient

    build_called = False

    async def reject_live_generation(self, *args, **kwargs):
        raise AssertionError("live LLM generation attempted")

    class FakeStrategy:
        async def build_index(self, documents):
            nonlocal build_called
            build_called = True
            client = object.__new__(LLMClient)
            await client.generate(query="q", context="", system_prompt="test")

    monkeypatch.setattr(LLMClient, "_call", reject_live_generation)
    monkeypatch.setattr(strategies, "get_strategy", lambda name: FakeStrategy())

    base = _base("naive_vector")
    cfg = base.model_copy(update={"chunk_tokens": 256})
    result = await _score_trial("naive_vector", cfg, [], [], "ndcg", base)

    assert build_called is True
    assert result.per_question_scores == []


@pytest.mark.asyncio
async def test_score_trial_raises_when_index_rebuild_fails(monkeypatch):
    from kb_arena import strategies
    from kb_arena.benchmark.optimizer import OptimizationTrialError, _score_trial

    class BrokenStrategy:
        async def build_index(self, documents):
            raise ConnectionError("vector store offline")

    monkeypatch.setattr(strategies, "get_strategy", lambda name: BrokenStrategy())
    base = _base("naive_vector")
    cfg = base.model_copy(update={"chunk_tokens": 256})

    with pytest.raises(OptimizationTrialError, match="vector store offline"):
        await _score_trial("naive_vector", cfg, [], [object()], "ndcg", base)


@pytest.mark.asyncio
async def test_score_trial_passes_selected_corpus_to_retrieval(monkeypatch):
    from types import SimpleNamespace

    from kb_arena import strategies
    from kb_arena.benchmark import retriever_lab
    from kb_arena.benchmark.optimizer import _score_trial
    from kb_arena.models.retrieval import RetrievalTrace

    class FakeStrategy:
        name = "naive_vector"

    seen_corpora = []

    async def fake_retrieve(strategy, question, top_k, corpus="all"):
        seen_corpora.append(corpus)
        return RetrievalTrace(query=question, retrieved=[], top_k=top_k)

    monkeypatch.setattr(strategies, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(retriever_lab, "_retrieve_only", fake_retrieve)
    question = SimpleNamespace(id="q1", question="question", expected_chunks=[], ground_truth=None)
    base = _base("naive_vector")

    await _score_trial("naive_vector", base, [], [question], "ndcg", base, corpus="alpha")

    assert seen_corpora == ["alpha"]
