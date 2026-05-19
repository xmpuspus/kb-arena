"""Optimizer orchestration — run_optimize and the dry-run planner.

Execution seams (document load, question load, per-trial scoring) are
monkeypatched so the loop is exercised with zero ChromaDB/Neo4j/network I/O.
"""

from __future__ import annotations

import json

import pytest

from kb_arena.benchmark import optimizer as opt


class _Q:
    def __init__(self, qid):
        self.id = qid
        self.question = f"q-{qid}"


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(opt, "load_documents", lambda corpus: ["doc"])
    monkeypatch.setattr(opt, "load_questions", lambda corpus: [_Q(1), _Q(2)])

    # Score: top_k=10 is best for naive_vector, baseline (top_k=5) mid, top_k=3 worst.
    async def fake_score(strategy, cfg, documents, questions, metric, prev_cfg):
        table = {3: 0.20, 5: 0.40, 10: 0.62}
        return table.get(cfg.top_k, 0.10)

    monkeypatch.setattr(opt, "_score_trial", fake_score)
    return tmp_path


@pytest.mark.asyncio
async def test_run_optimize_writes_report_with_best_and_delta(patched):
    code = await opt.run_optimize(
        "aws-compute",
        "naive_vector",
        top_ks=[3, 5, 10],
        chunk_sizes=[],
        embedding_providers=[],
        reranker_backends=[],
        metric="ndcg",
        out_dir=str(patched),
    )
    assert code == 0
    report = json.loads((patched / "optimize.json").read_text())
    nv = report["strategies"]["naive_vector"]
    assert nv["best_config"]["top_k"] == 10
    assert nv["best_score"] == 0.62
    assert nv["baseline_score"] == 0.40
    assert round(nv["delta"], 4) == 0.22
    assert nv["improved"] is True


@pytest.mark.asyncio
async def test_dry_run_plans_without_scoring(monkeypatch, tmp_path):
    monkeypatch.setattr(opt, "load_documents", lambda corpus: ["doc"])
    monkeypatch.setattr(opt, "load_questions", lambda corpus: [_Q(1)])

    called = False

    async def boom(*a, **k):
        nonlocal called
        called = True
        return 0.0

    monkeypatch.setattr(opt, "_score_trial", boom)

    plan = opt.plan_optimize(
        ["naive_vector", "bm25"],
        top_ks=[3, 5, 10],
        chunk_sizes=[256, 512],
        embedding_providers=["openai", "bge"],
        reranker_backends=[],
        method="grid",
        max_trials=0,
    )
    assert called is False
    by_strategy = {p["strategy"]: p for p in plan}
    # bm25 sweeps only top_k -> 3 trials, 1 rebuild (the first).
    assert by_strategy["bm25"]["n_trials"] == 3
    # naive_vector: 3 top_k x 2 chunk x 2 emb = 12 trials.
    assert by_strategy["naive_vector"]["n_trials"] == 12
    assert by_strategy["naive_vector"]["n_rebuilds"] >= 1


@pytest.mark.asyncio
async def test_run_optimize_dry_run_returns_zero_and_skips_scoring(patched, capsys):
    code = await opt.run_optimize(
        "aws-compute",
        "naive_vector",
        top_ks=[3, 5],
        chunk_sizes=[],
        embedding_providers=[],
        reranker_backends=[],
        dry_run=True,
        out_dir=str(patched),
    )
    assert code == 0
    assert not (patched / "optimize.json").exists()
