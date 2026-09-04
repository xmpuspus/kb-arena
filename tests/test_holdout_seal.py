"""The holdout split is sealed: the optimizer refuses it, and every use is written down."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kb_arena.benchmark import holdout
from kb_arena.settings import settings


def test_uses_are_appended_and_read_back(tmp_path):
    holdout.record_holdout_use(
        tmp_path, tool="benchmark", corpus="c", run_id="r1", strategies=["b", "a"]
    )
    holdout.record_holdout_use(tmp_path, tool="optimize", corpus="d", run_id="r2", strategies=["a"])
    with open(holdout.ledger_path(tmp_path), "a") as handle:
        handle.write("{torn\n")

    uses = holdout.holdout_uses(tmp_path)

    assert [u["run_id"] for u in uses] == ["r1", "r2"]
    assert uses[0]["strategies"] == ["a", "b"]
    assert holdout.holdout_uses(tmp_path, corpus="d")[0]["tool"] == "optimize"
    assert holdout.holdout_uses(tmp_path / "none") == []


def _holdout_questions(n: int = 3):
    from kb_arena.models.benchmark import GroundTruth, Question

    return [
        Question(
            id=f"q{i}",
            tier=1,
            type="factoid",
            hops=1,
            question=f"q{i}?",
            ground_truth=GroundTruth(answer="a"),
            split="holdout",
        )
        for i in range(n)
    ]


def _optimize_harness(monkeypatch, tmp_path):
    from kb_arena.benchmark import optimizer

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    monkeypatch.setattr(
        optimizer, "load_documents", lambda corpus, strict=True: [SimpleNamespace(id="d")]
    )
    monkeypatch.setattr(optimizer, "load_questions", lambda corpus: _holdout_questions())


def test_the_optimizer_refuses_the_holdout_split_without_confirmation(tmp_path, monkeypatch):
    from kb_arena.benchmark import optimizer

    _optimize_harness(monkeypatch, tmp_path)

    code = asyncio.run(optimizer.run_optimize("c", strategies_filter="bm25", split="holdout"))

    assert code == 1
    assert holdout.holdout_uses(tmp_path) == []


def test_a_confirmed_holdout_run_passes_the_gate(tmp_path, monkeypatch):
    from kb_arena.benchmark import optimizer

    _optimize_harness(monkeypatch, tmp_path)

    def stop_before_scoring(*args, **kwargs):
        raise RuntimeError("stop before any real scoring")

    # _trials_for runs right after the gate and outside any broad except
    monkeypatch.setattr(optimizer, "_trials_for", stop_before_scoring)

    with pytest.raises(RuntimeError, match="stop before"):
        asyncio.run(
            optimizer.run_optimize(
                "c", strategies_filter="bm25", split="holdout", allow_holdout=True
            )
        )


def test_a_holdout_benchmark_run_is_written_down(tmp_path, monkeypatch):
    from kb_arena.benchmark import runner
    from kb_arena.models.benchmark import AnswerRecord, Score

    async def fake_run_one(strat, qid, *args, **kwargs):
        return AnswerRecord(
            question_id=qid, strategy=strat.name, answer="a", score=Score(accuracy=1.0)
        )

    class _Strategy:
        name = "x"

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    monkeypatch.setattr(
        runner, "load_questions", lambda corpus, tier=0, split="": _holdout_questions(1)
    )
    monkeypatch.setattr(runner, "_load_strategies", lambda names: [_Strategy()])
    monkeypatch.setattr(runner, "LLMClient", lambda: object())
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    monkeypatch.setattr(settings, "benchmark_cost_cap_usd", 0.0)

    asyncio.run(runner.run_benchmark(corpus="c", strategy="any", parallel=False, split="holdout"))
    asyncio.run(
        runner.run_benchmark(corpus="c", strategy="any", parallel=False, split="development")
    )

    uses = holdout.holdout_uses(tmp_path)
    assert len(uses) == 1
    assert uses[0]["tool"] == "benchmark"
    assert uses[0]["strategies"] == ["x"]


def test_the_cli_lists_uses(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kb_arena.cli import app

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    empty = CliRunner().invoke(app, ["holdout-uses"])
    assert empty.exit_code == 0 and "not been opened" in empty.output

    holdout.record_holdout_use(
        tmp_path, tool="optimize", corpus="c", run_id="r9", strategies=["bm25"]
    )
    listed = CliRunner().invoke(app, ["holdout-uses", "--corpus", "c"])
    assert listed.exit_code == 0 and "r9" in listed.output
    assert json.loads(holdout.ledger_path(tmp_path).read_text().splitlines()[0])["run_id"] == "r9"
