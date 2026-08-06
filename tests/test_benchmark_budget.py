"""Regression coverage for the benchmark spend boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kb_arena.models.benchmark import AnswerRecord, Constraints, GroundTruth, Question, Score


@pytest.mark.asyncio
async def test_parallel_request_stops_launching_queries_after_observed_cost_cap(
    tmp_path, monkeypatch
):
    from kb_arena.benchmark import runner

    questions = [
        Question(
            id=f"q{i}",
            tier=1,
            type="factoid",
            hops=1,
            question=f"Question {i}",
            ground_truth=GroundTruth(answer="answer"),
            constraints=Constraints(),
        )
        for i in range(10)
    ]
    calls: list[tuple[str, str]] = []

    async def fake_run_one(strategy, question_id, *args, **kwargs):
        calls.append((strategy.name, question_id))
        return AnswerRecord(
            question_id=question_id,
            strategy=strategy.name,
            answer="answer",
            score=Score(accuracy=1.0),
            cost_usd=1.0,
        )

    monkeypatch.setattr(runner, "LLMClient", lambda: object())
    monkeypatch.setattr(runner, "load_questions", lambda corpus, tier=0, split="": questions)
    monkeypatch.setattr(
        runner,
        "_load_strategies",
        lambda strategy: [SimpleNamespace(name="one"), SimpleNamespace(name="two")],
    )
    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    monkeypatch.setattr(runner.settings, "results_path", str(tmp_path))
    monkeypatch.setattr(runner.settings, "benchmark_cost_cap_usd", 1.0)

    await runner.run_benchmark("sample", parallel=True)

    assert calls == [("one", "q0")]
    saved = json.loads((tmp_path / "sample_one.json").read_text())
    assert saved["stopped_by_cost_cap"] is True
    assert saved["config_snapshot"]["cost_cap_usd"] == 1.0
