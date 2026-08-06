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

    with pytest.raises(runner.BenchmarkIncompleteError, match="Cost cap reached"):
        await runner.run_benchmark("sample", parallel=True)

    assert calls == [("one", "q0")]
    saved = json.loads((tmp_path / "sample_one.json").read_text())
    assert saved["stopped_by_cost_cap"] is True
    assert saved["config_snapshot"]["cost_cap_usd"] == 1.0


@pytest.mark.asyncio
async def test_query_failure_aborts_instead_of_becoming_zero_score(monkeypatch):
    from kb_arena.benchmark import runner

    class BrokenStrategy:
        name = "broken"

        async def query(self, question, top_k):
            raise ConnectionError("index offline")

    monkeypatch.setattr(runner.settings, "benchmark_max_retries", 0)

    with pytest.raises(runner.BenchmarkExecutionError, match="index offline"):
        await runner._run_one(
            BrokenStrategy(),
            "q1",
            "question",
            GroundTruth(answer="answer"),
            Constraints(),
            [],
            object(),
            __import__("asyncio").Semaphore(1),
        )


@pytest.mark.asyncio
async def test_mock_strategy_result_aborts_benchmark_evidence(monkeypatch):
    import asyncio

    from kb_arena.benchmark import runner
    from kb_arena.strategies.base import AnswerResult

    class MockStrategy:
        name = "knowledge_graph"

        async def query(self, question, top_k):
            return AnswerResult(answer="demo", strategy=self.name, mock=True)

    monkeypatch.setattr(runner.settings, "benchmark_max_retries", 0)

    with pytest.raises(runner.BenchmarkExecutionError, match="mock result"):
        await runner._run_one(
            MockStrategy(),
            "q1",
            "question",
            GroundTruth(answer="answer"),
            Constraints(),
            [],
            object(),
            asyncio.Semaphore(1),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cost_cap", [float("nan"), float("inf"), -1.0])
async def test_benchmark_defensively_rejects_invalid_cost_cap(tmp_path, monkeypatch, cost_cap):
    from kb_arena.benchmark import runner

    monkeypatch.setattr(runner.settings, "benchmark_cost_cap_usd", cost_cap)
    monkeypatch.setattr(runner.settings, "results_path", str(tmp_path))

    with pytest.raises(runner.BenchmarkExecutionError, match="cost cap"):
        await runner.run_benchmark("sample")


@pytest.mark.asyncio
async def test_answer_record_cost_includes_generation_and_evaluation(monkeypatch):
    import asyncio

    from kb_arena.benchmark import runner
    from kb_arena.strategies.base import AnswerResult

    class WorkingStrategy:
        name = "working"

        async def query(self, question, top_k):
            return AnswerResult(
                answer="answer",
                strategy=self.name,
                cost_usd=0.2,
                tokens_used=10,
            )

    async def fake_evaluate(*args, **kwargs):
        return Score(accuracy=1.0, evaluation_cost_usd=0.3)

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)

    record = await runner._run_one(
        WorkingStrategy(),
        "q1",
        "question",
        GroundTruth(answer="answer"),
        Constraints(),
        [],
        object(),
        asyncio.Semaphore(1),
    )

    assert record.generation_cost_usd == pytest.approx(0.2)
    assert record.evaluation_cost_usd == pytest.approx(0.3)
    assert record.cost_usd == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_benchmark_fails_when_no_strategy_initializes(tmp_path, monkeypatch):
    from kb_arena.benchmark import runner

    monkeypatch.setattr(runner, "LLMClient", lambda: object())
    monkeypatch.setattr(runner, "_load_strategies", lambda strategy: [])
    monkeypatch.setattr(runner.settings, "results_path", str(tmp_path))

    with pytest.raises(runner.BenchmarkExecutionError, match="No strategies available"):
        await runner.run_benchmark("sample")


@pytest.mark.asyncio
async def test_benchmark_fails_when_split_selects_no_questions(tmp_path, monkeypatch):
    from kb_arena.benchmark import runner

    monkeypatch.setattr(runner, "LLMClient", lambda: object())
    monkeypatch.setattr(runner, "_load_strategies", lambda strategy: [SimpleNamespace(name="one")])
    monkeypatch.setattr(runner, "load_questions", lambda corpus, tier=0, split="": [])
    monkeypatch.setattr(runner.settings, "results_path", str(tmp_path))
    monkeypatch.setattr(runner.settings, "benchmark_cost_cap_usd", 0.0)

    with pytest.raises(runner.BenchmarkExecutionError, match="No questions selected"):
        await runner.run_benchmark("sample", split="holdout")


@pytest.mark.asyncio
async def test_bounded_scheduler_does_not_start_every_coroutine_at_once():
    import asyncio

    from kb_arena.benchmark.runner import _as_completed_bounded

    started: list[int] = []
    three_started = asyncio.Event()
    release = asyncio.Event()

    async def work(index: int) -> int:
        started.append(index)
        if len(started) == 3:
            three_started.set()
        await release.wait()
        return index

    async def consume() -> list[int]:
        return [result async for result in _as_completed_bounded((work(i) for i in range(20)), 3)]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(three_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert len(started) == 3
    release.set()
    assert sorted(await task) == list(range(20))
