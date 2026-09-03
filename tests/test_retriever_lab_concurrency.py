"""The Retriever Lab runs questions concurrently. The ceiling runs only when it means something."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from kb_arena.benchmark import retriever_lab, runner
from kb_arena.models.retrieval import RetrievalTrace
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult


def _questions(n: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=f"q{i}",
            question=f"question {i}",
            tier=1,
            expected_chunks=[],
            ground_truth=SimpleNamespace(source_refs=["doc"]),
        )
        for i in range(n)
    ]


class _SlowStrategy:
    """Each search parks for a moment and records how many ran at once."""

    name = "bm25"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self._gate = threading.Lock()

    async def query(self, question, top_k, corpus="all"):
        with self._gate:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.05)
        with self._gate:
            self.in_flight -= 1
        return AnswerResult(
            answer="",
            retrieval=RetrievalTrace(query=question, retrieved=[], top_k=top_k),
            strategy=self.name,
        )


def _wire(monkeypatch, tmp_path, strategy, questions):
    monkeypatch.setattr(retriever_lab, "load_questions", lambda corpus, split="": questions)
    monkeypatch.setattr(runner, "_load_strategies", lambda strategy_filter: [strategy])
    monkeypatch.setattr(settings, "results_path", str(tmp_path))


def _report(tmp_path) -> dict:
    return json.loads(next(tmp_path.glob("run_*/retriever_lab.json")).read_text())


@pytest.mark.asyncio
async def test_questions_run_concurrently_up_to_the_benchmark_limit(monkeypatch, tmp_path):
    strategy = _SlowStrategy()
    monkeypatch.setattr(settings, "benchmark_max_concurrent", 4)
    _wire(monkeypatch, tmp_path, strategy, _questions(10))

    code = await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert code == 0
    # A sequential loop never has two searches in flight. The bound caps it.
    assert 2 <= strategy.peak <= 4


@pytest.mark.asyncio
async def test_rows_keep_question_order_when_searches_finish_out_of_order(monkeypatch, tmp_path):
    strategy = _SlowStrategy()
    monkeypatch.setattr(settings, "benchmark_max_concurrent", 5)
    _wire(monkeypatch, tmp_path, strategy, _questions(6))

    await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    rows = [row["question_id"] for row in _report(tmp_path)["questions"]]
    assert rows == [f"q{i}" for i in range(6)]


@pytest.mark.asyncio
async def test_the_ceiling_is_skipped_when_no_strategy_ranks_over_the_vector_pool(
    monkeypatch, tmp_path
):
    strategy = _SlowStrategy()  # named bm25
    _wire(monkeypatch, tmp_path, strategy, _questions(2))
    calls: list[str] = []

    async def spy_ceiling(*args, **kwargs):
        calls.append("ran")
        return {"status": "ok", "questions": 2}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", spy_ceiling)

    code = await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert code == 0
    assert calls == []
    ceiling = _report(tmp_path)["retrieval_ceiling"]["test"]
    assert ceiling["status"] == "skipped"
    assert "--ceiling-k" in ceiling["reason"]


@pytest.mark.asyncio
async def test_an_explicit_ceiling_k_forces_the_diagnostic_for_any_run(monkeypatch, tmp_path):
    strategy = _SlowStrategy()
    _wire(monkeypatch, tmp_path, strategy, _questions(2))
    calls: list[int] = []

    async def spy_ceiling(questions, top_k, ceiling_k, corpus="all"):
        calls.append(ceiling_k)
        return {"status": "ok", "top_k": top_k, "ceiling_k": ceiling_k, "questions": 2}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", spy_ceiling)

    await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0, ceiling_k=20)

    assert calls == [20]


@pytest.mark.asyncio
async def test_a_vector_strategy_in_the_run_turns_the_ceiling_on(monkeypatch, tmp_path):
    strategy = _SlowStrategy()
    strategy.name = "rerank_vector"
    _wire(monkeypatch, tmp_path, strategy, _questions(2))
    calls: list[str] = []

    async def spy_ceiling(*args, **kwargs):
        calls.append("ran")
        return {"status": "ok", "questions": 2}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", spy_ceiling)

    await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert calls == ["ran"]
