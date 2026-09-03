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
async def test_a_reranker_over_the_naive_pool_turns_the_ceiling_on(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from kb_arena.strategies.naive_vector import NaiveVectorStrategy

    strategy = _SlowStrategy()
    strategy.name = "my_plugin_reranker"
    # Every reranker in the tree keeps its base here, plugin or built in.
    strategy._base = NaiveVectorStrategy(chroma_client=MagicMock())
    _wire(monkeypatch, tmp_path, strategy, _questions(2))
    calls: list[str] = []

    async def spy_ceiling(*args, **kwargs):
        calls.append("ran")
        return {"status": "ok", "questions": 2}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", spy_ceiling)

    await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert calls == ["ran"]


@pytest.mark.asyncio
async def test_hybrid_does_not_turn_the_ceiling_on_by_name(monkeypatch, tmp_path):
    # hybrid ranks over contextual_vector, which has its own collection.
    strategy = _SlowStrategy()
    strategy.name = "hybrid"
    _wire(monkeypatch, tmp_path, strategy, _questions(2))
    calls: list[str] = []

    async def spy_ceiling(*args, **kwargs):
        calls.append("ran")
        return {"status": "ok", "questions": 2}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", spy_ceiling)

    await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert calls == []


@pytest.mark.asyncio
async def test_a_raw_error_on_one_question_keeps_every_other_row(monkeypatch, tmp_path):
    # _retrieve_only wraps backend failures, but a strategy that returns the
    # wrong type raises a raw AttributeError past it. That must cost one row,
    # not the whole strategy.
    class MostlyWorking(_SlowStrategy):
        async def query(self, question, top_k, corpus="all"):
            if question == "question 2":
                return object()
            return await super().query(question, top_k, corpus)

    strategy = MostlyWorking()
    _wire(monkeypatch, tmp_path, strategy, _questions(5))

    async def no_ceiling(*args, **kwargs):
        return {}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", no_ceiling)

    code = await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert code == 1
    report = _report(tmp_path)
    rows = report["questions"]
    assert [row["question_id"] for row in rows] == [f"q{i}" for i in range(5)]
    assert "execution_error" in rows[2]
    assert all("recall_at_k" in row for i, row in enumerate(rows) if i != 2)
    assert report["corpora"]["test"]["bm25"]["execution_errors"] == 1
