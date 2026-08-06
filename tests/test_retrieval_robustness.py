"""Regression tests for retrieval failures that previously became zero scores."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kb_arena.benchmark.retriever_lab import RetrievalExecutionError, _retrieve_only
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult


def _candidates(n):
    return [
        RetrievedChunk(
            chunk_id=f"doc::sec::{i}",
            doc_id="doc",
            content=f"c{i}",
            score=0.5,
            rank=i + 1,
            source_strategy="naive_vector",
        )
        for i in range(n)
    ]


# --- Regression: rerank_vector must not crash on the c.source/c.doc_id path ---


@pytest.mark.asyncio
async def test_rerank_vector_returns_nonempty_trace(mock_chroma_client, mock_llm_client):
    from kb_arena.strategies.rerank_vector import RerankVectorStrategy

    strategy = RerankVectorStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)
    base = AnswerResult(
        answer="",
        sources=["doc"],
        retrieval=RetrievalTrace(query="Q", retrieved=_candidates(3), latency_ms=1.0, top_k=12),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=base)

    class _FakeReranker:
        def score(self, query, passages):
            return [float(len(passages) - i) for i in range(len(passages))]

    strategy._reranker = _FakeReranker()

    result = await strategy.query("Q", top_k=2)
    assert result.strategy == "rerank_vector"
    # The bug made this 0 (AttributeError on c.source swallowed into empty trace).
    assert len(result.retrieval.retrieved) == 2
    assert result.sources == ["doc"]  # exercises the c.doc_id source-building path


# --- Retrieval failures are errors, not valid zero-score observations ---


@pytest.mark.asyncio
async def test_retrieve_only_raises_execution_error_with_original_cause():
    class BrokenStrategy:
        name = "broken"

        async def query(self, question, top_k):
            raise ConnectionError("index offline")

    with pytest.raises(RetrievalExecutionError, match="index offline") as caught:
        await _retrieve_only(BrokenStrategy(), "Where is it?", 5)

    assert isinstance(caught.value.__cause__, ConnectionError)


@pytest.mark.asyncio
async def test_retrieve_only_preserves_legitimate_empty_result():
    class EmptyStrategy:
        name = "empty"

        async def query(self, question, top_k):
            return AnswerResult(
                answer="",
                retrieval=RetrievalTrace(query=question, retrieved=[], top_k=top_k),
                strategy=self.name,
            )

    trace = await _retrieve_only(EmptyStrategy(), "No match", 5)

    assert trace.retrieved == []


@pytest.mark.asyncio
async def test_retrieve_only_rejects_missing_retrieval_trace():
    class MissingTraceStrategy:
        name = "missing"

        async def query(self, question, top_k):
            return AnswerResult(answer="answer", retrieval=None, strategy=self.name)

    with pytest.raises(RetrievalExecutionError, match="no retrieval trace"):
        await _retrieve_only(MissingTraceStrategy(), "Question", 5)


@pytest.mark.asyncio
async def test_retrieval_only_llm_mode_does_not_stub_concurrent_task(monkeypatch):
    from kb_arena.benchmark.retriever_lab import _PatchLLMClient
    from kb_arena.llm.client import LLMClient, LLMResponse

    async def live_call(self, *args, **kwargs):
        return LLMResponse(text="LIVE", input_tokens=2, output_tokens=1, cost_usd=0.01)

    monkeypatch.setattr(LLMClient, "_call", live_call)
    client = object.__new__(LLMClient)
    patched = asyncio.Event()
    release = asyncio.Event()

    async def retrieval_task():
        with _PatchLLMClient():
            stubbed = await client.generate(query="q", context="", system_prompt="test")
            patched.set()
            await release.wait()
            return stubbed

    task = asyncio.create_task(retrieval_task())
    await patched.wait()
    live = await client.generate(query="q", context="", system_prompt="test")
    release.set()
    stubbed = await task

    assert stubbed.text == ""
    assert stubbed.total_tokens == 0
    assert live.text == "LIVE"
    assert live.total_tokens == 3


@pytest.mark.asyncio
async def test_retriever_lab_records_error_and_excludes_failed_query(monkeypatch, tmp_path):
    from kb_arena.benchmark import retriever_lab, runner
    from kb_arena.settings import settings

    class BrokenStrategy:
        name = "broken"

        async def query(self, question, top_k):
            raise RuntimeError("backend unavailable")

    question = SimpleNamespace(
        id="q1",
        question="What failed?",
        tier=1,
        expected_chunks=[],
        ground_truth=SimpleNamespace(source_refs=["doc"]),
    )
    monkeypatch.setattr(retriever_lab, "load_questions", lambda corpus, split="": [question])
    monkeypatch.setattr(runner, "_load_strategies", lambda strategy_filter: [BrokenStrategy()])
    monkeypatch.setattr(settings, "results_path", str(tmp_path))

    async def no_ceiling(*args, **kwargs):
        return {}

    monkeypatch.setattr(retriever_lab, "_retrieval_ceiling", no_ceiling)

    code = await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert code == 1
    report_path = next(tmp_path.glob("run_*/retriever_lab.json"))
    report = json.loads(report_path.read_text())
    summary = report["corpora"]["test"]["broken"]
    assert summary["questions"] == 0
    assert summary["execution_errors"] == 1
    assert "recall_at_k" not in report["questions"][0]
    assert report["questions"][0]["execution_error"]["type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_retriever_lab_fails_and_records_retrieval_ceiling_error(monkeypatch, tmp_path):
    from kb_arena import strategies
    from kb_arena.benchmark import retriever_lab, runner
    from kb_arena.settings import settings

    class WorkingStrategy:
        name = "working"

        async def query(self, question, top_k):
            return AnswerResult(
                answer="",
                retrieval=RetrievalTrace(query=question, retrieved=[], top_k=top_k),
                strategy=self.name,
            )

    class BrokenCeilingStrategy:
        name = "naive_vector"

        async def query(self, question, top_k):
            raise ConnectionError("deep index unavailable")

    question = SimpleNamespace(
        id="q1",
        question="What failed?",
        tier=1,
        expected_chunks=[],
        ground_truth=SimpleNamespace(source_refs=["doc"]),
    )
    monkeypatch.setattr(retriever_lab, "load_questions", lambda corpus, split="": [question])
    monkeypatch.setattr(runner, "_load_strategies", lambda strategy_filter: [WorkingStrategy()])
    monkeypatch.setattr(strategies, "get_strategy", lambda name: BrokenCeilingStrategy())
    monkeypatch.setattr(settings, "results_path", str(tmp_path))

    code = await retriever_lab.run_retriever_lab(corpus="test", min_recall=0.0)

    assert code == 1
    report_path = next(tmp_path.glob("run_*/retriever_lab.json"))
    report = json.loads(report_path.read_text())
    ceiling = report["retrieval_ceiling"]["test"]
    assert ceiling["status"] == "error"
    assert ceiling["execution_error"]["type"] == "ConnectionError"


@pytest.mark.asyncio
async def test_retriever_lab_fails_when_no_questions_are_selected(monkeypatch, tmp_path):
    from kb_arena.benchmark import retriever_lab, runner
    from kb_arena.settings import settings

    monkeypatch.setattr(retriever_lab, "load_questions", lambda corpus, split="": [])
    monkeypatch.setattr(
        runner,
        "_load_strategies",
        lambda strategy_filter: [SimpleNamespace(name="naive_vector")],
    )
    monkeypatch.setattr(settings, "results_path", str(tmp_path))

    code = await retriever_lab.run_retriever_lab(corpus="empty", split="holdout")

    assert code == 1
    report_path = next(tmp_path.glob("run_*/retriever_lab.json"))
    report = json.loads(report_path.read_text())
    assert report["corpora"] == {}
