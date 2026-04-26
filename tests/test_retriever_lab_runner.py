"""Synthetic integration test — runner attaches IR metrics, reporter emits them."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.benchmark.reporter import _build_markdown
from kb_arena.models.benchmark import (
    AnswerRecord,
    BenchmarkResult,
    GroundTruth,
    Question,
    Score,
)
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult


def _question(qid: str, expected: list[str]) -> Question:
    return Question(
        id=qid,
        tier=1,
        type="factoid",
        hops=1,
        question=f"q for {qid}",
        ground_truth=GroundTruth(answer="a", source_refs=expected),
    )


def _retrieved(chunk_ids: list[str]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=cid,
            doc_id=cid.split("::", 1)[0],
            content="x",
            score=1.0,
            rank=i + 1,
            source_strategy="fake",
        )
        for i, cid in enumerate(chunk_ids)
    ]


def test_compute_all_doc_level_match_for_aws_style_refs():
    """ground_truth.source_refs uses doc paths; doc-level fallback should match."""
    retrieved = _retrieved(["lambda/welcome::s1::0", "ec2/instance::s1::0"])
    # Simulate runner contract: expected_chunks empty, expected_doc_ids from source_refs
    metrics = compute_all(
        retrieved=retrieved,
        expected_ids=set(),
        k=5,
        expected_doc_ids={"lambda/welcome", "s3/welcome"},
    )
    assert metrics.fallback_doc_level is True
    # Wait — chunk_id has doc path embedded but doc_id was set to the prefix split
    # Verify that doc1 actually matched
    assert metrics.recall_at_k == 0.5  # 1 of 2 expected docs surfaced


def test_reporter_renders_ir_table_when_metrics_populated():
    bench = BenchmarkResult(
        corpus="test",
        strategy="vec",
        records=[
            AnswerRecord(
                question_id="q1", strategy="vec", answer="a", score=Score(accuracy=0.8)
            )
        ],
        mean_recall_at_k=0.65,
        mean_precision_at_k=0.4,
        mean_hit_at_k=0.9,
        mean_mrr=0.55,
        mean_ndcg_at_k=0.62,
        ir_top_k=5,
    )
    md = _build_markdown([bench])
    assert "Retrieval Quality (top-5)" in md
    assert "Recall@5" in md
    assert "65.0%" in md
    assert "0.550" in md  # MRR
    assert "0.620" in md  # NDCG


def test_reporter_skips_ir_table_when_metrics_zero():
    bench = BenchmarkResult(
        corpus="test",
        strategy="vec",
        records=[
            AnswerRecord(
                question_id="q1", strategy="vec", answer="a", score=Score(accuracy=0.8)
            )
        ],
    )
    md = _build_markdown([bench])
    assert "Retrieval Quality" not in md


@pytest.mark.asyncio
async def test_runner_attaches_retrieval_metrics(monkeypatch, tmp_path):
    """End-to-end: _run_one populates AnswerRecord.retrieval_metrics."""
    from kb_arena.benchmark.runner import _run_one

    fake_strategy = AsyncMock()
    fake_strategy.name = "fake"
    fake_strategy.query = AsyncMock(
        return_value=AnswerResult(
            answer="answer",
            sources=["doc1"],
            retrieval=RetrievalTrace(
                query="q",
                top_k=5,
                retrieved=_retrieved(["doc1::s1::0", "doc1::s1::1"]),
            ),
            strategy="fake",
        )
    )

    fake_llm = AsyncMock()
    # evaluate() uses LLM judge - mock it to avoid real call
    monkeypatch.setattr(
        "kb_arena.benchmark.runner.evaluate",
        AsyncMock(return_value=Score(accuracy=1.0, faithfulness=1.0)),
    )

    import asyncio as _aio

    sem = _aio.Semaphore(1)
    record = await _run_one(
        fake_strategy,
        question_id="q1",
        question_text="test q",
        ground_truth=GroundTruth(answer="a", source_refs=["doc1"]),
        constraints=None,
        expected_chunks=["doc1::s1::0"],
        llm=fake_llm,
        semaphore=sem,
        top_k=5,
    )

    assert record.retrieval_metrics is not None
    assert record.retrieval_metrics.k == 5
    # Expected chunk-level: ["doc1::s1::0"] — retrieved at rank 1
    assert record.retrieval_metrics.recall_at_k == 1.0
    assert record.retrieval_metrics.hit_at_k == 1
    assert record.retrieval_metrics.mrr == 1.0
