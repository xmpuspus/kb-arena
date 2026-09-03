"""A question's review label rides on the benchmark record, not just the loader."""

from __future__ import annotations

import asyncio

import pytest

from kb_arena.benchmark import runner
from kb_arena.models.benchmark import AnswerRecord, Constraints, GroundTruth, Score
from kb_arena.strategies.base import AnswerResult


class _WorkingStrategy:
    name = "working"

    async def query(self, question, top_k):
        return AnswerResult(answer="answer", strategy=self.name, cost_usd=0.0, tokens_used=1)


@pytest.mark.asyncio
async def test_run_one_stamps_the_review_label_on_the_record(monkeypatch):
    async def fake_evaluate(*args, **kwargs):
        return Score(accuracy=1.0)

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)

    record = await runner._run_one(
        _WorkingStrategy(),
        "nist-direct-001",
        "question",
        GroundTruth(answer="answer"),
        Constraints(),
        [],
        object(),
        asyncio.Semaphore(1),
        review_status="machine-assisted-draft",
        reviewed_by="Codex draft pass",
    )

    assert record.question_review_status == "machine-assisted-draft"
    assert record.question_reviewed_by == "Codex draft pass"
    # Both survive serialisation, which is what a results file is.
    dumped = record.model_dump()
    assert dumped["question_review_status"] == "machine-assisted-draft"
    assert dumped["question_reviewed_by"] == "Codex draft pass"


def test_a_record_built_without_a_label_reads_unspecified():
    record = AnswerRecord(question_id="q", strategy="s", answer="a", score=Score(accuracy=1.0))

    assert record.question_review_status == "unspecified"
