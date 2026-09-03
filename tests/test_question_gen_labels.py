"""Generated questions leave the generator labelled as machine drafts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from kb_arena.benchmark.question_gen import TIER_DEFS, _generate_tier_questions
from kb_arena.models.benchmark import Question


class _FakeLLM:
    def __init__(self, drafts: list[dict]) -> None:
        self._drafts = drafts

    async def generate(self, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(text=json.dumps(self._drafts))


def test_generated_questions_carry_the_machine_draft_label():
    drafts = [
        {"question": "What does Lambda bill for?", "answer": "Requests and duration."},
        {"question": "What is the default timeout?", "answer": "Three seconds."},
    ]

    questions = asyncio.run(
        _generate_tier_questions(_FakeLLM(drafts), 1, TIER_DEFS[1], "excerpt", "aws-compute", 2)
    )

    assert len(questions) == 2
    for raw in questions:
        assert raw["review_status"] == "machine-assisted-draft"
        assert raw["reviewed_by"] == "kb-arena generate-questions draft pass"
        loaded = Question.model_validate(raw)
        assert loaded.review_status == "machine-assisted-draft"
