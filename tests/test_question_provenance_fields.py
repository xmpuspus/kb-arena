"""The four review-provenance fields on a question survive the model, not just the YAML."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kb_arena.benchmark.questions import load_questions
from kb_arena.models.benchmark import Question
from kb_arena.settings import settings

ROOT = Path(__file__).resolve().parents[1]
NIST_DIRECT = ROOT / "datasets" / "nist-800-171-r3" / "questions" / "direct.yaml"


def test_the_real_nist_corpus_keeps_its_provenance_on_the_model(monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(ROOT / "datasets"))

    questions = {q.id: q for q in load_questions("nist-800-171-r3")}
    first = questions["nist-direct-001"]

    assert first.review_status == "machine-assisted-draft"
    assert first.reviewed_by == "Codex draft pass"
    assert first.rationale == "Maps the requirement's named activity to its control."
    assert first.source_anchors == ["sec-sec_03.01.08"]
    # Every one of the 80 drafts carries the label, so nothing downstream can
    # present a machine draft as a reviewed question.
    assert all(q.review_status == "machine-assisted-draft" for q in questions.values())
    assert len(questions) == 80


def test_the_four_fields_round_trip_through_dump_and_load():
    raw = yaml.safe_load(NIST_DIRECT.read_text())[0]
    loaded = Question.model_validate(raw)

    dumped = yaml.safe_dump(loaded.model_dump(), sort_keys=False)
    reloaded = Question.model_validate(yaml.safe_load(dumped))

    for field in ("review_status", "reviewed_by", "rationale", "source_anchors"):
        assert getattr(reloaded, field) == raw[field], field


def test_a_question_without_provenance_loads_with_honest_defaults():
    raw = {
        "id": "aws-001",
        "tier": 1,
        "type": "factoid",
        "hops": 1,
        "question": "What does Lambda charge for?",
        "ground_truth": {"answer": "Requests and compute time."},
    }

    q = Question.model_validate(raw)

    assert q.review_status == "unspecified"
    assert q.reviewed_by == ""
    assert q.rationale == ""
    assert q.source_anchors == []


def test_an_unknown_review_status_is_rejected():
    raw = yaml.safe_load(NIST_DIRECT.read_text())[0]
    raw["review_status"] = "reviewed"

    with pytest.raises(ValidationError, match="review_status"):
        Question.model_validate(raw)
