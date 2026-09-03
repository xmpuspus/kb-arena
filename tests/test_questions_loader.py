"""Question YAML errors name the file instead of raising a raw scanner error."""

from __future__ import annotations

import pytest

from kb_arena.benchmark.questions import load_questions
from kb_arena.settings import settings


def test_malformed_question_yaml_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    questions_dir = tmp_path / "broken" / "questions"
    questions_dir.mkdir(parents=True)
    bad = questions_dir / "tier1_factoid.yaml"
    bad.write_text("- id: [unclosed\n  question: what\n")

    with pytest.raises(ValueError, match="Invalid question YAML") as exc_info:
        load_questions("broken")

    assert "tier1_factoid.yaml" in str(exc_info.value)
