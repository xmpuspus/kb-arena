"""YAML question loader — reads benchmark questions from datasets/{corpus}/questions/."""

from __future__ import annotations

from pathlib import Path

import yaml

from kb_arena.models.benchmark import Question
from kb_arena.settings import settings


def load_questions(
    corpus: str,
    tier: int = 0,
    question_type: str = "",
) -> list[Question]:
    """Load and validate questions from YAML files for a corpus.

    Args:
        corpus: corpus name (e.g. aws-compute, my-docs)
        tier: filter to specific tier (0 = all tiers)
        question_type: filter to specific type (empty = all types)
    """
    questions_dir = Path(settings.datasets_path) / corpus / "questions"
    if not questions_dir.exists():
        raise FileNotFoundError(f"Questions directory not found: {questions_dir}")

    questions: list[Question] = []

    for yaml_file in sorted(questions_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_file.read_text())
        if not raw:
            continue
        for entry in raw:
            q = Question.model_validate(entry)
            if tier and q.tier != tier:
                continue
            if question_type and q.type != question_type:
                continue
            questions.append(q)

    return questions


def discover_corpora() -> list[str]:
    """Find all corpora that have a questions/ directory with YAML files."""
    datasets_dir = Path(settings.datasets_path)
    if not datasets_dir.exists():
        return []
    return sorted(
        d.name
        for d in datasets_dir.iterdir()
        if d.is_dir() and (d / "questions").is_dir() and list((d / "questions").glob("*.yaml"))
    )


def load_all_questions(tier: int = 0, question_type: str = "") -> list[Question]:
    """Load questions across all discovered corpora."""
    all_questions: list[Question] = []
    for corpus in discover_corpora():
        try:
            all_questions.extend(load_questions(corpus, tier=tier, question_type=question_type))
        except FileNotFoundError:
            pass
    return all_questions
