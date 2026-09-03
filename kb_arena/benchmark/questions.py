"""Load YAML benchmark questions from datasets/{corpus}/questions/."""

from __future__ import annotations

from pathlib import Path

import yaml

from kb_arena.models.benchmark import Question
from kb_arena.settings import settings

EXPECTED_CHUNKS_FILE = "expected_chunks.yaml"
QUESTION_SPLITS = frozenset({"development", "validation", "holdout", "unspecified"})


def validate_expected_chunks(raw: object, path: Path) -> dict[str, list[str]]:
    """Validate qrels without turning malformed labels into valid empty evidence."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a question-to-chunks mapping in {path}")

    validated: dict[str, list[str]] = {}
    for question_id, chunk_ids in raw.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Expected a non-empty question ID in {path}")
        if not isinstance(chunk_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id.strip() for chunk_id in chunk_ids
        ):
            raise ValueError(
                f"Expected a list of non-empty chunk IDs for {question_id!r} in {path}"
            )
        validated[question_id] = chunk_ids
    return validated


def load_questions(
    corpus: str,
    tier: int = 0,
    question_type: str = "",
    split: str = "",
) -> list[Question]:
    """Load and validate questions from YAML files for a corpus.

    Merges expected_chunks.yaml (if present) into Question.expected_chunks.

    Args:
        corpus: corpus name (e.g. aws-compute, my-docs)
        tier: filter to specific tier (0 = all tiers)
        question_type: filter to specific type (empty = all types)
        split: development, validation, holdout, unspecified, "all", or empty for all
    """
    split_filter = "" if split == "all" else split
    if split_filter and split_filter not in QUESTION_SPLITS:
        raise ValueError(f"Unknown question split {split!r}. Valid: {sorted(QUESTION_SPLITS)}")
    questions_dir = Path(settings.datasets_path) / corpus / "questions"
    if not questions_dir.exists():
        raise FileNotFoundError(f"Questions directory not found: {questions_dir}")

    expected_chunks_map: dict[str, list[str]] = {}
    expected_path = questions_dir / EXPECTED_CHUNKS_FILE
    if expected_path.exists():
        try:
            loaded = yaml.safe_load(expected_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid expected chunks YAML: {expected_path}") from exc
        expected_chunks_map = validate_expected_chunks(loaded, expected_path)

    questions: list[Question] = []

    for yaml_file in sorted(questions_dir.glob("*.yaml")):
        if yaml_file.name == EXPECTED_CHUNKS_FILE:
            continue
        try:
            raw = yaml.safe_load(yaml_file.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid question YAML: {yaml_file}") from exc
        if not raw:
            continue
        for entry in raw:
            q = Question.model_validate(entry)
            if q.id in expected_chunks_map:
                q = q.model_copy(update={"expected_chunks": expected_chunks_map[q.id]})
            if tier and q.tier != tier:
                continue
            if question_type and q.type != question_type:
                continue
            if split_filter and q.split != split_filter:
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


def load_all_questions(tier: int = 0, question_type: str = "", split: str = "") -> list[Question]:
    """Load questions across all discovered corpora."""
    all_questions: list[Question] = []
    for corpus in discover_corpora():
        try:
            all_questions.extend(
                load_questions(corpus, tier=tier, question_type=question_type, split=split)
            )
        except FileNotFoundError:
            pass
    return all_questions
