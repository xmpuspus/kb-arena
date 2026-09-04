"""Load YAML benchmark questions from datasets/{corpus}/questions/."""

from __future__ import annotations

from pathlib import Path

import yaml

from kb_arena.models.benchmark import Question
from kb_arena.settings import settings

EXPECTED_CHUNKS_FILE = "expected_chunks.yaml"
QUESTION_SPLITS = frozenset({"development", "validation", "holdout", "unspecified"})


QRELS_VERSION = 2
GRADES = (0, 1, 2)


def load_qrels(raw: object, path: Path) -> tuple[dict[str, dict[str, int]], int]:
    """Graded labels per question, and the file version they came from.

    Version 1 is a question-to-list mapping, every listed chunk grade 1.
    Version 2 wraps a labels mapping of question to chunk-to-grade, with
    grades 0, 1, or 2. A grade 0 records a judged negative and never counts
    as expected. Malformed labels raise, so a broken file never reads as
    valid empty evidence.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a question-to-chunks mapping in {path}")
    version = 1
    labels = raw
    if "labels" in raw and isinstance(raw.get("version"), int):
        version = int(raw["version"])
        labels = raw["labels"]
        if version != QRELS_VERSION:
            raise ValueError(f"Unknown qrels version {version} in {path}")
        if not isinstance(labels, dict):
            raise ValueError(f"Expected a labels mapping in {path}")
    graded: dict[str, dict[str, int]] = {}
    for question_id, value in labels.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Expected a non-empty question ID in {path}")
        if isinstance(value, list):
            if version == QRELS_VERSION:
                # A versioned file promises grades. A list carries none, so it
                # would read as every chunk at grade 1 and lose every judged
                # negative without a word.
                raise ValueError(
                    f"Expected a grade mapping for {question_id!r} in {path}, not a list. "
                    f"The file declares version {version}."
                )
            if not all(isinstance(c, str) and c.strip() for c in value):
                raise ValueError(
                    f"Expected a list of non-empty chunk IDs for {question_id!r} in {path}"
                )
            graded[question_id] = {c: 1 for c in value}
        elif isinstance(value, dict) and version == QRELS_VERSION:
            grades: dict[str, int] = {}
            for chunk_id, grade in value.items():
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(f"Expected a non-empty chunk ID for {question_id!r} in {path}")
                if isinstance(grade, bool) or grade not in GRADES:
                    raise ValueError(
                        f"Expected a grade of 0, 1, or 2 for {chunk_id!r} under {question_id!r} "
                        f"in {path}"
                    )
                grades[chunk_id] = int(grade)
            graded[question_id] = grades
        else:
            raise ValueError(
                f"Expected a list of chunk IDs or a chunk-to-grade mapping for "
                f"{question_id!r} in {path}"
            )
    return graded, version


def validate_expected_chunks(raw: object, path: Path) -> dict[str, list[str]]:
    """The expected chunk ids per question, from either file version. Grade 0 never counts."""
    graded, _ = load_qrels(raw, path)
    return {qid: [c for c, g in grades.items() if g > 0] for qid, grades in graded.items()}


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
    grades_map: dict[str, dict[str, int]] = {}
    expected_path = questions_dir / EXPECTED_CHUNKS_FILE
    if expected_path.exists():
        try:
            loaded = yaml.safe_load(expected_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid expected chunks YAML: {expected_path}") from exc
        graded, _ = load_qrels(loaded, expected_path)
        grades_map = {
            qid: {c: g for c, g in grades.items() if g > 0} for qid, grades in graded.items()
        }
        expected_chunks_map = {qid: list(grades) for qid, grades in grades_map.items()}
        negatives_map = {
            qid: sorted(c for c, g in grades.items() if g == 0) for qid, grades in graded.items()
        }

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
                q = q.model_copy(
                    update={
                        "expected_chunks": expected_chunks_map[q.id],
                        "expected_grades": grades_map.get(q.id, {}),
                        "judged_negatives": negatives_map.get(q.id, []),
                    }
                )
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
