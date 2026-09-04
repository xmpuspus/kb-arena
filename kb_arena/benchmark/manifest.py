"""Experiment manifest: what a run measured, so two runs can be compared honestly.

A leaderboard that averages runs made against different question sets,
different qrels, different judges, or different top_k values reports a
number that measured nothing. The manifest names each of those inputs and
folds the ones that must match into one compatibility key. Runs share a
row only when their keys match.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from kb_arena import __version__
from kb_arena.settings import settings

SCHEMA_VERSION = 2
LEGACY_KEY = "legacy"  # a result file written before manifests existed


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]


def _question_record(q) -> dict:
    # The judge reads the ground-truth answer, the required entities, the
    # constraints, and the source refs. A change to any of them changes the
    # score, so the whole record goes into the fingerprint, not a few fields.
    if hasattr(q, "model_dump"):
        return q.model_dump(mode="json")
    return {k: v for k, v in vars(q).items() if not k.startswith("_")}


def question_set_fingerprint(questions) -> str:
    """Stable across file order and process runs. Changes when any field of a question changes."""
    rows = sorted((_question_record(q) for q in questions), key=lambda r: str(r.get("id")))
    return _digest(rows)


def qrels_fingerprint(corpus: str) -> str | None:
    path = Path(settings.datasets_path) / corpus / "questions" / "expected_chunks.yaml"
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def judge_identity() -> dict[str, str]:
    provider = settings.llm_provider
    model = {
        "anthropic": settings.judge_model,
        "openai": settings.openai_judge_model,
        "ollama": settings.ollama_judge_model,
    }.get(provider, settings.judge_model)
    return {"provider": provider, "model": model}


def embedding_identity() -> dict[str, str]:
    provider = settings.embedding_provider
    model = {
        "openai": settings.embedding_model,
        "ollama": settings.ollama_embedding_model,
        "bge": "BAAI/bge-large-en-v1.5",
    }.get(provider, settings.embedding_model)
    return {"provider": provider, "model": model}


def git_sha() -> str | None:
    """Best effort. A wheel install has no repository, and that is fine."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# The fields that decide whether two runs compare. The reader recomputes the
# key from these, so a stale or edited stored key can never group two runs
# that differ in one of them.
CORE_FIELDS = (
    "schema_version",
    "corpus",
    "question_split",
    "question_set_fingerprint",
    "qrels_fingerprint",
    "judge",
    "embedding",
    "chunk",
    "top_k",
)


def core_of(manifest: dict) -> dict:
    return {field: manifest.get(field) for field in CORE_FIELDS}


def build_manifest(corpus: str, questions, *, top_k: int, split: str, reference_free: bool) -> dict:
    """The record a result file carries. The compatibility key covers the core."""
    core = {
        "schema_version": SCHEMA_VERSION,
        "corpus": corpus,
        "question_split": split or "all",
        "question_set_fingerprint": question_set_fingerprint(questions),
        "qrels_fingerprint": qrels_fingerprint(corpus),
        "judge": {
            **judge_identity(),
            "reference_free": reference_free,
            # RAGAS adds judge calls and changes the recorded scores and cost.
            "ragas": bool(settings.benchmark_enable_ragas),
        },
        "embedding": embedding_identity(),
        "chunk": {"tokens": settings.chunk_tokens, "overlap_tokens": settings.chunk_overlap_tokens},
        "top_k": top_k,
    }
    return {
        **core,
        "question_count": len(questions),
        "code_version": __version__,
        "git_sha": git_sha(),
        "compatibility_key": _digest(core_of(core)),
    }


def compatibility_key(data: dict) -> str:
    """The key a stored result groups under. A file without a manifest is legacy."""
    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        return LEGACY_KEY
    # A stamped manifest names its question set. Recompute the key from the
    # core fields instead of trusting the stored one, so a stale, edited, or
    # blank key can never group runs that differ.
    if isinstance(manifest.get("question_set_fingerprint"), str):
        return _digest(core_of(manifest))
    return LEGACY_KEY


def manifest_summary(data: dict) -> dict:
    """The few manifest fields a leaderboard row shows next to its numbers."""
    if "manifest" not in data:
        return {}
    manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
    judge = manifest.get("judge") if isinstance(manifest.get("judge"), dict) else {}
    return {
        "schema_version": manifest.get("schema_version", 1),
        "question_split": manifest.get("question_split"),
        "question_set_fingerprint": manifest.get("question_set_fingerprint"),
        "qrels_fingerprint": manifest.get("qrels_fingerprint"),
        "judge_model": judge.get("model"),
        "top_k": manifest.get("top_k"),
    }
