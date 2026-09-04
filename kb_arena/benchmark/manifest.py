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

import yaml

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
    """A digest of the parsed labels. A comment or a reflow in the YAML never moves it."""
    path = Path(settings.datasets_path) / corpus / "questions" / "expected_chunks.yaml"
    if not path.exists():
        return None
    try:
        parsed = yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return _digest(path.read_bytes().hex())
    return _digest(parsed)


def judge_identity() -> dict[str, str]:
    provider = settings.llm_provider
    model = {
        "anthropic": settings.judge_model,
        "openai": settings.openai_judge_model,
        "ollama": settings.ollama_judge_model,
    }.get(provider, settings.judge_model)
    return {"provider": provider, "model": model}


def generation_identity() -> dict[str, str]:
    """The provider and model that answer the questions. A different answerer is a different run."""
    provider = settings.llm_provider
    model = {
        "anthropic": settings.generate_model,
        "openai": settings.openai_generate_model,
        "ollama": settings.ollama_generate_model,
    }.get(provider, "")
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
    "generation",
    "scoring",
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
        "generation": generation_identity(),
        # RAGAS adds judge calls and changes the recorded scores and cost.
        "scoring": {
            "reference_free": reference_free,
            "ragas": bool(settings.benchmark_enable_ragas),
        },
        # A reference-free run never calls the judge, so its identity must
        # not split runs that scored the same way.
        "judge": None if reference_free else judge_identity(),
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


def judge_provider_of(manifest: dict) -> str:
    """The judge provider a manifest names, or empty when the run never judged."""
    judge = manifest.get("judge") if isinstance(manifest, dict) else None
    return str(judge.get("provider", "")) if isinstance(judge, dict) else ""


def compatibility_key(data: dict) -> str:
    """The key a stored result groups under. A file without a manifest is legacy."""
    manifest = data.get("manifest")
    if not isinstance(manifest, dict):
        return LEGACY_KEY
    # A stamped manifest names its question set. Recompute the key from the
    # core fields instead of trusting the stored one, so a stale, edited, or
    # blank key can never group runs that differ.
    fingerprint = manifest.get("question_set_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        key = _digest(core_of(manifest))
        # A run the cost cap stopped scored fewer questions than its manifest
        # names. It never blends with a full run of the same experiment.
        if _is_partial(data, manifest):
            return f"{key}-partial"
        return key
    return LEGACY_KEY


def _is_partial(data: dict, manifest: dict) -> bool:
    if data.get("stopped_by_cost_cap") is True:
        return True
    expected = manifest.get("question_count")
    records = data.get("records")
    return isinstance(expected, int) and isinstance(records, list) and len(records) < expected


def manifest_summary(data: dict) -> dict:
    """The few manifest fields a leaderboard row shows next to its numbers."""
    manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
    # A v1 file dumped through the v2 model carries an empty manifest. That
    # is still a legacy file, and a summary of nulls would say otherwise.
    fingerprint = manifest.get("question_set_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return {}
    judge = manifest.get("judge") if isinstance(manifest.get("judge"), dict) else {}
    return {
        "schema_version": manifest.get("schema_version", 1),
        "question_split": manifest.get("question_split"),
        "question_set_fingerprint": manifest.get("question_set_fingerprint"),
        "qrels_fingerprint": manifest.get("qrels_fingerprint"),
        "judge_model": judge.get("model"),
        "top_k": manifest.get("top_k"),
    }
