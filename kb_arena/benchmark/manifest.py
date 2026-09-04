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
    # A version 2 file wraps the labels with a record of the candidate pool.
    # The pool describes how the labels were made, so a change to
    # --n-candidates alone must not read as different ground truth.
    if isinstance(parsed, dict) and "labels" in parsed and "version" in parsed:
        return _digest(parsed["labels"])
    return _digest(parsed)


def judge_identity() -> dict[str, str]:
    """The provider and model that grade answers. The judge can sit on its own provider."""
    provider = settings.judge_provider or settings.llm_provider
    model = {
        "anthropic": settings.judge_model,
        "openai": settings.openai_judge_model,
        "ollama": settings.ollama_judge_model,
    }.get(provider, "")
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
    """Best effort. A wheel install has no repository, and that is fine.

    The whole commit, not the abbreviation. `kb-arena variance` compares this
    value for equality when it decides whether two runs came from one build,
    and an abbreviation is a prefix rather than an identity.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


def build_manifest(
    corpus: str,
    questions,
    *,
    top_k: int,
    split: str,
    reference_free: bool,
    seed_covers_whole_run: bool = True,
) -> dict:
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
        # Deliberately outside the core: two runs that differ only by seed
        # measured the same experiment, so they must group together and give
        # a spread instead of splitting into two keys of one run each.
        "seed": seed_identity(covers_whole_run=seed_covers_whole_run),
        "question_count": len(questions),
        "code_version": __version__,
        "git_sha": git_sha(),
        "compatibility_key": _digest(core_of(core)),
    }


def seed_identity(covers_whole_run: bool = True) -> dict:
    """The seed a run sets, and what that seed does and does not control.

    Nothing in KB Arena samples from a strategy today, and the judge runs at
    temperature 0, so a repeat still moves through provider-side variation
    this seed cannot reach. Saying so is the point: a reader must not read a
    captured seed as a promise of an identical run.
    """
    return {
        "value": int(settings.run_seed),
        # A resume of a checkpoint written before seeds existed inherits
        # records scored under an unknown seed. The reader is told, rather than
        # left to assume the value below covers the whole run.
        "covers_whole_run": bool(covers_whole_run),
        # Only what code in this package actually reads. A claim here that no
        # consumer honours is a record of work that never happened.
        "controls": ["optimize trial order", "bootstrap resampling"],
        "does_not_control": [
            "provider-side model sampling",
            "retrieval tie order",
            "judge output",
        ],
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
        scored = _scored_count(data)
        if scored is not None:
            # A partial run of 10 questions and one of 70 are not repeats of
            # one experiment. Neither are two runs of 10 that scored different
            # questions, so the suffix names which ones, not only how many.
            return f"{key}-partial-{scored}-{_scored_fingerprint(data)}"
        return key
    return LEGACY_KEY


def _scored_fingerprint(data: dict) -> str:
    """A short digest of which questions a partial run scored."""
    records = data.get("records")
    if not isinstance(records, list):
        return "none"
    ids = sorted(
        str(r.get("question_id") or r.get("id") or "") for r in records if isinstance(r, dict)
    )
    return _digest(ids)[:8] if ids else "none"


def _scored_count(data: dict) -> int | None:
    """How many questions a partial run scored, or None when the run is whole.

    A run the cost cap stopped scored fewer questions than its manifest names.
    It never blends with a full run, and it never blends with a partial run of
    a different size either.
    """
    manifest = data.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    records = data.get("records")
    count = len(records) if isinstance(records, list) else None
    expected = manifest.get("question_count")
    whole = isinstance(expected, int) and count is not None and count >= expected
    if whole:
        # The cap stopped the run after the last question, so nothing is
        # missing and the run compares with every other whole run.
        return None
    if data.get("stopped_by_cost_cap") is True:
        return count if count is not None else -1
    if isinstance(expected, int) and count is not None and count < expected:
        return count
    return None


def _is_partial(data: dict, manifest: dict) -> bool:
    """Kept for readers outside this module. The key uses `_scored_count`."""
    return _scored_count(data) is not None


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
