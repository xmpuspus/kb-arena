"""KB Arena MCP server: typed tools for corpus, strategy, and benchmark work.

Run with ``python -m kb_arena.mcp.server`` after ``pip install 'kb-arena[mcp]'``.
The server speaks stdio, so an MCP client starts it as a subprocess and talks
JSON-RPC over stdin/stdout. See docs/mcp-server.md for client setup.

Every tool here calls a plain kb_arena function, never a Typer command, so a
protocol test can mock the same function the tool calls. No tool returns an
empty result to mean "this failed" — a failure raises, so a caller can tell
"nothing here" from "could not check".
"""

from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from kb_arena.settings import settings


def _readable_errors(fn):
    """Carry a raised message to the client instead of the framework's generic text.

    The server wraps any exception that is not already a `ToolError` in a
    fixed "Error executing tool X" message and drops the original text. That
    reads as a crash. This re-raises as `ToolError(str(exc))` first, so a
    caller sees why a corpus was invalid or a file was missing.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


server = MCPServer(
    name="kb-arena",
    instructions=(
        "Compare retrieval strategies on a corpus with reproducible evidence. "
        "Start with list_corpora and list_strategies, validate a corpus before "
        "benchmarking it, and read a job's manifest before citing its numbers."
    ),
)

_CORPUS_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _corpus_dir(corpus: str) -> Path:
    """The corpus directory under the configured root, or a raised error.

    Refuses a name with a path separator or a `..` segment before it ever
    touches the filesystem, then resolves and re-checks so a symlink under
    the corpus root cannot point the caller outside it either.
    """
    if not _CORPUS_NAME_RE.match(corpus):
        raise ValueError(f"invalid corpus name: {corpus!r}")
    root = Path(settings.datasets_path).resolve()
    candidate = (root / corpus).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError(f"corpus path escapes the configured corpus root: {corpus!r}")
    return candidate


# ── list_corpora ──


@server.tool()
@_readable_errors
async def list_corpora() -> dict:
    """List every corpus under the configured datasets root, with pipeline status.

    Raises if the configured root itself is missing, so a misconfigured
    KB_ARENA_DATASETS_PATH reads as an error, not as "no corpora exist".
    """
    datasets_dir = Path(settings.datasets_path)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"configured corpus root does not exist: {datasets_dir}")

    results_dir = Path(settings.results_path)
    corpora = []
    for entry in sorted(datasets_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        processed_dir = entry / "processed"
        has_processed = processed_dir.is_dir() and any(processed_dir.glob("*.jsonl"))
        question_count = 0
        questions_dir = entry / "questions"
        if questions_dir.is_dir():
            for qfile in questions_dir.glob("*.yaml"):
                try:
                    question_count += qfile.read_text().count("- id:")
                except OSError:
                    pass
        has_results = results_dir.exists() and any(results_dir.glob(f"{entry.name}_*.json"))
        corpora.append(
            {
                "name": entry.name,
                "has_processed": has_processed,
                "question_count": question_count,
                "has_results": has_results,
            }
        )
    return {"corpora": corpora}


# ── list_strategies ──


@server.tool()
@_readable_errors
async def list_strategies() -> dict:
    """List every built-in strategy and its runtime status.

    Reads `kb_arena.strategies.catalog.STRATEGY_CATALOG` fresh on every call,
    so a strategy added to the catalog shows up here without a code change
    to this tool.
    """
    from kb_arena.strategies.catalog import STRATEGY_CATALOG, public_catalog

    # This server builds no strategy runtime of its own, so nothing is
    # "loaded" here. A strategy with a missing optional dependency still
    # reports its install hint; one with none reports "not loaded by this
    # runtime" rather than disappearing from the list.
    catalog = public_catalog(loaded_names=[])
    return {"strategies": [spec.name for spec in STRATEGY_CATALOG], "catalog": catalog}


# ── validate_corpus ──


@server.tool()
@_readable_errors
async def validate_corpus(corpus: str) -> dict:
    """Check whether a corpus exists under the configured root and is buildable.

    A corpus that does not exist is a normal, reported outcome (`valid: false`
    with a reason). An unsafe corpus name (path traversal) raises instead,
    because that is not a validation result, it is a rejected request.
    """
    corpus_path = _corpus_dir(corpus)
    if not corpus_path.is_dir():
        return {"corpus": corpus, "valid": False, "reason": "corpus directory not found"}

    processed_dir = corpus_path / "processed"
    has_processed = processed_dir.is_dir() and any(processed_dir.glob("*.jsonl"))
    questions_dir = corpus_path / "questions"
    has_questions = questions_dir.is_dir() and any(questions_dir.glob("*.yaml"))

    errors: list[str] = []
    question_count = 0
    if has_questions:
        from kb_arena.benchmark.questions import load_questions

        try:
            question_count = len(load_questions(corpus))
        except Exception as exc:  # a malformed question file, not a server bug
            errors.append(f"questions failed to load: {exc}")

    return {
        "corpus": corpus,
        "valid": has_processed and not errors,
        "has_processed": has_processed,
        "has_questions": has_questions,
        "question_count": question_count,
        "errors": errors,
    }


# ── start_benchmark / job_status ──


@dataclass
class _Job:
    job_id: str
    corpus: str
    strategy: str
    status: str  # queued | running | completed | failed
    created_at: str
    run_id: str | None = None
    error: str | None = None
    finished_at: str | None = None


_JOBS: dict[str, _Job] = {}
_TASKS: dict[str, object] = {}
_MAX_JOBS = 200


_FINISHED = ("completed", "failed")


def _trim_jobs() -> None:
    """Drop the oldest FINISHED job once the registry grows past its cap.

    A stdio server can run for a long session. Without a cap, a caller that
    starts many jobs and never reads their status would grow this dict forever.

    A running job is never dropped. The first version evicted by age alone, so
    a burst of new jobs made a running benchmark unreachable while it kept
    consuming resources and writing results. The cap bounds what the server
    remembers, and it must never bound it by forgetting live work.
    """
    while len(_JOBS) > _MAX_JOBS:
        stale = next((jid for jid, job in _JOBS.items() if job.status in _FINISHED), None)
        if stale is None:
            return
        _JOBS.pop(stale, None)
        _TASKS.pop(stale, None)


@server.tool()
@_readable_errors
async def start_benchmark(
    corpus: str = "all",
    strategy: str = "all",
    tier: int = 0,
    split: str = "",
    top_k: int = 5,
) -> dict:
    """Start a benchmark run in the background and return a job id to poll.

    A benchmark can run for minutes, so this schedules the run and returns
    right away. Poll `job_status` with the returned job id for the outcome.
    """
    import asyncio

    if corpus != "all":
        _corpus_dir(corpus)

    if strategy != "all":
        from kb_arena.strategies.catalog import STRATEGY_CATALOG

        known = {spec.name for spec in STRATEGY_CATALOG}
        for name in strategy.split(","):
            if name.strip() not in known:
                raise ValueError(f"unknown strategy: {name.strip()!r}")

    running = sum(1 for job in _JOBS.values() if job.status not in _FINISHED)
    if running >= _MAX_JOBS:
        raise ValueError(
            f"{running} benchmark jobs are already running, which is the cap. "
            f"Poll `job_status` and start another when one finishes."
        )

    job_id = uuid4().hex[:8]
    job = _Job(
        job_id=job_id,
        corpus=corpus,
        strategy=strategy,
        status="queued",
        created_at=datetime.now(UTC).isoformat(),
    )
    _JOBS[job_id] = job
    _trim_jobs()

    async def _run() -> None:
        job.status = "running"
        try:
            import contextlib
            import sys

            from kb_arena.benchmark.runner import run_benchmark

            # The runner prints its run id, its progress and its summary to
            # stdout. On a stdio server stdout carries JSON-RPC, so that text
            # lands between messages and the client fails to parse the stream.
            # Everything the runner says goes to stderr for the length of the
            # run, where a client reads it as a log.
            with contextlib.redirect_stdout(sys.stderr):
                job.run_id = await run_benchmark(
                    corpus=corpus,
                    strategy=strategy,
                    tier=tier,
                    split=split,
                    top_k=top_k,
                )
            job.status = "completed"
        except Exception as exc:
            # The job registry is the only place this failure is visible, so
            # it is recorded rather than left to an unretrieved task log.
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = datetime.now(UTC).isoformat()

    _TASKS[job_id] = asyncio.create_task(_run())
    return {"job_id": job_id, "status": job.status}


@server.tool()
@_readable_errors
async def job_status(job_id: str) -> dict:
    """Current state of a job started by `start_benchmark`.

    Raises for an unknown job id, rather than returning an empty or default
    status that would read as a real, if uneventful, job.
    """
    job = _JOBS.get(job_id)
    if job is None:
        raise ValueError(f"unknown job id: {job_id!r}")
    return asdict(job)


# ── compare ──


@server.tool()
@_readable_errors
async def compare(
    corpus: str,
    a: str,
    b: str,
    run_a: str = "",
    run_b: str = "",
    metric: str = "accuracy",
) -> dict:
    """Pair two strategies question by question on the same corpus. Delta is b minus a."""
    from kb_arena.benchmark.compare import compare_result_files, resolve_result_path

    results_dir = Path(settings.results_path)
    path_a = resolve_result_path(results_dir, corpus, a, run_a or None)
    path_b = resolve_result_path(results_dir, corpus, b, run_b or None)
    if not path_a.exists():
        raise FileNotFoundError(f"no result file at {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"no result file at {path_b}")
    return compare_result_files(path_a, path_b, metric=metric)


# ── get_manifest ──


@server.tool()
@_readable_errors
async def get_manifest(corpus: str, strategy: str, run_id: str = "") -> dict:
    """The manifest of one result file: what it measured, so two runs can be compared honestly."""
    import json

    from kb_arena.benchmark.compare import resolve_result_path
    from kb_arena.benchmark.manifest import manifest_summary

    results_dir = Path(settings.results_path)
    path = resolve_result_path(results_dir, corpus, strategy, run_id or None)
    if not path.exists():
        raise FileNotFoundError(f"no result file at {path}")
    data = json.loads(path.read_text())
    manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
    return {
        "corpus": corpus,
        "strategy": strategy,
        "run_id": run_id or data.get("run_id"),
        "manifest": manifest,
        "summary": manifest_summary(data),
    }


# ── export_evidence ──


@server.tool()
@_readable_errors
async def export_evidence(corpus: str, run_id: str) -> dict:
    """Write the evidence bundle for a completed run, or report why it cannot be written.

    Mirrors `kb-arena evidence`: a run whose questions are not fully human
    reviewed does not get written as citable evidence, it gets its problems
    listed instead.
    """
    from kb_arena.benchmark.evidence import (
        build_bundle,
        check_bundle,
        is_bundle_result,
        write_bundle,
    )
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.benchmark.review import review_summary
    from kb_arena.benchmark.runner import RUN_ID_PATTERN
    from kb_arena.cli import _run_scope

    _corpus_dir(corpus)
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")

    root = Path.cwd()
    run_dir = Path(settings.results_path) / f"run_{run_id}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run at {run_dir}")

    results = []
    for found in sorted(run_dir.glob("*.json")):
        if not is_bundle_result(found, corpus):
            continue
        resolved = found.resolve()
        results.append(resolved.relative_to(root) if root in resolved.parents else found)

    run_set, run_split = _run_scope(results, root)
    questions = load_questions(corpus, split=run_split or "all")
    review = review_summary(questions)

    bundle = build_bundle(
        command=["kb-arena", "retriever-lab", "--corpus", corpus],
        result_paths=list(results),
        review=review,
        corpus=corpus,
        seed=settings.run_seed,
        question_set_fingerprint=run_set,
        review_question_set=question_set_fingerprint(questions),
        review_split=run_split or "all",
    )
    problems = check_bundle(bundle, root)
    if problems:
        return {"written": False, "problems": problems, "bundle": bundle}

    path = write_bundle(run_dir, bundle)
    return {"written": True, "path": str(path), "citable": bundle["citable"], "bundle": bundle}


def main() -> None:
    """Entry point for `python -m kb_arena.mcp.server`."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
