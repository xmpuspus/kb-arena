"""Regression tests for the one-shot ``kb-arena run`` workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from kb_arena.cli import app
from kb_arena.settings import settings

runner = CliRunner()

MINIMAL_DOCUMENT = (
    '{"id":"doc","source":"guide.md","corpus":"sample","title":"Guide","sections":[]}\n'
)
MINIMAL_QUESTION = (
    "- id: q1\n"
    "  tier: 1\n"
    "  type: factoid\n"
    "  hops: 1\n"
    "  question: What?\n"
    "  ground_truth:\n"
    "    answer: Answer.\n"
)


def _corpus(tmp_path: Path, name: str = "sample") -> Path:
    base = tmp_path / "datasets" / name
    (base / "raw").mkdir(parents=True)
    (base / "processed").mkdir()
    (base / "questions").mkdir()
    (base / "questions" / "questions.yaml").write_text(MINIMAL_QUESTION)
    return base


def test_run_ingests_files_already_in_raw_directory(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "raw" / "guide.md").write_text("# Guide\n\nLocal documentation.\n")
    calls: list[tuple[str, str]] = []

    def fake_ingest(path: str, corpus: str, format: str) -> int:
        calls.append((path, corpus))
        (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
        return 1

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append(("vectors", corpus))

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        calls.append(("benchmark", strategy))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest", fake_ingest)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample", "--skip-graph"])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        (str(Path("datasets/sample/raw")), "sample"),
        ("vectors", "sample"),
        (
            "benchmark",
            "naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,rerank_vector,qiss",
        ),
    ]


def test_run_continues_after_graph_failure_without_checkpointing_it(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    calls: list[str] = []

    async def failing_graph(corpus: str) -> None:
        calls.append("graph")
        raise OSError("Neo4j is unavailable")

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append("vectors")

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        calls.append(f"benchmark:{strategy}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", failing_graph)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample"])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        "graph",
        "vectors",
        "benchmark:naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,rerank_vector,qiss",
    ]
    state_path = base / ".pipeline_state.json"
    state = json.loads(state_path.read_text())
    assert "build_graph" not in state
    assert state["build_vectors"] == "done"
    assert state["benchmark"] == "done"


def test_run_rejects_unsupported_raw_files_without_checkpointing_ingest(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "raw" / ".DS_Store").write_bytes(b"metadata")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)

    result = runner.invoke(app, ["run", "--corpus", "sample", "--skip-graph"])

    assert result.exit_code == 1
    assert "No documents found to process" in result.stdout
    assert not (base / ".pipeline_state.json").exists()


def test_run_rebuilds_empty_artifacts_instead_of_checkpointing_them(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "raw" / "guide.md").write_text("# Guide\n\nLocal documentation.\n")
    (base / "processed" / "documents.jsonl").write_text("")
    (base / "questions" / "questions.yaml").write_text("")
    calls: list[str] = []

    def fake_ingest(path: str, corpus: str, format: str) -> int:
        calls.append("ingest")
        (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
        return 1

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append("vectors")

    async def fake_questions(corpus: str, count: int) -> None:
        calls.append("questions")
        (base / "questions" / "questions.yaml").write_text(MINIMAL_QUESTION)

    async def fake_benchmark(corpus: str, strategy: str, **kwargs) -> None:
        calls.append("benchmark")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest", fake_ingest)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.question_gen.run_question_generation", fake_questions)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample", "--skip-graph"])

    assert result.exit_code == 0, result.stdout
    assert calls == ["ingest", "vectors", "questions", "benchmark"]
    state = json.loads((base / ".pipeline_state.json").read_text())
    assert state["ingest"] == "done"
    assert state["generate_questions"] == "done"


def test_run_does_not_swallow_unexpected_graph_failures(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)

    async def broken_graph(corpus: str) -> None:
        raise AttributeError("bad extractor state")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", broken_graph)

    result = runner.invoke(app, ["run", "--corpus", "sample"])

    assert result.exit_code == 1
    assert "Pipeline complete" not in result.stdout
    state_path = base / ".pipeline_state.json"
    assert not state_path.exists() or "build_graph" not in json.loads(state_path.read_text())


def test_build_vectors_preflights_llm_only_for_generation_strategies(monkeypatch):
    preflights: list[dict[str, bool]] = []

    async def fake_build(corpus: str, strategy: str) -> None:
        return None

    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: preflights.append(kwargs))
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_build)

    naive = runner.invoke(
        app, ["build-vectors", "--corpus", "sample", "--strategy", "naive_vector"]
    )
    contextual = runner.invoke(
        app, ["build-vectors", "--corpus", "sample", "--strategy", "contextual_vector"]
    )
    bm25 = runner.invoke(app, ["build-vectors", "--corpus", "sample", "--strategy", "bm25"])
    pageindex = runner.invoke(
        app, ["build-vectors", "--corpus", "sample", "--strategy", "pageindex"]
    )

    assert naive.exit_code == 0, naive.stdout
    assert contextual.exit_code == 0, contextual.stdout
    assert bm25.exit_code == 0, bm25.stdout
    assert pageindex.exit_code == 0, pageindex.stdout
    assert preflights == [
        {"needs_llm": False, "needs_embeddings": True},
        {"needs_llm": True, "needs_embeddings": True},
        {"needs_llm": False, "needs_embeddings": False},
        {"needs_llm": True, "needs_embeddings": False},
    ]


def test_demo_forces_read_only_mode_while_the_server_runs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    results.mkdir()
    (results / "aws-compute_naive_vector.json").write_text("{}\n")
    observed: dict[str, object] = {}

    class InertTimer:
        def __init__(self, *args, **kwargs):
            pass

        def start(self) -> None:
            pass

    def fake_uvicorn_run(*args, **kwargs) -> None:
        observed["environment"] = os.environ.get("KB_ARENA_DEMO_MODE")
        observed["setting"] = settings.demo_mode

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KB_ARENA_DEMO_MODE", raising=False)
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr("threading.Timer", InertTimer)
    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)

    result = runner.invoke(app, ["demo", "--host", "127.0.0.1", "--port", "8765"])

    assert result.exit_code == 0, result.stdout
    assert observed == {"environment": "true", "setting": True}
    assert "KB_ARENA_DEMO_MODE" not in os.environ
    assert settings.demo_mode is False
