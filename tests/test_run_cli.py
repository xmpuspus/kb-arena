"""Regression tests for the one-shot ``kb-arena run`` workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from kb_arena.cli import app
from kb_arena.settings import settings

runner = CliRunner()


def _corpus(tmp_path: Path, name: str = "sample") -> Path:
    base = tmp_path / "datasets" / name
    (base / "raw").mkdir(parents=True)
    (base / "processed").mkdir()
    (base / "questions").mkdir()
    (base / "questions" / "questions.yaml").write_text("questions: []\n")
    return base


def test_run_ingests_files_already_in_raw_directory(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "raw" / "guide.md").write_text("# Guide\n\nLocal documentation.\n")
    calls: list[tuple[str, str]] = []

    def fake_ingest(path: str, corpus: str, format: str) -> None:
        calls.append((path, corpus))
        (base / "processed" / "documents.jsonl").write_text("{}\n")

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append(("vectors", corpus))

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        calls.append(("benchmark", corpus))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest", fake_ingest)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample", "--skip-graph"])

    assert result.exit_code == 0, result.stdout
    assert calls == [
        (str(Path("datasets/sample/raw")), "sample"),
        ("vectors", "sample"),
        ("benchmark", "sample"),
    ]


def test_run_continues_after_graph_failure_without_checkpointing_it(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text("{}\n")
    calls: list[str] = []

    async def failing_graph(corpus: str) -> None:
        calls.append("graph")
        raise OSError("Neo4j is unavailable")

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append("vectors")

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        calls.append("benchmark")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", failing_graph)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample"])

    assert result.exit_code == 0, result.stdout
    assert calls == ["graph", "vectors", "benchmark"]
    state = json.loads((base / ".pipeline_state.json").read_text())
    assert "build_graph" not in state
    assert state["build_vectors"] == "done"
    assert state["benchmark"] == "done"


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
