"""Regression tests for the one-shot ``kb-arena run`` workflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kb_arena.cli import app
from kb_arena.exceptions import GraphError
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


def test_graph_schema_migration_requires_explicit_confirmation(monkeypatch):
    called = False

    async def fake_migration(database):
        nonlocal called
        called = True
        return ["topic_fqn"]

    monkeypatch.setattr(
        "kb_arena.graph.extractor.migrate_legacy_graph_schema",
        fake_migration,
    )

    result = runner.invoke(app, ["migrate-graph-schema", "--database", "kb_arena"])

    assert result.exit_code == 1
    assert "--confirm-dedicated-database" in result.stdout
    assert called is False


def test_graph_schema_migration_runs_after_confirmation(monkeypatch):
    migrated: list[str] = []

    async def fake_migration(database):
        migrated.append(database)
        return ["topic_fqn"]

    monkeypatch.setattr(
        "kb_arena.graph.extractor.migrate_legacy_graph_schema",
        fake_migration,
    )

    result = runner.invoke(
        app,
        [
            "migrate-graph-schema",
            "--database",
            "kb_arena",
            "--confirm-dedicated-database",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "topic_fqn" in result.stdout
    assert migrated == ["kb_arena"]


def test_health_checks_configured_database_and_closes_failed_driver(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    import neo4j

    driver = MagicMock()
    session = AsyncMock()
    session.run.side_effect = OSError("database unavailable")
    context = AsyncMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = context
    driver.close = AsyncMock()

    monkeypatch.setattr(settings, "neo4j_database", "kb_arena")
    monkeypatch.setattr(neo4j.AsyncGraphDatabase, "driver", MagicMock(return_value=driver))
    monkeypatch.setattr("chromadb.PersistentClient", lambda *args, **kwargs: MagicMock())

    result = runner.invoke(app, ["health", "--format", "json"])

    assert result.exit_code == 0, result.stdout
    driver.session.assert_called_once_with(database="kb_arena")
    driver.close.assert_awaited_once()


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
            "naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,metadata_filtered,temporal,qiss",
        ),
    ]


@pytest.mark.parametrize(
    ("source", "source_format"),
    [
        ("https://example.com/docs", "web"),
        ("HTTPS://example.com/docs", "web"),
        ("github:example/docs", "github"),
        ("GITHUB:example/docs", "github"),
    ],
)
def test_run_dispatches_special_docs_to_special_ingest(
    tmp_path, monkeypatch, source, source_format
):
    base = _corpus(tmp_path)
    calls: list[tuple[str, str, str]] = []

    def fake_special(source: str, corpus: str, format: str) -> int:
        calls.append((source, corpus, format))
        (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
        return 1

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        return None

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest_special", fake_special)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(
        app,
        [
            "run",
            "--corpus",
            "sample",
            "--docs",
            source,
            "--skip-graph",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert calls == [(source, "sample", source_format)]


def test_run_new_explicit_docs_invalidate_downstream_checkpoints(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    new_docs = tmp_path / "new.md"
    new_docs.write_text("# New documentation\n")
    (base / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "ingest": "done",
                "ingest_source": "old.md",
                "build_graph": "done",
                "build_vectors": "done",
                "generate_questions": "done",
                "benchmark": "done",
                "benchmark_strategies": "all",
            }
        )
    )
    calls: list[str] = []

    def fake_ingest(path: str, corpus: str, format: str) -> int:
        calls.append(f"ingest:{path}")
        return 1

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        calls.append("vectors")

    async def fake_questions(corpus: str, count: int) -> None:
        calls.append("questions")

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        calls.append(f"benchmark:{strategy}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest", fake_ingest)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.question_gen.run_question_generation", fake_questions)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    result = runner.invoke(
        app,
        ["run", "--corpus", "sample", "--docs", str(new_docs), "--skip-graph"],
    )

    assert result.exit_code == 0, result.stdout
    assert calls == [
        f"ingest:{new_docs}",
        "vectors",
        "questions",
        "benchmark:naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,metadata_filtered,temporal,qiss",
    ]
    state = json.loads((base / ".pipeline_state.json").read_text())
    assert state["ingest_source"] == str(new_docs)
    assert "build_graph" not in state


@pytest.mark.parametrize("failure", [OSError("Neo4j is unavailable"), GraphError("bad graph")])
def test_run_continues_after_graph_failure_without_checkpointing_it(tmp_path, monkeypatch, failure):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    calls: list[str] = []

    async def failing_graph(corpus: str) -> None:
        calls.append("graph")
        raise failure

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
        "benchmark:naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,metadata_filtered,temporal,qiss",
    ]
    state_path = base / ".pipeline_state.json"
    state = json.loads(state_path.read_text())
    assert "build_graph" not in state
    assert state["build_vectors"] == "done"
    assert state["benchmark"] == "done"


def test_run_reruns_benchmark_when_graph_recovers(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    graph_attempts = 0
    benchmark_strategies: list[str] = []

    async def recovering_graph(corpus: str) -> None:
        nonlocal graph_attempts
        graph_attempts += 1
        if graph_attempts == 1:
            raise OSError("Neo4j is unavailable")

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        return None

    async def fake_benchmark(corpus: str, strategy: str) -> None:
        benchmark_strategies.append(strategy)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.graph.extractor.run_extraction", recovering_graph)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_benchmark)

    first = runner.invoke(app, ["run", "--corpus", "sample"])
    second = runner.invoke(app, ["run", "--corpus", "sample"])

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert benchmark_strategies == [
        "naive_vector,contextual_vector,qna_pairs,raptor,pageindex,bm25,metadata_filtered,temporal,qiss",
        "all",
    ]
    state = json.loads((base / ".pipeline_state.json").read_text())
    assert state["build_graph"] == "done"
    assert state["benchmark"] == "done"
    assert state["benchmark_strategies"] == "all"


def test_run_does_not_checkpoint_cost_capped_benchmark(tmp_path, monkeypatch):
    from kb_arena.benchmark.runner import BenchmarkIncompleteError

    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)

    async def fake_vectors(corpus: str, strategy: str = "all") -> None:
        return None

    async def capped_benchmark(corpus: str, strategy: str) -> None:
        raise BenchmarkIncompleteError("cost cap reached before completion")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kb_arena.cli._preflight", lambda **kwargs: None)
    monkeypatch.setattr("kb_arena.strategies.build_vector_indexes", fake_vectors)
    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", capped_benchmark)

    result = runner.invoke(app, ["run", "--corpus", "sample", "--skip-graph"])

    assert result.exit_code == 1
    state = json.loads((base / ".pipeline_state.json").read_text())
    assert "benchmark" not in state
    assert "Pipeline complete" not in result.stdout


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


def test_run_invalidates_stale_checkpoints_for_empty_artifacts(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "raw" / "guide.md").write_text("# Guide\n\nLocal documentation.\n")
    (base / "processed" / "documents.jsonl").write_text("")
    (base / "questions" / "questions.yaml").write_text("")
    (base / ".pipeline_state.json").write_text(
        json.dumps(
            {
                "ingest": "done",
                "build_graph": "done",
                "build_vectors": "done",
                "generate_questions": "done",
                "benchmark": "done",
            }
        )
    )
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
    assert "build_graph" not in state
    assert all(
        state[name] == "done"
        for name in ("ingest", "build_vectors", "generate_questions", "benchmark")
    )


def test_run_completed_resume_does_not_require_credentials(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    completed = {
        "ingest": "done",
        "build_graph": "done",
        "build_vectors": "done",
        "generate_questions": "done",
        "benchmark": "done",
        "benchmark_strategies": "all",
    }
    (base / ".pipeline_state.json").write_text(json.dumps(completed))

    monkeypatch.chdir(tmp_path)

    def unexpected_preflight(**kwargs) -> None:
        raise AssertionError(f"completed resume preflighted credentials: {kwargs}")

    monkeypatch.setattr("kb_arena.cli._preflight", unexpected_preflight)

    result = runner.invoke(app, ["run", "--corpus", "sample"])

    assert result.exit_code == 0, result.stdout
    assert "Pipeline complete" in result.stdout
    assert json.loads((base / ".pipeline_state.json").read_text()) == completed


def test_run_uses_configured_datasets_path(tmp_path, monkeypatch):
    base = _corpus(tmp_path)
    (base / "processed" / "documents.jsonl").write_text(MINIMAL_DOCUMENT)
    completed = {
        "ingest": "done",
        "build_graph": "done",
        "build_vectors": "done",
        "generate_questions": "done",
        "benchmark": "done",
        "benchmark_strategies": "all",
    }
    (base / ".pipeline_state.json").write_text(json.dumps(completed))
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "datasets"))
    monkeypatch.setattr(
        "kb_arena.cli._preflight",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    result = runner.invoke(app, ["run", "--corpus", "sample"])

    assert result.exit_code == 0, result.stdout
    assert "Pipeline complete" in result.stdout


def test_init_corpus_uses_configured_datasets_path(tmp_path, monkeypatch):
    datasets = tmp_path / "custom-datasets"
    monkeypatch.setattr(settings, "datasets_path", str(datasets))

    result = runner.invoke(app, ["init-corpus", "configured"])

    assert result.exit_code == 0, result.stdout
    assert (datasets / "configured" / "raw").is_dir()
    assert (datasets / "configured" / "processed").is_dir()
    assert (datasets / "configured" / "questions").is_dir()


def test_ingest_exits_when_no_documents_are_produced(tmp_path, monkeypatch):
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest", lambda **kwargs: 0)

    result = runner.invoke(app, ["ingest", str(tmp_path), "--corpus", "sample"])

    assert result.exit_code == 1
    assert "Ingestion produced no documents" in result.stdout
    assert "Next:" not in result.stdout


def test_special_ingest_exits_when_no_documents_are_produced(monkeypatch):
    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest_special", lambda **kwargs: 0)

    result = runner.invoke(
        app,
        ["ingest", "https://example.com", "--corpus", "sample"],
    )

    assert result.exit_code == 1
    assert "Ingestion produced no documents" in result.stdout
    assert "Next:" not in result.stdout


def test_ingest_auto_detects_an_uppercase_url_scheme_as_web(monkeypatch):
    calls: list[dict] = []

    def fake_special(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr("kb_arena.ingest.pipeline.run_ingest_special", fake_special)

    runner.invoke(app, ["ingest", "HTTPS://example.com/docs", "--corpus", "sample"])

    assert [(c["source"], c["format"]) for c in calls] == [("HTTPS://example.com/docs", "web")]


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
