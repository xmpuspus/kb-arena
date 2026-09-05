"""Protocol tests for the KB Arena MCP server.

Each test calls `server.call_tool(name, args)`, the same dispatch path the
stdio transport uses for a `tools/call` request, so a passing test here is a
passing tool call over the wire. No test calls a real model or database:
`run_benchmark` is mocked wherever a test exercises `start_benchmark`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from kb_arena.mcp import server as mcp_server  # noqa: E402
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, Score  # noqa: E402
from kb_arena.settings import settings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TOOL_NAMES = {
    "list_corpora",
    "list_strategies",
    "validate_corpus",
    "start_benchmark",
    "job_status",
    "compare",
    "get_manifest",
    "export_evidence",
}


def _content(result) -> dict:
    """The tool's payload, the same shape a client reads.

    A tool typed to return a concrete model gets `structured_content` for
    free; one typed `-> dict` does not, so this falls back to the text
    block every tool call carries either way.
    """
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(result.content[0].text)


def _write_result(results_dir, run_id, corpus, strategy, scores: dict[str, float], manifest=None):
    run_dir = results_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bench = BenchmarkResult(
        corpus=corpus,
        strategy=strategy,
        run_id=run_id,
        records=[
            AnswerRecord(
                question_id=q,
                strategy=strategy,
                answer="a",
                score=Score(accuracy=v),
                latency_ms=10.0,
            )
            for q, v in scores.items()
        ],
    )
    data = json.loads(bench.model_dump_json())
    if manifest is not None:
        data["manifest"] = manifest
    (run_dir / f"{corpus}_{strategy}.json").write_text(json.dumps(data))
    (results_dir / f"{corpus}_{strategy}.json").write_text(json.dumps(data))


# ── server shape ──


@pytest.mark.asyncio
async def test_the_server_registers_exactly_the_eight_named_tools():
    tools = await mcp_server.server.list_tools()
    assert {t.name for t in tools} == TOOL_NAMES


# ── list_corpora ──


@pytest.mark.asyncio
async def test_list_corpora_reports_pipeline_status_per_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr(settings, "results_path", str(tmp_path / "results"))

    corpus_dir = tmp_path / "demo"
    (corpus_dir / "processed").mkdir(parents=True)
    (corpus_dir / "processed" / "docs.jsonl").write_text("{}\n")
    (corpus_dir / "questions").mkdir()
    (corpus_dir / "questions" / "q.yaml").write_text("- id: q1\n- id: q2\n")

    result = await mcp_server.server.call_tool("list_corpora", {})
    corpora = _content(result)["corpora"]

    assert corpora == [
        {"name": "demo", "has_processed": True, "question_count": 2, "has_results": False}
    ]


@pytest.mark.asyncio
async def test_list_corpora_raises_when_the_configured_root_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "does-not-exist"))

    with pytest.raises(ToolError, match="does not exist"):
        await mcp_server.server.call_tool("list_corpora", {})


# ── list_strategies ──


@pytest.mark.asyncio
async def test_list_strategies_lists_every_catalog_entry_with_a_status():
    from kb_arena.strategies.catalog import STRATEGY_CATALOG

    result = await mcp_server.server.call_tool("list_strategies", {})
    payload = _content(result)

    assert payload["strategies"] == [spec.name for spec in STRATEGY_CATALOG]
    assert len(payload["catalog"]) == len(STRATEGY_CATALOG)
    for entry in payload["catalog"]:
        assert entry["status"]


@pytest.mark.asyncio
async def test_list_strategies_reads_the_catalog_at_call_time_not_a_literal_list(monkeypatch):
    """Patching the catalog module's tuple must change the tool's output.

    A tool that copied the names into a literal list at import time would
    keep answering with the old set after this patch. This is what proves it
    does not.
    """
    import kb_arena.strategies.catalog as catalog_module

    fake_spec = catalog_module.StrategySpec("fake_strategy_for_this_test", "Fake", "test")
    monkeypatch.setattr(catalog_module, "STRATEGY_CATALOG", (fake_spec,))

    result = await mcp_server.server.call_tool("list_strategies", {})
    payload = _content(result)

    assert payload["strategies"] == ["fake_strategy_for_this_test"]
    assert [entry["name"] for entry in payload["catalog"]] == ["fake_strategy_for_this_test"]


# ── validate_corpus ──


@pytest.mark.asyncio
async def test_validate_corpus_reports_a_missing_directory_as_invalid_not_an_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    result = await mcp_server.server.call_tool("validate_corpus", {"corpus": "nope"})

    assert _content(result) == {
        "corpus": "nope",
        "valid": False,
        "reason": "corpus directory not found",
    }


@pytest.mark.asyncio
async def test_validate_corpus_reports_valid_when_processed_chunks_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    corpus_dir = tmp_path / "demo"
    (corpus_dir / "processed").mkdir(parents=True)
    (corpus_dir / "processed" / "docs.jsonl").write_text("{}\n")

    result = await mcp_server.server.call_tool("validate_corpus", {"corpus": "demo"})
    payload = _content(result)

    assert payload["valid"] is True
    assert payload["has_processed"] is True
    assert payload["has_questions"] is False
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_validate_corpus_refuses_a_traversal_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    with pytest.raises(ToolError, match="invalid corpus name"):
        await mcp_server.server.call_tool("validate_corpus", {"corpus": "../outside"})


# ── start_benchmark / job_status ──


@pytest.mark.asyncio
async def test_start_benchmark_reports_a_completed_job_after_the_run_finishes(monkeypatch):
    async def fake_run_benchmark(**kwargs):
        return "runid123"

    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_run_benchmark)

    started = await mcp_server.server.call_tool(
        "start_benchmark", {"corpus": "demo", "strategy": "bm25"}
    )
    job_id = _content(started)["job_id"]
    assert _content(started)["status"] == "queued"

    await mcp_server._TASKS[job_id]

    status = await mcp_server.server.call_tool("job_status", {"job_id": job_id})
    payload = _content(status)
    assert payload["status"] == "completed"
    assert payload["run_id"] == "runid123"
    assert payload["error"] is None


@pytest.mark.asyncio
async def test_start_benchmark_records_a_run_failure_in_job_status(monkeypatch):
    from kb_arena.benchmark.runner import BenchmarkExecutionError

    async def fake_run_benchmark(**kwargs):
        raise BenchmarkExecutionError("no strategies available")

    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_run_benchmark)

    started = await mcp_server.server.call_tool("start_benchmark", {"corpus": "all"})
    job_id = _content(started)["job_id"]

    await mcp_server._TASKS[job_id]

    status = await mcp_server.server.call_tool("job_status", {"job_id": job_id})
    payload = _content(status)
    assert payload["status"] == "failed"
    assert "no strategies available" in payload["error"]
    assert payload["run_id"] is None


@pytest.mark.asyncio
async def test_start_benchmark_refuses_an_unknown_strategy():
    with pytest.raises(ToolError, match="unknown strategy"):
        await mcp_server.server.call_tool(
            "start_benchmark", {"corpus": "all", "strategy": "not_a_real_strategy"}
        )


@pytest.mark.asyncio
async def test_start_benchmark_refuses_a_traversal_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    with pytest.raises(ToolError, match="invalid corpus name"):
        await mcp_server.server.call_tool("start_benchmark", {"corpus": "../outside"})


@pytest.mark.asyncio
async def test_job_status_raises_for_an_unknown_job_id():
    with pytest.raises(ToolError, match="unknown job id"):
        await mcp_server.server.call_tool("job_status", {"job_id": "never-started"})


# ── compare ──


@pytest.mark.asyncio
async def test_compare_pairs_two_strategies_on_the_same_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_result(tmp_path, "r1", "demo", "bm25", {"q1": 0.2, "q2": 0.4})
    _write_result(tmp_path, "r1", "demo", "naive_vector", {"q1": 0.6, "q2": 0.4})

    result = await mcp_server.server.call_tool(
        "compare", {"corpus": "demo", "a": "bm25", "b": "naive_vector"}
    )
    payload = _content(result)

    assert payload["mean_delta"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_compare_raises_when_a_result_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_result(tmp_path, "r1", "demo", "bm25", {"q1": 0.2})

    with pytest.raises(ToolError, match="no result file"):
        await mcp_server.server.call_tool(
            "compare", {"corpus": "demo", "a": "bm25", "b": "not_run"}
        )


@pytest.mark.asyncio
async def test_compare_against_the_committed_aws_compute_run(committed_results):
    """A real, read-only pair from the repository's own committed results."""
    result = await mcp_server.server.call_tool(
        "compare", {"corpus": "aws-compute", "a": "bm25", "b": "naive_vector"}
    )
    payload = _content(result)
    assert payload["n_paired"] > 0


@pytest.fixture
def committed_results(monkeypatch):
    """Point `results_path` at the repository's own committed results.

    These two tests read what the repository ships. Reading the ambient setting
    made them depend on whichever earlier test last pointed it somewhere else,
    and in a full run that is a tmp directory holding nothing.
    """
    monkeypatch.setattr(settings, "results_path", str(ROOT / "results"))


# ── get_manifest ──


@pytest.mark.asyncio
async def test_get_manifest_returns_the_stored_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write_result(tmp_path, "r1", "demo", "bm25", {"q1": 0.2}, manifest={"compatibility_key": "k1"})

    result = await mcp_server.server.call_tool(
        "get_manifest", {"corpus": "demo", "strategy": "bm25", "run_id": "r1"}
    )
    payload = _content(result)

    assert payload["manifest"]["compatibility_key"] == "k1"


@pytest.mark.asyncio
async def test_get_manifest_raises_when_the_result_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))

    with pytest.raises(ToolError, match="no result file"):
        await mcp_server.server.call_tool("get_manifest", {"corpus": "demo", "strategy": "bm25"})


@pytest.mark.asyncio
async def test_get_manifest_on_a_legacy_file_reports_an_empty_manifest_not_a_fabricated_one(
    committed_results,
):
    """The committed top-level aws-compute results predate manifests (schema v1)."""
    result = await mcp_server.server.call_tool(
        "get_manifest", {"corpus": "aws-compute", "strategy": "bm25"}
    )
    payload = _content(result)
    assert payload["manifest"] == {}
    assert payload["summary"] == {}


# ── export_evidence ──


@pytest.mark.asyncio
async def test_export_evidence_reports_problems_instead_of_writing_a_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr(settings, "results_path", str(tmp_path / "results"))
    (tmp_path / "demo" / "questions").mkdir(parents=True)
    run_dir = tmp_path / "results" / "run_abc12345"
    run_dir.mkdir(parents=True)

    result = await mcp_server.server.call_tool(
        "export_evidence", {"corpus": "demo", "run_id": "abc12345"}
    )
    payload = _content(result)

    assert payload["written"] is False
    assert any("no result files" in p for p in payload["problems"])
    assert not (run_dir / "evidence.json").exists()


@pytest.mark.asyncio
async def test_export_evidence_raises_when_the_run_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr(settings, "results_path", str(tmp_path / "results"))
    (tmp_path / "demo" / "questions").mkdir(parents=True)

    with pytest.raises(ToolError, match="no run at"):
        await mcp_server.server.call_tool(
            "export_evidence", {"corpus": "demo", "run_id": "missingrun"}
        )


@pytest.mark.asyncio
async def test_export_evidence_refuses_an_invalid_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    (tmp_path / "demo" / "questions").mkdir(parents=True)

    with pytest.raises(ToolError, match="invalid run id"):
        await mcp_server.server.call_tool(
            "export_evidence", {"corpus": "demo", "run_id": "../../etc"}
        )


# ── the stdio stream and the job registry ──


@pytest.mark.asyncio
async def test_the_runner_never_writes_to_the_protocol_stream(monkeypatch, capsys):
    """stdout carries JSON-RPC on a stdio server, and the runner prints to it.

    Its run id, its progress and its summary landed between messages, and the
    client then failed to parse the stream.
    """

    async def noisy_run(**kwargs):
        print("Run ID: deadbeef")
        return "deadbeef"

    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", noisy_run)
    monkeypatch.setattr(settings, "results_path", str(ROOT / "results"))

    started = await mcp_server.server.call_tool(
        "start_benchmark", {"corpus": "aws-compute", "strategy": "bm25"}
    )
    job_id = _content(started)["job_id"]
    await mcp_server._TASKS[job_id]

    captured = capsys.readouterr()
    assert "Run ID" not in captured.out
    assert "Run ID" in captured.err
    assert mcp_server._JOBS[job_id].status == "completed"


def test_the_registry_cap_never_forgets_a_running_job():
    """Evicting by age alone made a live benchmark unreachable while it ran on."""
    from datetime import UTC, datetime

    from kb_arena.mcp.server import _JOBS, _MAX_JOBS, _Job, _trim_jobs

    prior = dict(_JOBS)
    _JOBS.clear()
    try:
        now = datetime.now(UTC).isoformat()
        _JOBS["live"] = _Job(
            job_id="live", corpus="c", strategy="bm25", status="running", created_at=now
        )
        for i in range(_MAX_JOBS + 5):
            _JOBS[f"done{i}"] = _Job(
                job_id=f"done{i}", corpus="c", strategy="bm25", status="completed", created_at=now
            )

        _trim_jobs()

        assert "live" in _JOBS
        assert len(_JOBS) == _MAX_JOBS
    finally:
        _JOBS.clear()
        _JOBS.update(prior)
