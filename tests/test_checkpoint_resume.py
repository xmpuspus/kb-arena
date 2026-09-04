"""A crash leaves whole files, and a stopped run resumes from its checkpoint."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from kb_arena.benchmark import atomic, runner
from kb_arena.benchmark.atomic import append_jsonl, atomic_write_text, read_jsonl
from kb_arena.models.benchmark import AnswerRecord, GroundTruth, Question, Score
from kb_arena.settings import settings


def test_a_failed_write_leaves_the_old_file_and_no_temp(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    atomic_write_text(target, "old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "new")

    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_read_jsonl_drops_a_torn_last_line(tmp_path):
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    with open(path, "a") as handle:
        handle.write('{"a": 3, "tor')

    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def _record(qid: str, strategy: str = "x", accuracy: float = 1.0) -> AnswerRecord:
    return AnswerRecord(
        question_id=qid, strategy=strategy, answer="a", score=Score(accuracy=accuracy)
    )


def test_load_checkpoint_keeps_the_last_line_per_question_and_skips_junk(tmp_path):
    path = tmp_path / "c_x.records.jsonl"
    append_jsonl(path, _record("q1", accuracy=0.0).model_dump(mode="json"))
    append_jsonl(path, {"not": "a record"})
    append_jsonl(path, _record("q1", accuracy=1.0).model_dump(mode="json"))
    append_jsonl(path, _record("q2").model_dump(mode="json"))

    done = runner.load_checkpoint(path)

    assert sorted(done) == ["q1", "q2"]
    assert done["q1"].score.accuracy == 1.0


class _Strategy:
    def __init__(self, name: str):
        self.name = name


def _questions(n: int) -> list[Question]:
    return [
        Question(
            id=f"q{i}",
            tier=1,
            type="factoid",
            hops=1,
            question=f"q{i}?",
            ground_truth=GroundTruth(answer="a"),
        )
        for i in range(1, n + 1)
    ]


def _seed_run(tmp_path, run_id: str, **overrides):
    snap = {
        "llm_provider": settings.llm_provider,
        "generate_model": settings.generate_model,
        "judge_provider": "",
        "judge_model": "",
        "top_k": 5,
        "question_split": "all",
        "reference_free": False,
        "ragas_enabled": False,
    }
    snap.update(overrides)
    path = runner.run_manifest_path(tmp_path, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "config_snapshot": snap}))


def _harness(monkeypatch, tmp_path, strategies: list[str]):
    calls: list[tuple[str, str]] = []

    async def fake_run_one(strat, qid, *args, **kwargs):
        calls.append((strat.name, qid))
        return _record(qid, strategy=strat.name)

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    monkeypatch.setattr(runner, "load_questions", lambda corpus, tier=0, split="": _questions(3))
    monkeypatch.setattr(
        runner, "_load_strategies", lambda names: [_Strategy(n) for n in strategies]
    )
    monkeypatch.setattr(runner, "LLMClient", lambda: object())
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    monkeypatch.setattr(settings, "benchmark_cost_cap_usd", 0.0)
    return calls


@pytest.mark.parametrize("parallel,strategies", [(False, ["x"]), (True, ["x", "y"])])
def test_a_resumed_run_skips_checkpointed_questions(tmp_path, monkeypatch, parallel, strategies):
    calls = _harness(monkeypatch, tmp_path, strategies)
    run_id = "abc12345"
    _seed_run(tmp_path, run_id)
    for name in strategies:
        ckpt = runner.checkpoint_path(tmp_path, run_id, "c", name)
        append_jsonl(ckpt, _record("q1", strategy=name).model_dump(mode="json"))
        append_jsonl(ckpt, _record("q2", strategy=name).model_dump(mode="json"))

    asyncio.run(
        runner.run_benchmark(corpus="c", strategy="any", parallel=parallel, resume_run_id=run_id)
    )

    assert sorted(calls) == sorted((name, "q3") for name in strategies)
    for name in strategies:
        result = json.loads((tmp_path / f"run_{run_id}" / f"c_{name}.json").read_text())
        assert sorted(r["question_id"] for r in result["records"]) == ["q1", "q2", "q3"]
        assert result["run_id"] == run_id
        lines = read_jsonl(runner.checkpoint_path(tmp_path, run_id, "c", name))
        assert [row["question_id"] for row in lines] == ["q1", "q2", "q3"]


def test_a_fresh_run_writes_a_checkpoint_line_per_record(tmp_path, monkeypatch):
    calls = _harness(monkeypatch, tmp_path, ["x"])

    asyncio.run(runner.run_benchmark(corpus="c", strategy="any", parallel=False))

    assert len(calls) == 3
    run_dirs = [p for p in tmp_path.iterdir() if p.name.startswith("run_")]
    assert len(run_dirs) == 1
    rows = read_jsonl(run_dirs[0] / "c_x.records.jsonl")
    assert sorted(row["question_id"] for row in rows) == ["q1", "q2", "q3"]
    assert not list(tmp_path.glob("**/*.tmp"))


def test_a_resume_under_different_settings_is_refused(tmp_path, monkeypatch):
    _harness(monkeypatch, tmp_path, ["x"])
    _seed_run(tmp_path, "abc12345", top_k=10, reference_free=True)

    with pytest.raises(runner.BenchmarkExecutionError, match="top_k, reference_free"):
        asyncio.run(
            runner.run_benchmark(
                corpus="c", strategy="any", parallel=False, resume_run_id="abc12345"
            )
        )


def test_a_resume_of_an_unknown_or_unsafe_run_id_is_refused(tmp_path, monkeypatch):
    _harness(monkeypatch, tmp_path, ["x"])

    with pytest.raises(runner.BenchmarkExecutionError, match="no run to resume"):
        asyncio.run(
            runner.run_benchmark(
                corpus="c", strategy="any", parallel=False, resume_run_id="nothere1"
            )
        )
    with pytest.raises(runner.BenchmarkExecutionError, match="invalid run id"):
        asyncio.run(
            runner.run_benchmark(corpus="c", strategy="any", parallel=False, resume_run_id="../etc")
        )


def test_a_fresh_run_writes_its_settings_first(tmp_path, monkeypatch):
    _harness(monkeypatch, tmp_path, ["x"])

    asyncio.run(runner.run_benchmark(corpus="c", strategy="any", parallel=False, top_k=7))

    run_dir = next(p for p in tmp_path.iterdir() if p.name.startswith("run_"))
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["config_snapshot"]["top_k"] == 7
    assert manifest["run_id"] == run_dir.name.removeprefix("run_")


def test_resumed_records_count_toward_the_cost_cap(tmp_path, monkeypatch):
    calls = _harness(monkeypatch, tmp_path, ["x"])
    monkeypatch.setattr(settings, "benchmark_cost_cap_usd", 0.05)
    run_id = "abc12345"
    _seed_run(tmp_path, run_id)
    ckpt = runner.checkpoint_path(tmp_path, run_id, "c", "x")
    for qid in ("q1", "q2"):
        rec = _record(qid)
        rec.cost_usd = 0.03
        append_jsonl(ckpt, rec.model_dump(mode="json"))

    with pytest.raises(runner.BenchmarkIncompleteError):
        asyncio.run(
            runner.run_benchmark(corpus="c", strategy="any", parallel=False, resume_run_id=run_id)
        )

    # q3 ran once, then the cap that the first attempt already spent stopped the run
    assert calls == [("x", "q3")]


def test_the_cli_passes_the_resume_id(monkeypatch):
    from typer.testing import CliRunner

    from kb_arena.cli import app

    seen: list[dict] = []

    async def fake_run_benchmark(**kwargs):
        seen.append(kwargs)

    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(settings, "anthropic_api_key", "k")
    monkeypatch.setattr(settings, "openai_api_key", "k")

    result = CliRunner().invoke(
        app, ["benchmark", "--corpus", "aws-compute", "--resume", "abc12345"]
    )

    assert result.exit_code == 0, result.output
    assert seen[0]["resume_run_id"] == "abc12345"


def test_arena_state_saves_atomically(tmp_path, monkeypatch):
    from kb_arena.arena.engine import ArenaState

    state = ArenaState(elo={"a": 1000.0})
    path = tmp_path / "arena" / "arena_state.json"
    state.save(path)
    assert json.loads(path.read_text())["elo"] == {"a": 1000.0}
    assert not list(path.parent.glob("*.tmp"))
    assert atomic.read_jsonl(tmp_path / "none.jsonl") == []
