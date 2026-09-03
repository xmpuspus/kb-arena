"""A result file names what it measured, and the leaderboard never blends runs that differ."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from kb_arena.benchmark import manifest as mf
from kb_arena.chatbot import api
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, Score
from kb_arena.settings import settings


def _q(qid: str, text: str, chunks: list[str] | None = None, split: str = "development"):
    return SimpleNamespace(id=qid, question=text, expected_chunks=chunks or [], split=split)


def test_question_fingerprint_ignores_order_and_tracks_content():
    a = [_q("q1", "one", ["c1"]), _q("q2", "two")]
    b = [_q("q2", "two"), _q("q1", "one", ["c1"])]
    c = [_q("q1", "one changed", ["c1"]), _q("q2", "two")]

    assert mf.question_set_fingerprint(a) == mf.question_set_fingerprint(b)
    assert mf.question_set_fingerprint(a) != mf.question_set_fingerprint(c)


def test_compatibility_key_changes_with_judge_split_top_k_and_qrels(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    questions = [_q("q1", "one")]
    base = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)

    other_k = mf.build_manifest("c", questions, top_k=10, split="development", reference_free=False)
    other_split = mf.build_manifest("c", questions, top_k=5, split="holdout", reference_free=False)
    monkeypatch.setattr(settings, "judge_model", "some-other-judge")
    other_judge = mf.build_manifest(
        "c", questions, top_k=5, split="development", reference_free=False
    )
    qrels = tmp_path / "c" / "questions"
    qrels.mkdir(parents=True)
    (qrels / "expected_chunks.yaml").write_text("q1: [c1]\n")
    other_qrels = mf.build_manifest(
        "c", questions, top_k=5, split="development", reference_free=False
    )

    keys = {m["compatibility_key"] for m in (base, other_k, other_split, other_judge, other_qrels)}
    assert len(keys) == 5
    assert base["schema_version"] == mf.SCHEMA_VERSION
    assert base["qrels_fingerprint"] is None
    assert other_qrels["qrels_fingerprint"]


def test_a_result_file_without_a_manifest_is_legacy():
    assert mf.compatibility_key({"corpus": "c", "strategy": "s"}) == mf.LEGACY_KEY
    assert mf.compatibility_key({"manifest": {"compatibility_key": "abc123"}}) == "abc123"


def _write_run(
    results: Path, run_id: str, corpus: str, strategy: str, accuracy: float, manifest: dict | None
):
    run_dir = results / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bench = BenchmarkResult(
        corpus=corpus,
        strategy=strategy,
        run_id=run_id,
        records=[
            AnswerRecord(
                question_id="q1", strategy=strategy, answer="a", score=Score(accuracy=accuracy)
            )
        ],
        overall_accuracy=accuracy,
        manifest=manifest or {},
        schema_version=2 if manifest else 1,
    )
    (run_dir / f"{corpus}_{strategy}.json").write_text(bench.model_dump_json())


def test_leaderboard_groups_only_runs_that_share_a_compatibility_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    key_a = {"compatibility_key": "aaaaaaaaaaaa", "judge": {"model": "judge-a"}, "top_k": 5}
    key_b = {"compatibility_key": "bbbbbbbbbbbb", "judge": {"model": "judge-b"}, "top_k": 5}
    _write_run(tmp_path, "r1", "c", "naive_vector", 0.9, key_a)
    _write_run(tmp_path, "r2", "c", "naive_vector", 0.7, key_a)
    _write_run(tmp_path, "r3", "c", "naive_vector", 0.1, key_b)
    _write_run(tmp_path, "r4", "c", "naive_vector", 0.5, None)  # legacy, no manifest

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    rows = {row["compatibility_key"]: row for row in board}
    assert set(rows) == {"aaaaaaaaaaaa", "bbbbbbbbbbbb", mf.LEGACY_KEY}
    assert rows["aaaaaaaaaaaa"]["runs"] == 2
    assert rows["aaaaaaaaaaaa"]["mean_accuracy"] == 0.8
    assert rows["bbbbbbbbbbbb"]["runs"] == 1
    assert rows["bbbbbbbbbbbb"]["mean_accuracy"] == 0.1
    assert rows["aaaaaaaaaaaa"]["mixed_with"] == ["bbbbbbbbbbbb", mf.LEGACY_KEY]
    assert rows["aaaaaaaaaaaa"]["manifest"]["judge_model"] == "judge-a"


def test_benchmark_runs_option_repeats_the_run(monkeypatch):
    from typer.testing import CliRunner

    from kb_arena.cli import app

    calls: list[dict] = []

    async def fake_run_benchmark(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("kb_arena.benchmark.runner.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    result = CliRunner().invoke(app, ["benchmark", "--corpus", "aws-compute", "--runs", "3"])

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
