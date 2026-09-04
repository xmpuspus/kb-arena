"""A result file names what it measured, and the leaderboard never blends runs that differ."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from kb_arena.benchmark import manifest as mf
from kb_arena.chatbot import api
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, Score
from kb_arena.settings import settings


def _q(
    qid: str,
    text: str,
    chunks: list[str] | None = None,
    split: str = "development",
    answer: str = "a",
):
    return SimpleNamespace(
        id=qid,
        question=text,
        expected_chunks=chunks or [],
        split=split,
        ground_truth={"answer": answer},
    )


def test_question_fingerprint_ignores_order_and_tracks_content():
    a = [_q("q1", "one", ["c1"]), _q("q2", "two")]
    b = [_q("q2", "two"), _q("q1", "one", ["c1"])]
    c = [_q("q1", "one changed", ["c1"]), _q("q2", "two")]

    assert mf.question_set_fingerprint(a) == mf.question_set_fingerprint(b)
    assert mf.question_set_fingerprint(a) != mf.question_set_fingerprint(c)


def test_question_fingerprint_tracks_the_answer_key_the_judge_reads():
    from kb_arena.models.benchmark import GroundTruth, Question

    def make(answer: str, entities: list[str]):
        return Question(
            id="q1",
            tier=1,
            type="factoid",
            hops=1,
            question="one",
            ground_truth=GroundTruth(answer=answer, required_entities=entities),
        )

    base = mf.question_set_fingerprint([make("a", ["x"])])
    assert base != mf.question_set_fingerprint([make("b", ["x"])])
    assert base != mf.question_set_fingerprint([make("a", ["x", "y"])])
    assert base == mf.question_set_fingerprint([make("a", ["x"])])


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


def test_the_answering_model_is_part_of_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    questions = [_q("q1", "one")]
    monkeypatch.setattr(settings, "generate_model", "model-a")
    a = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)
    monkeypatch.setattr(settings, "generate_model", "model-b")
    b = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)
    assert a["compatibility_key"] != b["compatibility_key"]
    assert a["generation"]["model"] == "model-a"


def test_a_reference_free_run_ignores_the_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    questions = [_q("q1", "one")]
    monkeypatch.setattr(settings, "judge_model", "judge-a")
    a = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=True)
    monkeypatch.setattr(settings, "judge_model", "judge-b")
    b = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=True)
    assert a["compatibility_key"] == b["compatibility_key"]
    assert a["judge"] is None
    judged = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)
    assert judged["compatibility_key"] != a["compatibility_key"]


def test_a_comment_in_the_qrels_file_keeps_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    qrels = tmp_path / "c" / "questions"
    qrels.mkdir(parents=True)
    (qrels / "expected_chunks.yaml").write_text("q1: [c1]\n")
    before = mf.qrels_fingerprint("c")
    (qrels / "expected_chunks.yaml").write_text("# reviewed 2026-09-04\nq1:\n  - c1\n")
    assert mf.qrels_fingerprint("c") == before
    (qrels / "expected_chunks.yaml").write_text("q1: [c1, c2]\n")
    assert mf.qrels_fingerprint("c") != before


def test_a_cost_capped_partial_run_never_blends_with_a_full_one(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    manifest = mf.build_manifest(
        "c", [_q("q1", "one"), _q("q2", "two")], top_k=5, split="", reference_free=False
    )
    full = {"manifest": manifest, "records": [{}, {}]}
    short = {"manifest": manifest, "records": [{}]}
    # The cap stopped this run after the last question, so nothing is missing.
    capped_but_whole = {"manifest": manifest, "records": [{}, {}], "stopped_by_cost_cap": True}
    capped_early = {"manifest": manifest, "records": [{}], "stopped_by_cost_cap": True}
    assert mf.compatibility_key(full) == manifest["compatibility_key"]
    # The scored count rides in the suffix, so two partial runs that stopped at
    # different points never read as repeats of one experiment.
    assert mf.compatibility_key(short) == manifest["compatibility_key"] + "-partial-1"
    assert mf.compatibility_key(capped_but_whole) == manifest["compatibility_key"], (
        "a run that scored every question compares with every other whole run, "
        "whatever stopped it afterwards"
    )
    assert mf.compatibility_key(capped_early) == manifest["compatibility_key"] + "-partial-1"


def test_a_v1_file_dumped_by_the_v2_model_still_summarises_empty():
    assert mf.manifest_summary({"manifest": {}, "schema_version": 1}) == {}
    assert mf.manifest_summary({"manifest": {"compatibility_key": "x"}}) == {}


def test_a_reference_free_manifest_has_no_judge_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    free = mf.build_manifest("c", [_q("q1", "one")], top_k=5, split="", reference_free=True)
    judged = mf.build_manifest("c", [_q("q1", "one")], top_k=5, split="", reference_free=False)
    assert mf.judge_provider_of(free) == ""
    assert mf.judge_provider_of(judged) == settings.llm_provider
    assert mf.judge_provider_of({}) == ""


def test_an_empty_fingerprint_is_not_a_stamped_manifest():
    blank = {"manifest": {"question_set_fingerprint": "", "top_k": 5}}
    assert mf.compatibility_key(blank) == mf.LEGACY_KEY
    assert mf.manifest_summary(blank) == {}


def test_a_bad_top_level_copy_does_not_hide_the_run_directory_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    key = {"question_set_fingerprint": "f1", "judge": {"model": "j"}, "top_k": 5}
    _write_run(tmp_path, "r1", "c", "bm25", 1.0, key)
    good = json.loads((tmp_path / "run_r1" / "c_bm25.json").read_text())
    good["total_cost_usd"] = "not a number"  # the top-level copy is corrupt
    (tmp_path / "c_bm25.json").write_text(json.dumps(good))

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    assert len(board) == 1
    assert board[0]["runs"] == 1
    assert board[0]["mean_accuracy"] == 1.0


def test_a_result_file_without_a_manifest_is_legacy():
    assert mf.compatibility_key({"corpus": "c", "strategy": "s"}) == mf.LEGACY_KEY
    assert mf.compatibility_key({"manifest": {"compatibility_key": "abc123"}}) == mf.LEGACY_KEY
    assert mf.compatibility_key({"manifest": {"compatibility_key": ""}}) == mf.LEGACY_KEY
    assert mf.compatibility_key({"manifest": 7}) == mf.LEGACY_KEY


def test_the_reader_recomputes_the_key_and_ignores_a_stored_one(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    manifest = mf.build_manifest(
        "c", [_q("q1", "one")], top_k=5, split="development", reference_free=False
    )
    assert mf.compatibility_key({"manifest": manifest}) == manifest["compatibility_key"]

    edited = {**manifest, "compatibility_key": "tampered"}
    assert mf.compatibility_key({"manifest": edited}) == manifest["compatibility_key"]

    other_k = {**manifest, "top_k": 10}
    assert mf.compatibility_key({"manifest": other_k}) != manifest["compatibility_key"]


def test_ragas_changes_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    questions = [_q("q1", "one")]
    monkeypatch.setattr(settings, "benchmark_enable_ragas", False)
    plain = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)
    monkeypatch.setattr(settings, "benchmark_enable_ragas", True)
    ragas = mf.build_manifest("c", questions, top_k=5, split="development", reference_free=False)
    assert plain["compatibility_key"] != ragas["compatibility_key"]


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
    key_a = {"question_set_fingerprint": "f1", "judge": {"model": "judge-a"}, "top_k": 5}
    key_b = {"question_set_fingerprint": "f1", "judge": {"model": "judge-b"}, "top_k": 5}
    ka, kb = mf._digest(mf.core_of(key_a)), mf._digest(mf.core_of(key_b))
    _write_run(tmp_path, "r1", "c", "naive_vector", 0.9, key_a)
    _write_run(tmp_path, "r2", "c", "naive_vector", 0.7, key_a)
    _write_run(tmp_path, "r3", "c", "naive_vector", 0.1, key_b)
    _write_run(tmp_path, "r4", "c", "naive_vector", 0.5, None)  # legacy, no manifest

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    rows = {row["compatibility_key"]: row for row in board}
    assert set(rows) == {ka, kb, mf.LEGACY_KEY}
    assert rows[ka]["runs"] == 2
    assert rows[ka]["mean_accuracy"] == 0.8
    assert rows[kb]["runs"] == 1
    assert rows[kb]["mean_accuracy"] == 0.1
    assert rows[ka]["mixed_with"] == sorted([kb, mf.LEGACY_KEY])
    assert rows[ka]["manifest"]["judge_model"] == "judge-a"


def test_a_run_written_twice_counts_once(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    key = {"question_set_fingerprint": "f1", "judge": {"model": "j"}, "top_k": 5}
    _write_run(tmp_path, "r1", "c", "bm25", 0.0, key)
    _write_run(tmp_path, "r2", "c", "bm25", 1.0, key)
    # the runner also leaves the newest run at the top level
    (tmp_path / "c_bm25.json").write_text((tmp_path / "run_r2" / "c_bm25.json").read_text())

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    assert len(board) == 1
    assert board[0]["runs"] == 2
    assert board[0]["mean_accuracy"] == 0.5


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


def test_two_partial_runs_of_different_sizes_never_share_a_key():
    """A run that scored 10 questions and one that scored 70 are not repeats."""
    manifest = {
        "schema_version": 1,
        "corpus": "c",
        "question_split": "all",
        "question_set_fingerprint": "fp",
        "qrels_fingerprint": None,
        "generation": {},
        "scoring": {"reference_free": True, "ragas": False},
        "judge": None,
        "embedding": {},
        "chunk": {"tokens": 512, "overlap_tokens": 64},
        "top_k": 5,
        "question_count": 70,
    }
    ten = {"manifest": manifest, "records": [{}] * 10, "stopped_by_cost_cap": True}
    seventy = {"manifest": manifest, "records": [{}] * 70}

    assert mf.compatibility_key(ten) != mf.compatibility_key(seventy)
    assert mf.compatibility_key(ten).endswith("-partial-10")
    # A whole run carries no suffix at all.
    assert "partial" not in mf.compatibility_key(seventy)
