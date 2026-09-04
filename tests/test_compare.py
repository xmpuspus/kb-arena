"""Two strategies pair question by question, with a CI and a win count, never just two means."""

from __future__ import annotations

import asyncio
import json

import pytest

from kb_arena.benchmark import compare as cmp
from kb_arena.chatbot import api
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, Score
from kb_arena.settings import settings


def test_paired_compare_reports_deltas_wins_and_a_ci():
    a = {f"q{i}": 0.5 for i in range(10)}
    b = {f"q{i}": (0.9 if i < 7 else 0.5) for i in range(10)}
    b["q9"] = 0.1

    result = cmp.paired_compare(a, b, metric="accuracy", label_a="x", label_b="y")

    assert result["n_paired"] == 10
    assert (result["wins"], result["ties"], result["losses"]) == (7, 2, 1)
    assert result["mean_delta"] == pytest.approx((0.4 * 7 - 0.4) / 10)
    low, high = result["delta_ci_95"]
    assert low <= result["mean_delta"] <= high
    assert 0.0 < result["wilcoxon_p"] <= 1.0
    assert result["effect_size_d"] > 0
    assert result["per_question"][0] == {
        "question_id": "q0",
        "a": 0.5,
        "b": 0.9,
        "delta": pytest.approx(0.4),
    }


def test_effect_size_uses_the_sample_sd_and_is_none_for_a_uniform_shift():
    deltas = [0.4, 0.1, 0.0, 0.3, 0.0, 0.3, -0.1, 0.3, 0.1, 0.2]
    a = {f"q{i}": 0.5 for i in range(10)}
    b = {f"q{i}": 0.5 + d for i, d in enumerate(deltas)}
    result = cmp.paired_compare(a, b, metric="accuracy", label_a="x", label_b="y")
    import statistics

    assert result["effect_size_d"] == pytest.approx(
        statistics.fmean(deltas) / statistics.stdev(deltas)
    )

    uniform = cmp.paired_compare(
        {f"q{i}": 0.2 for i in range(6)},
        {f"q{i}": 0.6 for i in range(6)},
        metric="accuracy",
        label_a="x",
        label_b="y",
    )
    assert uniform["effect_size_d"] is None
    assert uniform["mean_delta"] == pytest.approx(0.4)


def test_too_few_pairs_never_fire_a_flag():
    result = cmp.paired_compare(
        {"q1": 0.2}, {"q1": 0.9}, metric="accuracy", label_a="x", label_b="y"
    )
    assert result["n_paired"] == 1
    assert result["ci_excludes_zero"] is False
    assert result["significant"] is False
    assert result["enough_pairs_for_inference"] is False


def test_no_shared_questions_raises():
    with pytest.raises(ValueError, match="no shared questions"):
        cmp.paired_compare({"q1": 0.2}, {"q2": 0.9}, metric="accuracy", label_a="x", label_b="y")


def test_significant_needs_b_to_be_better_in_the_metric_direction():
    a = {f"q{i}": 100.0 for i in range(10)}
    b = {f"q{i}": 100.0 + 50.0 + i for i in range(10)}  # b is slower on every question
    result = cmp.paired_compare(a, b, metric="latency_ms", label_a="x", label_b="y")
    assert result["wilcoxon_p"] < 0.05
    assert result["significant"] is False
    assert result["losses"] == 10


def test_identical_scores_give_no_difference():
    a = {"q1": 0.3, "q2": 0.7}

    result = cmp.paired_compare(a, dict(a), metric="accuracy", label_a="x", label_b="y")

    assert result["mean_delta"] == 0.0
    assert result["delta_ci_95"] == [0.0, 0.0]
    assert result["wilcoxon_p"] == 1.0
    assert result["significant"] is False
    assert result["ci_excludes_zero"] is False
    assert (result["wins"], result["ties"], result["losses"]) == (0, 2, 0)


def test_lower_is_better_flips_the_win_count():
    a = {"q1": 100.0, "q2": 200.0}
    b = {"q1": 50.0, "q2": 250.0}

    result = cmp.paired_compare(a, b, metric="latency_ms", label_a="x", label_b="y")

    assert result["lower_is_better"] is True
    assert (result["wins"], result["losses"]) == (1, 1)


def test_only_shared_questions_pair_and_the_rest_are_counted():
    result = cmp.paired_compare(
        {"q1": 1.0, "q2": 0.0}, {"q2": 1.0, "q3": 1.0}, metric="accuracy", label_a="x", label_b="y"
    )

    assert result["n_paired"] == 1
    assert (result["unpaired_a"], result["unpaired_b"]) == (1, 1)


def _write(results, run_id, corpus, strategy, scores: dict[str, float], manifest=None, errors=()):
    run_dir = results / f"run_{run_id}"
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
                latency_ms=0.0 if q in errors else 10.0,
                is_error=q in errors,
            )
            for q, v in scores.items()
        ],
    )
    data = json.loads(bench.model_dump_json())
    if manifest is not None:
        data["manifest"] = manifest
    (run_dir / f"{corpus}_{strategy}.json").write_text(json.dumps(data))
    (results / f"{corpus}_{strategy}.json").write_text(json.dumps(data))


def test_compare_result_files_flags_a_different_compatibility_key(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2, "q2": 0.4}, {"compatibility_key": "k1"})
    _write(tmp_path, "r2", "c", "naive", {"q1": 0.6, "q2": 0.4}, {"compatibility_key": "k2"})

    result = cmp.compare_result_files(tmp_path / "c_bm25.json", tmp_path / "c_naive.json")

    assert result["meta"]["comparable"] is False
    assert any("compatibility key" in r for r in result["meta"]["reasons"])
    assert result["mean_delta"] == pytest.approx(0.2)


def test_compare_result_files_on_the_same_key_is_clean(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2, "q2": 0.4}, {"compatibility_key": "k1"})
    _write(tmp_path, "r1", "c", "naive", {"q1": 0.6, "q2": 0.4}, {"compatibility_key": "k1"})

    result = cmp.compare_result_files(
        tmp_path / "c_bm25.json", tmp_path / "c_naive.json", metric="latency_ms"
    )

    assert result["meta"]["comparable"] is True
    assert result["metric"] == "latency_ms"
    assert result["ties"] == 2


def test_one_side_without_a_manifest_is_not_a_clean_comparison(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2}, {"compatibility_key": "k1"})
    _write(tmp_path, "r1", "c", "naive", {"q1": 0.6})

    result = cmp.compare_result_files(tmp_path / "c_bm25.json", tmp_path / "c_naive.json")

    assert result["meta"]["comparable"] is False
    assert any("b carries a manifest" in r or "unchecked" in r for r in result["meta"]["reasons"])


def test_an_error_record_leaves_the_pairing_instead_of_winning_on_latency(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.5, "q2": 0.5}, {"compatibility_key": "k"})
    _write(
        tmp_path,
        "r1",
        "c",
        "naive",
        {"q1": 0.0, "q2": 0.5},
        {"compatibility_key": "k"},
        errors=("q1",),
    )

    result = cmp.compare_result_files(
        tmp_path / "c_bm25.json", tmp_path / "c_naive.json", metric="latency_ms"
    )

    assert result["n_paired"] == 1
    assert (result["wins"], result["ties"], result["losses"]) == (0, 1, 0)
    assert result["meta"]["b"]["error_records"] == 1
    assert result["meta"]["comparable"] is False
    assert any("error records in b" in r for r in result["meta"]["reasons"])


def test_lab_rows_without_a_question_id_never_pair(tmp_path):
    rows = [
        {"strategy": "bm25", "ndcg_at_k": 0.5},
        {"strategy": "naive", "ndcg_at_k": 0.9},
        {"strategy": "bm25", "question_id": "q1", "ndcg_at_k": 0.5},
        {"strategy": "naive", "question_id": "q1", "ndcg_at_k": 0.9},
    ]
    path = tmp_path / "retriever_lab.json"
    path.write_text(json.dumps({"questions": rows}))

    result = cmp.compare_lab(path, "bm25", "naive")

    assert result["n_paired"] == 1
    assert result["per_question"][0]["question_id"] == "q1"


def test_non_finite_and_duplicate_records_are_handled(tmp_path):
    _write(
        tmp_path, "r1", "c", "bm25", {"q1": 0.2, "q2": 0.4, "q3": 0.5}, {"compatibility_key": "k"}
    )
    _write(
        tmp_path, "r1", "c", "naive", {"q1": 0.6, "q2": 0.4, "q3": 0.5}, {"compatibility_key": "k"}
    )
    path = tmp_path / "c_naive.json"
    data = json.loads(path.read_text())
    data["records"][2]["score"]["accuracy"] = float("nan")
    data["records"].append(dict(data["records"][0]))  # q1 twice
    path.write_text(json.dumps(data))

    result = cmp.compare_result_files(tmp_path / "c_bm25.json", path)

    assert result["n_paired"] == 2  # the NaN row left the pairing
    assert result["meta"]["b"]["duplicate_records"] == 1
    assert result["meta"]["comparable"] is False
    assert any("duplicate" in r for r in result["meta"]["reasons"])
    assert "path" not in result["meta"]["a"]
    assert result["meta"]["a"]["file"] == "c_bm25.json"


def test_a_file_without_a_corpus_is_not_a_clean_comparison(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2}, {"compatibility_key": "k"})
    _write(tmp_path, "r1", "c", "naive", {"q1": 0.6}, {"compatibility_key": "k"})
    for name in ("c_bm25.json", "c_naive.json"):
        data = json.loads((tmp_path / name).read_text())
        data["corpus"] = ""
        (tmp_path / name).write_text(json.dumps(data))

    result = cmp.compare_result_files(tmp_path / "c_bm25.json", tmp_path / "c_naive.json")

    assert result["meta"]["comparable"] is False
    assert any("names no corpus" in r for r in result["meta"]["reasons"])


def test_a_bad_metric_name_is_rejected(tmp_path):
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2}, {"compatibility_key": "k"})
    with pytest.raises(ValueError, match="invalid metric"):
        cmp.benchmark_scores(tmp_path / "c_bm25.json", "../../tmp/pwned")


def test_resolve_result_path_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        cmp.resolve_result_path(tmp_path, "../etc", "bm25", None)
    with pytest.raises(ValueError):
        cmp.resolve_result_path(tmp_path, "c", "bm25", "r1/../../x")
    with pytest.raises(ValueError):
        cmp.resolve_result_path(tmp_path, "c", "bm25\n", "r1")
    with pytest.raises(ValueError):
        cmp.resolve_result_path(tmp_path, "c", "..", "r1")
    assert (
        cmp.resolve_result_path(tmp_path, "c", "bm25", "r1") == tmp_path / "run_r1" / "c_bm25.json"
    )
    assert cmp.resolve_result_path(tmp_path, "c", "bm25", None) == tmp_path / "c_bm25.json"


def test_compare_lab_pairs_two_strategies_inside_one_run(tmp_path):
    rows = []
    for q, (x, y) in {"q1": (0.5, 0.9), "q2": (0.5, 0.5), "q3": (0.7, 0.2)}.items():
        rows.append({"strategy": "bm25", "question_id": q, "ndcg_at_k": x, "recall_at_k": 1.0})
        rows.append({"strategy": "naive", "question_id": q, "ndcg_at_k": y, "recall_at_k": 1.0})
    rows.append({"strategy": "naive", "question_id": "q4", "error": "boom"})
    path = tmp_path / "retriever_lab.json"
    path.write_text(json.dumps({"corpus": "c", "questions": rows}))

    result = cmp.compare_lab(path, "bm25", "naive")

    assert result["n_paired"] == 3
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert result["unpaired_b"] == 0  # the error row carried no metric, so it never paired
    assert result["meta"]["comparable"] is True
    assert result["meta"]["file"] == "retriever_lab.json"


def test_the_api_route_validates_ids_and_finds_files(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2}, {"compatibility_key": "k"})
    _write(tmp_path, "r1", "c", "naive", {"q1": 0.6}, {"compatibility_key": "k"})

    with pytest.raises(HTTPException) as bad:
        asyncio.run(api.compare_strategies(corpus="../c", a="bm25", b="naive"))
    assert bad.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        asyncio.run(api.compare_strategies(corpus="c", a="bm25", b="nope"))
    assert missing.value.status_code == 404
    (tmp_path / "c_dir.json").mkdir()
    with pytest.raises(HTTPException) as directory:
        asyncio.run(api.compare_strategies(corpus="c", a="bm25", b="dir"))
    assert directory.value.status_code == 400

    result = asyncio.run(
        api.compare_strategies(corpus="c", a="bm25", b="naive", run_a="r1", run_b="r1")
    )
    assert result["mean_delta"] == pytest.approx(0.4)
    assert result["meta"]["comparable"] is True


def test_the_cli_writes_the_artifact(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kb_arena.cli import app

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    _write(tmp_path, "r1", "c", "bm25", {"q1": 0.2, "q2": 0.2}, {"compatibility_key": "k"})
    _write(tmp_path, "r1", "c", "naive", {"q1": 0.6, "q2": 0.2}, {"compatibility_key": "k"})

    result = CliRunner().invoke(app, ["compare", "--corpus", "c", "--a", "bm25", "--b", "naive"])

    assert result.exit_code == 0, result.output
    artifact = json.loads(
        (tmp_path / "compare_c_bm25@latest_vs_naive@latest_accuracy.json").read_text()
    )
    assert artifact["n_paired"] == 2
    assert "Comparable" in result.output

    bad = CliRunner().invoke(
        app, ["compare", "--corpus", "c", "--a", "bm25", "--b", "naive", "--metric", "../x"]
    )
    assert bad.exit_code == 1
    lab_with_run = CliRunner().invoke(
        app,
        [
            "compare",
            "--lab",
            str(tmp_path / "x.json"),
            "--a",
            "bm25",
            "--b",
            "naive",
            "--run-a",
            "r1",
        ],
    )
    assert lab_with_run.exit_code == 1
