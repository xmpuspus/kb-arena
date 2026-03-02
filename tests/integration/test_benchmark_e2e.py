"""Integration test: benchmark pipeline — questions → evaluator → reporter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import yaml

from kb_arena.benchmark.evaluator import _structural_check, evaluate
from kb_arena.benchmark.reporter import _build_markdown, _build_summary
from kb_arena.models.benchmark import (
    AnswerRecord,
    BenchmarkResult,
    Constraints,
    GroundTruth,
    Question,
    Score,
)

# ---------------------------------------------------------------------------
# Sample questions for testing
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS_YAML = [
    {
        "id": "py-t1-001",
        "tier": 1,
        "type": "factoid",
        "hops": 1,
        "question": "What does json.loads do?",
        "ground_truth": {
            "answer": "json.loads deserializes a JSON string to a Python object.",
            "source_refs": ["json.html"],
            "required_entities": ["json.loads"],
        },
        "constraints": {
            "must_mention": ["json.loads", "deserializ"],
            "must_not_claim": ["encodes", "writes to file"],
            "max_tokens": 200,
        },
    },
    {
        "id": "py-t1-002",
        "tier": 1,
        "type": "factoid",
        "hops": 1,
        "question": "What does os.path.join do?",
        "ground_truth": {
            "answer": "os.path.join joins path components intelligently.",
            "source_refs": ["os.html"],
            "required_entities": ["os.path.join"],
        },
        "constraints": {
            "must_mention": ["path"],
            "must_not_claim": ["encodes JSON"],
            "max_tokens": 200,
        },
    },
    {
        "id": "py-t2-001",
        "tier": 2,
        "type": "comparison",
        "hops": 2,
        "question": "How does json.loads differ from json.load?",
        "ground_truth": {
            "answer": "json.loads reads from a string, json.load reads from a file object.",
            "source_refs": ["json.html"],
            "required_entities": ["json.loads", "json.load"],
        },
        "constraints": {
            "must_mention": ["string", "file"],
            "must_not_claim": [],
            "max_tokens": 300,
        },
    },
    {
        "id": "py-t3-001",
        "tier": 3,
        "type": "relational",
        "hops": 3,
        "question": "What exceptions can json.loads raise?",
        "ground_truth": {
            "answer": "json.loads raises JSONDecodeError when the input is not valid JSON.",
            "source_refs": ["json.html"],
            "required_entities": ["JSONDecodeError"],
        },
        "constraints": {
            "must_mention": ["JSONDecodeError"],
            "must_not_claim": ["FileNotFoundError"],
            "max_tokens": 300,
        },
    },
    {
        "id": "py-t4-001",
        "tier": 4,
        "type": "causal",
        "hops": 4,
        "question": "Why does json.loads raise ValueError on some inputs?",
        "ground_truth": {
            "answer": "JSONDecodeError is a subclass of ValueError, raised when parsing fails.",
            "source_refs": ["json.html"],
            "required_entities": ["ValueError", "JSONDecodeError"],
        },
        "constraints": {
            "must_mention": ["ValueError"],
            "must_not_claim": [],
            "max_tokens": 400,
        },
    },
]


@pytest.fixture
def questions():
    return [Question.model_validate(q) for q in SAMPLE_QUESTIONS_YAML]


@pytest.fixture
def questions_yaml_dir(tmp_path):
    q_dir = tmp_path / "datasets" / "python-stdlib" / "questions"
    q_dir.mkdir(parents=True)
    (q_dir / "tier1.yaml").write_text(yaml.dump(SAMPLE_QUESTIONS_YAML[:2]), encoding="utf-8")
    (q_dir / "tier2_3_4.yaml").write_text(yaml.dump(SAMPLE_QUESTIONS_YAML[2:]), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def test_questions_load_from_yaml(questions_yaml_dir, monkeypatch):
    from kb_arena.benchmark.questions import load_questions

    monkeypatch.setattr(
        "kb_arena.benchmark.questions.settings.datasets_path", str(questions_yaml_dir / "datasets")
    )
    qs = load_questions("python-stdlib")
    assert len(qs) == 5


def test_questions_tier_filter(questions_yaml_dir, monkeypatch):
    from kb_arena.benchmark.questions import load_questions

    monkeypatch.setattr(
        "kb_arena.benchmark.questions.settings.datasets_path", str(questions_yaml_dir / "datasets")
    )
    qs = load_questions("python-stdlib", tier=1)
    assert all(q.tier == 1 for q in qs)
    assert len(qs) == 2


def test_questions_type_filter(questions_yaml_dir, monkeypatch):
    from kb_arena.benchmark.questions import load_questions

    monkeypatch.setattr(
        "kb_arena.benchmark.questions.settings.datasets_path", str(questions_yaml_dir / "datasets")
    )
    qs = load_questions("python-stdlib", question_type="factoid")
    assert all(q.type == "factoid" for q in qs)
    assert len(qs) == 2


def test_missing_corpus_raises(tmp_path, monkeypatch):
    from kb_arena.benchmark.questions import load_questions

    monkeypatch.setattr(
        "kb_arena.benchmark.questions.settings.datasets_path", str(tmp_path / "datasets")
    )
    with pytest.raises(FileNotFoundError):
        load_questions("nonexistent-corpus")


# ---------------------------------------------------------------------------
# Structural evaluator
# ---------------------------------------------------------------------------


def test_structural_check_all_mentions_present():
    constraints = Constraints(must_mention=["json.loads", "deserializ"], must_not_claim=[])
    score = _structural_check(
        "json.loads deserializes a JSON string to a Python object.", constraints
    )
    assert score.structural_pass is True
    assert score.accuracy == 1.0
    assert set(score.mentions_found) == {"json.loads", "deserializ"}


def test_structural_check_partial_mentions():
    constraints = Constraints(must_mention=["json.loads", "unicode"], must_not_claim=[])
    score = _structural_check("json.loads parses JSON.", constraints)
    assert score.structural_pass is True
    assert score.accuracy == 0.5  # 1/2 mentions found


def test_structural_check_false_claim_kills_accuracy():
    constraints = Constraints(must_mention=["json.loads"], must_not_claim=["encodes"])
    score = _structural_check("json.loads encodes data to JSON.", constraints)
    assert score.structural_pass is False
    assert score.accuracy == 0.0
    assert "encodes" in score.false_claims


def test_structural_check_no_constraints():
    constraints = Constraints()
    score = _structural_check("Any answer here.", constraints)
    assert score.structural_pass is True
    assert score.accuracy == 1.0


def test_structural_check_empty_answer_fails_mentions():
    constraints = Constraints(must_mention=["json.loads"], must_not_claim=[])
    score = _structural_check("", constraints)
    assert score.structural_pass is True
    assert score.accuracy == 0.0
    assert score.mentions_found == []


@pytest.mark.asyncio
async def test_evaluate_skips_llm_on_structural_fail():
    constraints = Constraints(must_mention=[], must_not_claim=["wrong claim"])
    ground_truth = GroundTruth(answer="correct answer")
    mock_llm = AsyncMock()

    score = await evaluate("this contains wrong claim", ground_truth, constraints, llm=mock_llm)

    assert score.structural_pass is False
    assert score.accuracy == 0.0
    mock_llm.judge.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate_calls_llm_on_structural_pass():
    constraints = Constraints(must_mention=["json.loads"], must_not_claim=[])
    ground_truth = GroundTruth(answer="json.loads parses JSON strings.")
    mock_llm = AsyncMock()
    mock_llm.judge.return_value = '{"accuracy": 0.9, "completeness": 0.8, "faithfulness": 1.0}'

    score = await evaluate(
        "json.loads parses JSON strings into Python objects.",
        ground_truth,
        constraints,
        llm=mock_llm,
    )

    assert score.structural_pass is True
    mock_llm.judge.assert_called_once()
    assert score.accuracy == pytest.approx(0.9)
    assert score.completeness == pytest.approx(0.8)
    assert score.faithfulness == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_evaluate_graceful_on_llm_failure():
    constraints = Constraints(must_mention=["json.loads"], must_not_claim=[])
    ground_truth = GroundTruth(answer="json.loads parses JSON.")
    mock_llm = AsyncMock()
    mock_llm.judge.side_effect = RuntimeError("API timeout")

    score = await evaluate("json.loads parses JSON data.", ground_truth, constraints, llm=mock_llm)

    # Structural score should stand
    assert score.structural_pass is True
    assert score.accuracy >= 0.0


@pytest.mark.asyncio
async def test_evaluate_without_llm():
    constraints = Constraints(must_mention=["json.loads"], must_not_claim=[])
    ground_truth = GroundTruth(answer="json.loads parses JSON.")

    score = await evaluate("json.loads parses JSON data.", ground_truth, constraints, llm=None)

    assert score.structural_pass is True
    assert score.accuracy >= 0.0


# ---------------------------------------------------------------------------
# Accuracy by tier calculation
# ---------------------------------------------------------------------------


def _make_record(
    question_id: str, score_val: float, strategy: str = "naive_vector"
) -> AnswerRecord:
    return AnswerRecord(
        question_id=question_id,
        strategy=strategy,
        answer="some answer",
        score=Score(accuracy=score_val, completeness=score_val, faithfulness=1.0),
        latency_ms=100.0,
        cost_usd=0.001,
    )


def test_aggregate_accuracy_by_tier():
    from kb_arena.benchmark.runner import _aggregate

    bench = BenchmarkResult(corpus="python-stdlib", strategy="naive_vector")
    bench.records = [
        _make_record("py-t1-001", 1.0),
        _make_record("py-t1-002", 0.5),
        _make_record("py-t2-001", 0.8),
        _make_record("py-t3-001", 0.2),
    ]

    result = _aggregate(bench)

    assert 1 in result.accuracy_by_tier
    assert 2 in result.accuracy_by_tier
    assert 3 in result.accuracy_by_tier
    assert result.accuracy_by_tier[1] == pytest.approx(0.75)
    assert result.accuracy_by_tier[2] == pytest.approx(0.8)
    assert result.accuracy_by_tier[3] == pytest.approx(0.2)


def test_aggregate_avg_latency():
    from kb_arena.benchmark.runner import _aggregate

    bench = BenchmarkResult(corpus="python-stdlib", strategy="naive_vector")
    bench.records = [
        _make_record("py-t1-001", 1.0),
        _make_record("py-t1-002", 1.0),
    ]
    bench.records[0].latency_ms = 100.0
    bench.records[1].latency_ms = 200.0

    result = _aggregate(bench)
    assert result.avg_latency_ms == pytest.approx(150.0)


def test_aggregate_cost_per_correct():
    from kb_arena.benchmark.runner import _aggregate

    bench = BenchmarkResult(corpus="python-stdlib", strategy="naive_vector")
    bench.records = [
        _make_record("py-t1-001", 1.0),  # correct (>= 0.7)
        _make_record("py-t1-002", 0.3),  # incorrect
    ]
    bench.records[0].cost_usd = 0.01
    bench.records[1].cost_usd = 0.01

    result = _aggregate(bench)
    assert result.total_cost_usd == pytest.approx(0.02)
    # 1 correct answer at total cost 0.02
    assert result.cost_per_correct == pytest.approx(0.02)


def test_aggregate_question_id_without_tier():
    """Question IDs that don't match pattern should go to tier 0."""
    from kb_arena.benchmark.runner import _aggregate

    bench = BenchmarkResult(corpus="python-stdlib", strategy="naive_vector")
    bench.records = [_make_record("no-tier-in-id", 0.9)]

    result = _aggregate(bench)
    assert 0 in result.accuracy_by_tier


# ---------------------------------------------------------------------------
# Reporter: markdown and summary
# ---------------------------------------------------------------------------


def _make_bench_result(corpus: str, strategy: str, tier_accuracy: dict) -> BenchmarkResult:
    records = []
    for tier, acc in tier_accuracy.items():
        records.append(_make_record(f"py-t{tier}-001", acc, strategy))

    bench = BenchmarkResult(
        corpus=corpus,
        strategy=strategy,
        accuracy_by_tier=tier_accuracy,
        avg_latency_ms=150.0,
        total_cost_usd=0.05,
        cost_per_correct=0.025,
        total_questions=len(records),
        records=records,
    )
    return bench


def test_reporter_generates_markdown():
    results = [
        _make_bench_result("python-stdlib", "naive_vector", {1: 0.6, 2: 0.4}),
        _make_bench_result("python-stdlib", "knowledge_graph", {1: 0.9, 2: 0.8}),
    ]
    md = _build_markdown(results)

    assert "# KB Arena Benchmark Report" in md
    assert "python-stdlib" in md
    assert "naive_vector" in md
    assert "knowledge_graph" in md
    assert "Accuracy" in md


def test_reporter_markdown_has_tables():
    results = [_make_bench_result("python-stdlib", "naive_vector", {1: 0.75})]
    md = _build_markdown(results)
    # Should contain markdown table separators
    assert "|" in md
    assert "---" in md


def test_reporter_markdown_includes_tier_table():
    results = [_make_bench_result("python-stdlib", "naive_vector", {1: 0.6, 2: 0.5, 3: 0.4})]
    md = _build_markdown(results)
    assert "Tier 1" in md or "Factoid" in md


def test_reporter_formats_percentage():
    results = [_make_bench_result("python-stdlib", "naive_vector", {1: 0.75})]
    md = _build_markdown(results)
    assert "75.0%" in md


def test_reporter_summary_json():
    results = [
        _make_bench_result("python-stdlib", "naive_vector", {1: 0.6, 2: 0.4}),
        _make_bench_result("python-stdlib", "knowledge_graph", {1: 0.9, 2: 0.8}),
    ]
    summary = _build_summary(results)

    assert "corpora" in summary
    assert "python-stdlib" in summary["corpora"]
    assert "naive_vector" in summary["corpora"]["python-stdlib"]
    assert "knowledge_graph" in summary["corpora"]["python-stdlib"]

    naive = summary["corpora"]["python-stdlib"]["naive_vector"]
    assert "accuracy_overall" in naive
    assert "accuracy_by_tier" in naive
    assert "avg_latency_ms" in naive
    assert "total_cost_usd" in naive
    assert "total_questions" in naive


def test_reporter_summary_accuracy_overall_is_average():
    results = [_make_bench_result("python-stdlib", "naive_vector", {1: 0.6, 2: 0.4})]
    summary = _build_summary(results)
    overall = summary["corpora"]["python-stdlib"]["naive_vector"]["accuracy_overall"]
    assert overall == pytest.approx(0.5, abs=0.01)


def test_reporter_handles_multiple_corpora():
    results = [
        _make_bench_result("python-stdlib", "naive_vector", {1: 0.7}),
        _make_bench_result("kubernetes", "naive_vector", {1: 0.5}),
    ]
    md = _build_markdown(results)
    assert "python-stdlib" in md
    assert "kubernetes" in md


def test_reporter_writes_files(tmp_path, monkeypatch):
    from kb_arena.benchmark.reporter import generate_report

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr("kb_arena.benchmark.reporter.settings.results_path", str(results_dir))

    # Write a fake result file
    bench = _make_bench_result("python-stdlib", "naive_vector", {1: 0.7, 2: 0.5})
    out = results_dir / "python-stdlib_naive_vector.json"
    out.write_text(bench.model_dump_json(indent=2))

    generate_report(corpus="python-stdlib", output=str(results_dir / "report.md"))

    assert (results_dir / "report.md").exists()
    assert (results_dir / "summary.json").exists()

    md_text = (results_dir / "report.md").read_text()
    assert "naive_vector" in md_text

    summary = json.loads((results_dir / "summary.json").read_text())
    assert "corpora" in summary
