"""Live tests for the benchmark evaluator — structural checks and LLM judge."""

from __future__ import annotations

import pytest

from kb_arena.benchmark.evaluator import _structural_check, evaluate
from kb_arena.models.benchmark import Constraints, GroundTruth

pytestmark = pytest.mark.live


@pytest.fixture
def json_ground_truth():
    return GroundTruth(
        answer=(
            "json.loads(s) deserializes a JSON string to a Python object"
            " such as dict, list, or primitive."
        ),
        source_refs=["aws-compute-lambda"],
        required_entities=["json.loads", "JSONDecodeError"],
    )


@pytest.fixture
def json_constraints():
    return Constraints(
        must_mention=["json.loads", "deserializ"],
        must_not_claim=["gzip", "compress", "pickle"],
        max_tokens=500,
    )


# --- Structural checks (no LLM) ---


def test_structural_correct_answer(json_constraints):
    answer = "json.loads deserializes a JSON string into a Python object."
    score = _structural_check(answer, json_constraints)
    assert score.structural_pass is True
    assert score.accuracy == 1.0
    assert score.faithfulness == 1.0
    assert len(score.false_claims) == 0


def test_structural_missing_must_mention(json_constraints):
    answer = "This function converts data formats."
    score = _structural_check(answer, json_constraints)
    assert score.accuracy < 1.0
    assert score.mentions_found == []


def test_structural_must_not_claim_triggers(json_constraints):
    answer = "json.loads is used to deserializ data using gzip compression."
    score = _structural_check(answer, json_constraints)
    assert score.structural_pass is False
    assert score.accuracy == 0.0
    assert score.faithfulness == 0.0
    assert "gzip" in score.false_claims


def test_structural_partial_mentions():
    constraints = Constraints(must_mention=["json", "loads", "error"])
    answer = "json.loads parses JSON strings."
    score = _structural_check(answer, constraints)
    # "json" and "loads" are present, "error" is not
    assert 0.0 < score.accuracy < 1.0


def test_structural_empty_constraints():
    answer = "Any answer passes with no constraints."
    score = _structural_check(answer, Constraints())
    assert score.structural_pass is True
    assert score.accuracy == 1.0


def test_structural_no_llm_needed():
    """Structural check is synchronous and has no LLM dependency."""
    import time

    constraints = Constraints(must_mention=["word"])
    t0 = time.perf_counter()
    for _ in range(100):
        _structural_check("this word is here", constraints)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 100, f"100 structural checks took {elapsed_ms:.1f}ms (too slow)"


# --- LLM judge (live calls) ---


async def test_perfect_answer_high_score(live_llm_client, json_ground_truth, json_constraints):
    answer = "json.loads deserializes a JSON-formatted string into a Python object."
    score = await evaluate(answer, json_ground_truth, json_constraints, llm=live_llm_client)
    assert score.accuracy >= 0.7, f"Perfect answer scored {score.accuracy}"
    assert score.faithfulness >= 0.8


async def test_partially_correct_answer_mid_score(
    live_llm_client, json_ground_truth, json_constraints
):
    answer = "json.loads is used for deserializ data but I'm not sure about the details."
    score = await evaluate(answer, json_ground_truth, json_constraints, llm=live_llm_client)
    # Partial answer — expect middle range
    assert 0.0 <= score.accuracy <= 0.9


async def test_completely_wrong_answer_low_score(live_llm_client, json_constraints):
    ground_truth = GroundTruth(
        answer="json.loads converts JSON string to Python object.",
    )
    answer = "This function sorts a list in ascending order by comparing element values."
    score = await evaluate(answer, ground_truth, json_constraints, llm=live_llm_client)
    # Wrong answer AND missing must_mention → structural fails, score should be low
    assert score.accuracy <= 0.4


async def test_must_not_claim_short_circuits_judge(live_llm_client, json_constraints):
    """When must_not_claim fires, structural_pass=False, LLM judge is skipped."""
    ground_truth = GroundTruth(answer="json.loads deserializes JSON.")
    answer = "json.loads deserializ data using gzip encoding."
    score = await evaluate(answer, ground_truth, json_constraints, llm=live_llm_client)
    assert score.accuracy == 0.0
    assert score.structural_pass is False
    assert "gzip" in score.false_claims


async def test_missing_all_must_mention_low_score(live_llm_client):
    constraints = Constraints(must_mention=["alpha", "beta", "gamma"])
    ground_truth = GroundTruth(answer="alpha beta gamma are all required.")
    answer = "This answer mentions nothing relevant."
    score = await evaluate(answer, ground_truth, constraints, llm=live_llm_client)
    # mention_ratio = 0/3 → structural accuracy = 0.0
    assert score.accuracy == 0.0


async def test_judge_parses_json_output(live_llm_client):
    """LLM judge must return parseable JSON with required fields."""
    import json
    import re

    from kb_arena.benchmark.evaluator import JUDGE_SYSTEM_PROMPT

    raw = await live_llm_client.judge(
        answer="json.loads parses a JSON string.",
        reference="json.loads converts a JSON string to a Python object.",
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    assert m, f"No JSON object in judge output: {raw}"
    parsed = json.loads(m.group())
    assert "accuracy" in parsed
    assert "completeness" in parsed
    assert "faithfulness" in parsed
    for key in ["accuracy", "completeness", "faithfulness"]:
        assert 0.0 <= float(parsed[key]) <= 1.0


async def test_evaluate_without_llm_uses_structural():
    """evaluate() with llm=None returns structural score only."""
    constraints = Constraints(must_mention=["json"])
    ground_truth = GroundTruth(answer="json is great.")
    answer = "json is a module."
    score = await evaluate(answer, ground_truth, constraints, llm=None)
    assert score.structural_pass is True
    assert score.accuracy == 1.0  # "json" is mentioned


# --- Real question scenarios ---


@pytest.fixture
def known_qa_pairs():
    """Five Q&A pairs with known correctness direction."""
    return [
        {
            "question": "What does json.loads do?",
            "good_answer": "json.loads deserializes a JSON-formatted string into a Python object.",
            "bad_answer": "json.loads sorts a list using quicksort.",
            "constraints": Constraints(must_mention=["json.loads", "deserializ"]),
            "ground_truth": GroundTruth(
                answer="json.loads(s) converts a JSON string to a Python dict, list, or primitive."
            ),
        },
        {
            "question": "What exception does json.loads raise?",
            "good_answer": (
                "json.loads raises json.JSONDecodeError when the input is not valid JSON."
            ),
            "bad_answer": "json.loads raises FileNotFoundError when the file doesn't exist.",
            "constraints": Constraints(must_mention=["JSONDecodeError"]),
            "ground_truth": GroundTruth(
                answer="json.loads raises json.JSONDecodeError on invalid JSON input."
            ),
        },
        {
            "question": "What does json.dumps return?",
            "good_answer": "json.dumps serializes a Python object to a JSON formatted string.",
            "bad_answer": "json.dumps returns a dictionary.",
            "constraints": Constraints(must_mention=["json.dumps", "string"]),
            "ground_truth": GroundTruth(answer="json.dumps returns a JSON-formatted string."),
        },
    ]


async def test_good_answers_score_higher(live_llm_client, known_qa_pairs):
    """Good answers should score higher than bad answers for the same question."""
    for pair in known_qa_pairs:
        good_score = await evaluate(
            pair["good_answer"], pair["ground_truth"], pair["constraints"], llm=live_llm_client
        )
        bad_score = await evaluate(
            pair["bad_answer"], pair["ground_truth"], pair["constraints"], llm=live_llm_client
        )
        assert good_score.accuracy >= bad_score.accuracy, (
            f"Good answer scored lower: good={good_score.accuracy}, bad={bad_score.accuracy}\n"
            f"Q: {pair['question']}"
        )


async def test_five_wrong_answers_all_caught(live_llm_client):
    """Deliberately wrong answers should all score < 0.5."""
    wrong_pairs = [
        {
            "answer": "json.loads is used to establish database connections.",
            "ground_truth": GroundTruth(answer="json.loads parses JSON strings."),
            "constraints": Constraints(must_mention=["json.loads"]),
        },
        {
            "answer": "The os module handles JSON serialization.",
            "ground_truth": GroundTruth(
                answer="The os module provides OS-dependent functionality."
            ),
            "constraints": Constraints(must_mention=["os"]),
        },
        {
            "answer": "json.dumps reads data from a file on disk.",
            "ground_truth": GroundTruth(
                answer="json.dumps serializes Python objects to JSON strings."
            ),
            "constraints": Constraints(must_mention=["json.dumps", "serial"]),
        },
        {
            "answer": "JSONDecodeError is raised when the network is unavailable.",
            "ground_truth": GroundTruth(answer="JSONDecodeError is raised on invalid JSON input."),
            "constraints": Constraints(must_mention=["JSONDecodeError", "invalid"]),
        },
        {
            "answer": "Python lists are immutable and cannot be modified.",
            "ground_truth": GroundTruth(answer="Python lists are mutable sequences."),
            "constraints": Constraints(must_mention=["list", "mutable"]),
        },
    ]
    for pair in wrong_pairs:
        score = await evaluate(
            pair["answer"], pair["ground_truth"], pair["constraints"], llm=live_llm_client
        )
        assert (
            score.accuracy <= 0.6
        ), f"Wrong answer scored too high: {score.accuracy}\nAnswer: {pair['answer']}"
