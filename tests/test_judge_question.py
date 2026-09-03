"""The answer judge receives the question, rejects malformed verdicts, and can be calibrated."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb_arena.llm.client import LLMClient, LLMResponse
from kb_arena.llm.providers import ProviderResponse
from kb_arena.models.benchmark import Constraints, GroundTruth

_GOOD_VERDICT = '{"accuracy": 0.8, "completeness": 0.7, "faithfulness": 1.0}'


def _resp(text: str) -> ProviderResponse:
    return ProviderResponse(text=text, input_tokens=10, output_tokens=5, model="judge-model")


@pytest.fixture
def anthropic_provider():
    with patch("kb_arena.llm.providers.AnthropicProvider") as cls:
        instance = MagicMock()
        instance.complete = AsyncMock(return_value=_resp(_GOOD_VERDICT))
        cls.return_value = instance
        yield instance


@pytest.mark.asyncio
async def test_judge_prompt_carries_the_question_before_the_reference(anthropic_provider):
    client = LLMClient(api_key="test-key")
    await client.judge(
        answer="Port 443.",
        reference="It listens on 443.",
        system_prompt="Judge.",
        question="Which port does the gateway use?",
    )
    user = anthropic_provider.complete.call_args.kwargs["user"]
    assert "Which port does the gateway use?" in user
    q_idx = user.index("Question:")
    ref_idx = user.index("Reference answer:")
    cand_idx = user.index("Candidate answer:")
    assert q_idx < ref_idx < cand_idx


@pytest.mark.asyncio
async def test_judge_without_question_keeps_the_reference_candidate_prompt(anthropic_provider):
    client = LLMClient(api_key="test-key")
    await client.judge(answer="A", reference="R", system_prompt="Judge.")
    user = anthropic_provider.complete.call_args.kwargs["user"]
    assert "Question:" not in user
    assert "Reference answer:" in user


@pytest.mark.parametrize(
    ("provider", "cls_path"),
    [
        ("anthropic", "kb_arena.llm.providers.AnthropicProvider"),
        ("openai", "kb_arena.llm.providers.OpenAIProvider"),
        ("ollama", "kb_arena.llm.providers.OllamaProvider"),
    ],
)
@pytest.mark.asyncio
async def test_every_provider_receives_the_question(monkeypatch, provider, cls_path):
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "llm_provider", provider)
    with patch(cls_path) as cls:
        instance = MagicMock()
        instance.complete = AsyncMock(return_value=_resp(_GOOD_VERDICT))
        cls.return_value = instance
        client = LLMClient(api_key="test-key")
        await client.judge(answer="A", reference="R", system_prompt="S", question="Q?")
        assert "Question:\nQ?" in instance.complete.call_args.kwargs["user"]


def test_judge_system_prompt_names_the_question():
    from kb_arena.benchmark.evaluator import JUDGE_SYSTEM_PROMPT

    assert "Given the question" in JUDGE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_evaluator_passes_question_text_to_the_judge():
    from kb_arena.benchmark.evaluator import evaluate

    llm = MagicMock()
    llm.judge = AsyncMock(return_value=LLMResponse(text=_GOOD_VERDICT, cost_usd=0.001))
    score = await evaluate(
        "The gateway listens on port 443.",
        GroundTruth(answer="443"),
        Constraints(),
        llm=llm,
        question_text="Which port does the gateway use?",
    )
    assert llm.judge.call_args.kwargs["question"] == "Which port does the gateway use?"
    assert score.accuracy == pytest.approx(0.8)
    assert score.completeness == pytest.approx(0.7)


@pytest.mark.parametrize(
    "verdict",
    [
        "no json here",
        '{"accuracy": 0.9, "completeness": 0.9}',
        '{"accuracy": true, "completeness": 0.9, "faithfulness": 1.0}',
        '{"accuracy": 1.5, "completeness": 0.9, "faithfulness": 1.0}',
        '{"accuracy": "high", "completeness": 0.9, "faithfulness": 1.0}',
    ],
)
@pytest.mark.asyncio
async def test_malformed_judge_verdicts_fail_loudly(verdict):
    from kb_arena.benchmark.evaluator import EvaluationExecutionError, evaluate

    llm = MagicMock()
    llm.judge = AsyncMock(return_value=LLMResponse(text=verdict, cost_usd=0.0))
    with pytest.raises(EvaluationExecutionError, match="LLM judge failed"):
        await evaluate(
            f"answer for {verdict}",
            GroundTruth(answer="ref"),
            Constraints(),
            llm=llm,
            question_text="q",
        )


# --- Calibration harness ---


def _scripted_judge(verdicts: dict[str, float]) -> MagicMock:
    """Return a judge whose accuracy depends on which candidate it sees."""

    async def judge(answer: str, reference: str, system_prompt: str, question: str = "", **_):
        accuracy = verdicts[answer]
        payload = {"accuracy": accuracy, "completeness": accuracy, "faithfulness": 1.0}
        return LLMResponse(text=json.dumps(payload), cost_usd=0.001)

    llm = MagicMock()
    llm.judge = AsyncMock(side_effect=judge)
    return llm


def test_packaged_calibration_set_loads_and_has_both_bands():
    from kb_arena.benchmark.judge_calibration import load_calibration_items

    items = load_calibration_items()
    assert len(items) >= 6
    labels = {item.label for item in items}
    assert {"correct", "wrong"} <= labels
    for item in items:
        assert 0.0 <= item.expected_min <= item.expected_max <= 1.0
        assert item.question and item.reference and item.candidate


@pytest.mark.asyncio
async def test_calibration_reports_agreement_and_out_of_band_items():
    from kb_arena.benchmark.judge_calibration import CalibrationItem, run_calibration

    items = [
        CalibrationItem(
            id="c1",
            label="correct",
            question="Which port?",
            reference="443",
            candidate="It uses 443.",
            expected_min=0.7,
            expected_max=1.0,
        ),
        CalibrationItem(
            id="w1",
            label="wrong",
            question="Which port?",
            reference="443",
            candidate="It uses 80.",
            expected_min=0.0,
            expected_max=0.3,
        ),
        CalibrationItem(
            id="w2",
            label="wrong",
            question="Which region?",
            reference="us-east-1",
            candidate="eu-west-1",
            expected_min=0.0,
            expected_max=0.3,
        ),
    ]
    llm = _scripted_judge({"It uses 443.": 0.9, "It uses 80.": 0.1, "eu-west-1": 0.8})

    report = await run_calibration(llm, items)

    assert report.total == 3
    assert report.in_band == 2
    assert report.agreement == pytest.approx(2 / 3)
    assert [o.id for o in report.outcomes if not o.in_band] == ["w2"]
    assert all(o.accuracy is not None for o in report.outcomes)
    # The harness sends the question to the judge, the whole point of calibration.
    assert llm.judge.call_args.kwargs["question"] == "Which region?"


@pytest.mark.asyncio
async def test_calibration_records_judge_failures_instead_of_crashing():
    from kb_arena.benchmark.judge_calibration import CalibrationItem, run_calibration

    item = CalibrationItem(
        id="x",
        label="correct",
        question="q",
        reference="r",
        candidate="c",
        expected_min=0.5,
        expected_max=1.0,
    )
    llm = MagicMock()
    llm.judge = AsyncMock(return_value=LLMResponse(text="not json", cost_usd=0.0))

    report = await run_calibration(llm, [item])

    assert report.total == 1
    assert report.in_band == 0
    assert report.outcomes[0].accuracy is None
    assert "LLM judge failed" in report.outcomes[0].error
