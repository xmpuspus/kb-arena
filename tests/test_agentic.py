"""Agentic strategy — the retrieve-judge-refine loop must stop at its budget.

Regression coverage for the two ways an agentic loop turns into an open-ended
bill: a judge that never says "enough", and a judge call that fails outright.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.strategies.agentic import AgenticStrategy, _parse_judge_decision


def _not_enough(refined_query: str = "refined query") -> LLMResponse:
    return LLMResponse(text=json.dumps({"enough": False, "refined_query": refined_query}))


def _enough() -> LLMResponse:
    return LLMResponse(text=json.dumps({"enough": True, "refined_query": ""}))


def _answer(text: str = "final answer") -> LLMResponse:
    return LLMResponse(text=text, input_tokens=10, output_tokens=5, cost_usd=0.001)


@pytest.mark.asyncio
async def test_budget_cap_stops_the_loop_even_when_the_judge_always_wants_more(
    mock_chroma_client,
):
    """A judge that always says "not enough" must not buy an unbounded loop."""
    strategy = AgenticStrategy(chroma_client=mock_chroma_client, max_iterations=2, max_llm_calls=3)
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(side_effect=[_not_enough(), _not_enough(), _answer()])

    result = await strategy.query("What does json.loads do?")

    assert result.answer == "final answer"
    assert strategy._llm.generate.call_count == 3  # the hard cap, never exceeded
    assert strategy.last_budget["iterations_used"] == 2
    assert strategy.last_budget["llm_calls_used"] == 3
    assert strategy.last_budget["stop_reason"] == "iteration_budget"


@pytest.mark.asyncio
async def test_a_satisfied_judge_stops_before_the_budget_is_spent(mock_chroma_client):
    """The stopping rule: a judge that says "enough" ends the loop early."""
    strategy = AgenticStrategy(chroma_client=mock_chroma_client, max_iterations=5, max_llm_calls=10)
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(side_effect=[_enough(), _answer()])

    result = await strategy.query("What does json.loads do?")

    assert result.answer == "final answer"
    assert strategy._llm.generate.call_count == 2
    assert strategy.last_budget["iterations_used"] == 1
    assert strategy.last_budget["llm_calls_used"] == 2
    assert strategy.last_budget["stop_reason"] == "judge_satisfied"


@pytest.mark.asyncio
async def test_a_judge_call_that_fails_is_never_dressed_as_an_answer(mock_chroma_client):
    """A raised error from the judge call propagates instead of faking a result."""
    strategy = AgenticStrategy(chroma_client=mock_chroma_client)
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(side_effect=ConnectionError("provider offline"))

    with pytest.raises(ConnectionError, match="provider offline"):
        await strategy.query("What does json.loads do?")


@pytest.mark.asyncio
async def test_a_low_call_budget_stops_the_loop_before_the_iteration_cap(mock_chroma_client):
    """The LLM-call budget is its own hard cap, independent of max_iterations."""
    strategy = AgenticStrategy(chroma_client=mock_chroma_client, max_iterations=10, max_llm_calls=2)
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(side_effect=[_not_enough(), _answer()])

    await strategy.query("What does json.loads do?")

    assert strategy._llm.generate.call_count == 2
    assert strategy.last_budget["llm_calls_used"] == 2
    assert strategy.last_budget["stop_reason"] == "llm_call_budget"


def test_malformed_judge_output_stops_the_loop_instead_of_guessing():
    enough, refined = _parse_judge_decision("not json at all")
    assert enough is True
    assert refined == ""


def test_construction_rejects_a_budget_below_one():
    with pytest.raises(ValueError):
        AgenticStrategy(max_iterations=0)
    with pytest.raises(ValueError):
        AgenticStrategy(max_llm_calls=0)


@pytest.mark.asyncio
async def test_every_llm_call_counts_toward_the_reported_cost(mock_chroma_client):
    """Only the final answer used to count, so the cost cap undercounted a run.

    A judge round costs real tokens. A run reporting the answer alone read as
    one call, and it walked past a cap several calls ago.
    """
    strategy = AgenticStrategy(chroma_client=mock_chroma_client, max_iterations=2, max_llm_calls=3)
    strategy._llm = AsyncMock()
    strategy._llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                text=json.dumps({"enough": True, "refined_query": ""}),
                input_tokens=60,
                output_tokens=40,
                cost_usd=1.0,
            ),
            LLMResponse(text="final answer", input_tokens=10, output_tokens=5, cost_usd=0.01),
        ]
    )

    result = await strategy.query("What does json.loads do?")

    assert result.cost_usd == pytest.approx(1.01)
    assert result.tokens_used == 115
