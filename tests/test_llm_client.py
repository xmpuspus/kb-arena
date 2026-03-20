"""Tests for LLMClient and LLMResponse — mocked provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb_arena.llm.client import LLMClient, LLMResponse, _compute_cost

# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


def test_compute_cost_haiku_model():
    cost = _compute_cost("claude-haiku-4-5", 1000, 100)
    assert cost == pytest.approx(0.0008 + 0.0004)


def test_compute_cost_sonnet_model():
    cost = _compute_cost("claude-sonnet-4-6", 1000, 100)
    assert cost == pytest.approx(0.003 + 0.0015)


def test_compute_cost_opus_model():
    cost = _compute_cost("claude-opus-4", 1000, 100)
    assert cost == pytest.approx(0.015 + 0.0075)


def test_compute_cost_zero_tokens():
    assert _compute_cost("claude-haiku-4-5", 0, 0) == 0.0


def test_compute_cost_with_cache_creation():
    base = _compute_cost("claude-haiku-4-5", 0, 0)
    with_cache = _compute_cost("claude-haiku-4-5", 0, 0, cache_creation_tokens=1000)
    assert with_cache > base


def test_compute_cost_with_cache_read():
    cost = _compute_cost("claude-haiku-4-5", 0, 0, cache_read_tokens=1000)
    assert cost > 0
    assert cost < _compute_cost("claude-haiku-4-5", 1000, 0)


def test_compute_cost_unknown_model_defaults_sonnet():
    sonnet_cost = _compute_cost("claude-sonnet-4-6", 1000, 100)
    unknown_cost = _compute_cost("unknown-model-xyz", 1000, 100)
    assert unknown_cost == sonnet_cost


def test_compute_cost_gpt4o_mini_exact_match():
    # gpt-4o-mini should match exactly, not fall through to gpt-4o substring
    cost = _compute_cost("gpt-4o-mini", 1_000_000, 0)
    assert cost == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


def test_llm_response_total_tokens():
    r = LLMResponse(text="hello", input_tokens=100, output_tokens=50)
    assert r.total_tokens == 150


def test_llm_response_total_tokens_zero():
    r = LLMResponse(text="hello")
    assert r.total_tokens == 0


def test_llm_response_defaults():
    r = LLMResponse(text="hello")
    assert r.input_tokens == 0
    assert r.output_tokens == 0
    assert r.cost_usd == 0.0


def test_llm_response_text_stored():
    r = LLMResponse(text="some output", input_tokens=10, output_tokens=5, cost_usd=0.001)
    assert r.text == "some output"
    assert r.cost_usd == 0.001


# ---------------------------------------------------------------------------
# LLMClient — mock provider
# ---------------------------------------------------------------------------


def _make_provider_response(text: str, input_tokens: int = 50, output_tokens: int = 20):
    from kb_arena.llm.providers import ProviderResponse

    return ProviderResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model="claude-haiku-4-5-20251001",
    )


@pytest.fixture
def mock_provider():
    """Mock AnthropicProvider so LLMClient works without a real API key."""
    with patch("kb_arena.llm.providers.AnthropicProvider") as mock_cls:
        instance = MagicMock()
        instance.complete = AsyncMock(return_value=_make_provider_response("mocked response"))
        mock_cls.return_value = instance
        yield instance


@pytest.mark.asyncio
async def test_classify_returns_string(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("factoid")
    client = LLMClient(api_key="test-key")
    result = await client.classify("What is json?", system_prompt="Classify the query.")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_classify_matches_allowed_value(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("comparison")
    client = LLMClient(api_key="test-key")
    result = await client.classify(
        "compare x vs y",
        system_prompt="Classify.",
        allowed_values=["factoid", "comparison", "procedural"],
    )
    assert result == "comparison"


@pytest.mark.asyncio
async def test_classify_fallback_to_first_value(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("something unexpected")
    client = LLMClient(api_key="test-key")
    result = await client.classify(
        "gibberish",
        system_prompt="Classify.",
        allowed_values=["factoid", "comparison"],
    )
    assert result == "factoid"


@pytest.mark.asyncio
async def test_classify_case_insensitive_match(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("PROCEDURAL")
    client = LLMClient(api_key="test-key")
    result = await client.classify(
        "how do I setup?",
        system_prompt="Classify.",
        allowed_values=["factoid", "procedural"],
    )
    assert result == "procedural"


@pytest.mark.asyncio
async def test_classify_with_history_includes_context(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("factoid")
    client = LLMClient(api_key="test-key")
    history = [{"role": "user", "content": "previous question about json"}]
    await client.classify("follow-up?", system_prompt="Classify.", history=history)

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "previous question about json" in user_message


@pytest.mark.asyncio
async def test_classify_history_truncates_to_six(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("factoid")
    client = LLMClient(api_key="test-key")
    history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    await client.classify("latest?", system_prompt="Classify.", history=history)

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "msg 0" not in user_message
    assert "msg 9" in user_message


@pytest.mark.asyncio
async def test_classify_no_history_no_conversation_prefix(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("factoid")
    client = LLMClient(api_key="test-key")
    await client.classify("What is X?", system_prompt="Classify.")

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "Conversation:" not in user_message


@pytest.mark.asyncio
async def test_generate_builds_context_query_prompt(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("The answer.")
    client = LLMClient(api_key="test-key")
    result = await client.generate(
        query="What is json.loads?",
        context="json.loads parses JSON strings.",
        system_prompt="Be helpful.",
    )
    assert isinstance(result, LLMResponse)
    assert result.text == "The answer."

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "Context:" in user_message
    assert "Query:" in user_message


@pytest.mark.asyncio
async def test_generate_empty_context_no_prefix(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("Direct answer.")
    client = LLMClient(api_key="test-key")
    await client.generate(query="What is json?", context="", system_prompt="Be helpful.")

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "Context:" not in user_message


@pytest.mark.asyncio
async def test_generate_uses_generate_model(mock_provider):
    from kb_arena.settings import settings

    mock_provider.complete.return_value = _make_provider_response("ok")
    client = LLMClient(api_key="test-key")
    await client.generate(query="Q", context="C", system_prompt="S")

    call_args = mock_provider.complete.call_args
    assert call_args.kwargs["model"] == settings.generate_model


@pytest.mark.asyncio
async def test_extract_passes_text_as_user(mock_provider):
    mock_provider.complete.return_value = _make_provider_response('{"entities": []}')
    client = LLMClient(api_key="test-key")
    result = await client.extract(text="Document text.", system_prompt="Extract entities.")
    assert isinstance(result, LLMResponse)
    assert '{"entities": []}' in result.text

    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "Document text." in user_message


@pytest.mark.asyncio
async def test_judge_builds_reference_candidate_prompt(mock_provider):
    mock_provider.complete.return_value = _make_provider_response('{"accuracy": 0.9}')
    client = LLMClient(api_key="test-key")
    result = await client.judge(
        answer="Candidate.",
        reference="Reference.",
        system_prompt="Judge quality.",
    )
    assert isinstance(result, LLMResponse)
    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["user"]
    assert "Reference answer:" in user_message
    assert "Candidate answer:" in user_message


@pytest.mark.asyncio
async def test_call_uses_system_prompt(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("ok")
    client = LLMClient(api_key="test-key")
    await client._call("fast", "system text", "user text")

    call_args = mock_provider.complete.call_args
    assert call_args.kwargs["system"] == "system text"


@pytest.mark.asyncio
async def test_call_default_temperature_zero(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("ok")
    client = LLMClient(api_key="test-key")
    await client._call("fast", "sys", "user")

    call_args = mock_provider.complete.call_args
    assert call_args.kwargs["temperature"] == 0


@pytest.mark.asyncio
async def test_call_respects_max_tokens(mock_provider):
    mock_provider.complete.return_value = _make_provider_response("ok")
    client = LLMClient(api_key="test-key")
    await client._call("fast", "sys", "user", max_tokens=100)

    call_args = mock_provider.complete.call_args
    assert call_args.kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_call_returns_llm_response_with_cost(mock_provider):
    mock_provider.complete.return_value = _make_provider_response(
        "ok", input_tokens=100, output_tokens=50
    )
    client = LLMClient(api_key="test-key")
    result = await client._call("fast", "sys", "user")
    assert isinstance(result, LLMResponse)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cost_usd > 0
