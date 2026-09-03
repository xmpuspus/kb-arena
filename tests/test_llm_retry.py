"""Provider-aware retry, jittered backoff, and stream deadlines in LLMClient."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kb_arena.llm.client import LLMClient
from kb_arena.llm.providers import ProviderResponse


def _resp(text: str = "ok") -> ProviderResponse:
    return ProviderResponse(text=text, input_tokens=10, output_tokens=5, model="m")


def _anthropic_rate_limit():
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _openai_rate_limit():
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _ollama_connect_error():
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.ConnectError("connection refused", request=request)


def _ollama_status(code: int):
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"status {code}", request=request, response=response)


_PROVIDERS = {
    "anthropic": "kb_arena.llm.providers.AnthropicProvider",
    "openai": "kb_arena.llm.providers.OpenAIProvider",
    "ollama": "kb_arena.llm.providers.OllamaProvider",
}


@pytest.fixture
def no_backoff(monkeypatch):
    sleeper = AsyncMock()
    monkeypatch.setattr("kb_arena.llm.client._sleep", sleeper)
    return sleeper


def _client_with(monkeypatch, provider: str, complete: AsyncMock) -> tuple[LLMClient, MagicMock]:
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "llm_provider", provider)
    patcher = patch(_PROVIDERS[provider])
    cls = patcher.start()
    instance = MagicMock()
    instance.complete = complete
    cls.return_value = instance
    client = LLMClient(api_key="test-key")
    patcher.stop()
    return client, instance


@pytest.mark.parametrize(
    ("provider", "make_error"),
    [
        ("anthropic", _anthropic_rate_limit),
        ("openai", _openai_rate_limit),
        ("ollama", _ollama_connect_error),
        ("ollama", lambda: _ollama_status(503)),
        ("ollama", lambda: _ollama_status(429)),
    ],
)
@pytest.mark.asyncio
async def test_transient_provider_error_is_retried(monkeypatch, no_backoff, provider, make_error):
    complete = AsyncMock(side_effect=[make_error(), _resp("ok")])
    client, instance = _client_with(monkeypatch, provider, complete)

    result = await client._call("fast", "sys", "user")

    assert result.text == "ok"
    assert instance.complete.await_count == 2
    assert no_backoff.await_count == 1


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
@pytest.mark.asyncio
async def test_permanent_error_is_not_retried(monkeypatch, no_backoff, provider):
    complete = AsyncMock(side_effect=ValueError("bad request"))
    client, instance = _client_with(monkeypatch, provider, complete)

    with pytest.raises(ValueError):
        await client._call("fast", "sys", "user")

    assert instance.complete.await_count == 1
    assert no_backoff.await_count == 0


@pytest.mark.asyncio
async def test_ollama_client_error_status_is_not_retried(monkeypatch, no_backoff):
    complete = AsyncMock(side_effect=_ollama_status(400))
    client, instance = _client_with(monkeypatch, "ollama", complete)

    with pytest.raises(httpx.HTTPStatusError):
        await client._call("fast", "sys", "user")

    assert instance.complete.await_count == 1


@pytest.mark.asyncio
async def test_backoff_grows_and_carries_jitter(monkeypatch, no_backoff):
    monkeypatch.setattr("kb_arena.llm.client.random.uniform", lambda a, b: 0.25)
    complete = AsyncMock(
        side_effect=[_anthropic_rate_limit(), _anthropic_rate_limit(), _resp("ok")]
    )
    client, _ = _client_with(monkeypatch, "anthropic", complete)

    await client._call("fast", "sys", "user")

    delays = [call.args[0] for call in no_backoff.await_args_list]
    assert delays == [pytest.approx(1.25), pytest.approx(2.25)]


@pytest.mark.asyncio
async def test_timeout_sleeps_before_the_next_attempt(monkeypatch, no_backoff):
    complete = AsyncMock(side_effect=[TimeoutError(), _resp("ok")])
    client, instance = _client_with(monkeypatch, "anthropic", complete)

    result = await client._call("fast", "sys", "user")

    assert result.text == "ok"
    assert instance.complete.await_count == 2
    assert no_backoff.await_count == 1


@pytest.mark.asyncio
async def test_exhausted_attempts_raise_the_last_error(monkeypatch, no_backoff):
    complete = AsyncMock(side_effect=[_openai_rate_limit()] * 3)
    client, instance = _client_with(monkeypatch, "openai", complete)

    import openai

    with pytest.raises(openai.RateLimitError):
        await client._call("fast", "sys", "user")

    assert instance.complete.await_count == 3
    assert no_backoff.await_count == 2


# --- Streaming deadlines ---


def _stream_provider(monkeypatch, provider: str, generators: list) -> tuple[LLMClient, list]:
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "llm_provider", provider)
    calls: list = []
    with patch(_PROVIDERS[provider]) as cls:
        instance = MagicMock()

        def stream_text(**kwargs):
            calls.append(kwargs)
            return generators[len(calls) - 1]()

        instance.stream_text = stream_text
        cls.return_value = instance
        client = LLMClient(api_key="test-key")
    return client, calls


async def _collect(client: LLMClient) -> list:
    return [item async for item in client.stream(query="Q", context="C", system_prompt="S")]


@pytest.mark.asyncio
async def test_stream_retries_a_transient_error_before_the_first_token(monkeypatch, no_backoff):
    async def failing():
        raise _anthropic_rate_limit()
        yield  # pragma: no cover

    async def working():
        yield "hello"
        yield _resp()

    client, calls = _stream_provider(monkeypatch, "anthropic", [failing, working])

    output = await _collect(client)

    assert output == ["hello"]
    assert len(calls) == 2
    assert no_backoff.await_count == 1


@pytest.mark.asyncio
async def test_stream_does_not_retry_after_the_first_token(monkeypatch, no_backoff):
    async def partial():
        yield "hello"
        raise _anthropic_rate_limit()

    async def working():
        yield "never"

    client, calls = _stream_provider(monkeypatch, "anthropic", [partial, working])
    output = []
    import anthropic

    with pytest.raises(anthropic.RateLimitError):
        async for item in client.stream(query="Q", context="C", system_prompt="S"):
            output.append(item)

    assert output == ["hello"]
    assert len(calls) == 1
    assert no_backoff.await_count == 0


@pytest.mark.asyncio
async def test_stream_first_token_deadline(monkeypatch, no_backoff):
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "llm_stream_first_token_timeout_s", 0.02)
    monkeypatch.setattr(settings, "llm_max_attempts", 1)

    async def stalled():
        await asyncio.sleep(1.0)
        yield "late"

    client, calls = _stream_provider(monkeypatch, "anthropic", [stalled])

    with pytest.raises(TimeoutError):
        await _collect(client)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_stream_idle_deadline_between_tokens(monkeypatch, no_backoff):
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "llm_stream_idle_timeout_s", 0.02)

    async def slow_after_first():
        yield "first"
        await asyncio.sleep(1.0)
        yield "second"

    client, _ = _stream_provider(monkeypatch, "anthropic", [slow_after_first])
    output = []
    with pytest.raises(TimeoutError):
        async for item in client.stream(query="Q", context="C", system_prompt="S"):
            output.append(item)
    assert output == ["first"]


def test_stream_timeouts_are_settings_with_sane_defaults():
    from kb_arena.settings import Settings

    s = Settings(_env_file=None)
    assert s.llm_call_timeout_s == pytest.approx(60.0)
    assert s.llm_max_attempts == 3
    assert s.llm_stream_first_token_timeout_s > 0
    assert s.llm_stream_idle_timeout_s > 0
