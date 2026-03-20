"""Tests for LLM provider abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb_arena.llm.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)


def test_create_provider_anthropic():
    with patch("anthropic.AsyncAnthropic"):
        p = create_provider("anthropic", api_key="test")
        assert isinstance(p, AnthropicProvider)


def test_create_provider_openai():
    with patch("openai.AsyncOpenAI"):
        p = create_provider("openai", api_key="test")
        assert isinstance(p, OpenAIProvider)


def test_create_provider_ollama():
    p = create_provider("ollama", base_url="http://localhost:11434")
    assert isinstance(p, OllamaProvider)


def test_create_provider_unknown():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_provider("unknown")


@pytest.mark.asyncio
async def test_anthropic_complete():
    mock_client = AsyncMock()
    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello world")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        provider = AnthropicProvider(api_key="test-key")
        resp = await provider.complete(model="claude-sonnet-4-6", system="Be helpful", user="Hi")
        assert resp.text == "Hello world"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5


@pytest.mark.asyncio
async def test_openai_complete():
    mock_client = AsyncMock()
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from GPT"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 8
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        provider = OpenAIProvider(api_key="test-key")
        resp = await provider.complete(model="gpt-4o", system="Be helpful", user="Hi")
        assert resp.text == "Hello from GPT"
        assert resp.input_tokens == 15
        assert resp.output_tokens == 8
