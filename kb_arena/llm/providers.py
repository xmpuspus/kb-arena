"""LLM provider backends - Anthropic, OpenAI, Ollama."""

from __future__ import annotations

import abc
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """Raw response from any LLM provider."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


class LLMProvider(abc.ABC):
    """Abstract LLM backend."""

    @abc.abstractmethod
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0,
    ) -> ProviderResponse: ...

    @abc.abstractmethod
    async def stream_text(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        temperature: float = 0,
    ) -> AsyncIterator[str]: ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude backend."""

    def __init__(self, api_key: str):
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, model, system, user, max_tokens=4096, temperature=0):
        response = await self.client.messages.create(
            model=model,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = response.usage
        return ProviderResponse(
            text=response.content[0].text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model=model,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    async def stream_text(self, model, system, user, max_tokens=4096, temperature=0):
        async with self.client.messages.stream(
            model=model,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            # Capture usage after stream completes and store on instance
            final = await stream.get_final_message()
            usage = final.usage
            self.last_stream_response = ProviderResponse(
                text="",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                model=model,
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )


class OpenAIProvider(LLMProvider):
    """OpenAI GPT backend."""

    def __init__(self, api_key: str):
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, model, system, user, max_tokens=4096, temperature=0):
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    async def stream_text(self, model, system, user, max_tokens=4096, temperature=0):
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(LLMProvider):
    """Ollama local inference backend."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        import httpx

        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def complete(self, model, system, user, max_tokens=4096, temperature=0):
        resp = await self.client.post(
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return ProviderResponse(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            model=model,
        )

    async def stream_text(self, model, system, user, max_tokens=4096, temperature=0):
        import json as json_mod

        async with self.client.stream(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    data = json_mod.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content


def create_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Factory for LLM providers."""
    if provider_name == "anthropic":
        return AnthropicProvider(api_key=kwargs.get("api_key", ""))
    elif provider_name == "openai":
        return OpenAIProvider(api_key=kwargs.get("api_key", ""))
    elif provider_name == "ollama":
        return OllamaProvider(base_url=kwargs.get("base_url", "http://localhost:11434"))
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. Choose: anthropic, openai, ollama"
        )
