"""Dual-model LLM client with prompt caching (cloudwright pattern).

Haiku for classification (~20 tokens, <50ms).
Sonnet for generation, extraction, evaluation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import anthropic

from kb_arena.settings import settings

logger = logging.getLogger(__name__)

_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)

GENERATE_MODEL = settings.generate_model
FAST_MODEL = settings.fast_model
JUDGE_MODEL = settings.judge_model

# Per-million-token pricing (USD). Update when Anthropic changes pricing.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "haiku": {"input": 0.80, "output": 4.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
}


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost from token counts and model name."""
    pricing = _MODEL_PRICING["sonnet"]  # default
    for tier, p in _MODEL_PRICING.items():
        if tier in model:
            pricing = p
            break

    input_cost = input_tokens * pricing["input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    cache_create_cost = cache_creation_tokens * pricing["input"] * 1.25 / 1_000_000
    cache_read_cost = cache_read_tokens * pricing["input"] * 0.1 / 1_000_000
    return input_cost + output_cost + cache_create_cost + cache_read_cost


@dataclass
class LLMResponse:
    """Result from an LLM call, including text and usage metrics."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    def __init__(self, api_key: str | None = None):
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.anthropic_api_key,
        )

    async def classify(
        self,
        query: str,
        system_prompt: str,
        allowed_values: list[str] | None = None,
        history: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """Cheap classification call. Haiku, ~20 tokens, <50ms."""
        user_content = query
        if history:
            turns = history[-6:]  # last 6 turns
            ctx = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in turns)
            user_content = f"Conversation:\n{ctx}\n\nCurrent query: {query}"

        resp = await self._call(FAST_MODEL, system_prompt, user_content, max_tokens=100, **kwargs)
        result = resp.text.strip().lower()

        if allowed_values:
            for v in allowed_values:
                if v.lower() in result:
                    return v
            return allowed_values[0]  # fallback to first value

        return result

    async def generate(
        self,
        query: str,
        context: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Full generation call. Sonnet."""
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        return await self._call(GENERATE_MODEL, system_prompt, user_content, **kwargs)

    async def extract(
        self,
        text: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Entity/relationship extraction. Sonnet, structured output."""
        return await self._call(GENERATE_MODEL, system_prompt, text, **kwargs)

    async def judge(
        self,
        answer: str,
        reference: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """LLM-as-judge evaluation. Uses JUDGE_MODEL (defaults to Opus) to avoid same-model bias."""
        user_content = f"Reference answer:\n{reference}\n\nCandidate answer:\n{answer}"
        return await self._call(
            JUDGE_MODEL, system_prompt, user_content, max_tokens=300, **kwargs
        )

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Core API call with retry (3 attempts, exponential backoff) and 60s timeout."""
        last_exc: BaseException = RuntimeError("LLM call failed before any attempt")
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    self._call_once(model, system, user, max_tokens, **kwargs),
                    timeout=60.0,
                )
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("LLM call timed out (attempt %d/3)", attempt + 1)
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt < 2:
                    delay = 2**attempt
                    logger.warning(
                        "LLM call failed (attempt %d/3): %s. Retrying in %ds",
                        attempt + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_exc

    async def _call_once(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Single API call with cache_control on system prompt."""
        response = await self.client.messages.create(
            model=model,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
            temperature=kwargs.pop("temperature", 0),
            max_tokens=max_tokens,
            **kwargs,
        )
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = _compute_cost(model, input_tokens, output_tokens, cache_creation, cache_read)

        return LLMResponse(
            text=response.content[0].text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    async def stream(
        self,
        query: str,
        context: str,
        system_prompt: str,
        **kwargs,
    ):
        """Streaming generation. Yields text deltas."""
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        async with self.client.messages.stream(
            model=GENERATE_MODEL,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
            temperature=0,
            max_tokens=kwargs.pop("max_tokens", 4096),
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
