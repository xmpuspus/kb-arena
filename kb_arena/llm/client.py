"""Dual-model LLM client with prompt caching (cloudwright pattern).

Haiku for classification (~20 tokens, <50ms).
Sonnet for generation, extraction, evaluation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from kb_arena.settings import settings

logger = logging.getLogger(__name__)

# Per-million-token pricing (USD). Update when providers change pricing.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "haiku": {"input": 0.80, "output": 4.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Ollama (free local inference)
    "llama": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "qwen": {"input": 0.0, "output": 0.0},
    "phi": {"input": 0.0, "output": 0.0},
    "gemma": {"input": 0.0, "output": 0.0},
}


def _compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost from token counts and model name."""
    # Try exact match first (e.g. "gpt-4o-mini"), then substring fallback
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        for tier, p in _MODEL_PRICING.items():
            if tier in model:
                pricing = p
                break
    if pricing is None:
        pricing = _MODEL_PRICING["sonnet"]  # default

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


def _retryable_exceptions():
    """Build the retryable exception tuple for the active provider."""
    try:
        import anthropic

        return (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )
    except ImportError:
        return (OSError,)


class LLMClient:
    def __init__(self, api_key: str | None = None):
        from kb_arena.llm.providers import create_provider

        provider_name = settings.llm_provider

        if provider_name == "anthropic":
            key = api_key or settings.llm_api_key or settings.anthropic_api_key
            self._provider = create_provider("anthropic", api_key=key)
            self._models = {
                "generate": settings.generate_model,
                "fast": settings.fast_model,
                "judge": settings.judge_model,
            }
        elif provider_name == "openai":
            key = api_key or settings.llm_api_key or settings.openai_api_key
            self._provider = create_provider("openai", api_key=key)
            self._models = {
                "generate": settings.openai_generate_model,
                "fast": settings.openai_fast_model,
                "judge": settings.openai_judge_model,
            }
        elif provider_name == "ollama":
            self._provider = create_provider("ollama", base_url=settings.ollama_base_url)
            self._models = {
                "generate": settings.ollama_generate_model,
                "fast": settings.ollama_fast_model,
                "judge": settings.ollama_judge_model,
            }
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

        self._last_stream_usage: LLMResponse | None = None

    async def classify(
        self,
        query: str,
        system_prompt: str,
        allowed_values: list[str] | None = None,
        history: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """Cheap classification call. Fast model, ~20 tokens, <50ms."""
        user_content = query
        if history:
            turns = history[-6:]  # last 6 turns
            ctx = "\n".join(f"{t['role']}: {t['content'][:500]}" for t in turns)
            user_content = f"Conversation:\n{ctx}\n\nCurrent query: {query}"

        resp = await self._call("fast", system_prompt, user_content, max_tokens=100, **kwargs)
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
        """Full generation call. Generate model."""
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        return await self._call("generate", system_prompt, user_content, **kwargs)

    async def extract(
        self,
        text: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Entity/relationship extraction. Generate model, structured output."""
        return await self._call("generate", system_prompt, text, **kwargs)

    async def judge(
        self,
        answer: str,
        reference: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """LLM-as-judge evaluation. Uses judge model to avoid same-model bias."""
        user_content = f"Reference answer:\n{reference}\n\nCandidate answer:\n{answer}"
        return await self._call("judge", system_prompt, user_content, max_tokens=300, **kwargs)

    async def _call(
        self,
        model_key: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Core API call with retry (3 attempts, exponential backoff) and 60s timeout."""
        _retryable = _retryable_exceptions()
        last_exc: BaseException = RuntimeError("LLM call failed before any attempt")
        for attempt in range(3):
            try:
                return await asyncio.wait_for(
                    self._call_once(model_key, system, user, max_tokens, **kwargs),
                    timeout=60.0,
                )
            except TimeoutError as exc:
                last_exc = exc
                logger.warning("LLM call timed out (attempt %d/3)", attempt + 1)
            except _retryable as exc:
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
        model_key: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Single API call delegated to the active provider."""
        model = self._models[model_key]
        temperature = kwargs.pop("temperature", 0)
        resp = await self._provider.complete(
            model=model,
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cost = _compute_cost(
            model,
            resp.input_tokens,
            resp.output_tokens,
            resp.cache_creation_tokens,
            resp.cache_read_tokens,
        )
        return LLMResponse(
            text=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
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
        model = self._models["generate"]
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        max_tokens = kwargs.pop("max_tokens", 4096)

        async for text in self._provider.stream_text(
            model=model,
            system=system_prompt,
            user=user_content,
            max_tokens=max_tokens,
        ):
            yield text

        # Capture usage if the provider recorded it (Anthropic does, others may not)
        raw = getattr(self._provider, "last_stream_response", None)
        if raw is not None:
            cost = _compute_cost(
                model,
                raw.input_tokens,
                raw.output_tokens,
                raw.cache_creation_tokens,
                raw.cache_read_tokens,
            )
            self._last_stream_usage = LLMResponse(
                text="",
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                cost_usd=cost,
            )
