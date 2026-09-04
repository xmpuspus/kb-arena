"""Dual-model LLM client with prompt caching (cloudwright pattern).

Haiku for classification (~20 tokens, <50ms).
Sonnet for generation, extraction, evaluation.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass

from kb_arena.settings import settings

logger = logging.getLogger(__name__)
_retrieval_only_mode: ContextVar[bool] = ContextVar("kb_arena_retrieval_only", default=False)

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


# The judge's user message, section by section. The evaluator hashes this
# with the system prompt, so a reorder or a reword moves the prompt hash.
JUDGE_USER_TEMPLATE = {
    "question": "Question:\n{question}",
    "reference": "Reference answer:\n{reference}",
    "candidate": "Candidate answer:\n{answer}",
}


@dataclass
class LLMResponse:
    """Result from an LLM call, including text and usage metrics."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Which provider and model produced the text, so a judge verdict can
    # name its judge in the result file.
    provider: str = ""
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_BACKOFF_CAP_S = 30.0
_BACKOFF_JITTER_S = 0.5


async def _sleep(seconds: float) -> None:
    """Backoff sleep, kept separate so tests can replace it without touching asyncio."""
    await asyncio.sleep(seconds)


def _is_retryable(exc: BaseException, provider_name: str) -> bool:
    """Decide whether an error from the active provider is transient.

    Each SDK raises its own classes, so the old Anthropic-only tuple let every OpenAI
    and Ollama rate limit fail on the first attempt. A timeout is transient everywhere.
    """
    if isinstance(exc, TimeoutError):
        return True
    if provider_name == "anthropic":
        try:
            import anthropic
        except ImportError:
            return isinstance(exc, OSError)
        return isinstance(
            exc,
            anthropic.RateLimitError
            | anthropic.APIConnectionError
            | anthropic.APITimeoutError
            | anthropic.InternalServerError,
        )
    if provider_name == "openai":
        try:
            import openai
        except ImportError:
            return isinstance(exc, OSError)
        return isinstance(
            exc,
            openai.RateLimitError
            | openai.APIConnectionError
            | openai.APITimeoutError
            | openai.InternalServerError,
        )
    if provider_name == "ollama":
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in _RETRYABLE_HTTP_STATUS
        return isinstance(exc, httpx.TransportError | OSError)
    return isinstance(exc, OSError)


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter so concurrent callers do not retry in lockstep."""
    return min(float(2**attempt), _BACKOFF_CAP_S) + random.uniform(0.0, _BACKOFF_JITTER_S)


def _provider_setup(provider_name: str, api_key: str | None = None, *, generic_key: bool = True):
    """The provider object and its model names for one provider.

    KB_ARENA_LLM_API_KEY belongs to the generation provider. A judge on a
    second provider must not inherit it, so that path passes generic_key=False
    and reads only its own provider's key.
    """
    from kb_arena.llm.providers import create_provider

    shared = settings.llm_api_key if generic_key else ""
    if provider_name == "anthropic":
        key = api_key or shared or settings.anthropic_api_key
        return create_provider("anthropic", api_key=key), {
            "generate": settings.generate_model,
            "fast": settings.fast_model,
            "judge": settings.judge_model,
        }
    if provider_name == "openai":
        key = api_key or shared or settings.openai_api_key
        return create_provider("openai", api_key=key), {
            "generate": settings.openai_generate_model,
            "fast": settings.openai_fast_model,
            "judge": settings.openai_judge_model,
        }
    if provider_name == "ollama":
        return create_provider("ollama", base_url=settings.ollama_base_url), {
            "generate": settings.ollama_generate_model,
            "fast": settings.ollama_fast_model,
            "judge": settings.ollama_judge_model,
        }
    raise ValueError(f"Unknown provider: {provider_name}")


class LLMClient:
    def __init__(self, api_key: str | None = None):
        provider_name = settings.llm_provider
        self._provider_name = provider_name
        self._provider, self._models = _provider_setup(provider_name, api_key)

        # The judge can live on another provider. The api_key argument belongs
        # to the generation provider, so the judge provider reads its own key
        # from settings.
        judge_name = settings.judge_provider or provider_name
        self._judge_provider_name = judge_name
        if judge_name == provider_name:
            self._judge_provider = self._provider
        else:
            self._judge_provider, judge_models = _provider_setup(judge_name, generic_key=False)
            self._models["judge"] = judge_models["judge"]

        self._last_stream_usage: LLMResponse | None = None

    @property
    def judge_identity(self) -> dict[str, str]:
        """The provider and model that grade answers, for a run's snapshot."""
        return {"provider": self._judge_provider_name, "model": self._models["judge"]}

    def _provider_for(self, model_key: str):
        if model_key == "judge":
            return self._judge_provider, self._judge_provider_name
        return self._provider, self._provider_name

    async def classify(
        self,
        query: str,
        system_prompt: str,
        allowed_values: list[str] | None = None,
        history: list[dict] | None = None,
        **kwargs,
    ) -> str:
        """Cheap classification call. Fast model, ~20 tokens, <50ms."""
        if _retrieval_only_mode.get():
            return ""
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
        if _retrieval_only_mode.get():
            return LLMResponse(text="", input_tokens=0, output_tokens=0, cost_usd=0.0)
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        return await self._call("generate", system_prompt, user_content, **kwargs)

    async def extract(
        self,
        text: str,
        system_prompt: str,
        **kwargs,
    ) -> LLMResponse:
        """Entity/relationship extraction. Generate model, structured output."""
        if _retrieval_only_mode.get():
            return LLMResponse(text="", input_tokens=0, output_tokens=0, cost_usd=0.0)
        return await self._call("generate", system_prompt, text, **kwargs)

    async def judge(
        self,
        answer: str,
        reference: str,
        system_prompt: str,
        question: str = "",
        **kwargs,
    ) -> LLMResponse:
        """LLM-as-judge evaluation. Uses judge model to avoid same-model bias.

        The judge scores whether the candidate answers the question, so the question
        goes first. Without it the judge can only measure similarity to the reference.
        """
        parts = []
        if question:
            parts.append(JUDGE_USER_TEMPLATE["question"].format(question=question))
        parts.append(JUDGE_USER_TEMPLATE["reference"].format(reference=reference))
        parts.append(JUDGE_USER_TEMPLATE["candidate"].format(answer=answer))
        user_content = "\n\n".join(parts)
        return await self._call("judge", system_prompt, user_content, max_tokens=400, **kwargs)

    async def _call(
        self,
        model_key: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Core API call with a per-attempt deadline and provider-aware retry."""
        attempts = max(1, settings.llm_max_attempts)
        last_exc: BaseException = RuntimeError("LLM call failed before any attempt")
        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    self._call_once(model_key, system, user, max_tokens, **kwargs),
                    timeout=settings.llm_call_timeout_s,
                )
            except Exception as exc:
                if not _is_retryable(exc, self._provider_for(model_key)[1]):
                    raise
                last_exc = exc
                if attempt >= attempts - 1:
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                await _sleep(delay)
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
        provider, provider_name = self._provider_for(model_key)
        temperature = kwargs.pop("temperature", 0)
        resp = await provider.complete(
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
            provider=provider_name,
            model=model,
        )

    async def stream(
        self,
        query: str,
        context: str,
        system_prompt: str,
        include_usage: bool = False,
        **kwargs,
    ) -> AsyncIterator[str | LLMResponse]:
        """Stream text deltas and optionally a final per-call usage marker."""
        from kb_arena.llm.providers import ProviderResponse

        model = self._models["generate"]
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        max_tokens = kwargs.pop("max_tokens", 4096)
        attempts = max(1, settings.llm_max_attempts)

        raw: ProviderResponse | None = None
        for attempt in range(attempts):
            iterator = self._provider.stream_text(
                model=model,
                system=system_prompt,
                user=user_content,
                max_tokens=max_tokens,
            ).__aiter__()
            first_token_seen = False
            try:
                while True:
                    # Before the first token a stall is a setup problem and can retry.
                    # After it, a retry would repeat text the caller already showed.
                    deadline = (
                        settings.llm_stream_idle_timeout_s
                        if first_token_seen
                        else settings.llm_stream_first_token_timeout_s
                    )
                    try:
                        item = await asyncio.wait_for(iterator.__anext__(), timeout=deadline)
                    except StopAsyncIteration:
                        break
                    if isinstance(item, ProviderResponse):
                        raw = item
                        continue
                    first_token_seen = True
                    yield item
                break
            except Exception as exc:
                if (
                    first_token_seen
                    or not _is_retryable(exc, self._provider_name)
                    or attempt >= attempts - 1
                ):
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    "LLM stream failed before the first token (attempt %d/%d): %s. "
                    "Retrying in %.1fs",
                    attempt + 1,
                    attempts,
                    exc,
                    delay,
                )
                await _sleep(delay)
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001 - closing a broken stream must not mask the cause
                        pass

        # Backward compatibility for third-party providers that only expose the
        # legacy instance field. Built-in providers emit a per-call marker.
        if raw is None:
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
            if include_usage:
                yield self._last_stream_usage
