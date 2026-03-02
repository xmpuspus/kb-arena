"""Dual-model LLM client with prompt caching (cloudwright pattern).

Haiku for classification (~20 tokens, <50ms).
Sonnet for generation, extraction, evaluation.
"""

from __future__ import annotations

import anthropic

from kb_arena.settings import settings

GENERATE_MODEL = settings.generate_model
FAST_MODEL = settings.fast_model


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

        result = await self._call(FAST_MODEL, system_prompt, user_content, max_tokens=100, **kwargs)
        result = result.strip().lower()

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
    ) -> str:
        """Full generation call. Sonnet."""
        user_content = f"Context:\n{context}\n\nQuery: {query}" if context else query
        return await self._call(GENERATE_MODEL, system_prompt, user_content, **kwargs)

    async def extract(
        self,
        text: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Entity/relationship extraction. Sonnet, structured output."""
        return await self._call(GENERATE_MODEL, system_prompt, text, **kwargs)

    async def judge(
        self,
        answer: str,
        reference: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """LLM-as-judge evaluation. Sonnet."""
        user_content = f"Reference answer:\n{reference}\n\nCandidate answer:\n{answer}"
        return await self._call(
            GENERATE_MODEL, system_prompt, user_content, max_tokens=300, **kwargs
        )

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Core API call with cache_control on system prompt."""
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
        return response.content[0].text

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
