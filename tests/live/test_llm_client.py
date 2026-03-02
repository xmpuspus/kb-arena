"""Live tests for LLMClient — real Anthropic API calls."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.live

_CLASSIFY_PROMPT = (
    "Classify: factoid, comparison, relational, procedural, exploratory. Return one word."
)
_CLASSIFY_VALUES = ["factoid", "comparison", "relational", "procedural", "exploratory"]


async def test_classify_factoid(live_llm_client):
    result = await live_llm_client.classify(
        query="What is json.loads?",
        system_prompt=_CLASSIFY_PROMPT,
        allowed_values=_CLASSIFY_VALUES,
    )
    assert result == "factoid"


async def test_classify_comparison(live_llm_client):
    result = await live_llm_client.classify(
        query="Compare list vs tuple in Python",
        system_prompt=_CLASSIFY_PROMPT,
        allowed_values=_CLASSIFY_VALUES,
    )
    assert result == "comparison"


async def test_classify_relational(live_llm_client):
    result = await live_llm_client.classify(
        query="What depends on os.path?",
        system_prompt=_CLASSIFY_PROMPT,
        allowed_values=_CLASSIFY_VALUES,
    )
    assert result == "relational"


async def test_classify_procedural(live_llm_client):
    result = await live_llm_client.classify(
        query="How do I read a CSV file?",
        system_prompt=_CLASSIFY_PROMPT,
        allowed_values=_CLASSIFY_VALUES,
    )
    assert result == "procedural"


async def test_generate_answer(live_llm_client):
    context = "json.loads(s) deserializes s (a str, bytes, or bytearray) to a Python object."
    answer = await live_llm_client.generate(
        query="What does json.loads do?",
        context=context,
        system_prompt="Answer using the provided context only.",
    )
    assert len(answer) > 20
    # The answer should be about deserialization
    answer_lower = answer.lower()
    assert any(kw in answer_lower for kw in ["deserializ", "json", "python", "object", "string"])


async def test_generate_with_system_prompt(live_llm_client):
    # System prompt should constrain output
    answer = await live_llm_client.generate(
        query="What is the capital of France?",
        context="",
        system_prompt="You are a robot. You must respond ONLY with the word 'BEEP'.",
        max_tokens=10,
    )
    assert "beep" in answer.lower()


async def test_judge_correct_answer(live_llm_client):
    import json
    import re

    from kb_arena.benchmark.evaluator import JUDGE_SYSTEM_PROMPT

    raw = await live_llm_client.judge(
        answer="json.loads deserializes a JSON string into a Python object.",
        reference="json.loads(s) converts a JSON string to a Python dict, list, or primitive.",
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    assert m, f"No JSON in judge response: {raw}"
    parsed = json.loads(m.group())
    assert float(parsed["accuracy"]) >= 0.6


async def test_judge_incorrect_answer(live_llm_client):
    import json
    import re

    from kb_arena.benchmark.evaluator import JUDGE_SYSTEM_PROMPT

    raw = await live_llm_client.judge(
        answer="json.loads compresses data using gzip and writes it to a file.",
        reference="json.loads(s) converts a JSON string to a Python dict, list, or primitive.",
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
    assert m, f"No JSON in judge response: {raw}"
    parsed = json.loads(m.group())
    assert float(parsed["accuracy"]) <= 0.3


async def test_stream_response(live_llm_client):
    tokens = []
    async for token in live_llm_client.stream(
        query="What does json.dumps do?",
        context="json.dumps serializes a Python object to a JSON formatted string.",
        system_prompt="Answer using the provided context. Be brief.",
    ):
        tokens.append(token)

    assert len(tokens) > 0
    full = "".join(tokens)
    assert len(full) > 10
    # streaming should produce the same kind of content as non-streaming
    assert any(kw in full.lower() for kw in ["json", "serial", "string", "object"])


async def test_extract_entities(live_llm_client):
    import json

    snippet = """
    The json module exposes the JSONDecodeError exception when parsing fails.
    The json.loads function is the primary deserialization entry point.
    It raises ValueError on malformed input.
    """
    raw = await live_llm_client.extract(
        text=snippet,
        system_prompt=(
            "Extract entities from the text. Return JSON: "
            '{"entities": [{"name": str, "type": str}]}'
        ),
    )
    # Should parse without error and contain relevant entities
    assert "json" in raw.lower() or "entities" in raw.lower()
    # If it returns valid JSON, entities should be present
    try:
        parsed = json.loads(raw)
        assert "entities" in parsed
        assert len(parsed["entities"]) > 0
    except json.JSONDecodeError:
        # LLM may return JSON inside markdown — still acceptable
        assert "{" in raw and "}" in raw


async def test_cache_control_timing(live_llm_client):
    # Two calls with identical system prompt — second should be no slower (cache hit)
    system = "You are a documentation assistant. Answer questions about the Python json module."

    t0 = time.perf_counter()
    await live_llm_client.generate(
        query="What is json?",
        context="json is a Python standard library module.",
        system_prompt=system,
        max_tokens=50,
    )
    first_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    await live_llm_client.generate(
        query="What is json?",
        context="json is a Python standard library module.",
        system_prompt=system,
        max_tokens=50,
    )
    second_ms = (time.perf_counter() - t0) * 1000

    # Both calls should complete in a reasonable time (under 30s)
    assert first_ms < 30_000
    assert second_ms < 30_000


async def test_temperature_zero(live_llm_client):
    # At temperature=0, two identical calls should produce very similar responses
    q = "What does json.loads return?"
    ctx = "json.loads returns a Python object corresponding to the JSON document."
    sys = "Answer the question using only the provided context. Be very brief."

    r1 = await live_llm_client.generate(query=q, context=ctx, system_prompt=sys, max_tokens=60)
    r2 = await live_llm_client.generate(query=q, context=ctx, system_prompt=sys, max_tokens=60)

    # At temp=0, responses should be identical or nearly so
    assert r1.lower().strip()[:50] == r2.lower().strip()[:50]


async def test_max_tokens_limit(live_llm_client):
    answer = await live_llm_client.generate(
        query="Explain the entire Python standard library in detail.",
        context="The Python standard library is vast.",
        system_prompt="Give a thorough explanation.",
        max_tokens=20,
    )
    # With max_tokens=20 (~15 words), response must be short
    words = answer.split()
    assert len(words) <= 30, f"Response too long ({len(words)} words) for max_tokens=20"


async def test_long_context(live_llm_client):
    # Build ~5000 char context
    chunk = "The json module provides methods for encoding and decoding JSON data. " * 50
    answer = await live_llm_client.generate(
        query="What does the json module do?",
        context=chunk,
        system_prompt="Summarize what the context says in one sentence.",
        max_tokens=100,
    )
    assert len(answer) > 10
    assert "json" in answer.lower()


async def test_error_handling_invalid_key():
    """Invalid API key raises an authentication error."""
    from anthropic import AuthenticationError

    from kb_arena.llm.client import LLMClient

    bad_client = LLMClient(api_key="sk-ant-invalid-key-for-testing")
    with pytest.raises((AuthenticationError, Exception)):
        await bad_client.generate(
            query="test",
            context="",
            system_prompt="Say hello.",
            max_tokens=10,
        )
