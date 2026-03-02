"""Live test fixtures — real API keys, real LLM/embedding calls."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def live_settings():
    """Load real settings from .env."""
    from kb_arena.settings import Settings
    return Settings()


@pytest.fixture(scope="session")
def live_llm_client(live_settings):
    """Real LLM client backed by actual Anthropic API key."""
    from kb_arena.llm.client import LLMClient
    if not live_settings.anthropic_api_key:
        pytest.skip("KB_ARENA_ANTHROPIC_API_KEY not set in .env")
    return LLMClient(api_key=live_settings.anthropic_api_key)


@pytest.fixture(scope="session")
def live_openai_key():
    """Return the raw OpenAI API key from the environment (no KB_ARENA_ prefix)."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    return key


def pytest_collection_modifyitems(items):
    for item in items:
        if "live" in str(item.fspath):
            item.add_marker(pytest.mark.live)
