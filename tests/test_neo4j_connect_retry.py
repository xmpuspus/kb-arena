"""Neo4jStore.connect retries a database that still warms up."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import ServiceUnavailable

from kb_arena.graph import neo4j_store


def _fake_driver(failures: int) -> MagicMock:
    driver = MagicMock()
    driver.close = AsyncMock()
    attempts = {"n": 0}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def run(self, query):
            attempts["n"] += 1
            if attempts["n"] <= failures:
                raise ServiceUnavailable("still starting")
            result = MagicMock()
            result.consume = AsyncMock()
            return result

    driver.session = lambda **kwargs: _Session()
    driver.attempts = attempts
    return driver


@pytest.mark.asyncio
async def test_connect_retries_until_neo4j_answers(monkeypatch):
    driver = _fake_driver(failures=2)
    monkeypatch.setattr(neo4j_store.AsyncGraphDatabase, "driver", lambda *a, **k: driver)
    sleeper = AsyncMock()
    monkeypatch.setattr(neo4j_store, "_sleep", sleeper)

    store = await neo4j_store.Neo4jStore.connect(uri="bolt://x", user="u", password="p")

    assert store is not None
    assert driver.attempts["n"] == 3
    assert sleeper.await_count == 2
    driver.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_connect_gives_up_after_the_last_attempt(monkeypatch):
    driver = _fake_driver(failures=10)
    monkeypatch.setattr(neo4j_store.AsyncGraphDatabase, "driver", lambda *a, **k: driver)
    monkeypatch.setattr(neo4j_store, "_sleep", AsyncMock())

    with pytest.raises(ServiceUnavailable):
        await neo4j_store.Neo4jStore.connect(uri="bolt://x", user="u", password="p")

    assert driver.attempts["n"] == 3
    driver.close.assert_awaited_once()
