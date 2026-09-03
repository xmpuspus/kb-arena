"""A blocking Chroma call inside an async strategy no longer freezes the event loop."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from kb_arena.strategies import chroma_index
from kb_arena.strategies.naive_vector import NaiveVectorStrategy
from kb_arena.strategies.raptor import RaptorStrategy

_DELAY = 0.3
_HIT = {
    "ids": [["doc1::s1::0"]],
    "documents": [["one chunk"]],
    "metadatas": [[{"source_id": "doc1", "chunk_id": "doc1::s1::0"}]],
    "distances": [[0.1]],
}


def _slow_client(delay: float = _DELAY) -> MagicMock:
    # Chroma's query() and upsert() embed on the calling thread. This stand-in
    # sleeps the same way, so a frozen loop shows up as missing heartbeats.
    client = MagicMock()
    collection = MagicMock()

    def slow_query(**kwargs):
        time.sleep(delay)
        return _HIT

    def slow_upsert(**kwargs):
        time.sleep(delay)

    collection.query.side_effect = slow_query
    collection.upsert.side_effect = slow_upsert
    collection.count.return_value = 1
    client.get_or_create_collection.return_value = collection
    return client


async def _heartbeat(stop: asyncio.Event, ticks: list[float]) -> None:
    while not stop.is_set():
        ticks.append(time.perf_counter())
        await asyncio.sleep(0.01)


async def _ticks_during(coro) -> int:
    ticks: list[float] = []
    stop = asyncio.Event()
    beat = asyncio.create_task(_heartbeat(stop, ticks))
    try:
        await coro
    finally:
        stop.set()
        await beat
    return len(ticks)


@pytest.mark.asyncio
async def test_two_naive_queries_overlap_instead_of_running_one_at_a_time(mock_llm_client):
    strategy = NaiveVectorStrategy(chroma_client=_slow_client())
    strategy._llm = mock_llm_client
    # The first call also builds the embedding function, which takes longer
    # than the search itself. Warm that up so the timing below sees only
    # the two searches.
    await strategy.query("warm-up")

    started = time.perf_counter()
    await asyncio.gather(strategy.query("first"), strategy.query("second"))
    elapsed = time.perf_counter() - started

    # Two 0.3 s searches in well under 0.6 s means they ran at the same time.
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_the_loop_keeps_ticking_during_a_naive_search(mock_llm_client):
    strategy = NaiveVectorStrategy(chroma_client=_slow_client())
    strategy._llm = mock_llm_client

    ticks = await _ticks_during(strategy.query("question"))

    # A 10 ms heartbeat sees about 30 ticks across a 0.3 s search. A loop
    # frozen by the search records one or two.
    assert ticks >= 10


@pytest.mark.asyncio
async def test_the_loop_keeps_ticking_during_a_naive_index_build(sample_documents):
    strategy = NaiveVectorStrategy(chroma_client=_slow_client())

    ticks = await _ticks_during(strategy.build_index(sample_documents))

    assert ticks >= 10


@pytest.mark.asyncio
async def test_the_loop_keeps_ticking_during_a_raptor_search(mock_llm_client):
    strategy = RaptorStrategy(chroma_client=_slow_client(0.1))
    strategy._llm = mock_llm_client

    # Three levels, 0.1 s each, all on one worker thread: 0.3 s of blocking.
    ticks = await _ticks_during(strategy.query("question"))

    assert ticks >= 10


@pytest.mark.asyncio
async def test_the_loop_keeps_ticking_through_a_cold_start(mock_llm_client):
    # The first request builds the client, the embedding function, and the
    # collection. That setup used to run on the loop thread.
    client = _slow_client(0.0)

    def slow_setup(**kwargs):
        time.sleep(_DELAY)
        return client.get_or_create_collection.return_value

    collection = client.get_or_create_collection.return_value
    client.get_or_create_collection.side_effect = slow_setup
    client.get_or_create_collection.return_value = collection
    strategy = NaiveVectorStrategy(chroma_client=client)
    strategy._llm = mock_llm_client

    ticks = await _ticks_during(strategy.query("first ever question"))

    assert ticks >= 10


@pytest.mark.asyncio
async def test_a_cancelled_build_keeps_the_lock_until_its_worker_finishes(sample_documents):
    # asyncio.to_thread cancels only the coroutine. The publish keeps running
    # on its thread, so the build lock must stay held until it is done.
    entered = threading.Event()
    release = threading.Event()
    client = MagicMock()
    collection = MagicMock()

    def blocking_upsert(**kwargs):
        entered.set()
        release.wait(5)

    collection.upsert.side_effect = blocking_upsert
    client.get_or_create_collection.return_value = collection
    strategy = NaiveVectorStrategy(chroma_client=client)

    build = asyncio.create_task(strategy.build_index(sample_documents))
    await asyncio.to_thread(entered.wait, 5)
    build.cancel()
    await asyncio.sleep(0.1)

    lock_path = chroma_index._index_directory() / chroma_index._LOCK_FILENAME
    with lock_path.open("a+b") as handle:
        # Still held: the worker has not finished, so no second build may start.
        assert chroma_index._try_lock(handle) is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await build
        assert chroma_index._try_lock(handle) is True
        chroma_index._unlock(handle)
