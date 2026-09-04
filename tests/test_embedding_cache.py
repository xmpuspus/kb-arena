"""An embedded text is paid for once, per provider and model, and reused by every strategy."""

from __future__ import annotations

import json
import sqlite3

import pytest

from kb_arena.settings import settings
from kb_arena.strategies import embeddings
from kb_arena.strategies.embedding_cache import CachedEmbedding, cache_key


class _Provider:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, input):
        self.calls.append(list(input))
        return [[float(len(text)), 1.0] for text in input]


def _plain(vectors):
    # chromadb's EmbeddingFunction wrapper hands back numpy arrays
    return [[float(x) for x in vector] for vector in vectors]


def test_a_second_call_only_embeds_the_new_texts(tmp_path):
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m1", path=tmp_path / "c.sqlite")

    first = _plain(cache(["alpha", "beta"]))
    second = _plain(cache(["beta", "gamma", "alpha"]))

    assert first == [[5.0, 1.0], [4.0, 1.0]]
    assert second == [[4.0, 1.0], [5.0, 1.0], [5.0, 1.0]]
    assert provider.calls == [["alpha", "beta"], ["gamma"]]
    assert (cache.hits, cache.misses) == (2, 3)


def test_duplicates_in_one_call_embed_once_and_keep_their_order(tmp_path):
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m1", path=tmp_path / "c.sqlite")

    vectors = _plain(cache(["same", "other", "same"]))

    assert vectors == [[4.0, 1.0], [5.0, 1.0], [4.0, 1.0]]
    assert provider.calls == [["same", "other"]]


def test_the_cache_persists_across_instances_and_is_keyed_by_model(tmp_path):
    path = tmp_path / "c.sqlite"
    first = _Provider()
    CachedEmbedding(first, provider="fake", model="m1", path=path)(["alpha"])

    second = _Provider()
    again = CachedEmbedding(second, provider="fake", model="m1", path=path)
    assert _plain(again(["alpha"])) == [[5.0, 1.0]]
    assert second.calls == [], "the vector came from disk"

    other_model = _Provider()
    CachedEmbedding(other_model, provider="fake", model="m2", path=path)(["alpha"])
    assert other_model.calls == [["alpha"]], "another model never shares a vector"
    assert cache_key("fake", "m1", "alpha") != cache_key("fake", "m2", "alpha")


def test_a_corrupt_row_is_embedded_again(tmp_path):
    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m1", path=path)
    cache(["alpha"])
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE vectors SET vector = ?", ("not json",))

    assert _plain(cache(["alpha"])) == [[5.0, 1.0]]
    assert provider.calls == [["alpha"], ["alpha"]]
    with sqlite3.connect(path) as conn:
        assert json.loads(conn.execute("SELECT vector FROM vectors").fetchone()[0]) == [5.0, 1.0]


def test_a_provider_that_returns_the_wrong_count_is_an_error(tmp_path):
    class _Short:
        def __call__(self, input):
            return [[1.0]]

    cache = CachedEmbedding(_Short(), provider="fake", model="m1", path=tmp_path / "c.sqlite")
    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        cache(["a", "b"])


def test_the_factory_wraps_the_provider_unless_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "bge")
    monkeypatch.setattr(settings, "embedding_cache_path", str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(embeddings, "_PROVIDERS", {"bge": _Provider})

    monkeypatch.setattr(settings, "embedding_cache_enabled", True)
    wrapped = embeddings.get_embedding_function()
    assert isinstance(wrapped, CachedEmbedding)
    assert wrapped.identity["provider"] == "bge"

    monkeypatch.setattr(settings, "embedding_cache_enabled", False)
    assert isinstance(embeddings.get_embedding_function(), _Provider)
