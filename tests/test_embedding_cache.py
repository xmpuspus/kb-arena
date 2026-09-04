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


def test_the_endpoint_is_part_of_the_key_for_a_self_hosted_provider(tmp_path):
    path = tmp_path / "c.sqlite"
    one = _Provider()
    CachedEmbedding(one, provider="ollama", model="m", path=path, endpoint="http://a:11434")(
        ["alpha"]
    )
    two = _Provider()
    CachedEmbedding(two, provider="ollama", model="m", path=path, endpoint="http://b:11434")(
        ["alpha"]
    )

    assert two.calls == [["alpha"]], "another server never shares a vector"
    assert cache_key("ollama", "m", "alpha", "http://a:11434") != cache_key(
        "ollama", "m", "alpha", "http://b:11434"
    )
    assert cache_key("openai", "m", "alpha") == cache_key("openai", "m", "alpha", "")


def test_two_wrappers_on_one_file_share_one_lock(tmp_path):
    path = tmp_path / "c.sqlite"
    first = CachedEmbedding(_Provider(), provider="fake", model="m", path=path)
    second = CachedEmbedding(_Provider(), provider="fake", model="m", path=path)
    other = CachedEmbedding(_Provider(), provider="fake", model="m", path=tmp_path / "d.sqlite")

    assert first._lock is second._lock
    assert first._lock is not other._lock


def test_the_factory_gives_ollama_its_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "embedding_provider", "ollama")
    monkeypatch.setattr(settings, "embedding_cache_path", str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(settings, "ollama_base_url", "http://box:11434")
    monkeypatch.setattr(embeddings, "_PROVIDERS", {"ollama": _Provider})

    wrapped = embeddings.get_embedding_function()

    assert wrapped.identity["endpoint"] == "http://box:11434"


def test_two_callers_that_miss_on_one_text_pay_for_it_once(tmp_path):
    import threading
    import time

    class _Slow:
        def __init__(self):
            self.calls: list[list[str]] = []

        def __call__(self, input):
            self.calls.append(list(input))
            time.sleep(0.3)
            return [[float(len(text)), 1.0] for text in input]

    provider = _Slow()
    path = tmp_path / "c.sqlite"
    first = CachedEmbedding(provider, provider="fake", model="m", path=path)
    second = CachedEmbedding(provider, provider="fake", model="m", path=path)
    results: dict[str, list] = {}

    def run(name, cache):
        results[name] = _plain(cache(["shared", name]))

    threads = [
        threading.Thread(target=run, args=("a", first)),
        threading.Thread(target=run, args=("b", second)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    shared_calls = sum(1 for call in provider.calls if "shared" in call)
    assert shared_calls == 1, "the second caller waited for the first instead of paying again"
    assert results["a"][0] == [6.0, 1.0] and results["b"][0] == [6.0, 1.0]


def test_the_cache_never_lives_under_the_chroma_path(tmp_path, monkeypatch):
    from kb_arena.strategies.embedding_cache import default_cache_path

    monkeypatch.setattr(settings, "embedding_cache_path", "")
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "trial-temp"))
    assert str(tmp_path / "trial-temp") not in str(default_cache_path().resolve())
    monkeypatch.setattr(settings, "embedding_cache_path", str(tmp_path / "keep.sqlite"))
    assert default_cache_path() == tmp_path / "keep.sqlite"


def test_a_local_provider_names_its_own_model_in_the_key(tmp_path, monkeypatch):
    class _Local:
        def __init__(self, model="local-weights-v1"):
            self._model = model

        def __call__(self, input):
            return [[1.0] for _ in input]

    monkeypatch.setattr(settings, "embedding_provider", "bge")
    monkeypatch.setattr(settings, "embedding_cache_path", str(tmp_path / "c.sqlite"))
    monkeypatch.setattr(settings, "embedding_model", "text-embedding-3-large")
    monkeypatch.setattr(embeddings, "_PROVIDERS", {"bge": _Local})

    wrapped = embeddings.get_embedding_function()

    assert wrapped.identity["model"] == "local-weights-v1"


def test_the_bge_provider_records_its_model_name():
    import inspect

    from kb_arena.strategies.embeddings import BGEEmbedding

    source = inspect.getsource(BGEEmbedding.__init__)
    assert "self._model = model" in source


def test_the_cache_file_uses_wal(tmp_path):
    import sqlite3

    cache = CachedEmbedding(_Provider(), provider="fake", model="m", path=tmp_path / "c.sqlite")
    cache(["alpha"])
    with sqlite3.connect(tmp_path / "c.sqlite") as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_a_corrupt_vector_row_is_dropped_and_embedded_again(tmp_path):
    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m", path=path)
    cache(["alpha"])
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE vectors SET vector = ?", (json.dumps([1.0, "x"]),))

    assert _plain(cache(["alpha"])) == [[5.0, 1.0]]
    assert provider.calls == [["alpha"], ["alpha"]]
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE vectors SET vector = ?", (json.dumps([float("nan"), 1.0]),))
    assert _plain(cache(["alpha"])) == [[5.0, 1.0]]


def test_a_row_with_the_wrong_length_for_its_identity_is_dropped(tmp_path):
    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m", path=path)
    cache(["alpha", "beta", "gamma"])
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE vectors SET vector = ? WHERE key = ?",
            (json.dumps([9.0]), cache_key("fake", "m", "beta")),
        )

    vectors = _plain(cache(["alpha", "beta", "gamma"]))

    assert vectors[1] == [4.0, 1.0], "the short row went back to the provider"
    assert provider.calls[-1] == ["beta"]


def test_the_salt_retires_old_vectors(tmp_path, monkeypatch):
    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="ollama", model="m", path=path)
    monkeypatch.setattr(settings, "embedding_cache_salt", "")
    cache(["alpha"])
    monkeypatch.setattr(settings, "embedding_cache_salt", "rev-2")
    cache(["alpha"])

    assert provider.calls == [["alpha"], ["alpha"]]


def test_a_claim_in_the_database_stops_a_second_process_from_paying_twice(tmp_path):
    import sqlite3 as _sqlite
    import time as _time

    path = tmp_path / "c.sqlite"
    first = CachedEmbedding(_Provider(), provider="fake", model="m", path=path)
    key = cache_key("fake", "m", "alpha")
    # another process claimed the key a moment ago
    with _sqlite.connect(path) as conn:
        conn.execute("INSERT INTO claims (key, claimed_at) VALUES (?, ?)", (key, _time.time()))
        conn.commit()

    import threading

    def other_process_finishes():
        _time.sleep(0.2)
        with _sqlite.connect(path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)",
                (key, json.dumps([7.0, 7.0])),
            )
            conn.execute("DELETE FROM claims WHERE key = ?", (key,))
            conn.commit()

    threading.Thread(target=other_process_finishes).start()
    vectors = _plain(first(["alpha"]))

    assert vectors == [
        [7.0, 7.0]
    ], "the vector came from the other process, not a second provider call"
    assert first._inner.calls == []


def test_a_stale_claim_is_taken_over(tmp_path):
    import sqlite3 as _sqlite

    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m", path=path)
    with _sqlite.connect(path) as conn:
        conn.execute(
            "INSERT INTO claims (key, claimed_at) VALUES (?, ?)",
            (cache_key("fake", "m", "alpha"), 1.0),
        )
        conn.commit()

    assert _plain(cache(["alpha"])) == [[5.0, 1.0]]
    assert provider.calls == [["alpha"]]
    with _sqlite.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 0


def test_the_cache_file_is_ignored_by_git():
    from pathlib import Path

    assert "embedding_cache.sqlite" in Path(".gitignore").read_text()


def test_a_lone_row_with_the_wrong_length_for_its_identity_is_dropped(tmp_path):
    path = tmp_path / "c.sqlite"
    provider = _Provider()
    cache = CachedEmbedding(provider, provider="fake", model="m", path=path)
    cache(["alpha"])  # records the identity's length, 2
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE vectors SET vector = ?", (json.dumps([9.0, 9.0, 9.0]),))

    assert _plain(cache(["alpha"])) == [
        [5.0, 1.0]
    ], "a lone wrong-length row went back to the provider"
    assert provider.calls == [["alpha"], ["alpha"]]


def test_provider_output_with_a_bad_element_is_rejected_not_stored(tmp_path):
    class _Bad:
        def __call__(self, input):
            return [[1.0, "2.0"] for _ in input]

    class _Bool:
        def __call__(self, input):
            return [[True, 1.0] for _ in input]

    for bad in (_Bad(), _Bool()):
        cache = CachedEmbedding(bad, provider="fake", model="m", path=tmp_path / "c.sqlite")
        with pytest.raises(RuntimeError, match="non-numeric"):
            cache(["alpha"])
    with sqlite3.connect(tmp_path / "c.sqlite") as conn:
        assert conn.execute("SELECT count(*) FROM vectors").fetchone()[0] == 0
