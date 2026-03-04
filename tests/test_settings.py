"""Tests for Settings — defaults and env var overrides."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_settings_default_generate_model():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.generate_model == "claude-sonnet-4-6"


def test_settings_default_fast_model():
    from kb_arena.settings import Settings

    s = Settings()
    assert "haiku" in s.fast_model.lower()


def test_settings_default_neo4j_uri():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.neo4j_uri == "bolt://localhost:7687"


def test_settings_default_neo4j_user():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.neo4j_user == "neo4j"


def test_settings_default_chroma_path():
    from kb_arena.settings import Settings

    s = Settings()
    assert "chroma" in s.chroma_path.lower()


def test_settings_default_port():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.port == 8000


def test_settings_default_host():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.host == "0.0.0.0"


def test_settings_default_debug_false():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.debug is False


def test_settings_default_datasets_path():
    from kb_arena.settings import Settings

    s = Settings()
    assert "datasets" in s.datasets_path


def test_settings_default_results_path():
    from kb_arena.settings import Settings

    s = Settings()
    assert "results" in s.results_path


def test_settings_default_benchmark_max_concurrent():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.benchmark_max_concurrent == 5


def test_settings_default_benchmark_max_retries():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.benchmark_max_retries == 2


def test_settings_default_benchmark_temperature():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.benchmark_temperature == 0.0


def test_settings_default_embedding_model():
    from kb_arena.settings import Settings

    s = Settings()
    assert "text-embedding" in s.embedding_model


def test_settings_default_anthropic_api_key_empty():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.anthropic_api_key == ""


def test_settings_default_query_timeout():
    from kb_arena.settings import Settings

    s = Settings()
    assert s.benchmark_query_timeout_s == 120


# ---------------------------------------------------------------------------
# Custom env vars via monkeypatch
# ---------------------------------------------------------------------------


def test_settings_port_via_env(monkeypatch):
    monkeypatch.setenv("KB_ARENA_PORT", "9999")
    from kb_arena.settings import Settings

    s = Settings()
    assert s.port == 9999


def test_settings_debug_true_via_env(monkeypatch):
    monkeypatch.setenv("KB_ARENA_DEBUG", "true")
    from kb_arena.settings import Settings

    s = Settings()
    assert s.debug is True


def test_settings_neo4j_uri_via_env(monkeypatch):
    monkeypatch.setenv("KB_ARENA_NEO4J_URI", "bolt://remote:7687")
    from kb_arena.settings import Settings

    s = Settings()
    assert s.neo4j_uri == "bolt://remote:7687"


def test_settings_datasets_path_via_env(monkeypatch):
    monkeypatch.setenv("KB_ARENA_DATASETS_PATH", "/tmp/my-datasets")
    from kb_arena.settings import Settings

    s = Settings()
    assert s.datasets_path == "/tmp/my-datasets"


def test_settings_extra_fields_ignored(monkeypatch):
    monkeypatch.setenv("KB_ARENA_NONEXISTENT_FIELD", "value")
    from kb_arena.settings import Settings

    s = Settings()
    assert s is not None
