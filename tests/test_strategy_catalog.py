"""Contracts for the strategy catalog shared by runtime surfaces."""

from __future__ import annotations

import pytest

from kb_arena.benchmark.runner import STRATEGY_NAMES
from kb_arena.strategies import STRATEGY_REGISTRY, get_strategy
from kb_arena.strategies.catalog import (
    STRATEGY_CATALOG,
    default_strategy_names,
    missing_optional_modules,
    public_catalog,
)


def test_catalog_matches_the_registered_strategy_order():
    assert tuple(spec.name for spec in STRATEGY_CATALOG) == tuple(STRATEGY_REGISTRY)


def test_catalog_separates_registered_and_default_strategies():
    assert len(STRATEGY_CATALOG) == 13
    assert default_strategy_names() == tuple(STRATEGY_NAMES)
    assert len(default_strategy_names()) == 9


def test_catalog_labels_quantum_strategies_as_experiments():
    by_name = {spec.name: spec for spec in STRATEGY_CATALOG}

    assert by_name["qiss"].experimental is True
    assert by_name["qiss"].optional_extra is None
    assert by_name["rerank_vector"].default_benchmark is False
    assert by_name["rerank_vector"].optional_extra == "rerank"
    assert by_name["rerank_vector"].required_modules == ("sentence_transformers",)
    assert by_name["sqr"].experimental is True
    assert by_name["sqr"].default_benchmark is False
    assert by_name["sqr"].optional_extra == "quantum"
    assert by_name["sqr"].required_modules == ("qiskit", "qiskit_aer", "sklearn")


def test_reranker_missing_extra_is_explicit(monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def without_reranker(name: str, *args, **kwargs):
        if name == "sentence_transformers":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_reranker)

    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "reranker_backend", "bge")
    with pytest.raises(ImportError, match=r"kb-arena\[rerank\]"):
        get_strategy("rerank_vector")

    rerank = next(row for row in public_catalog([]) if row["name"] == "rerank_vector")
    assert rerank["status"] == "unavailable"
    assert "kb-arena[rerank]" in rerank["unavailable_reason"]


@pytest.mark.parametrize(
    ("backend", "module"),
    [("cohere", "cohere"), ("voyage", "voyageai")],
)
def test_remote_reranker_does_not_require_local_bge(
    monkeypatch, mock_chroma_client, backend, module
):
    import importlib.util

    from kb_arena.settings import settings

    real_find_spec = importlib.util.find_spec

    def selected_backend_only(name: str, *args, **kwargs):
        if name == "sentence_transformers":
            return None
        if name == module:
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(settings, "reranker_backend", backend)
    monkeypatch.setattr(importlib.util, "find_spec", selected_backend_only)
    monkeypatch.setattr("chromadb.PersistentClient", lambda **kwargs: mock_chroma_client)

    assert (
        missing_optional_modules(
            next(spec for spec in STRATEGY_CATALOG if spec.name == "rerank_vector")
        )
        == ()
    )
    assert get_strategy("rerank_vector").name == "rerank_vector"
