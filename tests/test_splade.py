"""Tests for Strategy 13: SPLADE, learned sparse retrieval over its own index.

The sparse-vector math is pure NumPy and Python (no transformers, no torch),
so these run in core CI. `[splade]` is exercised only through a mocked encoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from kb_arena.models.document import Document, Section
from kb_arena.strategies.catalog import STRATEGY_CATALOG, public_catalog
from kb_arena.strategies.splade import (
    SPLADEStrategy,
    sparse_dot,
    sparse_vector_to_terms,
    splade_weights_from_logits,
)


def test_splade_weights_from_logits_ignores_padding_and_stays_nonnegative():
    # Two tokens, three vocab terms. Padding (mask 0) must not win the max even
    # though its raw logit is the largest value in the batch.
    logits = np.array([[1.0, -2.0, 0.5], [9.0, 9.0, 9.0]])
    mask = np.array([1.0, 0.0])
    weights = splade_weights_from_logits(logits, mask)
    assert weights.shape == (3,)
    assert weights[0] == pytest.approx(np.log1p(1.0))
    assert weights[1] == pytest.approx(0.0)  # relu(-2.0) -> 0
    assert (weights >= 0).all()


def test_sparse_dot_scores_shared_terms_only():
    query_terms = sparse_vector_to_terms(np.array([1.0, 0.0, 2.0]))
    doc_terms = sparse_vector_to_terms(np.array([3.0, 5.0, 0.0]))
    # Term 0 is the only term both sides carry: 1.0 * 3.0.
    assert sparse_dot(query_terms, doc_terms) == pytest.approx(3.0)


def test_splade_missing_extra_is_explicit(monkeypatch):
    import importlib.util

    from kb_arena.strategies import get_strategy

    real_find_spec = importlib.util.find_spec

    def without_transformers(name: str, *args, **kwargs):
        if name in ("transformers", "torch"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_transformers)

    with pytest.raises(ImportError, match=r"kb-arena\[splade\]"):
        get_strategy("splade")

    spec = next(s for s in STRATEGY_CATALOG if s.name == "splade")
    assert spec.needs_embeddings is False
    row = next(r for r in public_catalog([]) if r["name"] == "splade")
    assert row["status"] == "unavailable"
    assert "kb-arena[splade]" in row["unavailable_reason"]


@pytest.mark.asyncio
async def test_splade_builds_and_queries_its_own_index(tmp_path, monkeypatch, mock_llm_client):
    from kb_arena.settings import settings

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    table = {
        "What does c1 say?": {1: 1.0},
        "c0 text": {2: 1.0},
        "c1 text": {1: 1.0, 2: 0.2},
    }

    class _FakeEncoder:
        def encode(self, texts):
            return [table[t] for t in texts]

    doc = Document(
        id="doc1",
        source="doc1.md",
        corpus="sample",
        title="Doc 1",
        sections=[
            Section(id="s0", title="Zero", content="c0 text"),
            Section(id="s1", title="One", content="c1 text"),
        ],
    )

    builder = SPLADEStrategy()
    builder._encoder = _FakeEncoder()
    await builder.build_index([doc])

    # A fresh instance must round-trip the index through disk (JSON keys are
    # strings, term ids are ints) rather than reuse the builder's in-memory state.
    strategy = SPLADEStrategy()
    strategy._encoder = _FakeEncoder()
    strategy._llm = mock_llm_client

    result = await strategy.query("What does c1 say?", top_k=1, corpus="sample")
    kept = result.retrieval.retrieved
    assert kept[0].content == "c1 text"
    assert kept[0].chunk_id == "doc1::s1"
    assert kept[0].doc_id == "doc1"
