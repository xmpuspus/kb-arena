"""Chunk size / overlap must be settings-driven so the optimizer can sweep them.

Before v0.7 chunk size was a hardcoded module constant (CHUNK_TOKENS=512),
unreachable from the optimizer. These tests pin the new contract: _chunk_text()
resolves None args from settings, and an explicit arg still wins.
"""

from __future__ import annotations

import pytest

from kb_arena.settings import Settings, settings
from kb_arena.strategies.naive_vector import _chunk_text
from kb_arena.strategies.raptor import _chunk_text as _raptor_chunk_text


def test_chunk_text_uses_settings_when_args_omitted(monkeypatch):
    long_text = "lambda invocation context " * 200
    monkeypatch.setattr(settings, "chunk_tokens", 16)
    monkeypatch.setattr(settings, "chunk_overlap_tokens", 0)
    small = _chunk_text(long_text)

    monkeypatch.setattr(settings, "chunk_tokens", 100000)
    big = _chunk_text(long_text)

    assert len(small) > len(big)
    assert len(big) == 1


def test_chunk_text_explicit_arg_overrides_settings(monkeypatch):
    long_text = "lambda invocation context " * 200
    monkeypatch.setattr(settings, "chunk_tokens", 16)
    explicit = _chunk_text(long_text, chunk_tokens=100000, overlap_tokens=0)
    assert len(explicit) == 1


def test_settings_expose_chunk_defaults():
    fresh = type(settings)()
    assert fresh.chunk_tokens == 512
    assert fresh.chunk_overlap_tokens == 50


@pytest.mark.parametrize("chunker", [_chunk_text, _raptor_chunk_text])
@pytest.mark.parametrize(
    ("chunk_tokens", "overlap_tokens"),
    [(0, 0), (2, -1), (2, 2), (2, 3)],
)
def test_chunkers_reject_nonprogressing_windows(chunker, chunk_tokens, overlap_tokens):
    with pytest.raises(ValueError, match="chunk overlap"):
        chunker("one two three four", chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)


def test_settings_reject_nonprogressing_chunk_window():
    with pytest.raises(ValueError, match="chunk overlap"):
        Settings(chunk_tokens=2, chunk_overlap_tokens=2)
