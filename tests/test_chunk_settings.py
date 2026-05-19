"""Chunk size / overlap must be settings-driven so the optimizer can sweep them.

Before v0.7 chunk size was a hardcoded module constant (CHUNK_TOKENS=512),
unreachable from the optimizer. These tests pin the new contract: _chunk_text()
resolves None args from settings, and an explicit arg still wins.
"""

from __future__ import annotations

from kb_arena.settings import settings
from kb_arena.strategies.naive_vector import _chunk_text


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
