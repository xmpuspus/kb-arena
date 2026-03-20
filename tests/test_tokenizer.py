"""Tests for the shared tokenizer module."""

from kb_arena.tokenizer import detokenize, token_count, tokenize


def test_tokenize_returns_ints():
    tokens = tokenize("Hello world")
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) >= 2


def test_roundtrip():
    text = "The quick brown fox jumps over the lazy dog."
    assert detokenize(tokenize(text)) == text


def test_token_count():
    count = token_count("Hello")
    assert count >= 1


def test_token_count_vs_whitespace():
    text = "API-Gateway configuration requires multi-step setup"
    ws_count = len(text.split())
    tk_count = token_count(text)
    # BPE typically produces MORE tokens than whitespace splitting for hyphenated/compound words
    assert tk_count >= ws_count
