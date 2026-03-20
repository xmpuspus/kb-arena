"""Shared tokenizer - tiktoken cl100k_base for accurate BPE token counting."""

from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def tokenize(text: str) -> list[int]:
    """Return BPE token IDs for accurate chunk sizing."""
    return _enc.encode(text)


def detokenize(tokens: list[int]) -> str:
    """Decode token IDs back to text."""
    return _enc.decode(tokens)


def token_count(text: str) -> int:
    """Count BPE tokens in text."""
    return len(_enc.encode(text))
