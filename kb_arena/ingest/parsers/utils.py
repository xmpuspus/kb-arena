"""Shared parser utilities — slugify, ID generation, token counting."""

from __future__ import annotations

import re


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s/-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text or "section"


def unique_id(slug: str, seen: set[str]) -> str:
    """Generate a unique ID by appending a suffix if needed."""
    candidate = slug
    n = 1
    while candidate in seen:
        candidate = f"{slug}-{n}"
        n += 1
    seen.add(candidate)
    return candidate


def token_count(text: str) -> int:
    """Approximate BPE token count from whitespace word count."""
    return int(len(text.split()) * 1.3)


def read_text(path, encoding: str = "utf-8") -> str:
    """Read file text with UTF-8, falling back to latin-1."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
