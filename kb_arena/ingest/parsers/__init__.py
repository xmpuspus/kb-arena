"""Parser registry."""

from __future__ import annotations

from kb_arena.ingest.parsers.html import HtmlParser
from kb_arena.ingest.parsers.markdown import MarkdownParser
from kb_arena.ingest.parsers.sec_edgar import SecEdgarParser

PARSERS = {
    "markdown": MarkdownParser,
    "html": HtmlParser,
    "sec-edgar": SecEdgarParser,
}

__all__ = ["PARSERS", "MarkdownParser", "HtmlParser", "SecEdgarParser"]
