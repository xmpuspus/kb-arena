"""Parser registry."""

from __future__ import annotations

from kb_arena.ingest.parsers.base import Parser
from kb_arena.ingest.parsers.csv_parser import CsvParser
from kb_arena.ingest.parsers.docx import DocxParser
from kb_arena.ingest.parsers.github import GitHubParser
from kb_arena.ingest.parsers.html import HtmlParser
from kb_arena.ingest.parsers.markdown import MarkdownParser
from kb_arena.ingest.parsers.pdf import PdfParser
from kb_arena.ingest.parsers.plaintext import PlaintextParser
from kb_arena.ingest.parsers.sec_edgar import SecEdgarParser
from kb_arena.ingest.parsers.web import WebParser

PARSERS: dict[str, type] = {
    "markdown": MarkdownParser,
    "html": HtmlParser,
    "sec-edgar": SecEdgarParser,
    "pdf": PdfParser,
    "docx": DocxParser,
    "plaintext": PlaintextParser,
    "web": WebParser,
    "csv": CsvParser,
    "github": GitHubParser,
}

__all__ = [
    "PARSERS",
    "Parser",
    "MarkdownParser",
    "HtmlParser",
    "SecEdgarParser",
    "PdfParser",
    "DocxParser",
    "PlaintextParser",
    "WebParser",
    "CsvParser",
    "GitHubParser",
]
