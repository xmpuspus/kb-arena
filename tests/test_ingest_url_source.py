"""`kb-arena ingest <url> --format web` hands the parser the URL the user typed."""

from __future__ import annotations

from pathlib import Path

from kb_arena.ingest import pipeline
from kb_arena.ingest.parsers.web import WebParser
from kb_arena.settings import settings


def test_special_ingest_passes_a_url_through_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "datasets"))
    seen: list[str] = []

    def fake_scrape(self, url, corpus):
        seen.append(url)
        return []

    monkeypatch.setattr(WebParser, "_scrape", fake_scrape)

    pipeline.run_ingest_special("https://example.com/docs", corpus="c", format="web")

    assert seen == ["https://example.com/docs"]


def test_web_parser_still_reads_a_url_from_a_file(tmp_path, monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(WebParser, "_scrape", lambda self, url, corpus: seen.append(url) or [])
    url_file = tmp_path / "source.url"
    url_file.write_text("https://example.com/from-file\n")

    WebParser().parse(url_file, "c")

    assert seen == ["https://example.com/from-file"]


def test_a_path_wrapped_url_is_what_broke_before():
    # Documents the defect: Path() drops one slash, and the parser then sees
    # no scheme, so it tries to read a file that does not exist.
    assert str(Path("https://example.com")) == "https:/example.com"
    assert WebParser().parse(Path("https://example.com"), "c") == []
