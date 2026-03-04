"""Tests for new parsers: plaintext, CSV, PDF, DOCX, web, GitHub."""

from __future__ import annotations

import json

import pytest

from kb_arena.ingest.parsers.csv_parser import CsvParser
from kb_arena.ingest.parsers.plaintext import PlaintextParser
from kb_arena.ingest.pipeline import run_ingest
from kb_arena.models.document import Document

# ---------------------------------------------------------------------------
# Plaintext parser tests
# ---------------------------------------------------------------------------

SAMPLE_PLAINTEXT = """\
INTRODUCTION

This is a guide to cloud computing services.
It covers various platforms and deployment models.

COMPUTE SERVICES

Virtual machines and container orchestration are key
building blocks. Instance types vary by CPU, memory,
and storage configuration.

STORAGE OPTIONS

Object storage provides scalable, durable data persistence.
Block storage is available for VM-attached volumes.
"""

SAMPLE_PLAINTEXT_NO_HEADINGS = """\
This is just a plain document with no headings at all.
It has multiple lines but no structure.
Just raw text content that should still be parseable.
"""


@pytest.fixture
def txt_parser(tmp_path):
    p = tmp_path / "guide.txt"
    p.write_text(SAMPLE_PLAINTEXT, encoding="utf-8")
    return PlaintextParser(), p


def test_plaintext_returns_one_document(txt_parser):
    parser, path = txt_parser
    docs = parser.parse(path, "test-corpus")
    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].corpus == "test-corpus"


def test_plaintext_detects_caps_headings(txt_parser):
    parser, path = txt_parser
    doc = parser.parse(path, "test-corpus")[0]
    titles = [s.title for s in doc.sections]
    assert any("INTRODUCTION" in t for t in titles)
    assert any("COMPUTE" in t for t in titles)
    assert any("STORAGE" in t for t in titles)


def test_plaintext_sections_have_content(txt_parser):
    parser, path = txt_parser
    doc = parser.parse(path, "test-corpus")[0]
    for section in doc.sections:
        assert section.content.strip()


def test_plaintext_token_count(txt_parser):
    parser, path = txt_parser
    doc = parser.parse(path, "test-corpus")[0]
    assert doc.raw_token_count > 0


def test_plaintext_no_headings_fallback(tmp_path):
    p = tmp_path / "flat.txt"
    p.write_text(SAMPLE_PLAINTEXT_NO_HEADINGS, encoding="utf-8")
    parser = PlaintextParser()
    docs = parser.parse(p, "test-corpus")
    assert len(docs) == 1
    doc = docs[0]
    assert len(doc.sections) >= 1
    assert doc.sections[0].content.strip()


def test_plaintext_empty_file(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    parser = PlaintextParser()
    docs = parser.parse(p, "test-corpus")
    assert len(docs) == 0


# ---------------------------------------------------------------------------
# CSV parser tests
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
Name,Type,Region,Cost
t3.micro,General Purpose,us-east-1,$0.0104
m5.large,General Purpose,us-west-2,$0.0960
c5.xlarge,Compute Optimized,eu-west-1,$0.1700
p3.2xlarge,GPU,us-east-1,$3.0600
r5.large,Memory Optimized,ap-southeast-1,$0.1260
"""

SAMPLE_TSV = """\
Name\tType\tRegion
t3.micro\tGeneral Purpose\tus-east-1
m5.large\tGeneral Purpose\tus-west-2
"""


@pytest.fixture
def csv_parser(tmp_path):
    p = tmp_path / "instances.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return CsvParser(), p


def test_csv_returns_one_document(csv_parser):
    parser, path = csv_parser
    docs = parser.parse(path, "test-corpus")
    assert len(docs) == 1
    assert docs[0].corpus == "test-corpus"


def test_csv_has_metadata(csv_parser):
    parser, path = csv_parser
    doc = parser.parse(path, "test-corpus")[0]
    assert doc.metadata["source_type"] == "csv"
    assert doc.metadata["row_count"] == 5
    assert doc.metadata["column_count"] == 4
    assert "Name" in doc.metadata["columns"]


def test_csv_sections_have_table(csv_parser):
    parser, path = csv_parser
    doc = parser.parse(path, "test-corpus")[0]
    assert len(doc.sections) >= 1
    tables = [t for s in doc.sections for t in s.tables]
    assert len(tables) >= 1
    assert "Name" in tables[0].headers


def test_csv_content_readable(csv_parser):
    parser, path = csv_parser
    doc = parser.parse(path, "test-corpus")[0]
    full_content = " ".join(s.content for s in doc.sections)
    assert "t3.micro" in full_content
    assert "General Purpose" in full_content


def test_tsv_auto_detected(tmp_path):
    p = tmp_path / "data.tsv"
    p.write_text(SAMPLE_TSV, encoding="utf-8")
    parser = CsvParser()
    docs = parser.parse(p, "test-corpus")
    assert len(docs) == 1
    assert docs[0].metadata["row_count"] == 2


def test_csv_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    parser = CsvParser()
    docs = parser.parse(p, "test-corpus")
    assert len(docs) == 0


def test_csv_header_only(tmp_path):
    p = tmp_path / "header_only.csv"
    p.write_text("Name,Type,Region\n", encoding="utf-8")
    parser = CsvParser()
    docs = parser.parse(p, "test-corpus")
    assert len(docs) == 0


# ---------------------------------------------------------------------------
# Pipeline integration tests for new formats
# ---------------------------------------------------------------------------


def test_pipeline_plaintext(tmp_path):
    import os

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc.txt").write_text(SAMPLE_PLAINTEXT, encoding="utf-8")

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(raw_dir), corpus="txt-test", format="auto")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "txt-test" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    doc = Document(**json.loads(lines[0]))
    assert doc.corpus == "txt-test"


def test_pipeline_csv(tmp_path):
    import os

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "data.csv").write_text(SAMPLE_CSV, encoding="utf-8")

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(raw_dir), corpus="csv-test", format="auto")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "csv-test" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Parser registry tests
# ---------------------------------------------------------------------------


def test_all_parsers_registered():
    from kb_arena.ingest.parsers import PARSERS

    expected = {"markdown", "html", "sec-edgar", "pdf", "docx", "plaintext", "web", "csv", "github"}
    assert set(PARSERS.keys()) == expected


def test_parser_protocol_compliance():
    from kb_arena.ingest.parsers import PARSERS
    from kb_arena.ingest.parsers.base import Parser

    for name, cls in PARSERS.items():
        instance = cls()
        assert isinstance(instance, Parser), f"{name} parser does not satisfy Parser protocol"


# ---------------------------------------------------------------------------
# Extension map tests
# ---------------------------------------------------------------------------


def test_ext_map_covers_new_formats():
    from kb_arena.ingest.pipeline import _EXT_MAP

    assert _EXT_MAP[".pdf"] == "pdf"
    assert _EXT_MAP[".docx"] == "docx"
    assert _EXT_MAP[".txt"] == "plaintext"
    assert _EXT_MAP[".csv"] == "csv"
    assert _EXT_MAP[".tsv"] == "csv"


# ---------------------------------------------------------------------------
# Shared utils tests
# ---------------------------------------------------------------------------


def test_slugify_basic():
    from kb_arena.ingest.parsers.utils import slugify

    assert slugify("Hello World") == "hello-world"
    assert slugify("AWS Lambda Config") == "aws-lambda-config"
    assert slugify("") == "section"
    assert slugify("  Spaces  ") == "spaces"


def test_unique_id_dedup():
    from kb_arena.ingest.parsers.utils import unique_id

    seen: set[str] = set()
    id1 = unique_id("foo", seen)
    id2 = unique_id("foo", seen)
    id3 = unique_id("foo", seen)
    assert id1 == "foo"
    assert id2 == "foo-1"
    assert id3 == "foo-2"


def test_token_count_approximation():
    from kb_arena.ingest.parsers.utils import token_count

    text = "the quick brown fox jumps over the lazy dog"  # 9 words
    count = token_count(text)
    assert count == int(9 * 1.3)


def test_read_text_utf8(tmp_path):
    from kb_arena.ingest.parsers.utils import read_text

    p = tmp_path / "utf8.txt"
    p.write_text("Hello Unicode: é à ñ", encoding="utf-8")
    assert "Hello Unicode" in read_text(p)


def test_read_text_latin1_fallback(tmp_path):
    from kb_arena.ingest.parsers.utils import read_text

    p = tmp_path / "latin1.txt"
    p.write_bytes("Hello\xe9World".encode("latin-1"))
    result = read_text(p)
    assert "Hello" in result
