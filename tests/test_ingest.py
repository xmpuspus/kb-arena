"""Tests for the document ingestion pipeline."""

from __future__ import annotations

import json

import pytest

from kb_arena.ingest.parsers.html import HtmlParser
from kb_arena.ingest.parsers.markdown import MarkdownParser
from kb_arena.ingest.parsers.sec_edgar import SecEdgarParser
from kb_arena.ingest.pipeline import run_ingest
from kb_arena.models.document import Document

# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

SAMPLE_MARKDOWN = """\
# json — JSON encoder and decoder

JSON (JavaScript Object Notation) is a lightweight data interchange format.

## json.loads

Deserialize `s` to a Python object using :func:`json.JSONDecodeError` on failure.

```python
>>> import json
>>> json.loads('{"key": "value"}')
{'key': 'value'}
```

| Parameter | Type   | Description           |
|-----------|--------|-----------------------|
| s         | str    | JSON string to parse  |
| cls       | type   | Custom decoder class  |

## json.dumps

Serialize *obj* to a JSON formatted string.
"""

SAMPLE_RST = """\
json module
===========

JSON (JavaScript Object Notation), specified by :rfc:`7159`.

json.loads
----------

Deserialize ``s`` using :func:`json.loads`. Raises :class:`json.JSONDecodeError`.

Example::

    import json
    json.loads('{"x": 1}')

json.dumps
----------

Serialize *obj* using :meth:`json.JSONEncoder.encode`.
"""

SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>json — JSON encoder and decoder</title></head>
<body>
  <div class="section" id="json-module">
    <h1>json — JSON encoder and decoder</h1>
    <p>JSON is a lightweight data interchange format.</p>
    <dl class="function">
      <dt id="json.loads">json.loads(s, *, cls=None)</dt>
      <dd>
        <p>Deserialize s to a Python object.</p>
        <table>
          <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>s</td><td>str</td><td>JSON string</td></tr>
          </tbody>
        </table>
        <pre><code class="language-python">>>> json.loads('{"x": 1}')</code></pre>
        <a class="reference internal" href="exceptions.html#json.JSONDecodeError">JSONDecodeError</a>
      </dd>
    </dl>
  </div>
</body>
</html>
"""

SAMPLE_SEC_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Acme Corp 10-K 2023</title></head>
<body>
  <p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
  <p>Item 1. Business</p>
  <p>Acme Corp. is a technology company incorporated in Delaware.
  We employ approximately 5,000 people. Revenue was $1.2 billion in fiscal 2023.</p>
  <p>Item 1A. Risk Factors</p>
  <p>We face significant competition. Our Chief Executive Officer John Smith
  oversees strategy. Total liabilities were $500,000,000.</p>
  <p>Item 7. Management Discussion and Analysis</p>
  <p>Net income increased to $200 million. Chief Financial Officer Jane Doe
  confirmed guidance of $1.5 billion for 2024.</p>
  <table>
    <tr><th>Year</th><th>Revenue</th><th>Net Income</th></tr>
    <tr><td>2023</td><td>$1.2B</td><td>$200M</td></tr>
    <tr><td>2022</td><td>$1.0B</td><td>$150M</td></tr>
  </table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Markdown parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def md_parser(tmp_path):
    p = tmp_path / "json.md"
    p.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return MarkdownParser(), p


def test_markdown_returns_one_document(md_parser):
    parser, path = md_parser
    docs = parser.parse(path, "python-stdlib")
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.corpus == "python-stdlib"


def test_markdown_title_from_h1(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert "json" in doc.title.lower()


def test_markdown_sections_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert len(doc.sections) == 3  # h1 + two h2s
    titles = [s.title for s in doc.sections]
    assert any("loads" in t for t in titles)
    assert any("dumps" in t for t in titles)


def test_markdown_heading_path(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    loads_section = next(s for s in doc.sections if "loads" in s.title)
    assert loads_section.heading_path[-1] == "json.loads"
    # heading_path should include parent heading
    assert len(loads_section.heading_path) == 2


def test_markdown_code_block_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    loads_section = next(s for s in doc.sections if "loads" in s.title)
    assert len(loads_section.code_blocks) == 1
    assert loads_section.code_blocks[0].language == "python"
    assert "json.loads" in loads_section.code_blocks[0].code


def test_markdown_table_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    loads_section = next(s for s in doc.sections if "loads" in s.title)
    assert len(loads_section.tables) == 1
    tbl = loads_section.tables[0]
    assert "Parameter" in tbl.headers
    assert len(tbl.rows) == 2


def test_markdown_token_count(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert doc.raw_token_count > 0


# ---------------------------------------------------------------------------
# RST parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def rst_parser(tmp_path):
    p = tmp_path / "json.rst"
    p.write_text(SAMPLE_RST, encoding="utf-8")
    return MarkdownParser(), p


def test_rst_sections_extracted(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert len(doc.sections) >= 2
    titles = [s.title for s in doc.sections]
    assert any("loads" in t for t in titles)


def test_rst_cross_references(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "python-stdlib")[0]
    all_links = [link for s in doc.sections for link in s.links]
    targets = {l.target for l in all_links}
    assert "json.loads" in targets or "json.JSONDecodeError" in targets


def test_rst_heading_path(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "python-stdlib")[0]
    loads_section = next(s for s in doc.sections if "loads" in s.title)
    assert len(loads_section.heading_path) >= 1
    assert "loads" in loads_section.heading_path[-1]


def test_rst_code_block_extracted(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "python-stdlib")[0]
    all_code = [cb for s in doc.sections for cb in s.code_blocks]
    assert len(all_code) >= 1
    assert "json.loads" in all_code[0].code or "json" in all_code[0].code


# ---------------------------------------------------------------------------
# HTML parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def html_parser(tmp_path):
    p = tmp_path / "json.html"
    p.write_text(SAMPLE_HTML, encoding="utf-8")
    return HtmlParser(), p


def test_html_returns_one_document(html_parser):
    parser, path = html_parser
    docs = parser.parse(path, "python-stdlib")
    assert len(docs) == 1


def test_html_title_from_title_tag(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert "json" in doc.title.lower()


def test_html_section_from_div(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    assert len(doc.sections) >= 1
    titles = [s.title for s in doc.sections]
    assert any("json" in t.lower() for t in titles)


def test_html_dl_section_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    titles = [s.title for s in doc.sections]
    assert any("loads" in t for t in titles)


def test_html_table_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    all_tables = [t for s in doc.sections for t in s.tables]
    assert len(all_tables) >= 1
    tbl = all_tables[0]
    assert "Parameter" in tbl.headers


def test_html_code_block_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    all_code = [cb for s in doc.sections for cb in s.code_blocks]
    assert len(all_code) >= 1


def test_html_cross_ref_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "python-stdlib")[0]
    all_links = [link for s in doc.sections for link in s.links]
    assert len(all_links) >= 1
    assert any("JSONDecodeError" in l.label for l in all_links)


# ---------------------------------------------------------------------------
# SEC EDGAR parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sec_parser(tmp_path):
    p = tmp_path / "acme-10k.html"
    p.write_text(SAMPLE_SEC_HTML, encoding="utf-8")
    return SecEdgarParser(), p


def test_sec_returns_one_document(sec_parser):
    parser, path = sec_parser
    docs = parser.parse(path, "sec-edgar")
    assert len(docs) == 1


def test_sec_item_sections_extracted(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "sec-edgar")[0]
    assert len(doc.sections) >= 2
    titles = [s.title for s in doc.sections]
    assert any("Business" in t or "1" in t for t in titles)
    assert any("Risk" in t or "1A" in t for t in titles)


def test_sec_table_extracted(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "sec-edgar")[0]
    all_tables = [t for s in doc.sections for t in s.tables]
    assert len(all_tables) >= 1


def test_sec_named_entities_in_metadata(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "sec-edgar")[0]
    assert "named_entities" in doc.metadata
    entities = doc.metadata["named_entities"]
    assert len(entities.get("dollar_amounts", [])) >= 1


def test_sec_heading_path(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "sec-edgar")[0]
    for section in doc.sections:
        assert len(section.heading_path) >= 1
        assert "10-K" in section.heading_path or any(section.heading_path)


# ---------------------------------------------------------------------------
# Pipeline JSONL output tests
# ---------------------------------------------------------------------------


def test_pipeline_writes_jsonl(tmp_path):
    # Create a small markdown corpus
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "doc1.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    (raw_dir / "doc2.md").write_text("# Second Doc\n\nSome content here.\n", encoding="utf-8")

    # Override datasets dir to tmp_path
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(raw_dir), corpus="test-corpus", format="markdown")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "test-corpus" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 2

    for line in lines:
        data = json.loads(line)
        doc = Document(**data)
        assert doc.corpus == "test-corpus"
        assert len(doc.sections) >= 1


def test_pipeline_each_line_valid_document(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "test.md").write_text(SAMPLE_MARKDOWN, encoding="utf-8")

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(raw_dir), corpus="validate-corpus")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "validate-corpus" / "processed" / "documents.jsonl"
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        assert "id" in parsed
        assert "sections" in parsed
        assert "corpus" in parsed


def test_pipeline_auto_detects_html(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "page.html").write_text(SAMPLE_HTML, encoding="utf-8")

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(raw_dir), corpus="html-corpus", format="auto")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "html-corpus" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    doc = Document(**json.loads(lines[0]))
    assert doc.corpus == "html-corpus"


def test_pipeline_single_file(tmp_path):
    md_file = tmp_path / "single.md"
    md_file.write_text("# Solo Doc\n\nJust one file.\n", encoding="utf-8")

    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_ingest(path=str(md_file), corpus="single-corpus", format="markdown")
    finally:
        os.chdir(original_cwd)

    out = tmp_path / "datasets" / "single-corpus" / "processed" / "documents.jsonl"
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Heading path depth tests
# ---------------------------------------------------------------------------


def test_heading_path_three_levels(tmp_path):
    content = "# Top\n\n## Middle\n\n### Deep\n\nContent here.\n"
    p = tmp_path / "deep.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    deep = next(s for s in doc.sections if s.title == "Deep")
    assert deep.heading_path == ["Top", "Middle", "Deep"]
    assert deep.level == 3


def test_section_id_uniqueness(tmp_path):
    # Two sections with the same title should get distinct IDs
    content = "# Foo\n\n## Bar\n\n## Bar\n\nDuplicate heading.\n"
    p = tmp_path / "dupe.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    ids = [s.id for s in doc.sections]
    assert len(ids) == len(set(ids)), "Section IDs must be unique"
