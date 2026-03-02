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
# AWS Lambda

AWS Lambda lets you run code without provisioning or managing servers.

## Lambda Configuration

Configure your Lambda function's memory, timeout, and runtime settings.

```bash
aws lambda update-function-configuration \\
  --function-name my-function \\
  --timeout 300 \\
  --memory-size 512
```

| Setting    | Type | Description                              |
|------------|------|------------------------------------------|
| Timeout    | int  | Maximum execution time in seconds (1-900)|
| MemorySize | int  | Memory allocation in MB (128-10240)      |

## Lambda Execution Role

A Lambda function's execution role grants it permission to access AWS services.
"""

SAMPLE_RST = """\
AWS Lambda
==========

AWS Lambda lets you run code without provisioning or managing servers.

Lambda Configuration
--------------------

Configure your Lambda function using
:func:`update-function-configuration`.
Raises :class:`ClientError`.

Example::

    aws lambda update-function-configuration \\
      --function-name my-function --timeout 300

Lambda Execution Role
---------------------

Configure :meth:`execution role` permissions using :func:`IAM`.
"""

SAMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>AWS Lambda — Configuration</title></head>
<body>
  <div class="section" id="lambda-configuration">
    <h1>AWS Lambda — Configuration</h1>
    <p>AWS Lambda lets you run code without provisioning or managing servers.</p>
    <dl class="function">
      <dt id="lambda.update-function-configuration">update-function-configuration</dt>
      <dd>
        <p>Updates a Lambda function's configuration.</p>
        <table>
          <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td>Timeout</td><td>int</td><td>Maximum execution time (seconds)</td></tr>
          </tbody>
        </table>
        <pre><code class="language-bash">aws lambda update-function-configuration \
--timeout 300</code></pre>
        <a class="reference internal"
           href="errors.html#ClientError">ClientError</a>
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
    p = tmp_path / "lambda.md"
    p.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    return MarkdownParser(), p


def test_markdown_returns_one_document(md_parser):
    parser, path = md_parser
    docs = parser.parse(path, "aws-compute")
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.corpus == "aws-compute"


def test_markdown_title_from_h1(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert "lambda" in doc.title.lower()


def test_markdown_sections_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert len(doc.sections) == 3  # h1 + two h2s
    titles = [s.title for s in doc.sections]
    assert any("Configuration" in t for t in titles)
    assert any("Execution Role" in t for t in titles)


def test_markdown_heading_path(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    config_section = next(s for s in doc.sections if "Configuration" in s.title)
    assert "Lambda Configuration" in config_section.heading_path[-1]
    # heading_path should include parent heading
    assert len(config_section.heading_path) == 2


def test_markdown_code_block_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    config_section = next(s for s in doc.sections if "Configuration" in s.title)
    assert len(config_section.code_blocks) == 1
    assert config_section.code_blocks[0].language == "bash"
    assert "lambda" in config_section.code_blocks[0].code


def test_markdown_table_extracted(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    config_section = next(s for s in doc.sections if "Configuration" in s.title)
    assert len(config_section.tables) == 1
    tbl = config_section.tables[0]
    assert "Setting" in tbl.headers
    assert len(tbl.rows) == 2


def test_markdown_token_count(md_parser):
    parser, path = md_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert doc.raw_token_count > 0


# ---------------------------------------------------------------------------
# RST parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def rst_parser(tmp_path):
    p = tmp_path / "lambda.rst"
    p.write_text(SAMPLE_RST, encoding="utf-8")
    return MarkdownParser(), p


def test_rst_sections_extracted(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert len(doc.sections) >= 2
    titles = [s.title for s in doc.sections]
    assert any("Configuration" in t for t in titles)


def test_rst_cross_references(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "aws-compute")[0]
    all_links = [link for s in doc.sections for link in s.links]
    targets = {lk.target for lk in all_links}
    assert "update-function-configuration" in targets or "ClientError" in targets


def test_rst_heading_path(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "aws-compute")[0]
    config_section = next(s for s in doc.sections if "Configuration" in s.title)
    assert len(config_section.heading_path) >= 1
    assert "Configuration" in config_section.heading_path[-1]


def test_rst_code_block_extracted(rst_parser):
    parser, path = rst_parser
    doc = parser.parse(path, "aws-compute")[0]
    all_code = [cb for s in doc.sections for cb in s.code_blocks]
    assert len(all_code) >= 1
    assert "lambda" in all_code[0].code or "aws" in all_code[0].code


# ---------------------------------------------------------------------------
# HTML parser tests
# ---------------------------------------------------------------------------


@pytest.fixture
def html_parser(tmp_path):
    p = tmp_path / "lambda.html"
    p.write_text(SAMPLE_HTML, encoding="utf-8")
    return HtmlParser(), p


def test_html_returns_one_document(html_parser):
    parser, path = html_parser
    docs = parser.parse(path, "aws-compute")
    assert len(docs) == 1


def test_html_title_from_title_tag(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert "lambda" in doc.title.lower()


def test_html_section_from_div(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    assert len(doc.sections) >= 1
    titles = [s.title for s in doc.sections]
    assert any("lambda" in t.lower() for t in titles)


def test_html_dl_section_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    titles = [s.title for s in doc.sections]
    assert any("configuration" in t.lower() for t in titles)


def test_html_table_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    all_tables = [t for s in doc.sections for t in s.tables]
    assert len(all_tables) >= 1
    tbl = all_tables[0]
    assert "Parameter" in tbl.headers


def test_html_code_block_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    all_code = [cb for s in doc.sections for cb in s.code_blocks]
    assert len(all_code) >= 1


def test_html_cross_ref_extracted(html_parser):
    parser, path = html_parser
    doc = parser.parse(path, "aws-compute")[0]
    all_links = [link for s in doc.sections for link in s.links]
    assert len(all_links) >= 1
    assert any("ClientError" in lk.label for lk in all_links)


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
    docs = parser.parse(path, "aws-storage")
    assert len(docs) == 1


def test_sec_item_sections_extracted(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "aws-storage")[0]
    assert len(doc.sections) >= 2
    titles = [s.title for s in doc.sections]
    assert any("Business" in t or "1" in t for t in titles)
    assert any("Risk" in t or "1A" in t for t in titles)


def test_sec_table_extracted(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "aws-storage")[0]
    all_tables = [t for s in doc.sections for t in s.tables]
    assert len(all_tables) >= 1


def test_sec_named_entities_in_metadata(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "aws-storage")[0]
    assert "named_entities" in doc.metadata
    entities = doc.metadata["named_entities"]
    assert len(entities.get("dollar_amounts", [])) >= 1


def test_sec_heading_path(sec_parser):
    parser, path = sec_parser
    doc = parser.parse(path, "aws-storage")[0]
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
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
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
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
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
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
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
