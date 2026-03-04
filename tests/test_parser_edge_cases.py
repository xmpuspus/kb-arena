"""Parser edge case tests — markdown, HTML, RST internals."""

from __future__ import annotations

from kb_arena.ingest.parsers.html import HtmlParser
from kb_arena.ingest.parsers.markdown import (
    MarkdownParser,
    _parse_md_code_blocks,
    _parse_md_tables,
)
from kb_arena.ingest.parsers.utils import slugify as _slugify
from kb_arena.ingest.parsers.utils import token_count as _token_count
from kb_arena.ingest.parsers.utils import unique_id as _unique_id

html_slugify = _slugify


# ---------------------------------------------------------------------------
# _slugify (markdown)
# ---------------------------------------------------------------------------


def test_slugify_basic():
    assert _slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    result = _slugify("Hello (World)!")
    assert "hello" in result
    assert "world" in result


def test_slugify_empty_string():
    assert _slugify("") == "section"


def test_slugify_multiple_spaces():
    assert _slugify("  foo   bar  ") == "foo-bar"


def test_slugify_already_lowercase():
    assert _slugify("already") == "already"


# ---------------------------------------------------------------------------
# _unique_id
# ---------------------------------------------------------------------------


def test_unique_id_first_occurrence():
    seen: set[str] = set()
    assert _unique_id("foo", seen) == "foo"
    assert "foo" in seen


def test_unique_id_collision():
    seen = {"foo"}
    assert _unique_id("foo", seen) == "foo-1"


def test_unique_id_multiple_collisions():
    seen = {"foo", "foo-1", "foo-2"}
    assert _unique_id("foo", seen) == "foo-3"


def test_unique_id_no_collision():
    seen: set[str] = set()
    _unique_id("alpha", seen)
    assert _unique_id("beta", seen) == "beta"


# ---------------------------------------------------------------------------
# _token_count
# ---------------------------------------------------------------------------


def test_token_count_empty():
    assert _token_count("") == 0


def test_token_count_single_word():
    # int(1 * 1.3) = 1
    assert _token_count("hello") == 1


def test_token_count_ten_words():
    text = "one two three four five six seven eight nine ten"
    assert _token_count(text) == 13  # int(10 * 1.3)


def test_token_count_scales():
    text = " ".join(["word"] * 100)
    assert _token_count(text) == 130


# ---------------------------------------------------------------------------
# _parse_md_code_blocks
# ---------------------------------------------------------------------------


def test_parse_md_code_blocks_empty():
    assert _parse_md_code_blocks([]) == []


def test_parse_md_code_blocks_single():
    lines = ["```python", "x = 1", "```"]
    blocks = _parse_md_code_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0].language == "python"
    assert "x = 1" in blocks[0].code


def test_parse_md_code_blocks_no_language():
    lines = ["```", "plain text", "```"]
    blocks = _parse_md_code_blocks(lines)
    assert len(blocks) == 1
    assert blocks[0].language == ""


def test_parse_md_code_blocks_multiple():
    lines = ["```bash", "ls -la", "```", "prose", "```python", "print('hi')", "```"]
    blocks = _parse_md_code_blocks(lines)
    assert len(blocks) == 2


def test_parse_md_code_blocks_multiline():
    lines = ["```yaml", "key: value", "nested:", "  a: 1", "```"]
    blocks = _parse_md_code_blocks(lines)
    assert len(blocks) == 1
    assert "nested" in blocks[0].code


# ---------------------------------------------------------------------------
# _parse_md_tables
# ---------------------------------------------------------------------------


def test_parse_md_tables_empty():
    assert _parse_md_tables([]) == []


def test_parse_md_tables_basic():
    lines = [
        "| Name | Age |",
        "|------|-----|",
        "| Alice | 30 |",
        "| Bob | 25 |",
    ]
    tables = _parse_md_tables(lines)
    assert len(tables) == 1
    assert "Name" in tables[0].headers
    assert len(tables[0].rows) == 2


def test_parse_md_tables_no_separator_not_parsed():
    lines = ["| Name | Age |", "| Alice | 30 |"]
    tables = _parse_md_tables(lines)
    assert len(tables) == 0


def test_parse_md_tables_single_column():
    lines = ["| Item |", "|------|", "| foo |", "| bar |"]
    tables = _parse_md_tables(lines)
    assert len(tables) == 1
    assert len(tables[0].headers) == 1


# ---------------------------------------------------------------------------
# Markdown parser — edge cases
# ---------------------------------------------------------------------------


def test_markdown_empty_document(tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    assert docs == []


def test_markdown_no_headings(tmp_path):
    p = tmp_path / "flat.md"
    p.write_text("Just some text.\nNo headings here.\n", encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    assert docs == []


def test_markdown_unicode_content(tmp_path):
    content = "# Unicode\n\nHello 日本語 Ñoño\n"
    p = tmp_path / "unicode.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    assert len(docs) == 1
    assert docs[0].sections[0].title == "Unicode"


def test_markdown_nested_headings_three_levels(tmp_path):
    content = "# L1\n\n## L2\n\n### L3\n\nDeep content.\n"
    p = tmp_path / "nested.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    l3 = next(s for s in doc.sections if s.title == "L3")
    assert l3.level == 3
    assert l3.heading_path == ["L1", "L2", "L3"]


def test_markdown_heading_resets_on_shallower(tmp_path):
    content = "# Top\n\n## Mid\n\n### Deep\n\n## Back to Mid\n\nContent.\n"
    p = tmp_path / "reset.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    back = next(s for s in doc.sections if s.title == "Back to Mid")
    assert back.heading_path == ["Top", "Back to Mid"]


def test_markdown_special_chars_in_heading(tmp_path):
    content = "# Config (v2.0)\n\nSome content.\n"
    p = tmp_path / "special.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    assert "Config" in doc.sections[0].title


def test_markdown_empty_section_body(tmp_path):
    content = "# Title\n\n## Empty Section\n\n## Has Content\n\nsome text\n"
    p = tmp_path / "empty_section.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    empty_sec = next(s for s in doc.sections if s.title == "Empty Section")
    assert empty_sec.content == ""


def test_markdown_very_long_document(tmp_path):
    lines = ["# Big Doc\n"]
    for i in range(50):
        lines.append(f"## Section {i}\n\nContent for section {i}. " * 3 + "\n")
    p = tmp_path / "big.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    assert len(doc.sections) == 51  # h1 + 50 h2s
    assert doc.raw_token_count > 0


def test_markdown_code_block_yaml(tmp_path):
    content = "# Doc\n\n```yaml\nkey: value\nnested:\n  a: 1\n```\n\nText after.\n"
    p = tmp_path / "code.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    assert len(doc.sections[0].code_blocks) == 1
    assert doc.sections[0].code_blocks[0].language == "yaml"
    assert "key: value" in doc.sections[0].code_blocks[0].code


def test_markdown_table_then_text(tmp_path):
    content = "# Doc\n\n## Section\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nText after table.\n"
    p = tmp_path / "table_text.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    sec = next(s for s in doc.sections if s.title == "Section")
    assert len(sec.tables) == 1
    assert "Text after table" in sec.content


def test_markdown_duplicate_headings_unique_ids(tmp_path):
    content = "# Doc\n\n## Foo\n\ncontent1\n\n## Foo\n\ncontent2\n"
    p = tmp_path / "dupe.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    ids = [s.id for s in doc.sections]
    assert len(ids) == len(set(ids))


def test_markdown_single_h1_no_children(tmp_path):
    content = "# Just a Title\n\nSome content without subsections.\n"
    p = tmp_path / "single.md"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    doc = parser.parse(p, "test")[0]
    assert len(doc.sections) == 1
    assert doc.sections[0].level == 1


# ---------------------------------------------------------------------------
# HTML parser — edge cases
# ---------------------------------------------------------------------------


def test_html_empty_body(tmp_path):
    content = "<html><head><title>Empty</title></head><body></body></html>"
    p = tmp_path / "empty.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    assert len(docs) == 1
    assert docs[0].title == "Empty"
    assert docs[0].sections == []


def test_html_malformed_no_closing_tags(tmp_path):
    content = "<html><body><h1>Title<p>Content<h2>Sub"
    p = tmp_path / "malformed.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    # BeautifulSoup handles malformed HTML gracefully
    docs = parser.parse(p, "test")
    assert isinstance(docs, list)


def test_html_entities_decoded(tmp_path):
    content = (
        "<html><body>"
        '<div class="section"><h1>Test &amp; Example</h1>'
        "<p>Use &lt;br&gt; here.</p></div>"
        "</body></html>"
    )
    p = tmp_path / "entities.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    assert len(docs) == 1
    all_text = " ".join(s.title + " " + s.content for s in docs[0].sections)
    # HTML entities should be decoded by BeautifulSoup
    assert "&amp;" not in all_text


def test_html_no_title_tag_uses_h1(tmp_path):
    content = "<html><body><h1>My Heading</h1><p>Content.</p></body></html>"
    p = tmp_path / "notitle.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    assert docs[0].title == "My Heading"


def test_html_multiple_tables(tmp_path):
    content = (
        "<html><body>"
        '<div class="section"><h1>Title</h1>'
        "<table><thead><tr><th>A</th></tr></thead>"
        "<tbody><tr><td>1</td></tr></tbody></table>"
        "<table><thead><tr><th>B</th></tr></thead>"
        "<tbody><tr><td>2</td></tr></tbody></table>"
        "</div></body></html>"
    )
    p = tmp_path / "tables.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    all_tables = [t for s in docs[0].sections for t in s.tables]
    assert len(all_tables) == 2


def test_html_slugify_empty():
    assert html_slugify("") == "section"


def test_html_slugify_basic():
    result = html_slugify("Hello World")
    assert result == "hello-world"


def test_html_code_block_extracted(tmp_path):
    content = (
        "<html><body>"
        '<div class="section"><h1>Title</h1>'
        '<pre><code class="language-python">x = 1</code></pre>'
        "</div></body></html>"
    )
    p = tmp_path / "code.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    all_code = [cb for s in docs[0].sections for cb in s.code_blocks]
    assert len(all_code) >= 1
    assert any("python" in cb.language for cb in all_code)


def test_html_no_title_no_h1_fallback_to_stem(tmp_path):
    content = "<html><body><p>No heading at all.</p></body></html>"
    p = tmp_path / "mystem.html"
    p.write_text(content, encoding="utf-8")
    parser = HtmlParser()
    docs = parser.parse(p, "test")
    assert "mystem" in docs[0].title.lower()


# ---------------------------------------------------------------------------
# RST parser edge cases
# ---------------------------------------------------------------------------


def test_rst_single_heading(tmp_path):
    content = "My Title\n========\n\nSome content here.\n"
    p = tmp_path / "single.rst"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    assert len(docs) == 1
    titles = [s.title for s in docs[0].sections]
    assert "My Title" in titles


def test_rst_code_block_via_double_colon(tmp_path):
    content = "Title\n=====\n\nExample::\n\n   some code here\n   more code\n\nEnd.\n"
    p = tmp_path / "colon_code.rst"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    all_code = [cb for s in docs[0].sections for cb in s.code_blocks]
    assert len(all_code) >= 1


def test_rst_xref_func_extracted(tmp_path):
    content = "Title\n=====\n\nUse :func:`my_function` to do things.\n"
    p = tmp_path / "xref.rst"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    all_links = [lk for s in docs[0].sections for lk in s.links]
    assert any("my_function" in lk.target for lk in all_links)


def test_rst_xref_class_ref_type(tmp_path):
    content = "Title\n=====\n\nSee :class:`MyClass` for details.\n"
    p = tmp_path / "xref_class.rst"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    all_links = [lk for s in docs[0].sections for lk in s.links]
    assert any(lk.ref_type == "class" for lk in all_links)


def test_rst_multiple_sections(tmp_path):
    content = (
        "First\n=====\n\nContent one.\n\n"
        "Second\n======\n\nContent two.\n\n"
        "Third\n=====\n\nContent three.\n"
    )
    p = tmp_path / "multi.rst"
    p.write_text(content, encoding="utf-8")
    parser = MarkdownParser()
    docs = parser.parse(p, "test")
    assert len(docs) == 1
    assert len(docs[0].sections) >= 2
