"""Markdown and RST parser.

Handles:
- Markdown: ATX headings (#, ##, ...), fenced code blocks, pipe tables
- RST: underline headings, directive blocks (.. function::, .. class::, etc.),
  cross-reference roles (:func:, :class:, :mod:, :meth:), :: code blocks, grid tables
"""

from __future__ import annotations

import re
from pathlib import Path

from kb_arena.models.document import CodeBlock, CrossRef, Document, Section, Table


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s/-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text or "section"


def _unique_id(slug: str, seen: set[str]) -> str:
    candidate = slug
    n = 1
    while candidate in seen:
        candidate = f"{slug}-{n}"
        n += 1
    seen.add(candidate)
    return candidate


def _token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_MD_FENCE_OPEN = re.compile(r"^```(\w*)$")
_MD_FENCE_CLOSE = re.compile(r"^```\s*$")
_MD_TABLE_ROW = re.compile(r"^\|(.+)\|$")
_MD_TABLE_SEP = re.compile(r"^\|[-| :]+\|$")


def _parse_md_tables(lines: list[str]) -> list[Table]:
    tables = []
    i = 0
    while i < len(lines):
        if _MD_TABLE_ROW.match(lines[i]):
            header_line = lines[i]
            if i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1]):
                headers = [c.strip() for c in header_line.strip("|").split("|")]
                rows = []
                j = i + 2
                while j < len(lines) and _MD_TABLE_ROW.match(lines[j]):
                    rows.append([c.strip() for c in lines[j].strip("|").split("|")])
                    j += 1
                tables.append(Table(headers=headers, rows=rows))
                i = j
                continue
        i += 1
    return tables


def _parse_md_code_blocks(lines: list[str]) -> list[CodeBlock]:
    blocks = []
    i = 0
    while i < len(lines):
        m = _MD_FENCE_OPEN.match(lines[i])
        if m:
            lang = m.group(1)
            code_lines = []
            i += 1
            while i < len(lines) and not _MD_FENCE_CLOSE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            blocks.append(CodeBlock(language=lang, code="\n".join(code_lines)))
        i += 1
    return blocks


def _strip_code_and_tables(lines: list[str]) -> list[str]:
    """Remove fenced code blocks and table rows for clean content text."""
    out = []
    in_fence = False
    for line in lines:
        if _MD_FENCE_OPEN.match(line) or _MD_FENCE_CLOSE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _MD_TABLE_ROW.match(line) or _MD_TABLE_SEP.match(line):
            continue
        out.append(line)
    return out


def _parse_markdown(text: str, source: str, corpus: str) -> list[Document]:
    lines = text.splitlines()

    # Split into logical heading-sections
    # Each entry: (level, title, body_lines)
    sections_raw: list[tuple[int, str, list[str]]] = []
    # None signals "before any heading" — preamble body is collected but not titled
    current_level: int | None = None
    current_title: str | None = None
    current_body: list[str] = []

    for line in lines:
        m = _MD_HEADING.match(line)
        if m:
            # Flush previous section only if it has a real heading title
            if current_title is not None:
                sections_raw.append((current_level or 1, current_title, current_body))
            current_level = len(m.group(1))
            current_title = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)

    # Flush final section
    if current_title is not None:
        sections_raw.append((current_level or 1, current_title, current_body))

    if not sections_raw:
        return []

    # Build heading stack to compute heading_path
    doc_title = sections_raw[0][1] if sections_raw else Path(source).stem
    doc_id = _slugify(Path(source).stem)

    seen_ids: set[str] = set()
    sections: list[Section] = []
    heading_stack: list[str] = []  # tracks titles at each level

    for level, title, body_lines in sections_raw:
        # Trim stack to current depth
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        heading_path = list(heading_stack)

        content = "\n".join(_strip_code_and_tables(body_lines)).strip()
        code_blocks = _parse_md_code_blocks(body_lines)
        tables = _parse_md_tables(body_lines)
        section_id = _unique_id(_slugify(title), seen_ids)

        sections.append(
            Section(
                id=section_id,
                title=title,
                content=content,
                heading_path=heading_path,
                tables=tables,
                code_blocks=code_blocks,
                links=[],
                level=level,
            )
        )

    full_text = " ".join(s.content for s in sections)
    doc = Document(
        id=_slugify(doc_id),
        source=source,
        corpus=corpus,
        title=doc_title,
        sections=sections,
        raw_token_count=_token_count(full_text),
    )
    return [doc]


# ---------------------------------------------------------------------------
# RST parser
# ---------------------------------------------------------------------------

# RST heading underline chars in conventional order
_RST_UNDERLINE_CHARS = set("=-~^\"'`:#*+<>_")
_RST_DIRECTIVE = re.compile(r"^\.\.\s+([\w:-]+)::\s*(.*)?$")
_RST_ROLE = re.compile(r":(\w+):`([^`]+)`")
_RST_XREF_TYPES = {"func", "class", "mod", "meth", "attr", "data", "obj", "exc", "ref"}


def _is_rst_underline(line: str, prev_line: str) -> bool:
    """True if `line` is an RST heading underline for `prev_line`."""
    if not line or not prev_line:
        return False
    ch = line[0]
    if ch not in _RST_UNDERLINE_CHARS:
        return False
    return len(line.strip()) >= len(prev_line.strip()) and len(set(line.strip())) == 1


def _extract_rst_xrefs(text: str) -> list[CrossRef]:
    refs = []
    for m in _RST_ROLE.finditer(text):
        role, target = m.group(1), m.group(2)
        if role in _RST_XREF_TYPES:
            ref_type = {
                "func": "function",
                "class": "class",
                "mod": "module",
                "meth": "method",
                "attr": "attribute",
            }.get(role, role)
            refs.append(CrossRef(target=target, label=target, ref_type=ref_type))
    return refs


def _extract_rst_code_blocks(lines: list[str]) -> list[CodeBlock]:
    """Extract :: literal blocks and .. code-block:: directives."""
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # .. code-block:: lang  or  .. code::
        m = _RST_DIRECTIVE.match(line)
        if m and m.group(1) in ("code-block", "code"):
            lang = m.group(2).strip() if m.group(2) else ""
            # skip options and blank lines
            i += 1
            while i < len(lines) and (lines[i].startswith("   :") or lines[i].strip() == ""):
                i += 1
            code_lines = []
            while i < len(lines) and (lines[i].startswith("   ") or lines[i].strip() == ""):
                code_lines.append(lines[i][3:] if lines[i].startswith("   ") else "")
                i += 1
            blocks.append(CodeBlock(language=lang, code="\n".join(code_lines).strip()))
            continue
        # bare :: at end of paragraph
        if line.rstrip().endswith("::"):
            i += 1
            # skip blank
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            code_lines = []
            while i < len(lines) and (
                lines[i].startswith("   ") or lines[i].startswith("\t") or lines[i].strip() == ""
            ):
                code_lines.append(lines[i].strip())
                i += 1
            if code_lines:
                blocks.append(CodeBlock(language="", code="\n".join(code_lines).strip()))
            continue
        i += 1
    return blocks


def _extract_rst_grid_tables(lines: list[str]) -> list[Table]:
    """Parse RST grid tables (+---+---+ style)."""
    tables = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("+") and re.match(r"^\+[-+=+]+\+$", lines[i]):
            table_lines = [lines[i]]
            j = i + 1
            while j < len(lines) and (lines[j].startswith("|") or lines[j].startswith("+")):
                table_lines.append(lines[j])
                j += 1
            # parse cells from | rows
            headers: list[str] = []
            rows: list[list[str]] = []
            is_header = True
            for tl in table_lines:
                if tl.startswith("+"):
                    if "=" in tl:
                        is_header = False
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                if is_header:
                    if not headers:
                        headers = cells
                    else:
                        # multi-line header cell — append
                        headers = [h + " " + c for h, c in zip(headers, cells)]
                else:
                    rows.append(cells)
            if headers:
                tables.append(Table(headers=headers, rows=rows))
            i = j
            continue
        i += 1
    return tables


def _parse_rst(text: str, source: str, corpus: str) -> list[Document]:
    lines = text.splitlines()
    doc_title = Path(source).stem

    # Detect heading levels by first-encounter of underline char
    level_map: dict[str, int] = {}

    sections_raw: list[tuple[int, str, list[str]]] = []
    current_level = 0
    current_title = doc_title
    current_body: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        if _is_rst_underline(next_line, line):
            ch = next_line[0]
            if ch not in level_map:
                level_map[ch] = len(level_map) + 1
            lvl = level_map[ch]

            if current_title or current_body:
                sections_raw.append((current_level, current_title, current_body))
            current_level = lvl
            current_title = line.strip()
            current_body = []
            i += 2  # skip title + underline
            continue

        # Overline+title+underline style (decorative heading)
        if (
            line
            and len(set(line.strip())) == 1
            and line[0] in _RST_UNDERLINE_CHARS
            and i + 2 < len(lines)
            and _is_rst_underline(lines[i + 2], lines[i + 1])
        ):
            ch = line[0]
            if ch not in level_map:
                level_map[ch] = len(level_map) + 1
            lvl = level_map[ch]
            title = lines[i + 1].strip()
            if current_title or current_body:
                sections_raw.append((current_level, current_title, current_body))
            current_level = lvl
            current_title = title
            current_body = []
            i += 3
            continue

        current_body.append(line)
        i += 1

    if current_title or current_body:
        sections_raw.append((current_level, current_title, current_body))

    if sections_raw:
        doc_title = sections_raw[0][1]

    seen_ids: set[str] = set()
    sections: list[Section] = []
    heading_stack: list[str] = []

    for level, title, body_lines in sections_raw:
        heading_stack = heading_stack[: level - 1]
        heading_stack.append(title)
        heading_path = list(heading_stack)

        # Strip directive lines from content text
        content_lines = [l for l in body_lines if not _RST_DIRECTIVE.match(l)]
        content = "\n".join(content_lines).strip()

        links = _extract_rst_xrefs("\n".join(body_lines))
        code_blocks = _extract_rst_code_blocks(body_lines)
        tables = _extract_rst_grid_tables(body_lines)
        section_id = _unique_id(_slugify(title), seen_ids)

        sections.append(
            Section(
                id=section_id,
                title=title,
                content=content,
                heading_path=heading_path,
                tables=tables,
                code_blocks=code_blocks,
                links=links,
                level=level,
            )
        )

    full_text = " ".join(s.content for s in sections)
    return [
        Document(
            id=_slugify(Path(source).stem),
            source=source,
            corpus=corpus,
            title=doc_title,
            sections=sections,
            raw_token_count=_token_count(full_text),
        )
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class MarkdownParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        source = str(path)
        if path.suffix.lower() == ".rst":
            return _parse_rst(text, source, corpus)
        return _parse_markdown(text, source, corpus)
