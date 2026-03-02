"""HTML parser for Python documentation and generic HTML files.

Handles:
- Nested <div class="section"> structure
- h1-h6 heading hierarchy
- <dl> blocks for function/class/method/attribute signatures
- <table> for parameter tables
- <pre>/<code> for code blocks
- <a class="reference"> for cross-references
- Python docs specific: classes "function", "method", "class", "attribute"
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

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


def _heading_level(tag: Tag) -> int | None:
    if tag.name and re.match(r"^h([1-6])$", tag.name):
        return int(tag.name[1])
    return None


def _extract_tables(container: Tag) -> list[Table]:
    tables = []
    for tbl in container.find_all("table", recursive=True):
        headers = []
        rows = []
        thead = tbl.find("thead")
        if thead:
            hr = thead.find("tr")
            if hr:
                headers = [th.get_text(strip=True) for th in hr.find_all(["th", "td"])]
        tbody = tbl.find("tbody") or tbl
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)
        if headers or rows:
            tables.append(Table(headers=headers, rows=rows))
    return tables


def _extract_code_blocks(container: Tag) -> list[CodeBlock]:
    blocks = []
    for pre in container.find_all("pre"):
        code_tag = pre.find("code")
        code_text = (code_tag or pre).get_text()
        lang = ""
        if code_tag:
            for cls in code_tag.get("class", []):
                if cls.startswith("language-"):
                    lang = cls[len("language-") :]
                    break
        blocks.append(CodeBlock(language=lang, code=code_text.strip()))
    return blocks


def _extract_links(container: Tag) -> list[CrossRef]:
    refs = []
    for a in container.find_all("a"):
        classes = a.get("class", [])
        href = a.get("href", "")
        label = a.get_text(strip=True)
        if "reference" in classes or "internal" in classes or "external" in classes:
            ref_type = "external" if "external" in classes else "internal"
            if href and label:
                refs.append(CrossRef(target=href, label=label, ref_type=ref_type))
    return refs


def _extract_dl_sections(
    container: Tag, corpus: str, source: str, heading_path: list[str], seen_ids: set[str]
) -> list[Section]:
    """Extract <dl> entries as sub-sections (function/class/method signatures)."""
    sections = []
    py_types = {"function", "method", "class", "attribute", "exception", "data"}

    for dl in container.find_all("dl", recursive=False):
        dl_class = set(dl.get("class", []))
        obj_type = next((c for c in dl_class if c in py_types), "")

        dt = dl.find("dt")
        dd = dl.find("dd")
        if not dt:
            continue

        title = dt.get_text(strip=True)
        content = dd.get_text(separator=" ", strip=True) if dd else ""
        dd_tag = dd if dd else dl

        child_path = heading_path + [title]
        sid = _unique_id(_slugify(title), seen_ids)

        sections.append(
            Section(
                id=sid,
                title=title,
                content=content,
                heading_path=child_path,
                tables=_extract_tables(dd_tag),
                code_blocks=_extract_code_blocks(dd_tag),
                links=_extract_links(dd_tag),
                level=len(child_path),
                parent_id=None,
            )
        )
        # Recurse into nested dl entries
        if dd:
            sections.extend(_extract_dl_sections(dd, corpus, source, child_path, seen_ids))
    return sections


def _walk_sections(
    container: Tag, corpus: str, source: str, heading_stack: list[str], seen_ids: set[str]
) -> list[Section]:
    sections: list[Section] = []

    for child in container.children:
        if not isinstance(child, Tag):
            continue

        level = _heading_level(child)
        if level is not None:
            title = child.get_text(strip=True)
            heading_stack = heading_stack[: level - 1] + [title]
            sid = _unique_id(_slugify(title), seen_ids)
            sections.append(
                Section(
                    id=sid,
                    title=title,
                    content="",
                    heading_path=list(heading_stack),
                    level=level,
                )
            )
            continue

        # <div class="section"> or <section>
        if child.name in ("div", "section"):
            classes = child.get("class", [])
            if "section" in classes or child.name == "section":
                # Get heading inside this block
                h_tag = child.find(re.compile(r"^h[1-6]$"))
                if h_tag:
                    title = h_tag.get_text(strip=True)
                    level = _heading_level(h_tag) or 2
                    heading_stack = heading_stack[: level - 1] + [title]
                    content = child.get_text(separator=" ", strip=True)
                    sid = _unique_id(_slugify(title), seen_ids)
                    dl_sections = _extract_dl_sections(
                        child, corpus, source, list(heading_stack), seen_ids
                    )
                    sections.append(
                        Section(
                            id=sid,
                            title=title,
                            content=content,
                            heading_path=list(heading_stack),
                            tables=_extract_tables(child),
                            code_blocks=_extract_code_blocks(child),
                            links=_extract_links(child),
                            level=level,
                            children=[s.id for s in dl_sections],
                        )
                    )
                    sections.extend(dl_sections)
                else:
                    sections.extend(_walk_sections(child, corpus, source, heading_stack, seen_ids))
                continue

        # Recurse into any other container
        if child.name in ("article", "main", "body"):
            sections.extend(_walk_sections(child, corpus, source, heading_stack, seen_ids))

    return sections


class HtmlParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        soup = BeautifulSoup(text, "html.parser")

        # Document title from <title> or first h1
        title_tag = soup.find("title")
        h1 = soup.find("h1")
        doc_title = (
            title_tag.get_text(strip=True)
            if title_tag
            else (h1.get_text(strip=True) if h1 else path.stem)
        )

        source = str(path)
        seen_ids: set[str] = set()
        body = soup.find("body") or soup
        sections = _walk_sections(body, corpus, source, [], seen_ids)

        # Fallback: flat heading extraction when no section divs present
        if not sections:
            for tag in body.find_all(re.compile(r"^h[1-6]$")):
                title = tag.get_text(strip=True)
                level = _heading_level(tag) or 1
                sid = _unique_id(_slugify(title), seen_ids)
                sections.append(
                    Section(
                        id=sid,
                        title=title,
                        content="",
                        heading_path=[title],
                        level=level,
                    )
                )

        full_text = " ".join(s.content for s in sections)
        return [
            Document(
                id=_slugify(path.stem),
                source=source,
                corpus=corpus,
                title=doc_title,
                sections=sections,
                raw_token_count=_token_count(full_text),
            )
        ]
