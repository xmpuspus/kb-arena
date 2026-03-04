"""SEC EDGAR 10-K HTML parser.

Standard 10-K structure:
  Item 1  — Business
  Item 1A — Risk Factors
  Item 2  — Properties
  Item 3  — Legal Proceedings
  Item 7  — MD&A
  Item 8  — Financial Statements
  Item 9  — Changes in Accountants
  Item 13 — Executive Compensation

Named-entity patterns extracted from text:
  - Dollar amounts  ($X.X billion / $X,XXX,XXX)
  - Executive names (via title heuristics)
  - Company names (via "Inc.", "Corp.", "LLC", etc.)
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from kb_arena.ingest.parsers.utils import read_text, slugify, token_count, unique_id
from kb_arena.models.document import Document, Section, Table

# 10-K item header patterns
_ITEM_HEADER = re.compile(
    r"^item\s+(\d+[a-c]?)\s*[.—–-]?\s*(.+)$",
    re.IGNORECASE,
)

# Named-entity extraction patterns
_DOLLAR_AMOUNT = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?", re.IGNORECASE
)
_COMPANY_NAME = re.compile(r"\b[A-Z][A-Za-z0-9&,.\s]{2,40}(?:Inc\.|Corp\.|LLC|Ltd\.|L\.P\.|Co\.)")
_EXECUTIVE_TITLE = re.compile(
    r"(?:Chief\s+\w+\s+Officer|President|Chairman|CEO|CFO|COO|CTO|Director)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
)


def _extract_named_entities(text: str) -> dict[str, list[str]]:
    return {
        "dollar_amounts": _DOLLAR_AMOUNT.findall(text)[:20],
        "companies": list({m.group().strip() for m in _COMPANY_NAME.finditer(text)})[:10],
        "executives": list({m.group(1) for m in _EXECUTIVE_TITLE.finditer(text)})[:10],
    }


def _parse_table_tag(tbl: Tag) -> Table | None:
    headers: list[str] = []
    rows: list[list[str]] = []
    first_row = True
    for tr in tbl.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        if not cells:
            continue
        th_cells = tr.find_all("th")
        if first_row and th_cells:
            headers = cells
            first_row = False
        else:
            rows.append(cells)
            first_row = False
    return Table(headers=headers, rows=rows) if rows else None


def _extract_tables(container: Tag) -> list[Table]:
    # If container itself is a <table>, parse it directly
    if container.name == "table":
        t = _parse_table_tag(container)
        return [t] if t else []
    tables = []
    for tbl in container.find_all("table", recursive=True):
        t = _parse_table_tag(tbl)
        if t:
            tables.append(t)
    return tables


def _text_is_item_header(text: str) -> re.Match | None:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return _ITEM_HEADER.match(cleaned)


class SecEdgarParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        text = read_text(path)
        soup = BeautifulSoup(text, "html.parser")
        source = str(path)

        # Try to extract company name and period from document metadata
        doc_title = path.stem
        title_tag = soup.find("title")
        if title_tag:
            doc_title = title_tag.get_text(strip=True)

        seen_ids: set[str] = set()
        sections: list[Section] = []

        # Strategy 1: find <div> or <p> tags that look like Item headers
        # SEC EDGAR HTML uses inconsistent structure — scan all block elements
        body = soup.find("body") or soup
        # Include "table" so whole table elements are captured as body content.
        # Exclude "td"/"tr" to avoid double-counting individual cells when the parent
        # table is already in the list.
        all_blocks = body.find_all(
            ["p", "div", "span", "h1", "h2", "h3", "h4", "table"], recursive=True
        )

        current_item_num = ""
        current_title = ""
        current_body_tags: list[Tag] = []

        def _flush_section() -> None:
            if not current_title:
                return
            content_parts = []
            tables = []
            for tag in current_body_tags:
                content_parts.append(tag.get_text(separator=" ", strip=True))
                tables.extend(_extract_tables(tag))
            content = " ".join(content_parts).strip()
            sid = unique_id(
                slugify(f"item-{current_item_num}-{current_title}"),
                seen_ids,
            )
            sections.append(
                Section(
                    id=sid,
                    title=f"Item {current_item_num}: {current_title}",
                    content=content,
                    heading_path=["10-K", f"Item {current_item_num}"],
                    tables=tables,
                    code_blocks=[],
                    links=[],
                    level=2,
                    parent_id="10-k-root",
                )
            )
            # attach entity metadata to section via a separate document-level store
            # (entities go into document metadata, not section model)
            sections[-1].content = content  # already set above

        seen_header_text: set[str] = set()

        for block in all_blocks:
            raw = block.get_text(separator=" ", strip=True)
            if len(raw) > 200:
                # Too long to be a header
                if current_title:
                    current_body_tags.append(block)
                continue

            m = _text_is_item_header(raw)
            if m and raw not in seen_header_text:
                seen_header_text.add(raw)
                _flush_section()
                current_item_num = m.group(1).upper()
                current_title = m.group(2).strip()
                current_body_tags = []
            elif current_title:
                current_body_tags.append(block)

        _flush_section()

        # Fallback: if no Item headers found, treat whole doc as one section
        if not sections:
            full_text = body.get_text(separator=" ", strip=True)
            sid = unique_id(slugify(path.stem), seen_ids)
            sections.append(
                Section(
                    id=sid,
                    title=doc_title,
                    content=full_text[:50000],  # cap enormous filings
                    heading_path=["10-K"],
                    tables=_extract_tables(body),
                    level=1,
                )
            )

        full_text = " ".join(s.content for s in sections)
        global_entities = _extract_named_entities(full_text)

        return [
            Document(
                id=slugify(path.stem),
                source=source,
                corpus=corpus,
                title=doc_title,
                sections=sections,
                metadata={"named_entities": global_entities},
                raw_token_count=token_count(full_text),
            )
        ]
