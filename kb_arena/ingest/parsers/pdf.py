"""PDF parser — requires: pip install kb-arena[pdf]"""

from __future__ import annotations

import logging
from pathlib import Path

from kb_arena.ingest.parsers.utils import slugify, token_count, unique_id
from kb_arena.models.document import Document, Section, Table

log = logging.getLogger(__name__)


def _try_import_fitz():
    try:
        import fitz  # noqa: F811

        return fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install with: pip install kb-arena[pdf]"
        ) from None


def _extract_tables_from_page(page) -> list[Table]:
    tables = []
    try:
        tab_finder = page.find_tables()
        for tab in tab_finder:
            data = tab.extract()
            if not data or len(data) < 2:
                continue
            headers = [str(c) if c else "" for c in data[0]]
            rows = [[str(c) if c else "" for c in row] for row in data[1:]]
            tables.append(Table(headers=headers, rows=rows))
    except Exception:  # noqa: BLE001
        log.debug("Table extraction failed for page", exc_info=True)
    return tables


def _detect_headings(blocks: list[dict], median_size: float) -> list[tuple[int, str, float]]:
    headings = []
    for i, block in enumerate(blocks):
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0)
                text = span.get("text", "").strip()
                if not text:
                    continue
                if size > median_size * 1.15 and len(text) < 200:
                    headings.append((i, text, size))
    return headings


class PdfParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        fitz = _try_import_fitz()
        doc = fitz.open(str(path))
        source = str(path)

        # First pass: collect font sizes for median calculation
        all_blocks = []
        all_sizes = []
        page_tables: dict[int, list[Table]] = {}

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            all_blocks.append(blocks)
            page_tables[page_num] = _extract_tables_from_page(page)

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size", 0)
                        if size > 0:
                            all_sizes.append(size)

        if not all_sizes:
            doc.close()
            return []

        all_sizes.sort()
        median_size = all_sizes[len(all_sizes) // 2]

        # Map distinct heading sizes → levels (largest = 1)
        heading_sizes: set[float] = set()
        for blocks in all_blocks:
            for _, _, size in _detect_headings(blocks, median_size):
                heading_sizes.add(round(size, 1))

        sorted_sizes = sorted(heading_sizes, reverse=True)
        size_to_level = {s: i + 1 for i, s in enumerate(sorted_sizes)}
        sections_raw: list[tuple[int, str, list[str], list[Table]]] = []
        current_title: str | None = None
        current_level = 1
        current_body: list[str] = []
        current_tables: list[Table] = []

        for page_num, blocks in enumerate(all_blocks):
            headings = _detect_headings(blocks, median_size)
            heading_indices = {h[0] for h in headings}

            for i, block in enumerate(blocks):
                if block.get("type") != 0:
                    continue

                if i in heading_indices:
                    if current_title is not None:
                        sections_raw.append(
                            (current_level, current_title, current_body, current_tables)
                        )

                    heading = next(h for h in headings if h[0] == i)
                    current_title = heading[1]
                    current_level = size_to_level.get(round(heading[2], 1), 1)
                    current_body = []
                    current_tables = list(page_tables.get(page_num, []))
                else:
                    text_parts = []
                    for line in block.get("lines", []):
                        line_text = " ".join(span.get("text", "") for span in line.get("spans", []))
                        if line_text.strip():
                            text_parts.append(line_text.strip())
                    if text_parts:
                        current_body.append(" ".join(text_parts))

        if current_title is not None:
            sections_raw.append((current_level, current_title, current_body, current_tables))
        elif current_body:
            # No headings found — single section fallback
            sections_raw.append((1, path.stem, current_body, current_tables))

        doc.close()

        if not sections_raw:
            return []

        doc_title = sections_raw[0][1] if sections_raw else path.stem
        seen_ids: set[str] = set()
        sections: list[Section] = []
        heading_stack: list[str] = []

        for level, title, body_lines, tables in sections_raw:
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)

            content = "\n".join(body_lines).strip()
            section_id = unique_id(slugify(title), seen_ids)

            sections.append(
                Section(
                    id=section_id,
                    title=title,
                    content=content,
                    heading_path=list(heading_stack),
                    tables=tables,
                    code_blocks=[],
                    links=[],
                    level=level,
                )
            )

        full_text = " ".join(s.content for s in sections)
        return [
            Document(
                id=slugify(path.stem),
                source=source,
                corpus=corpus,
                title=doc_title,
                sections=sections,
                raw_token_count=token_count(full_text),
            )
        ]
