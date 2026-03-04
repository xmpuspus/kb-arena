"""CSV/TSV parser with auto-delimiter detection."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from kb_arena.ingest.parsers.utils import read_text, slugify, token_count, unique_id
from kb_arena.models.document import Document, Section, Table


def _detect_delimiter(text: str) -> str:
    first_line = text.split("\n")[0]
    if "\t" in first_line:
        return "\t"
    return ","


class CsvParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        text = read_text(path)
        if not text.strip():
            return []

        source = str(path)
        delimiter = _detect_delimiter(text)

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows_raw = list(reader)

        if len(rows_raw) < 2:
            return []

        headers = [h.strip() for h in rows_raw[0]]
        data_rows = [
            [c.strip() for c in row] for row in rows_raw[1:] if any(c.strip() for c in row)
        ]

        if not data_rows:
            return []

        chunk_size = 20
        seen_ids: set[str] = set()
        sections: list[Section] = []

        for chunk_start in range(0, len(data_rows), chunk_size):
            chunk = data_rows[chunk_start : chunk_start + chunk_size]
            chunk_end = min(chunk_start + chunk_size, len(data_rows))

            content_parts = []
            for row in chunk:
                row_text = "; ".join(f"{h}: {v}" for h, v in zip(headers, row) if v)
                content_parts.append(row_text)

            title = f"Rows {chunk_start + 1}-{chunk_end}"
            section_id = unique_id(slugify(title), seen_ids)

            sections.append(
                Section(
                    id=section_id,
                    title=title,
                    content="\n".join(content_parts),
                    heading_path=[path.stem, title],
                    tables=[Table(headers=headers, rows=chunk)],
                    level=1,
                )
            )

        full_text = " ".join(s.content for s in sections)
        return [
            Document(
                id=slugify(path.stem),
                source=source,
                corpus=corpus,
                title=path.stem,
                sections=sections,
                metadata={
                    "source_type": "csv",
                    "row_count": len(data_rows),
                    "column_count": len(headers),
                    "columns": headers,
                },
                raw_token_count=token_count(full_text),
            )
        ]
