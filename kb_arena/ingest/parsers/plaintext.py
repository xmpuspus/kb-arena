"""Plain text parser with ALL CAPS heading detection."""

from __future__ import annotations

import re
from pathlib import Path

from kb_arena.ingest.parsers.utils import read_text, slugify, token_count, unique_id
from kb_arena.models.document import Document, Section

_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9 :,/&-]{2,80}$")


def _looks_like_heading(line: str, next_line: str | None) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    if _HEADING_PATTERN.match(stripped):
        return next_line is None or next_line.strip() == "" or next_line.startswith(" ")
    return False


class PlaintextParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        text = read_text(path)
        lines = text.splitlines()
        source = str(path)

        if not text.strip():
            return []

        sections_raw: list[tuple[str, list[str]]] = []
        current_title = path.stem
        current_body: list[str] = []

        for i, line in enumerate(lines):
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if _looks_like_heading(line, next_line):
                if current_body or current_title != path.stem:
                    sections_raw.append((current_title, current_body))
                current_title = line.strip()
                current_body = []
            else:
                current_body.append(line)

        sections_raw.append((current_title, current_body))

        seen_ids: set[str] = set()
        sections: list[Section] = []

        for title, body_lines in sections_raw:
            content = "\n".join(body_lines).strip()
            if not content and title == path.stem:
                continue
            section_id = unique_id(slugify(title), seen_ids)
            sections.append(
                Section(
                    id=section_id,
                    title=title,
                    content=content,
                    heading_path=[title],
                    level=1,
                )
            )

        if not sections:
            section_id = unique_id(slugify(path.stem), seen_ids)
            sections.append(
                Section(
                    id=section_id,
                    title=path.stem,
                    content=text.strip(),
                    heading_path=[path.stem],
                    level=1,
                )
            )

        doc_title = sections[0].title if sections else path.stem
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
