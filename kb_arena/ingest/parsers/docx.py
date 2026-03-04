"""DOCX parser — requires: pip install kb-arena[docx]"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from kb_arena.ingest.parsers.utils import slugify
from kb_arena.models.document import Document

log = logging.getLogger(__name__)


def _try_import_mammoth():
    try:
        import mammoth

        return mammoth
    except ImportError:
        raise ImportError(
            "mammoth is required for DOCX parsing. Install with: pip install kb-arena[docx]"
        ) from None


class DocxParser:
    def parse(self, path: Path, corpus: str) -> list[Document]:
        mammoth = _try_import_mammoth()

        with open(path, "rb") as f:
            result = mammoth.convert_to_html(f)
            html = result.value

        if not html.strip():
            return []

        from kb_arena.ingest.parsers.html import HtmlParser

        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as tmp:
            tmp.write(html)
            tmp_path = Path(tmp.name)

        try:
            html_parser = HtmlParser()
            docs = html_parser.parse(tmp_path, corpus)
        finally:
            tmp_path.unlink(missing_ok=True)

        source = str(path)
        for doc in docs:
            doc.source = source
            doc.id = slugify(path.stem)
            if doc.title == tmp_path.stem:
                doc.title = path.stem

        return docs
