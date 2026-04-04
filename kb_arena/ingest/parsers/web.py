"""Web scraper with llms.txt fast-path — requires: pip install kb-arena[web]"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

from kb_arena.ingest.parsers.utils import slugify, token_count, unique_id
from kb_arena.models.document import Document, Section

log = logging.getLogger(__name__)

_MAX_DEPTH = 3
_MAX_PAGES = 50


def _try_import_httpx():
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for web scraping. Install with: pip install kb-arena[web]"
        ) from None


def _try_import_bs4():
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup
    except ImportError:
        raise ImportError(
            "beautifulsoup4 is required for web scraping. Install with: pip install kb-arena[web]"
        ) from None


def _check_llms_txt(base_url: str, client) -> str | None:
    parsed = urlparse(base_url)
    llms_url = f"{parsed.scheme}://{parsed.netloc}/llms.txt"
    try:
        resp = client.get(llms_url, timeout=10, follow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 50:
            log.info("Found llms.txt at %s — using as primary source", llms_url)
            return resp.text
    except Exception:  # noqa: BLE001
        log.debug("llms.txt check failed for %s", llms_url, exc_info=True)
    return None


def _clean_html(html: str, bs_class) -> str:
    soup = bs_class(html, "html.parser")

    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    return main.get_text(separator="\n", strip=True)


def _extract_links(html: str, base_url: str, bs_class) -> list[str]:
    soup = bs_class(html, "html.parser")
    parsed_base = urlparse(base_url)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc == parsed_base.netloc and parsed.scheme in ("http", "https"):
            # Normalize: strip fragments and query params
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean not in links:
                links.append(clean)
    return links


def _fetch_page(url: str, client) -> str | None:
    try:
        resp = client.get(url, timeout=15, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "text/html" in content_type:
            return resp.text
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to fetch %s: %s", url, exc)
    return None


class WebParser:
    def __init__(self, max_depth: int = _MAX_DEPTH, max_pages: int = _MAX_PAGES):
        self.max_depth = max_depth
        self.max_pages = max_pages

    def parse(self, path: Path, corpus: str) -> list[Document]:
        # Path is either a URL string or a file containing one
        url = str(path)
        if not url.startswith(("http://", "https://")):
            # Try reading URL from file
            try:
                url = path.read_text().strip()
            except Exception:  # noqa: BLE001
                log.warning("Failed to read URL from %s", path, exc_info=True)
                return []
            if not url.startswith(("http://", "https://")):
                return []

        return self._scrape(url, corpus)

    def _scrape(self, url: str, corpus: str) -> list[Document]:
        httpx = _try_import_httpx()
        bs_class = _try_import_bs4()

        with httpx.Client(
            headers={"User-Agent": "kb-arena/1.0 (documentation indexer)"},
        ) as client:
            # llms.txt takes priority over crawling
            llms_txt = _check_llms_txt(url, client)
            if llms_txt:
                return self._parse_llms_txt(llms_txt, url, corpus)

            return self._crawl(url, corpus, client, bs_class)

    def _parse_llms_txt(self, text: str, url: str, corpus: str) -> list[Document]:
        from kb_arena.ingest.parsers.markdown import MarkdownParser

        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            parser = MarkdownParser()
            docs = parser.parse(tmp_path, corpus)
        finally:
            tmp_path.unlink(missing_ok=True)

        for doc in docs:
            doc.source = url
            doc.id = slugify(urlparse(url).netloc)
            doc.metadata = {"source_type": "llms.txt", "url": url}

        return docs

    def _crawl(self, start_url: str, corpus: str, client, bs_class) -> list[Document]:
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(start_url, 0)]
        pages: list[tuple[str, str]] = []  # (url, text_content)

        while queue and len(pages) < self.max_pages:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            html = _fetch_page(url, client)
            if not html:
                continue

            text = _clean_html(html, bs_class)
            if text.strip():
                pages.append((url, text))

            if depth < self.max_depth:
                for link in _extract_links(html, url, bs_class):
                    if link not in visited:
                        queue.append((link, depth + 1))

        if not pages:
            return []

        docs = []
        for page_url, text in pages:
            parsed = urlparse(page_url)
            page_slug = slugify(parsed.path.strip("/") or parsed.netloc)

            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

            seen_ids: set[str] = set()
            sections: list[Section] = []

            for i, para in enumerate(paragraphs):
                first_line = para.split("\n")[0][:100]
                title = first_line if len(first_line) < 80 else f"Section {i + 1}"
                section_id = unique_id(slugify(title), seen_ids)

                sections.append(
                    Section(
                        id=section_id,
                        title=title,
                        content=para,
                        heading_path=[title],
                        level=1,
                    )
                )

            if not sections:
                continue

            full_text = " ".join(s.content for s in sections)
            docs.append(
                Document(
                    id=page_slug,
                    source=page_url,
                    corpus=corpus,
                    title=sections[0].title,
                    sections=sections,
                    metadata={"source_type": "web", "url": page_url},
                    raw_token_count=token_count(full_text),
                )
            )

        return docs
