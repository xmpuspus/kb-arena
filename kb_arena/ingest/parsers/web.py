"""Web scraper with llms.txt fast-path — requires: pip install kb-arena[web]"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

from kb_arena.ingest.parsers.utils import slugify, token_count, unique_id
from kb_arena.models.document import Document, Section

log = logging.getLogger(__name__)

_MAX_DEPTH = 3
_MAX_PAGES = 50
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB cap — bounds memory use from one page

# SSRF guard — block file://, internal IPs, and metadata endpoints.
_BLOCKED_HOSTS = {
    "metadata.google.internal",
    "metadata.gce.internal",
    "instance-data.ec2.internal",
}


class SSRFBlockedError(ValueError):
    """Raised when a URL targets a private or blocked host."""


# Backward-compat alias for any external callers that imported the old name.
SSRFBlocked = SSRFBlockedError


class ResponseTooLargeError(ValueError):
    """Raised when a fetched response body exceeds the size cap."""


def _validate_url(url: str) -> str:
    """Reject non-HTTP(S) schemes and IPs in private/loopback/link-local ranges.

    Resolves DNS and checks every resolved IP, so attackers can't use a domain
    that resolves to 127.0.0.1 or 169.254.169.254 (AWS metadata). Returns the
    first checked IP as a string. The caller must connect to that exact IP
    instead of resolving the host again: a second resolution can return a
    different address (DNS rebinding), which would skip this check.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFBlocked(f"only http/https schemes allowed, got {parsed.scheme!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise SSRFBlocked("missing host")
    if host in _BLOCKED_HOSTS:
        raise SSRFBlocked(f"blocked host: {host}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SSRFBlocked(f"dns resolution failed for {host}: {exc}") from exc
    checked_ips: list[str] = []
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            raise SSRFBlocked(f"refusing to fetch private address {ip_str} ({host})")
        if ip.is_reserved or ip.is_unspecified:
            raise SSRFBlocked(f"refusing to fetch reserved address {ip_str} ({host})")
        checked_ips.append(ip_str)
    if not checked_ips:
        raise SSRFBlocked(f"dns resolution returned no usable address for {host}")
    return checked_ips[0]


def _pin_url_to_ip(url: str, ip: str) -> str:
    """Rewrite a URL's host to a literal IP, keeping the scheme, port, path, and query.

    _safe_get sends the request to this URL so the connection goes to the
    exact IP _validate_url checked, not to a host that DNS could resolve
    differently a second time.
    """
    parsed = urlparse(url)
    netloc = f"[{ip}]" if ":" in ip else ip
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


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


def _safe_get(client, url: str, timeout: int = 15, max_redirects: int = 5):
    """GET a URL with SSRF validation on every hop. Disables auto-follow_redirects.

    Connects to the exact IP _validate_url checked, not the hostname, at
    every hop. This closes a DNS-rebinding gap: without the pin, the check
    resolves the host once, and the HTTP client resolves it again to open
    the socket. A rebinding host can answer the check with a public IP and
    the connect with a private one. The Host header and TLS SNI still carry
    the real hostname, so the server sees a normal request and certificate
    checks still pass.

    Streams the body and aborts as soon as it passes _MAX_RESPONSE_BYTES, at
    every redirect hop, so a huge or malicious page cannot buffer its full
    body in memory before the cap gets checked.
    """
    current = url
    for _ in range(max_redirects + 1):
        host = urlparse(current).hostname or ""
        pinned_ip = _validate_url(current)
        pinned_url = _pin_url_to_ip(current, pinned_ip)
        with client.stream(
            "GET",
            pinned_url,
            timeout=timeout,
            follow_redirects=False,
            headers={"Host": host},
            extensions={"sni_hostname": host},
        ) as resp:
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > _MAX_RESPONSE_BYTES:
                raise ResponseTooLargeError(
                    f"{current}: content-length {content_length} exceeds "
                    f"the {_MAX_RESPONSE_BYTES} byte cap"
                )

            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > _MAX_RESPONSE_BYTES:
                    raise ResponseTooLargeError(
                        f"{current}: response body exceeded the "
                        f"{_MAX_RESPONSE_BYTES} byte cap while streaming"
                    )
                chunks.append(chunk)
            # Same field httpx's own Response.read() sets, so resp.text and
            # resp.content keep working for callers after the stream closes.
            resp._content = b"".join(chunks)

        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("location")
            if not location:
                return resp
            current = urljoin(current, location)
            continue
        return resp
    raise SSRFBlocked(f"too many redirects starting from {url}")


def _check_llms_txt(base_url: str, client) -> str | None:
    parsed = urlparse(base_url)
    llms_url = f"{parsed.scheme}://{parsed.netloc}/llms.txt"
    try:
        resp = _safe_get(client, llms_url, timeout=10)
        if resp.status_code == 200 and len(resp.text) > 50:
            log.info("Found llms.txt at %s — using as primary source", llms_url)
            return resp.text
    except SSRFBlocked as exc:
        log.warning("llms.txt blocked: %s", exc)
    except ResponseTooLargeError as exc:
        log.warning("llms.txt too large, skipped: %s", exc)
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
        resp = _safe_get(client, url, timeout=15)
        content_type = resp.headers.get("content-type", "")
        if resp.status_code == 200 and "text/html" in content_type:
            return resp.text
    except SSRFBlocked as exc:
        log.warning("Refusing to fetch %s: %s", url, exc)
    except ResponseTooLargeError as exc:
        log.warning("Skipping oversized page %s: %s", url, exc)
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
        try:
            _validate_url(url)
        except SSRFBlocked as exc:
            log.error("Refusing to scrape %s: %s", url, exc)
            return []

        httpx = _try_import_httpx()
        bs_class = _try_import_bs4()

        with httpx.Client(
            headers={"User-Agent": "kb-arena/1.0 (documentation indexer)"},
            follow_redirects=False,
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
