"""Web scraper with llms.txt fast-path — requires: pip install kb-arena[web]"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

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


def _validate_url(url: str) -> list[str]:
    """Reject non-HTTP(S) schemes and IPs in private/loopback/link-local ranges.

    Resolves DNS and checks every resolved IP, so attackers can't use a domain
    that resolves to 127.0.0.1 or 169.254.169.254 (AWS metadata). Returns the
    checked IPs in resolver order. The connection must go to one of them, not
    to a fresh lookup: a second resolution can return a different address
    (DNS rebinding), which would skip this check.
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
        # getaddrinfo repeats each address once per socket type.
        if ip_str not in checked_ips:
            checked_ips.append(ip_str)
    if not checked_ips:
        raise SSRFBlocked(f"dns resolution returned no usable address for {host}")
    return checked_ips


class _PinnedBackend(httpcore.SyncBackend):
    """Open each socket to an IP _validate_url checked, never to a fresh DNS answer.

    httpcore hands connect_tcp the URL's hostname and then starts TLS against
    that same hostname, so certificate checks, the Host header, cookies, and
    the connection-pool key all keep the real host. Only the socket target
    changes. Rewriting the URL to the IP instead would pool two hostnames that
    share an IP onto one TLS session and scope cookies to the IP.
    """

    def __init__(self, pins: dict[str, list[str]]) -> None:
        super().__init__()
        self._pins = pins

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        targets = self._pins.get(host.lower())
        if not targets:
            # Fail closed: a host that never passed _validate_url gets no socket.
            raise httpcore.ConnectError(f"no checked address for {host}")
        # Every address here passed the check, so falling through to the next
        # one on a refused connect keeps the multi-address fallback a plain
        # hostname connect would have had.
        for ip in targets[:-1]:
            try:
                return super().connect_tcp(
                    ip,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout):
                continue
        return super().connect_tcp(
            targets[-1],
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


class _PinnedTransport(httpx.HTTPTransport):
    """HTTPTransport whose connection pool dials through _PinnedBackend."""

    def __init__(self, pins: dict[str, list[str]]) -> None:
        super().__init__()
        # HTTPTransport builds its pool with no way to pass a network backend,
        # so this swaps in one that does. Same pool class and SSL context.
        self._pool.close()
        self._pool = httpcore.ConnectionPool(
            ssl_context=httpx.create_ssl_context(),
            network_backend=_PinnedBackend(pins),
        )


class _PinnedClient(httpx.Client):
    """httpx.Client that only connects to addresses _validate_url checked.

    _safe_get stores the checked IPs for a host in `pins` right before each
    request, and the transport dials those. Tests can pass a MockTransport;
    the pins still record what the guard checked. With a proxy set in the
    environment, httpx routes through the proxy transport instead, and the
    proxy resolves the host itself, so the pin does not apply on that path.
    """

    def __init__(self, **kwargs) -> None:
        self.pins: dict[str, list[str]] = {}
        kwargs.setdefault("transport", _PinnedTransport(self.pins))
        super().__init__(**kwargs)


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

    Pins the socket, at every hop, to the IPs _validate_url checked. This
    closes a DNS-rebinding gap: without the pin, the check resolves the host
    once, and the HTTP client resolves it again to open the socket. A
    rebinding host can answer the check with a public IP and the connect with
    a private one. `client` must be a _PinnedClient; its transport refuses
    any host that has no entry in `client.pins`.

    Streams the body and aborts as soon as it passes _MAX_RESPONSE_BYTES, at
    every redirect hop, so a huge or malicious page cannot buffer its full
    body in memory before the cap gets checked.
    """
    current = url
    for _ in range(max_redirects + 1):
        host = (urlparse(current).hostname or "").lower()
        client.pins[host] = _validate_url(current)
        with client.stream("GET", current, timeout=timeout, follow_redirects=False) as resp:
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

    def parse(self, path: Path | str, corpus: str) -> list[Document]:
        # Either the URL itself, as a str, or a Path to a file that holds one.
        # A URL must not arrive wrapped in Path(): that collapses "https://"
        # to "https:/" and the scheme check below then reads it as a file.
        url = str(path)
        if urlparse(url).scheme.lower() not in ("http", "https"):
            # Try reading URL from file
            try:
                url = path.read_text().strip()
            except Exception:  # noqa: BLE001
                log.warning("Failed to read URL from %s", path, exc_info=True)
                return []
            if urlparse(url).scheme.lower() not in ("http", "https"):
                return []

        return self._scrape(url, corpus)

    def _scrape(self, url: str, corpus: str) -> list[Document]:
        try:
            _validate_url(url)
        except SSRFBlocked as exc:
            log.error("Refusing to scrape %s: %s", url, exc)
            return []

        bs_class = _try_import_bs4()

        with _PinnedClient(
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
