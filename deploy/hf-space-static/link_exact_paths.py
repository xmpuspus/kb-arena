#!/usr/bin/env python3
"""Point every in-page link at a file the static Space actually serves.

A Hugging Face static Space serves exact file paths. It does not resolve a
directory path to the `index.html` inside it, and it answers a directory path
with a redirect that leaves the Space. So `href="/benchmark/"` sent a reader to
huggingface.co/benchmark, while `/benchmark/index.html` was there the whole time
and answered 200.

The export cannot emit these links, because the packaged bundle behind the API
is served by StaticFiles, which does resolve a directory. So the rewrite runs on
the Space copy only. Run it against the copied tree, never against
`kb_arena/static`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# A route link the export writes: an absolute path with a trailing slash and no
# file name. The root stays as it is, because the Space serves it.
ROUTE_LINK = re.compile(r'(href=")(/[A-Za-z0-9][A-Za-z0-9._~-]*(?:/[A-Za-z0-9._~-]+)*/)(")')


def rewrite(text: str, known: set[str] | None = None) -> tuple[str, int, list[str]]:
    """Rewrite a route link, and name a directory link with no page behind it.

    Rewriting every directory-shaped link turned `/missing/` into
    `/missing/index.html`, which is a 404 wearing the right shape. With the
    routes the build wrote, an unknown one is reported instead.
    """
    count = 0
    unknown: list[str] = []

    def swap(match: re.Match[str]) -> str:
        nonlocal count
        route = match.group(2)
        if known is not None and route not in known:
            unknown.append(route)
            return match.group(0)
        count += 1
        return f"{match.group(1)}{route}index.html{match.group(3)}"

    return ROUTE_LINK.sub(swap, text), count, unknown


def rewrite_routes(text: str, known: set[str]) -> tuple[str, int]:
    """Point every quoted route string at the file the Space serves.

    A chunk holds the navigation list, so the anchor React draws after hydration
    comes from here and not from the HTML the deploy rewrote.
    """
    count = 0
    for route in sorted(known, key=len, reverse=True):
        for quote in ('"', "'"):
            needle = f"{quote}{route}{quote}"
            found = text.count(needle)
            if found:
                text = text.replace(needle, f"{quote}{route}index.html{quote}")
                count += found
    return text, count


def routes(root: Path) -> list[str]:
    """Every path the export writes as a directory holding an index.html."""
    found = []
    for page in sorted(root.rglob("index.html")):
        rel = page.parent.relative_to(root).as_posix()
        if rel != ".":
            found.append(f"/{rel}/")
    return found


def remaining(root: Path, wanted: list[str]) -> list[str]:
    """Route paths still written as a directory, in any file the Space serves.

    The rewrite reaches the anchors. A reviewer asked whether a route also sits
    in the hydration payload, where the client router would read it after the
    page loads. This checks rather than reasons: every file the browser can
    fetch is read, and a directory route left anywhere fails the deploy.
    """
    left = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".html", ".js", ".txt", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for route in wanted:
            if f'"{route}"' in text or f"'{route}'" in text:
                left.append(f"{path.relative_to(root)} holds {route}")
    return left


def main(root: Path) -> int:
    if not (root / "index.html").is_file():
        print(f"No dashboard at {root}.", file=sys.stderr)
        return 1
    # Resolve first. `kb_arena/static/../static` reaches the same directory, and
    # an unresolved comparison read its parent as `..` and let the guard pass.
    root = root.resolve()
    if root.name == "static" and root.parent.name == "kb_arena":
        print(
            "Refusing to rewrite the packaged bundle. Run this on the Space copy.", file=sys.stderr
        )
        return 1

    known = set(routes(root))
    files = 0
    links = 0
    unknown: list[str] = []
    for page in sorted(root.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        changed, count, missing = rewrite(text, known)
        unknown.extend(f"{page.relative_to(root)} links to {route}" for route in missing)
        if count:
            page.write_text(changed, encoding="utf-8")
            files += 1
            links += count

    # The navigation list is a data array, so the bundler writes each route into
    # a chunk as well. React re-renders the anchor from that copy when the page
    # hydrates, which would put the directory path back over the rewrite. The
    # check below caught exactly that, so the chunks are rewritten too.
    for chunk in sorted(root.rglob("*.js")):
        text = chunk.read_text(encoding="utf-8", errors="replace")
        changed, count = rewrite_routes(text, known)
        if count:
            chunk.write_text(changed, encoding="utf-8")
            files += 1
            links += count

    print(f"exact paths written: {links} links across {files} files")

    if unknown:
        print("A link names a directory this build never wrote:", file=sys.stderr)
        for line in unknown[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1

    left = remaining(root, sorted(known))
    if left:
        print(
            "A route is still written as a directory, which this Space redirects:", file=sys.stderr
        )
        for line in left[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
