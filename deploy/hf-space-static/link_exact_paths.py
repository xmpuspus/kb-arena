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


def rewrite(text: str) -> tuple[str, int]:
    count = 0

    def swap(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group(1)}{match.group(2)}index.html{match.group(3)}"

    return ROUTE_LINK.sub(swap, text), count


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
    if root.name == "static" and root.parent.name == "kb_arena":
        print(
            "Refusing to rewrite the packaged bundle. Run this on the Space copy.", file=sys.stderr
        )
        return 1

    files = 0
    links = 0
    for page in sorted(root.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        changed, count = rewrite(text)
        if count:
            page.write_text(changed, encoding="utf-8")
            files += 1
            links += count
    print(f"exact paths written: {links} links across {files} files")

    left = remaining(root, routes(root))
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
