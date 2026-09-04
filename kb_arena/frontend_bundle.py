"""The digest that says whether the packaged frontend matches its sources.

`kb-arena serve` ships the pre-built pages under `kb_arena/static`. That
directory is a build output, so it goes stale the moment somebody edits a page
and forgets to rebuild. It happened twice: once for the demo page states, and
once for the scoped arena leaderboard. Both times the fix looked shipped and
was not.

The build records the digest of its sources next to the bundle. A test
recomputes the digest and fails when the two disagree, so a stale bundle stops
the run instead of reaching a user.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

STAMP_NAME = "build_source.json"

# Everything the pages are built from. `node_modules` and `out` are inputs
# nobody edits and outputs nobody reads, so both stay out of the digest.
SOURCE_DIRS = ("app", "components", "lib")
SOURCE_FILES = (
    "next.config.mjs",
    "package.json",
    "postcss.config.mjs",
    "tailwind.config.ts",
    "tsconfig.json",
)


def _source_paths(web: Path) -> list[Path]:
    paths: list[Path] = []
    for name in SOURCE_DIRS:
        root = web / name
        if root.is_dir():
            paths += [p for p in root.rglob("*") if p.is_file()]
    paths += [web / name for name in SOURCE_FILES if (web / name).is_file()]
    return sorted(paths)


def source_digest(web: Path) -> tuple[str, int]:
    """A digest over every frontend source file, with the file count."""
    h = hashlib.sha256()
    paths = _source_paths(web)
    for path in paths:
        h.update(str(path.relative_to(web)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest(), len(paths)


def write_stamp(web: Path, static: Path) -> dict:
    digest, count = source_digest(web)
    stamp = {"digest": digest, "files": count}
    (static / STAMP_NAME).write_text(json.dumps(stamp, indent=2) + "\n")
    return stamp


def read_stamp(static: Path) -> dict | None:
    path = static / STAMP_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
