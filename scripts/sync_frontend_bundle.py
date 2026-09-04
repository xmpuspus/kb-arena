#!/usr/bin/env python3
"""Copy the built frontend into the package and stamp it with its sources.

Run `npx next build` in `web/` first. This script then replaces
`kb_arena/static` with `web/out` and writes the source digest beside it, so
`test_the_packaged_bundle_matches_its_sources` can tell a fresh bundle from a
stale one.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb_arena.frontend_bundle import write_stamp  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    web, out, static = root / "web", root / "web" / "out", root / "kb_arena" / "static"
    if not out.is_dir():
        print("web/out is missing. Run `npx next build` in web/ first.", file=sys.stderr)
        return 1
    if static.exists():
        shutil.rmtree(static)
    shutil.copytree(out, static)
    stamp = write_stamp(web, static)
    print(f"bundle synced, {stamp['files']} source files, digest {stamp['digest'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
