"""Writes that leave a whole file or no file, and checkpoints a crash cannot tear."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file next to path, fsync it, then rename it over path.

    A reader never sees a half-written file, and a crash mid-write leaves
    the old file in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        # mkstemp opens 0600. Keep the mode the old file had, or the mode a
        # plain open would give, so a result file stays readable by the group
        # and the web container that serves it.
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o666 & ~_umask()
        os.chmod(temporary, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable. Best effort: some filesystems refuse it."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line and fsync it, so a crash loses at most the line in flight."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    """Every whole line in order. A torn line from a crash is dropped, not raised."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows
