"""The holdout split is sealed: every read is written down, and the optimizer never tunes on it.

A holdout question set only proves something when nobody has fitted to it.
Every tool that scores the holdout split appends one line here, so a reader
of a published number can count how many times the split was opened.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

HOLDOUT_SPLIT = "holdout"
LEDGER_NAME = "holdout_uses.jsonl"


def touches_holdout(questions) -> bool:
    """True when any selected question belongs to the holdout split.

    A run on "all" or on the unfiltered default reads the holdout questions
    too, so the split name alone never decides whether the seal was opened.
    """
    return any(getattr(q, "split", "unspecified") == HOLDOUT_SPLIT for q in questions)


def ledger_path(results_dir: Path | str) -> Path:
    return Path(results_dir) / LEDGER_NAME


def record_holdout_use(
    results_dir: Path | str, *, tool: str, corpus: str, run_id: str, strategies: list[str]
) -> dict:
    """Append one line for a run that scored the holdout split, and return it."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "tool": tool,
        "corpus": corpus,
        "run_id": run_id,
        "strategies": sorted(strategies),
    }
    path = ledger_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
    return entry


def read_ledger(results_dir: Path | str) -> tuple[list[dict], int]:
    """Every recorded use, oldest first, and the count of lines that did not parse.

    A line that did not parse is a use the count would otherwise hide, so
    the reader gets the number and can say the count is a floor.
    """
    path = ledger_path(results_dir)
    if not path.exists():
        return [], 0
    uses: list[dict] = []
    corrupt = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            if isinstance(entry, dict):
                uses.append(entry)
            else:
                corrupt += 1
    return uses, corrupt


def holdout_uses(results_dir: Path | str, corpus: str | None = None) -> list[dict]:
    """Every recorded use, oldest first. read_ledger reports the lines that did not parse."""
    uses, _ = read_ledger(results_dir)
    return [u for u in uses if corpus is None or u.get("corpus") == corpus]
