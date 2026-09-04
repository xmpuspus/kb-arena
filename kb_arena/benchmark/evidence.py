"""A run bundle a reader can check without trusting the person who made it.

A benchmark result on its own says a number. A bundle says which corpus, which
questions, which code, which seed, which command, and what the result may not be
used for. The last one matters most: a run against labels no human checked is a
development signal, and a bundle that does not say so invites a citation it
cannot support.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from kb_arena.benchmark.atomic import atomic_write_text

BUNDLE_VERSION = 1


def _python_identity() -> dict:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _package_version() -> str:
    from kb_arena import __version__

    return __version__


def _git_sha() -> str | None:
    from kb_arena.benchmark.manifest import git_sha

    return git_sha()


def build_bundle(
    *,
    command: list[str],
    result_paths: list[Path],
    review: dict,
    corpus: str,
    seed: int,
    notes: str = "",
) -> dict:
    """The record that travels with a committed run.

    `command` is what a reader types to repeat this. `review` is the verdict
    from `review_summary`, and it decides whether the bundle calls itself
    citable evidence or a development signal.
    """
    return {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "command": command,
        "results": sorted(str(p) for p in result_paths),
        "environment": {
            "kb_arena": _package_version(),
            "git_sha": _git_sha(),
            "python": _python_identity(),
            "platform": platform.platform(),
        },
        "seed": seed,
        "review": review,
        # The claim the bundle makes about itself, stated rather than implied.
        "citable": bool(review.get("publishable")),
        "why_not_citable": "" if review.get("publishable") else review.get("note", ""),
        "notes": notes,
    }


def write_bundle(directory: Path, bundle: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evidence.json"
    atomic_write_text(path, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return path


def repeat_command(bundle: dict) -> str:
    """The exact command line, so a reader repeats the run rather than guesses."""
    return " ".join(bundle.get("command") or [])


def check_bundle(bundle: dict, root: Path) -> list[str]:
    """What is missing or untrue in a bundle, as a list a caller can print."""
    problems: list[str] = []
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        problems.append(f"bundle_version is {bundle.get('bundle_version')!r}")
    if not bundle.get("command"):
        problems.append("no command, so nobody can repeat the run")
    for name in bundle.get("results") or []:
        if not (root / name).exists():
            problems.append(f"missing result file {name}")
    if not bundle.get("results"):
        problems.append("no result files, so the bundle describes nothing")
    review = bundle.get("review") or {}
    if bundle.get("citable") and not review.get("publishable"):
        problems.append("calls itself citable while its own review verdict refuses")
    if not bundle.get("citable") and not bundle.get("why_not_citable"):
        problems.append("is not citable and does not say why")
    env = bundle.get("environment") or {}
    for field in ("kb_arena", "python", "platform"):
        if not env.get(field):
            problems.append(f"environment records no {field}")
    return problems


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    """Check a bundle from the command line, for CI or a reviewer."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m kb_arena.benchmark.evidence <evidence.json>")
        return 2
    path = Path(args[0])
    bundle = json.loads(path.read_text())
    problems = check_bundle(bundle, path.parent.parent.parent)
    for problem in problems:
        print(f"{path}: {problem}")
    if not problems:
        print(f"{path}: complete. citable={bundle['citable']}")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
