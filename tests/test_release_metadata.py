"""Release metadata must identify the same published version."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET_VERSION = "0.10.0"
TARGET_DATE = "2026-08-05"


def test_release_metadata_is_aligned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    changelog = (ROOT / "CHANGELOG.md").read_text()

    assert project["project"]["version"] == TARGET_VERSION
    assert citation["version"] == TARGET_VERSION
    assert str(citation["date-released"]) == TARGET_DATE
    assert f"## [{TARGET_VERSION}] - {TARGET_DATE}" in changelog


def test_readme_links_resolve_from_package_indexes() -> None:
    readme = (ROOT / "README.md").read_text()
    relative_targets = []

    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", readme):
        target = match.group(1).strip()
        if not target.startswith(("https://", "http://", "mailto:", "#")):
            relative_targets.append(target)

    assert relative_targets == []


def test_sdist_excludes_local_and_generated_work() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    excluded = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert {"tmp/", ".venv/", "results/run_*/", "uv.lock"} <= excluded
