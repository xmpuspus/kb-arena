"""The registry entry names a version PyPI holds, or the publish fails there.

`mcp-publisher publish` resolves the package from PyPI. A `server.json` that
names a version PyPI does not carry fails at the registry, after the release is
already out. This check moves that failure into the suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _package_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    assert match, "pyproject.toml must declare a version"
    return match.group(1)


def test_the_registry_entry_names_the_released_version():
    entry = json.loads((ROOT / "server.json").read_text())
    version = _package_version()

    assert entry["version"] == version
    assert [p["version"] for p in entry["packages"]] == [version]


def test_the_registry_entry_points_at_this_repository():
    entry = json.loads((ROOT / "server.json").read_text())

    assert entry["name"] == "io.github.xmpuspus/kb-arena"
    assert entry["repository"]["url"] == "https://github.com/xmpuspus/kb-arena"
    # The registry caps the description at 100 characters and answers 422 above it.
    assert len(entry["description"]) <= 100


def test_the_registry_entry_speaks_stdio_from_pypi():
    entry = json.loads((ROOT / "server.json").read_text())
    package = entry["packages"][0]

    assert package["registryType"] == "pypi"
    assert package["identifier"] == "kb-arena"
    assert package["transport"]["type"] == "stdio"
