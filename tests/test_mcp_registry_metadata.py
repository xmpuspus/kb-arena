"""The registry entry agrees with the repository it describes, offline.

Every check here reads this repository: `server.json`, `pyproject.toml`,
`README.md`, and the Typer app itself. Together they catch
the publish-time failures this repository can see on its own: a version that
disagrees with the package, a description the registry answers 422 for, a
package argument that names no command, and a missing ownership marker.

No check here reaches PyPI. A unit test makes no network call, so nothing
below proves PyPI carries the named version or the marker. `mcp-publisher
publish` still fails when PyPI does not hold them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.main import get_command

from kb_arena.cli import app

ROOT = Path(__file__).resolve().parents[1]

# The registry reads the PyPI description, and PyPI takes that from the long
# description file. A marker in any other file never reaches the registry.
MARKER = re.compile(r"^<!-- mcp-name: (\S+) -->$", re.M)


def _package_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    assert match, "pyproject.toml must declare a version"
    return match.group(1)


def _entry() -> dict:
    return json.loads((ROOT / "server.json").read_text())


def test_the_registry_entry_names_the_released_version():
    entry = _entry()
    version = _package_version()

    assert entry["version"] == version
    assert [p["version"] for p in entry["packages"]] == [version]


def test_the_registry_entry_points_at_this_repository():
    entry = _entry()

    assert entry["name"] == "io.github.xmpuspus/kb-arena"
    assert entry["repository"]["url"] == "https://github.com/xmpuspus/kb-arena"
    # The registry caps the description at 100 characters and answers 422 above it.
    assert len(entry["description"]) <= 100


def test_the_registry_entry_speaks_stdio_from_pypi():
    entry = _entry()
    package = entry["packages"][0]

    assert package["registryType"] == "pypi"
    assert package["identifier"] == "kb-arena"
    assert package["transport"]["type"] == "stdio"


def test_the_ownership_marker_sits_in_the_package_long_description():
    """The registry proves ownership of the PyPI package by this marker.

    It reads the PyPI description, which pyproject takes from `README.md`.
    A marker in another file leaves the publish unproven.
    """
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert re.search(
        r'^readme = "README.md"', pyproject, re.M
    ), "the long description must stay README.md, because the marker below lives there"

    match = MARKER.search((ROOT / "README.md").read_text())
    assert match, "README.md must carry the <!-- mcp-name: ... --> marker on its own line"
    assert match.group(1) == _entry()["name"]


def test_every_package_argument_names_a_command_the_cli_has():
    """`uvx` starts the `kb-arena` console script, so the argument has to be a command.

    An empty or wrong argument list leaves the client at the ordinary CLI,
    which never speaks the protocol.
    """
    commands = set(get_command(app).commands)
    arguments = _entry()["packages"][0]["packageArguments"]
    positional = [a["value"] for a in arguments if a["type"] == "positional"]

    assert positional, "the entry must name the command that starts the server"
    for value in positional:
        assert value in commands, f"server.json names {value!r}, and the CLI has no such command"


def test_the_runtime_arguments_install_the_optional_extra():
    """A bare `uvx kb-arena` installs no extras, and the server needs one.

    Without this entry a registry client ran `uvx kb-arena mcp`, which printed
    the install line for the `mcp` extra and exited 1. The client never reached
    the server, and the entry looked correct because the schema accepted it.
    """
    entry = json.loads((ROOT / "server.json").read_text())
    package = entry["packages"][0]

    runtime = package.get("runtimeArguments") or []
    passed = {(a.get("name"), a.get("value")) for a in runtime}

    assert (
        "--from",
        "kb-arena[mcp]",
    ) in passed, "uvx installs no extras, so the registry entry must pass --from kb-arena[mcp]"
