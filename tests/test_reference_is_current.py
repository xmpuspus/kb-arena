"""The reference docs are generated, so a stale one cannot merge.

This session added three commands and four flags that no document mentioned.
A hand-written list goes stale the first time somebody adds a command and
forgets the doc, and nobody notices until a user cannot find the feature.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_reference.py"


def _generator():
    """Load the generator to read its helpers.

    Never render through this. Another test can register a command on the
    shared Typer app, and the render would then describe a command that exists
    only inside that test session.
    """
    spec = importlib.util.spec_from_file_location("generate_reference", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _render_in_a_clean_process(target: str) -> str:
    """What the generator writes on its own, which is what a user runs."""
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('g', {str(GENERATOR)!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        f"sys.stdout.write(m.TARGETS[{target!r}]())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    return result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "docs/reference-cli.md",
        "docs/reference-http.md",
        "docs/reference-environment.md",
        "docs/strategy-catalog.md",
    ],
)
def test_the_reference_on_disk_matches_the_code(name):
    """Run the generator in memory and compare. A difference means it is stale."""
    expected = _render_in_a_clean_process(name)
    actual = (ROOT / name).read_text()

    assert (
        actual == expected
    ), f"{name} is out of date. Run `python3 scripts/generate_reference.py`."


def test_every_command_appears_in_the_command_reference():
    """The check that would have caught variance, evidence and datasets."""
    module = _generator()
    reference = (ROOT / "docs" / "reference-cli.md").read_text()

    for name, _summary, _flags in module._cli_commands():
        assert f"`kb-arena {name}`" in reference, f"{name} is missing from the reference"


def test_every_long_flag_appears_in_the_command_reference():
    module = _generator()
    reference = (ROOT / "docs" / "reference-cli.md").read_text()

    for _name, _summary, flags in module._cli_commands():
        for flag, _help in flags:
            assert f"`{flag}`" in reference, f"{flag} is missing from the reference"


def test_every_route_appears_in_the_http_reference():
    module = _generator()
    reference = (ROOT / "docs" / "reference-http.md").read_text()

    for path, _method, _gate in module._api_routes():
        assert f"`{path}`" in reference, f"{path} is missing from the reference"


def test_the_reference_lists_every_route_the_app_itself_publishes():
    """The check above reads the generator, so it cannot catch what the generator misses.

    It did miss four. The generator read `@app` decorators out of `api.py` with
    a regex, and `api.py` mounts a router holding `/api/tools/generate`,
    `/api/tools/audit`, `/api/tools/fix` and `/api/tools/qa-pairs`. The
    reference said it listed every route and left all four out.

    The app's own OpenAPI document is the independent list, so this test fails
    when a route exists and the reference does not name it.
    """
    from kb_arena.chatbot.api import app

    reference = (ROOT / "docs" / "reference-http.md").read_text()
    missing = [p for p in app.openapi()["paths"] if f"`{p}`" not in reference]

    assert not missing, f"the app serves {missing}, and the reference names none of them"


def test_the_http_reference_states_the_gate_each_route_carries():
    """A reader planning a deployment needs to know which routes need a token."""
    module = _generator()
    routes = dict((p, g) for p, _m, g in module._api_routes())

    # The four content routes are the ones this project gates.
    for path in ("/api/benchmark/results", "/api/graph/data", "/api/compare"):
        assert routes[path].startswith("token"), f"{path} should carry the content gate"
    # And the ones a demo must serve stay open.
    assert routes["/health"] == "open"


def test_every_setting_appears_in_the_environment_reference():
    from kb_arena.settings import Settings

    reference = (ROOT / "docs" / "reference-environment.md").read_text()

    for name in Settings.model_fields:
        assert f"`KB_ARENA_{name.upper()}`" in reference, f"{name} is missing"


def test_no_secret_default_is_printed():
    """A reference that prints a token is a reference that leaks one."""
    reference = (ROOT / "docs" / "reference-environment.md").read_text()
    rows = [r for r in reference.splitlines() if r.startswith("| `KB_ARENA_")]

    for row in rows:
        name, default = row.split("|")[1].strip(), row.split("|")[2].strip()
        if any(word in name.lower() for word in ("key", "token", "password")):
            assert default in {"`(empty)`", "`(set)`"}, f"{name} prints its value"


@pytest.mark.parametrize(
    "name",
    [
        "docs/reference-cli.md",
        "docs/reference-http.md",
        "docs/reference-environment.md",
        "docs/strategy-catalog.md",
    ],
)
def test_each_reference_says_it_is_generated(name):
    """A reader who edits one by hand loses the edit on the next run."""
    text = (ROOT / name).read_text()

    assert text.startswith("<!-- Generated by scripts/generate_reference.py")


def test_every_strategy_appears_in_the_strategy_catalog_reference():
    """Only 3 of 19 catalog names appeared in the hand-written doc before this.

    Generation closes that gap. This test guards it: a strategy added to the
    catalog with no matching heading here fails the build.
    """
    from kb_arena.strategies.catalog import STRATEGY_CATALOG

    reference = (ROOT / "docs" / "strategy-catalog.md").read_text()

    for spec in STRATEGY_CATALOG:
        assert f"## {spec.label}" in reference, f"{spec.name} is missing from the strategy catalog"


def test_the_readme_links_every_reference():
    readme = (ROOT / "README.md").read_text()

    for name in ("reference-cli", "reference-http", "reference-environment"):
        assert name in readme, f"the README does not link docs/{name}.md"
