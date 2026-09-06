"""`kb-arena mcp` is the entry a registry client starts, so the CLI has to carry it.

A registry client runs the PyPI package through `uvx`, which starts the
`kb-arena` console script. Without this command, that client reaches the
ordinary CLI and never reaches the MCP server.
"""

from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

from kb_arena.cli import app

runner = CliRunner()


def test_the_mcp_command_names_the_extra_when_the_package_is_missing(monkeypatch):
    """The missing extra reads as an instruction, not as a traceback."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "mcp" else real_find_spec(name, *args, **kwargs)
        ),
    )

    result = runner.invoke(app, ["mcp"])

    assert result.exit_code == 1
    # Rich eats an unescaped [mcp], so this asserts the extra survives the render.
    assert "pip install 'kb-arena[mcp]'" in result.output


def test_the_mcp_command_starts_the_stdio_server(monkeypatch):
    """The command runs the server's own entry function, not a copy of it."""
    pytest.importorskip("mcp")
    from kb_arena.mcp import server as mcp_server

    started = []
    monkeypatch.setattr(mcp_server, "main", lambda: started.append(True))

    result = runner.invoke(app, ["mcp"])

    assert result.exit_code == 0, result.output
    assert started == [True]
