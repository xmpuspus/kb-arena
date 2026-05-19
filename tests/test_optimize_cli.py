"""Smoke tests for the `kb-arena optimize` CLI command."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from kb_arena.cli import app

_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clean(text: str) -> str:
    return _ANSI.sub("", text)


runner = CliRunner()


def test_optimize_help():
    result = runner.invoke(app, ["optimize", "--help"])
    assert result.exit_code == 0
    out = _clean(result.stdout).lower()
    assert "--top-ks" in out
    assert "--chunk-sizes" in out
    assert "--embedding-providers" in out
    assert "--reranker-backends" in out
    assert "--method" in out
    assert "--dry-run" in out
    assert "--metric" in out


def test_optimize_dry_run_no_keys_needed(monkeypatch):
    # dry-run must not require API keys or hit any service.
    result = runner.invoke(
        app,
        [
            "optimize",
            "--corpus",
            "aws-compute",
            "--strategies",
            "naive_vector,bm25",
            "--top-ks",
            "3,5,10",
            "--chunk-sizes",
            "256,512",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    out = _clean(result.stdout).lower()
    assert "plan" in out
    assert "naive_vector" in out
    assert "bm25" in out
