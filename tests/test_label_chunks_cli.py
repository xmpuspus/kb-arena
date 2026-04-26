"""Smoke tests for retriever-lab and label-chunks CLI commands."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from kb_arena.cli import app

# CI runs Rich-styled help output through ANSI escape codes that break naive
# substring searches. Strip them before asserting.
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clean(text: str) -> str:
    return _ANSI.sub("", text)


runner = CliRunner()


def test_retriever_lab_help():
    result = runner.invoke(app, ["retriever-lab", "--help"])
    assert result.exit_code == 0
    out = _clean(result.stdout).lower()
    assert "retrieval-only" in out
    assert "--top-k" in out


def test_retriever_lab_min_recall_flag():
    result = runner.invoke(app, ["retriever-lab", "--help"])
    assert "--min-recall" in _clean(result.stdout)


def test_label_chunks_help():
    result = runner.invoke(app, ["label-chunks", "--help"])
    assert result.exit_code == 0
    out = _clean(result.stdout)
    assert "--force" in out
    assert "expected_chunks.yaml" in out.lower() or "label" in out.lower()


def test_benchmark_top_k_flag_present():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--top-k" in _clean(result.stdout)
