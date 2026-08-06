"""Smoke tests for retriever-lab and label-chunks CLI commands."""

from __future__ import annotations

import re

import pytest
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


def test_benchmark_dry_run_does_not_preflight_credentials(monkeypatch):
    def unexpected_preflight(**kwargs) -> None:
        raise AssertionError(f"dry run preflighted credentials: {kwargs}")

    monkeypatch.setattr("kb_arena.cli._preflight", unexpected_preflight)

    result = runner.invoke(app, ["benchmark", "--corpus", "missing", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    assert "Dry run: benchmark" in _clean(result.stdout)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "1.1"])
@pytest.mark.parametrize(
    "command,option",
    [
        ("benchmark", "--fail-below"),
        ("retriever-lab", "--min-recall"),
    ],
)
def test_quality_floor_rejects_invalid_values_before_preflight(monkeypatch, command, option, value):
    def unexpected_preflight(**kwargs) -> None:
        raise AssertionError(f"invalid threshold preflighted credentials: {kwargs}")

    monkeypatch.setattr("kb_arena.cli._preflight", unexpected_preflight)

    result = runner.invoke(app, [command, option, value])

    assert result.exit_code == 1
    assert "finite number between 0 and 1" in _clean(result.stdout)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_eval_threshold_rejects_invalid_values(value):
    result = runner.invoke(app, ["eval", "--ci", "--threshold", f"accuracy={value}"])

    assert result.exit_code == 1
    assert "finite number between 0 and 1" in _clean(result.stdout)
