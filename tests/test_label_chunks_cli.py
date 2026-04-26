"""Smoke tests for retriever-lab and label-chunks CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from kb_arena.cli import app

runner = CliRunner()


def test_retriever_lab_help():
    result = runner.invoke(app, ["retriever-lab", "--help"])
    assert result.exit_code == 0
    assert "retrieval-only" in result.stdout.lower()
    assert "--top-k" in result.stdout


def test_retriever_lab_min_recall_flag():
    result = runner.invoke(app, ["retriever-lab", "--help"])
    assert "--min-recall" in result.stdout


def test_label_chunks_help():
    result = runner.invoke(app, ["label-chunks", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.stdout
    assert "expected_chunks.yaml" in result.stdout.lower() or "label" in result.stdout.lower()


def test_benchmark_top_k_flag_present():
    result = runner.invoke(app, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--top-k" in result.stdout
