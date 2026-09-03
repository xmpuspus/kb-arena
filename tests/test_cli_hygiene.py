"""CLI hygiene: valid scaffold example, format allow-list, quiet telemetry, log level, env typos."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from kb_arena.cli import app
from kb_arena.models.benchmark import Question
from kb_arena.settings import settings

runner = CliRunner()


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    path = tmp_path / "datasets"
    path.mkdir()
    monkeypatch.setattr(settings, "datasets_path", str(path))
    return path


def test_init_corpus_example_question_is_schema_valid(datasets_dir):
    result = runner.invoke(app, ["init-corpus", "my-docs"])
    assert result.exit_code == 0, result.output

    example = datasets_dir / "my-docs" / "questions" / "tier1_factoid.yaml.example"
    entries = yaml.safe_load(example.read_text())
    assert isinstance(entries, list) and entries
    for entry in entries:
        question = Question.model_validate(entry)
        assert question.split == "development"
        assert question.ground_truth.answer


def test_init_corpus_example_loads_through_the_question_loader(datasets_dir):
    from kb_arena.benchmark.questions import load_questions

    runner.invoke(app, ["init-corpus", "my-docs"])
    questions_dir = datasets_dir / "my-docs" / "questions"
    example = questions_dir / "tier1_factoid.yaml.example"
    example.rename(questions_dir / "tier1_factoid.yaml")

    loaded = load_questions("my-docs")
    assert len(loaded) == 1
    assert loaded[0].tier == 1


def test_report_rejects_an_unknown_format():
    result = runner.invoke(app, ["report", "--corpus", "aws-compute", "--format", "docx"])
    assert result.exit_code == 1
    assert "docx" in result.output
    assert "markdown" in result.output


def test_report_markdown_writes_a_markdown_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(Path(__file__).parents[1] / "results"))
    out = tmp_path / "report.md"
    result = runner.invoke(
        app,
        ["report", "--corpus", "aws-compute", "--format", "markdown", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert text.startswith("#")
    assert "bm25" in text


def test_cli_silences_chromadb_telemetry_noise(datasets_dir, monkeypatch):
    monkeypatch.delenv("ANONYMIZED_TELEMETRY", raising=False)
    runner.invoke(app, ["init-corpus", "quiet"])
    assert logging.getLogger("chromadb.telemetry").level == logging.CRITICAL
    assert logging.getLogger("chromadb.telemetry.product.posthog").level == logging.CRITICAL
    assert os.environ.get("ANONYMIZED_TELEMETRY") == "False"


def test_cli_applies_the_configured_log_level(datasets_dir, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "ERROR")
    runner.invoke(app, ["init-corpus", "levels"])
    assert logging.getLogger().level == logging.ERROR


def test_verbose_flag_overrides_the_log_level(datasets_dir, monkeypatch):
    monkeypatch.setattr(settings, "log_level", "ERROR")
    runner.invoke(app, ["--verbose", "init-corpus", "verbose"])
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.parametrize("value", ["verbose", "trace", ""])
def test_settings_reject_an_unknown_log_level(monkeypatch, value):
    from pydantic import ValidationError

    from kb_arena.settings import Settings

    monkeypatch.setenv("KB_ARENA_LOG_LEVEL", value)
    with pytest.raises(ValidationError, match="log level"):
        Settings(_env_file=None)


def test_settings_normalise_the_log_level(monkeypatch):
    from kb_arena.settings import Settings

    monkeypatch.setenv("KB_ARENA_LOG_LEVEL", "debug")
    assert Settings(_env_file=None).log_level == "DEBUG"


def test_unknown_env_keys_are_reported():
    from kb_arena.logging_config import unknown_env_keys

    env = {
        "KB_ARENA_NEO4J_PASSWROD": "x",
        "KB_ARENA_NEO4J_PASSWORD": "y",
        "KB_ARENA_LOG_LEVEL": "INFO",
        "PATH": "/bin",
    }
    assert unknown_env_keys(env) == ["KB_ARENA_NEO4J_PASSWROD"]


def test_cli_warns_about_a_misspelled_env_var(datasets_dir, monkeypatch, caplog):
    monkeypatch.setenv("KB_ARENA_NEO4J_PASSWROD", "oops")
    with caplog.at_level(logging.WARNING, logger="kb_arena"):
        runner.invoke(app, ["init-corpus", "typo"])
    assert "KB_ARENA_NEO4J_PASSWROD" in caplog.text
