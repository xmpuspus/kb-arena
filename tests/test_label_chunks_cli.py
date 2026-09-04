"""Smoke tests for retriever-lab and label-chunks CLI commands."""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.mark.asyncio
async def test_label_candidates_are_scoped_to_selected_corpus():
    from kb_arena.benchmark.expected_chunks import label_one_question

    empty_result = MagicMock()
    empty_result.retrieval.retrieved = []
    empty_result.cost_usd = 0.0
    bm25 = MagicMock()
    bm25.query = AsyncMock(return_value=empty_result)
    extra = MagicMock()
    extra.name = "naive_vector"
    extra.query = AsyncMock(return_value=empty_result)

    ids, cost = await label_one_question(
        "question", bm25, AsyncMock(), "alpha", extra_retrievers=[extra]
    )

    assert ids == []
    assert cost == 0.0
    bm25.query.assert_awaited_once_with("question", top_k=20, corpus="alpha")
    extra.query.assert_awaited_once_with("question", top_k=20, corpus="alpha")


@pytest.mark.asyncio
async def test_label_candidates_fail_when_an_enabled_retriever_fails():
    from kb_arena.benchmark.expected_chunks import label_one_question

    empty_result = MagicMock()
    empty_result.retrieval.retrieved = []
    empty_result.cost_usd = 0.0
    bm25 = MagicMock()
    bm25.query = AsyncMock(return_value=empty_result)
    extra = MagicMock()
    extra.name = "naive_vector"
    extra.query = AsyncMock(side_effect=ConnectionError("index offline"))
    llm = AsyncMock()

    with pytest.raises(RuntimeError, match="naive_vector") as caught:
        await label_one_question("question", bm25, llm, "alpha", extra_retrievers=[extra])

    assert isinstance(caught.value.__cause__, ConnectionError)
    llm.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_label_corpus_checkpoints_success_before_later_failure(tmp_path, monkeypatch):
    import yaml

    from kb_arena.benchmark import expected_chunks
    from kb_arena.settings import settings

    questions = [
        SimpleNamespace(id="q1", question="first"),
        SimpleNamespace(id="q2", question="second"),
    ]
    bm25 = MagicMock()
    bm25._ensure_index.return_value = True
    monkeypatch.setattr(expected_chunks, "load_questions", lambda corpus: questions)
    monkeypatch.setattr(expected_chunks, "BM25Strategy", lambda: bm25)
    monkeypatch.setattr(expected_chunks, "LLMClient", MagicMock)
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    monkeypatch.setattr(
        expected_chunks,
        "label_one_question",
        AsyncMock(side_effect=[({"doc::section::0": 2}, 0.1), ConnectionError("judge offline")]),
    )

    with pytest.raises(ConnectionError, match="judge offline"):
        await expected_chunks.label_corpus("alpha")

    saved = yaml.safe_load((tmp_path / "alpha" / "questions" / "expected_chunks.yaml").read_text())
    assert saved["version"] == 2
    assert saved["labels"] == {"q1": {"doc::section::0": 2}}


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
