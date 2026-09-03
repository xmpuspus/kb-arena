"""The GitHub parser reads the same owner/repo from every spelling of the spec."""

from __future__ import annotations

from pathlib import Path

import pytest

from kb_arena.ingest.parsers.github import GitHubParser


@pytest.mark.parametrize(
    "spec",
    [
        "github:owner/repo",
        Path("github:owner/repo"),
        # Path() collapses the double slash to "github:/owner/repo".
        Path("github://owner/repo"),
        "github://owner/repo",
    ],
)
def test_every_spec_spelling_yields_owner_slash_repo(monkeypatch, spec):
    seen: list[str] = []
    monkeypatch.setattr(
        GitHubParser, "_parse_remote", lambda self, repo_spec, corpus: seen.append(repo_spec) or []
    )

    GitHubParser().parse(spec, "c")

    assert seen == ["owner/repo"]
