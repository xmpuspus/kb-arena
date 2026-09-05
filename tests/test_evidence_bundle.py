"""A committed run says what it is, and refuses to say what it is not."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb_arena.benchmark.evidence import BUNDLE_VERSION, build_bundle, check_bundle

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "results" / "run_59b5b60d"


def _bundle(**overrides) -> dict:
    base = build_bundle(
        command=["kb-arena", "retriever-lab", "--corpus", "c"],
        result_paths=[Path("results/run_x/retriever_lab.json")],
        review={"publishable": False, "note": "nobody checked these"},
        corpus="c",
        seed=0,
    )
    return {**base, **overrides}


def test_a_bundle_records_what_a_reader_needs_to_repeat_the_run():
    bundle = _bundle()

    assert bundle["command"], "a run nobody can repeat is not evidence"
    assert bundle["environment"]["kb_arena"]
    assert bundle["environment"]["python"]["version"]
    assert bundle["environment"]["platform"]
    assert bundle["seed"] == 0
    assert bundle["created_at"]


def test_a_bundle_refuses_to_call_itself_citable_when_the_review_does():
    assert _bundle()["citable"] is False
    reviewed = _bundle(review={"publishable": True, "note": ""})
    assert (
        build_bundle(
            command=["x"],
            result_paths=[Path("y")],
            review={"publishable": True},
            corpus="c",
            seed=1,
        )["citable"]
        is True
    )
    assert reviewed["review"]["publishable"] is True


def test_a_bundle_that_is_not_citable_has_to_say_why():
    """Silence here reads as an oversight rather than a verdict."""
    problems = check_bundle(_bundle(why_not_citable=""), ROOT)

    assert any("does not say why" in p for p in problems)


def test_a_bundle_cannot_claim_more_than_its_own_review_allows():
    problems = check_bundle(_bundle(citable=True), ROOT)

    assert any("calls itself citable while its own review verdict refuses" in p for p in problems)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("command", [], "nobody can repeat"),
        ("results", [], "describes nothing"),
        ("bundle_version", 99, "bundle_version"),
    ],
)
def test_an_incomplete_bundle_names_what_is_missing(key, value, expected):
    problems = check_bundle(_bundle(**{key: value}), ROOT)

    assert any(expected in p for p in problems), problems


def test_a_bundle_notices_a_result_file_that_is_not_there():
    problems = check_bundle(_bundle(results=["results/run_x/gone.json"]), ROOT)

    assert any("missing result file" in p for p in problems)


def test_the_committed_run_is_complete():
    """The run in this repository has to pass the check it ships."""
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert check_bundle(bundle, ROOT) == []
    assert bundle["bundle_version"] == BUNDLE_VERSION


def test_the_committed_run_does_not_claim_to_be_citable():
    """All 75 questions carry no review status, and the bundle says so."""
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert bundle["citable"] is False
    assert bundle["why_not_citable"]
    assert bundle["review"]["counts"]["human-reviewed"] == 0


def test_the_committed_run_carries_a_readme_that_states_both_halves():
    """A number with no provenance beside it invites a citation it cannot support."""
    readme = " ".join((COMMITTED / "README.md").read_text().split())

    assert "Repeat it" in readme
    assert "What this run does NOT show" in readme
    assert "citable: false" in readme
    assert "Only a human can do that" in readme


def test_the_check_command_the_readme_documents_runs_as_written():
    """`--corpus` and `--run-id` were required options, so `--check` exited 2.

    The README told a reader to run `kb-arena evidence --check <path>`, and the
    command answered "Missing option '--corpus'". That is a document naming a
    command that does not run.
    """
    import subprocess
    import sys

    readme = (ROOT / "README.md").read_text()
    assert "kb-arena evidence --check <path>" in readme, "the README documents this command"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kb_arena.cli",
            "evidence",
            "--check",
            str(COMMITTED / "evidence.json"),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "complete" in result.stdout


def test_writing_a_bundle_still_asks_for_the_corpus_and_the_run():
    """The two options stay needed for a write, and the command says which is missing."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "kb_arena.cli", "evidence", "--corpus", "aws-compute"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    assert result.returncode == 2
    assert "--run-id" in result.stdout
