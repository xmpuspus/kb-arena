"""A committed run says what it is, and refuses to say what it is not."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb_arena.benchmark.evidence import BUNDLE_VERSION, build_bundle, check_bundle

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "results" / "run_b84eba57"


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


def test_the_committed_run_claims_exactly_what_its_own_review_supports():
    """The bundle must never claim more than its review verdict allows.

    The earlier version of this test pinned `citable is False`, which was the
    truth while the corpus carried no review status. Pinning the value instead
    of the rule made the test fail for a correct change. The rule is that the
    claim and the verdict agree, whichever way they point.
    """
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert bundle["citable"] == bundle["review"]["publishable"]
    if bundle["citable"]:
        assert bundle["review"]["counts"]["machine-assisted-draft"] == 0
        assert bundle["review"]["counts"]["unspecified"] == 0
        assert not bundle["why_not_citable"]
    else:
        assert bundle["why_not_citable"]


def test_the_committed_bundle_passes_its_own_checker():
    """A committed bundle nobody checks is a record asserting itself.

    `check_bundle` covers the question set too, so this fails when the run
    measured questions the review verdict does not describe.
    """
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert check_bundle(bundle, ROOT) == []


def test_the_committed_run_names_the_question_set_its_review_covers():
    """A review is a verdict about a set of questions, so the bundle names the set."""
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert bundle["question_set_fingerprint"] == question_set_fingerprint(
        load_questions(bundle["corpus"])
    ), "the corpus changed and the committed run no longer measures it"


def test_the_committed_run_carries_a_readme_that_states_both_halves():
    """A number with no provenance beside it invites a citation it cannot support."""
    readme = " ".join((COMMITTED / "README.md").read_text().split())

    assert "Repeat it" in readme
    assert "What this run does NOT show" in readme
    assert "does not rank retrieval architectures" in readme
    assert "cannot answer most of its own questions" in readme


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


@pytest.mark.parametrize("bad", [None, "", False, 0, [], {}, "   "])
def test_a_fingerprint_nobody_can_read_never_passes_as_a_match(tmp_path, bad):
    """The guard asked whether the stored value was truthy, so null skipped it.

    A value that names no question set is not a match. Reading it as one let a
    citable bundle pass without proving its review covers the run.
    """
    run = tmp_path / "results" / "run_x"
    run.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    data["manifests"]["aws-compute"]["question_set_fingerprint"] = bad
    (run / "retriever_lab.json").write_text(json.dumps(data))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": ["results/run_x/retriever_lab.json"]}

    assert check_bundle(bundle, tmp_path), f"a fingerprint of {bad!r} proves nothing"


def test_a_result_with_no_manifest_never_backs_a_citable_bundle(tmp_path):
    """A file that cannot say which questions it scored cannot support a citation."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "legacy.json").write_text(json.dumps({"corpus": "c", "strategy": "bm25"}))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": ["results/legacy.json"]}

    problems = check_bundle(bundle, tmp_path)

    assert any("carries no manifest" in p for p in problems), problems


def test_a_filtered_run_is_not_rejected_for_scoring_a_subset(tmp_path):
    """`--tier 1` and `--split holdout` score a subset on purpose.

    The bundle used to hash every question in the corpus, so it named a set no
    filtered run could match, and `--check` refused every one of them.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    subset = question_set_fingerprint([q for q in load_questions("aws-compute") if q.tier == 1])
    run = tmp_path / "results" / "run_t1"
    run.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    data["manifests"]["aws-compute"]["question_set_fingerprint"] = subset
    (run / "retriever_lab.json").write_text(json.dumps(data))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "results": ["results/run_t1/retriever_lab.json"],
        "question_set_fingerprint": subset,
    }

    assert check_bundle(bundle, tmp_path) == []


def test_the_bundle_never_lists_itself_as_one_of_the_run_results():
    """Listing it made a second `kb-arena evidence` write a different bundle.

    The bundle also carries no manifest, so it could never back its own claim.
    """
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert not any(name.endswith("evidence.json") for name in bundle["results"]), bundle["results"]
