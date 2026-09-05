"""A committed run says what it is, and refuses to say what it is not."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kb_arena.benchmark.evidence import BUNDLE_VERSION, build_bundle, check_bundle
from kb_arena.benchmark.review import review_summary

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


def _run_with(tmp_path, name, *, fingerprint=None, split=None):
    """A copy of the committed lab result, with its manifest edited."""
    run = tmp_path / "results" / name
    run.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    entry = data["manifests"]["aws-compute"]
    if fingerprint is not None:
        entry["question_set_fingerprint"] = fingerprint
    if split is not None:
        entry["question_split"] = split
    (run / "retriever_lab.json").write_text(json.dumps(data))
    return f"results/{name}/retriever_lab.json"


def _reviewed_corpus(tmp_path, monkeypatch, name="split-demo"):
    """A small corpus whose questions are all reviewed, split two ways."""
    import yaml

    from kb_arena.settings import settings

    rows = []
    for i in range(4):
        rows.append(
            {
                "id": f"{name}-{i:03d}",
                "tier": 1,
                "type": "factoid",
                "hops": 1,
                "split": "holdout" if i < 2 else "development",
                "review_status": "human-reviewed",
                "reviewed_by": "Xavier Puspus",
                "question": f"Question {i}?",
                "ground_truth": {"answer": f"Answer {i}."},
            }
        )
    questions = tmp_path / "datasets" / name / "questions"
    questions.mkdir(parents=True)
    (questions / "tier1.yaml").write_text(yaml.safe_dump(rows))
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "datasets"))
    return name


def test_a_split_filtered_run_stays_citable_because_its_manifest_names_the_split(
    tmp_path, monkeypatch
):
    """`--split holdout` scores a subset on purpose, and its manifest says so.

    So the review runs over that same split, and the two fingerprints agree.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    holdout = question_set_fingerprint(load_questions(corpus, split="holdout"))
    run = tmp_path / "results" / "run_h"
    run.mkdir(parents=True)
    (run / "retriever_lab.json").write_text(
        json.dumps(
            {
                "corpora": {corpus: {"bm25": {"questions": 2, "execution_errors": 0}}},
                "manifests": {
                    corpus: {
                        "question_set_fingerprint": holdout,
                        "question_split": "holdout",
                        "question_count": 2,
                    }
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": ["results/run_h/retriever_lab.json"],
        "question_set_fingerprint": holdout,
        "review_question_set": holdout,
        "review_split": "holdout",
        "review": review_summary(load_questions(corpus, split="holdout")),
    }

    assert check_bundle(bundle, tmp_path) == []


def test_a_run_whose_every_query_failed_is_not_citable(tmp_path, monkeypatch):
    """An outage writes a full summary, and every mean in it reads `0.0`.

    The review verdict describes the corpus, not the run, so it stays true while
    the run measures nothing. Only the scored count tells the two apart.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_dead"
    run.mkdir(parents=True)
    (run / "retriever_lab.json").write_text(
        json.dumps(
            {
                "corpora": {
                    corpus: {"bm25": {"questions": 0, "execution_errors": 4}},
                },
                "manifests": {
                    corpus: {
                        "question_set_fingerprint": fingerprint,
                        "question_split": "all",
                        "question_count": 4,
                    }
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": ["results/run_dead/retriever_lab.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("scored no questions" in p for p in problems), problems


def test_a_bundle_cannot_carry_a_review_verdict_the_questions_do_not_support(tmp_path):
    """The stored review is data in the bundle, so a hand edit used to pass.

    Matching fingerprints prove WHICH questions. They prove nothing about the
    verdict over them, so the checker recomputes the verdict from the corpus.
    """
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    review = {**bundle["review"], "counts": {"human-reviewed": 0, "machine-assisted-draft": 75}}
    bundle = {**bundle, "review": review}

    problems = check_bundle(bundle, ROOT)

    assert any("the questions it names count" in p for p in problems), problems


def test_a_tier_filtered_run_cannot_prove_that_its_review_covers_it(tmp_path):
    """`--tier 1` scores a subset and records `question_split: all`.

    Nothing in the manifest tells that subset apart from a corpus somebody
    edited, so the check refuses the bundle instead of widening until it passes.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    subset = question_set_fingerprint([q for q in load_questions("aws-compute") if q.tier == 1])
    result = _run_with(tmp_path, "run_t1", fingerprint=subset)
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": [result], "question_set_fingerprint": subset}

    problems = check_bundle(bundle, tmp_path)

    assert any("describes a different set of questions" in p for p in problems), problems


def test_a_citable_bundle_that_names_no_reviewed_set_is_refused(tmp_path):
    """The guard used to copy the run's fingerprint and compare it to the run.

    That compares a value to a copy of itself. A bundle written by the old code
    carries no `review_question_set`, so it can never prove the review covers it.
    """
    result = _run_with(tmp_path, "run_old")
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": [result], "review_question_set": ""}

    problems = check_bundle(bundle, tmp_path)

    assert any("does not say which question set its review covers" in p for p in problems)


def test_an_edited_question_stops_the_committed_bundle_from_reading_as_citable(tmp_path):
    """A review verdict describes the questions as they were, not as they are.

    The bundle records what the review covered, so a later edit to a question
    moves the corpus fingerprint and the check says which two values differ.
    """
    stale = "0000deadbeef"
    result = _run_with(tmp_path, "run_stale", fingerprint=stale)
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "results": [result],
        "question_set_fingerprint": stale,
        "review_question_set": stale,
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("Somebody changed a question" in p for p in problems), problems


def test_a_citable_bundle_over_a_corpus_nobody_can_read_is_refused(tmp_path):
    """No corpus on disk means no way to recheck the verdict."""
    result = _run_with(tmp_path, "run_gone")
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "corpus": "no-such-corpus", "results": [result]}

    problems = check_bundle(bundle, tmp_path)

    assert any("cannot be read" in p for p in problems), problems


def test_a_readable_manifest_does_not_stand_in_for_a_broken_one(tmp_path):
    """Bad fingerprints were dropped one by one, so a good sibling hid them."""
    run = tmp_path / "results" / "run_pair"
    run.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    good = data["manifests"]["aws-compute"]
    data["manifests"]["other"] = {**good, "question_set_fingerprint": None}
    (run / "retriever_lab.json").write_text(json.dumps(data))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": ["results/run_pair/retriever_lab.json"]}

    problems = check_bundle(bundle, tmp_path)

    assert any("nobody can read" in p for p in problems), problems


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("retriever_lab.json", True),
        ("aws-compute_bm25.json", True),
        ("aws-compute_my_plugin.json", True),
        ("evidence.json", False),
        ("run.json", False),
        ("summary.json", False),
        ("compare_lab_bm25_vs_naive_vector_ndcg_at_k.json", False),
        ("notes.json", False),
        ("compare_a_vs_b.json", False),
    ],
)
def test_only_a_measurement_goes_into_a_bundle(tmp_path, name, kept):
    """Naming the non-results one at a time cost three rounds.

    `summary.json`, then `evidence.json`, then `run.json`, then a comparison
    whose name carries two strategies and a metric. A deny-list cannot hold a
    name that varies, so the rule reads the name against the corpus.
    """
    from kb_arena.benchmark.evidence import is_bundle_result

    path = tmp_path / name
    path.write_text(json.dumps({"corpus": "aws-compute"}))

    assert is_bundle_result(path, "aws-compute") is kept


def test_a_plugin_result_stays_in_the_bundle_whatever_it_holds(tmp_path):
    """A plugin strategy is not in the catalog, and its result is still a result.

    Deciding by contents dropped it twice: once truncated, once carrying no
    manifest. Both times it left in silence and the rest still read as citable.
    The corpus prefix identifies it, so its contents cannot hide it.
    """
    from kb_arena.benchmark.evidence import is_bundle_result

    for body in ("{", json.dumps({"records": []}), json.dumps({"manifest": {}})):
        path = tmp_path / "aws-compute_my_plugin.json"
        path.write_text(body)
        assert is_bundle_result(path, "aws-compute") is True, body


def test_the_bundle_never_lists_itself_as_one_of_the_run_results():
    """Listing it made a second `kb-arena evidence` write a different bundle.

    The bundle also carries no manifest, so it could never back its own claim.
    """
    bundle = json.loads((COMMITTED / "evidence.json").read_text())

    assert not any(name.endswith("evidence.json") for name in bundle["results"]), bundle["results"]


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda d: d["manifests"].update(other=None), id="null-entry"),
        pytest.param(lambda d: d["manifests"].update(other=[]), id="list-entry"),
        pytest.param(lambda d: d["manifests"].update(other="x"), id="string-entry"),
        pytest.param(lambda d: d.update(manifest=None), id="null-singular-key"),
        pytest.param(lambda d: d.update(manifest=7), id="number-singular-key"),
    ],
)
def test_a_manifest_entry_that_is_not_a_record_never_passes_as_absent(tmp_path, edit):
    """The same defect appeared at three depths, so the reader discards nothing.

    First a bad fingerprint was dropped. Then a bad entry under `manifests`.
    Then a `manifest` key holding something other than a record. Each drop let a
    readable sibling speak for the whole file. A present key always counts now.
    """
    run = tmp_path / "results" / "run_null"
    run.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    edit(data)
    (run / "retriever_lab.json").write_text(json.dumps(data))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "results": ["results/run_null/retriever_lab.json"]}

    problems = check_bundle(bundle, tmp_path)

    assert any("nobody can read" in p for p in problems), problems


def test_writing_a_bundle_refuses_what_checking_it_would_reject(tmp_path, monkeypatch):
    """The write path announced a citable run without reading its own checker."""
    import subprocess

    results = tmp_path / "results" / "run_stale"
    results.mkdir(parents=True)
    data = json.loads((COMMITTED / "retriever_lab.json").read_text())
    data["manifests"]["aws-compute"]["question_set_fingerprint"] = "3aecce3d26b1"
    (results / "retriever_lab.json").write_text(json.dumps(data))

    result = subprocess.run(
        ["python3", "-m", "kb_arena.cli", "evidence"]
        + ["--corpus", "aws-compute", "--run-id", "stale"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "KB_ARENA_RESULTS_PATH": str(tmp_path / "results")},
    )

    assert result.returncode == 1, result.stdout
    assert "does not back a bundle" in result.stdout
    assert not (results / "evidence.json").exists(), "a rejected bundle must not land on disk"


def test_a_partial_run_does_not_read_as_a_whole_one(tmp_path, monkeypatch):
    """A run that scored 60 of 75 read as a 75-question run.

    The review verdict then covered 15 questions nobody measured.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_part"
    run.mkdir(parents=True)
    (run / "retriever_lab.json").write_text(
        json.dumps(
            {
                "corpora": {corpus: {"bm25": {"questions": 3, "execution_errors": 1}}},
                "manifests": {
                    corpus: {
                        "question_set_fingerprint": fingerprint,
                        "question_split": "all",
                        "question_count": 4,
                    }
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": ["results/run_part/retriever_lab.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("scored 3 of the 4 questions" in p for p in problems), problems


def test_a_question_file_holding_a_scalar_reports_instead_of_crashing(tmp_path, monkeypatch):
    """`load_questions` raises more than it documents.

    A bare scalar reaches `for entry in raw` and raises TypeError. That escaped
    the named catch, so `kb-arena evidence --check` crashed on a broken corpus.
    """
    from kb_arena.settings import settings

    questions = tmp_path / "datasets" / "broken" / "questions"
    questions.mkdir(parents=True)
    (questions / "tier1.yaml").write_text("42\n")
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path / "datasets"))
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {**bundle, "corpus": "broken", "results": []}

    problems = check_bundle(bundle, tmp_path)

    assert any("cannot be read" in p for p in problems), problems


def test_a_strategy_that_scored_everything_does_not_speak_for_one_that_failed(
    tmp_path, monkeypatch
):
    """The count used to be the maximum over the strategies.

    So one sound row spoke for a sibling that scored nothing, and the bundle
    cited a table with an outage in it. The weakest row decides now.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_mixed"
    run.mkdir(parents=True)
    (run / "retriever_lab.json").write_text(
        json.dumps(
            {
                "corpora": {
                    corpus: {
                        "bm25": {"questions": 4, "execution_errors": 0},
                        "naive_vector": {"questions": 0, "execution_errors": 4},
                    }
                },
                "manifests": {
                    corpus: {
                        "question_set_fingerprint": fingerprint,
                        "question_split": "all",
                        "question_count": 4,
                    }
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": ["results/run_mixed/retriever_lab.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("scored no questions" in p for p in problems), problems


def test_a_benchmark_record_marked_as_an_error_is_not_a_measurement(tmp_path, monkeypatch):
    """Counting the rows counted an outage as 75 measurements."""
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_err"
    run.mkdir(parents=True)
    (run / f"{corpus}_bm25.json").write_text(
        json.dumps(
            {
                "corpus": corpus,
                "strategy": "bm25",
                "records": [{"question_id": f"q{i}", "is_error": True} for i in range(4)],
                "manifest": {
                    "question_set_fingerprint": fingerprint,
                    "question_split": "all",
                    "question_count": 4,
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": [f"results/run_err/{corpus}_bm25.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("scored no questions" in p for p in problems), problems


def test_a_corrupt_result_never_vanishes_from_a_bundle(tmp_path):
    """A truncated plugin result answered no to the manifest clause and left.

    The same truncation in a built-in result was caught by its name and refused.
    So corruption hid one file and blocked another, and the quieter half is the
    dangerous one: the bundle then described less than the run.
    """
    from kb_arena.benchmark.evidence import is_bundle_result

    for name in ("aws-compute_my_plugin.json", "aws-compute_bm25.json", "retriever_lab.json"):
        path = tmp_path / name
        path.write_text("{")
        assert is_bundle_result(path, "aws-compute") is True, name


def test_a_run_holding_a_corrupt_result_writes_no_bundle(tmp_path, monkeypatch):
    """The corrupt file reaches the checker, which says it carries no manifest."""
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_torn"
    run.mkdir(parents=True)
    (run / f"{corpus}_my_plugin.json").write_text("{")
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": [f"results/run_torn/{corpus}_my_plugin.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("carries no manifest" in p for p in problems), problems


def test_a_manifest_that_names_fewer_questions_than_the_review_covers_is_refused(
    tmp_path, monkeypatch
):
    """The counts agreed with each other and with nothing else.

    A manifest naming 1 question and a summary scoring 1 matched, and the bundle
    still carried the fingerprint of every reviewed question. So a one-question
    run read as covering the whole reviewed set.
    """
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions

    corpus = _reviewed_corpus(tmp_path, monkeypatch)
    questions = load_questions(corpus)
    fingerprint = question_set_fingerprint(questions)
    run = tmp_path / "results" / "run_thin"
    run.mkdir(parents=True)
    (run / "retriever_lab.json").write_text(
        json.dumps(
            {
                "corpora": {corpus: {"bm25": {"questions": 1, "execution_errors": 0}}},
                "manifests": {
                    corpus: {
                        "question_set_fingerprint": fingerprint,
                        "question_split": "all",
                        "question_count": 1,
                    }
                },
            }
        )
    )
    bundle = json.loads((COMMITTED / "evidence.json").read_text())
    bundle = {
        **bundle,
        "corpus": corpus,
        "results": ["results/run_thin/retriever_lab.json"],
        "question_set_fingerprint": fingerprint,
        "review_question_set": fingerprint,
        "review_split": "all",
        "review": review_summary(questions),
    }

    problems = check_bundle(bundle, tmp_path)

    assert any("names 1 questions in its manifest" in p for p in problems), problems
