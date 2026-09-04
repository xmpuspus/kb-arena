"""The repository must not claim more than it does.

Each test here pins one sentence in a document against the code that sentence
describes. A doc drifts silently; a test does not.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_readme_promises_only_what_the_demo_bundles():
    """The demo ships benchmark results and no run directories."""
    bundled = sorted(p.name for p in (ROOT / "kb_arena" / "data").glob("*.json"))
    results = [n for n in bundled if n.startswith("aws-compute_")]
    assert results, "the demo has to bundle something for the README to describe"

    readme = (ROOT / "README.md").read_text()
    assert (
        f"{len(results)} benchmark result files" in readme
    ), f"the README must say how many result files ship. There are {len(results)}."
    # Nothing under kb_arena/data seeds the Retriever Lab or a spread over repeats.
    assert not [n for n in bundled if "retriever_lab" in n or n.startswith("run_")]
    assert "The Retriever Lab and the spread across repeated runs are empty" in readme


def test_the_labeling_doc_names_the_model_the_labeler_calls():
    """The doc said Haiku. Labeling calls `LLMClient.extract`, which is the generate model."""
    from kb_arena.benchmark import expected_chunks

    source = inspect.getsource(expected_chunks.label_one_question)
    assert "llm.extract(" in source, "if this moves, the doc below is wrong"

    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert "KB_ARENA_GENERATE_MODEL" in doc
    assert (
        "Claude Haiku to mark which are actually relevant" not in doc
    ), "labeling does not call the fast model"


def test_the_labeling_doc_names_the_pool_the_labeler_builds():
    """The doc said BM25 alone. The pool is a union plus a random sample."""
    from kb_arena.benchmark import expected_chunks

    source = inspect.getsource(expected_chunks)
    for name in (
        "NaiveVectorStrategy",
        "ContextualVectorStrategy",
        "QnAPairsStrategy",
        "RaptorStrategy",
    ):
        assert name in source, f"{name} is in the pool, so the doc must name it"

    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert "union of BM25 and every retrieval-only index" in doc
    assert "seeded random" in doc


def test_the_labeling_doc_describes_the_file_the_writer_produces():
    """The doc described a version 1 map after the writer moved to version 2."""
    from kb_arena.benchmark.questions import QRELS_VERSION

    assert QRELS_VERSION == 2
    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert "{version, pool, labels}" in doc
    assert "`{question_id: [chunk_id, ...]}` map" not in doc


@pytest.mark.parametrize(
    "claim",
    [
        "Contextual Vector wins on ranking, not coverage.",
        "RAPTOR's L0 layer is doing the work.",
        "Hybrid drops to 8% because Neo4j wasn't running.",
    ],
)
def test_the_lab_doc_makes_no_causal_claim_from_one_run(claim):
    """One run has no spread, so a gap of a few points is not a finding."""
    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert claim not in doc, "a single incomplete run cannot support a causal claim"


def test_the_lab_doc_says_what_the_one_run_cannot_show():
    """A reader must not take the table as a benchmark result."""
    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert "One run has no spread" in doc
    assert "kb-arena variance" in doc, "the doc must name the command that measures spread"
    assert "an outage, not a result" in doc, "the hybrid row measures the deployment"
    assert "A zero here means unmeasured" in doc


def test_the_leaderboard_asks_for_no_submission_it_cannot_process():
    """It invited a pull request against a review process that does not exist."""
    page = (ROOT / "web" / "app" / "leaderboard" / "page.tsx").read_text()
    assert "submit a run" not in page
    assert "open a PR" not in page
    # It says instead what the rows mean, which is the thing a reader needs.
    assert "measured different things" in page.replace("\n", " ").replace("  ", " ")
