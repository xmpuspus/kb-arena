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

    doc = _flat((ROOT / "docs" / "retriever-lab.md").read_text())
    assert "union of BM25 and every retrieval-only index that answers a probe" in doc
    assert "seeded random" in doc
    # A probe cannot tell a missing index from a provider outage, so the doc
    # must point at the pool record rather than promise which retrievers ran.
    assert "an index you built can still drop out" in doc
    assert "names the retrievers that actually answered" in doc
    assert "refuses to write" in doc, "the doc must state the refusal the code makes"
    assert "--allow-bm25-only" in doc
    assert "The file carries one pool record, so it describes one pool" in doc


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
        "Neo4j was not running, so the graph",
    ],
)
def test_the_lab_doc_makes_no_causal_claim_from_one_run(claim):
    """One run has no spread, so a gap of a few points is not a finding."""
    doc = (ROOT / "docs" / "retriever-lab.md").read_text()
    assert claim not in doc, "a single incomplete run cannot support a causal claim"


def _flat(text: str) -> str:
    """One line, single-spaced, so a wrapped sentence still matches."""
    return " ".join(text.split())


def test_the_lab_doc_says_what_the_one_run_cannot_show():
    """A reader must not take the table as a benchmark result."""
    doc = _flat((ROOT / "docs" / "retriever-lab.md").read_text())
    assert "One run has no spread" in doc
    # The doc must not send a reader to a command that cannot read lab metrics.
    # `kb-arena variance` reads a lab run as of N-33, so the doc names it.
    assert "kb-arena variance --corpus aws-compute --metric mean_recall_at_k" in doc
    assert "never averaged" in doc, "the doc must say what the command refuses"
    # The run does not record whether Neo4j answered, so neither reading is supported.
    assert "does not record whether Neo4j answered" in doc
    assert "A zero here means unmeasured" in doc


def test_the_leaderboard_asks_for_no_submission_it_cannot_process():
    """It invited a pull request against a review process that does not exist."""
    page = (ROOT / "web" / "app" / "leaderboard" / "page.tsx").read_text()
    assert "submit a run" not in page
    assert "open a PR" not in page
    # It says instead what the rows mean, which is the thing a reader needs.
    assert "measured different things" in page.replace("\n", " ").replace("  ", " ")


def test_the_cli_help_says_what_label_chunks_really_does():
    """The help repeated the claim the doc stopped making."""
    from kb_arena import cli

    help_text = inspect.getdoc(cli.label_chunks) or ""
    assert "Haiku judge" not in help_text
    assert "BM25 + Haiku" not in help_text
    assert "KB_ARENA_GENERATE_MODEL" in help_text
    assert "seeded random sample" in help_text


def test_the_leaderboard_shows_the_build_its_copy_names():
    """The copy said rows are distinguished by build and the table never showed one."""
    page = (ROOT / "web" / "app" / "leaderboard" / "page.tsx").read_text()
    # The row shape moved into the one parser that turns the answer into rows,
    # so the field is asserted where it now lives.
    client = (ROOT / "web" / "lib" / "api.ts").read_text()
    assert "build?: string;" in client, "the row type must carry what the API returns"
    assert "row.build" in page, "and the table must render it"
    assert "build unrecorded" in page, "a run with no version or commit says so"


def test_the_cli_help_names_a_setting_that_exists():
    """It named KB_ARENA_COST_CAP_USD, and the setting is benchmark_cost_cap_usd."""
    from kb_arena import cli
    from kb_arena.settings import Settings

    help_text = inspect.getdoc(cli.label_chunks) or ""
    named = [w.strip(".,`") for w in help_text.split() if w.startswith("KB_ARENA_")]
    assert named, "the help has to name the cap it is capped by"
    for var in named:
        field = var.removeprefix("KB_ARENA_").lower()
        assert field in Settings.model_fields, f"{var} is not a setting"


def test_the_doc_does_not_promise_a_grade_for_every_candidate():
    """The prompt asks. The parser accepts whatever comes back."""
    from kb_arena.benchmark import expected_chunks

    source = inspect.getsource(expected_chunks._parse_grades)
    # Nothing here requires the judge to cover the candidate set.
    assert "len(out) == len(valid)" not in source
    doc = _flat((ROOT / "docs" / "retriever-lab.md").read_text())
    assert "A judge that returns grades for only some of the candidates is accepted" in doc
    assert "a missing chunk means unjudged, not rejected" in doc.lower()

    # The CLI help is the surface a user reads first, so it must not disagree.
    from kb_arena import cli

    help_text = _flat(inspect.getdoc(cli.label_chunks) or "").lower()
    assert "a partial answer is accepted" in help_text
    assert "unjudged, not rejected" in help_text


def test_no_file_names_a_settings_variable_that_does_not_exist():
    """One wrong name was fixed in the CLI help and left in a module docstring.

    Grepping the whole tree closes the class instead of the instance. The
    exception is this test file, which quotes the wrong name to explain it.
    """
    import re

    from kb_arena.settings import Settings

    pattern = re.compile(r"KB_ARENA_[A-Z0-9_]+")
    bad: list[str] = []
    for path in list(ROOT.glob("kb_arena/**/*.py")) + list(ROOT.glob("docs/*.md")):
        if path.name == "settings.py":
            continue
        for name in set(pattern.findall(path.read_text())):
            field = name.removeprefix("KB_ARENA_").lower()
            if field not in Settings.model_fields:
                bad.append(f"{path.relative_to(ROOT)}: {name}")
    assert not bad, "these name a setting that does not exist: " + "; ".join(sorted(bad))


def test_a_bm25_only_gold_set_is_a_decision_and_not_a_default():
    """A provider outage looks exactly like an index that was never built.

    Both leave BM25 alone in the pool, and the labels that come out are drawn
    from what BM25 ranks high. That file then scores every strategy for the
    rest of its life, so writing it has to be asked for.
    """
    from kb_arena.benchmark.expected_chunks import NarrowPoolError, label_corpus

    signature = inspect.signature(label_corpus)
    assert signature.parameters["allow_bm25_only"].default is False
    assert issubclass(NarrowPoolError, RuntimeError)

    source = inspect.getsource(label_corpus)
    assert "if not extra_retrievers and not allow_bm25_only:" in source
    # The pool record says whether the narrow pool was asked for.
    assert '"bm25_only_by_request"' in source


def test_the_candidate_count_is_bounded():
    """Every candidate goes in the judge prompt, so the count drives the cost."""
    from kb_arena import cli

    for param in inspect.signature(cli.label_chunks).parameters.values():
        if param.name == "n_candidates":
            option = param.default
            assert option.min == 1
            assert option.max == 200
            break
    else:  # pragma: no cover - the parameter exists
        raise AssertionError("label-chunks must take --n-candidates")


def test_labels_judged_with_another_pool_are_never_relabelled_by_description():
    """The file carries one pool record, so it must describe one pool.

    Adding to a file whose labels were judged with a different pool puts this
    run's retrievers on somebody else's judgments. That is the misattribution
    the record exists to prevent.
    """
    from kb_arena.benchmark.expected_chunks import PoolChangedError, label_corpus

    source = inspect.getsource(label_corpus)
    assert "earlier_pool" in source
    assert "PoolChangedError(" in source
    assert "not force" in source, "--force relabels everything, so it is allowed"
    assert issubclass(PoolChangedError, RuntimeError)


def test_a_bm25_only_run_does_not_demand_the_provider_it_avoids():
    """The documented reason to pass the flag is that the provider is down."""
    from kb_arena import cli

    source = inspect.getsource(cli.label_chunks)
    assert (
        "needs_embeddings=not allow_bm25_only" in source
    ), "demanding embeddings would make the flag unusable in the case it exists for"


@pytest.mark.parametrize("name", ["demo-variance.gif", "demo-evidence.gif"])
def test_every_demo_the_readme_shows_exists_and_has_a_tape(name):
    """A README image that 404s is worse than no image.

    Each recording also keeps its tape, so a reader can see what produced it and
    remake it after the command changes.
    """
    root = Path(__file__).resolve().parents[1]
    gif = root / "docs" / name

    assert gif.is_file(), f"{name} is linked and missing"
    assert gif.stat().st_size > 10_000, f"{name} is too small to be a real recording"

    tape = root / "docs" / "tapes" / (name.removeprefix("demo-").removesuffix(".gif") + ".tape")
    assert tape.is_file(), f"{name} has no tape, so nobody can remake it"
    assert name in tape.read_text(), "the tape must name the file it writes"


def test_the_demo_tapes_run_the_module_from_this_checkout():
    """An editable install points at whichever working copy made it.

    The first take of the evidence demo recorded "No such command 'evidence'"
    for exactly that reason, so both tapes call the module instead.
    """
    root = Path(__file__).resolve().parents[1]

    for name in ("evidence.tape", "variance.tape"):
        tape = (root / "docs" / "tapes" / name).read_text()
        assert "python3 -m kb_arena.cli" in tape, f"{name} must not call the installed script"
