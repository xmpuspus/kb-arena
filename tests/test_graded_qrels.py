"""Graded ground truth from every retriever plus random chunks, in a versioned file."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import yaml

from kb_arena.benchmark import expected_chunks as labeler
from kb_arena.benchmark.ir_metrics import compute_all
from kb_arena.benchmark.questions import load_qrels, validate_expected_chunks
from kb_arena.models.retrieval import RetrievedChunk


def test_a_version_one_file_reads_as_grade_one_everywhere(tmp_path):
    graded, version = load_qrels({"q1": ["a", "b"], "q2": []}, tmp_path / "x.yaml")
    assert version == 1
    assert graded == {"q1": {"a": 1, "b": 1}, "q2": {}}
    assert validate_expected_chunks({"q1": ["a", "b"]}, tmp_path / "x.yaml") == {"q1": ["a", "b"]}


def test_a_version_two_file_keeps_grades_and_drops_judged_negatives(tmp_path):
    raw = {
        "version": 2,
        "pool": {"retrievers": ["bm25"]},
        "labels": {"q1": {"a": 2, "b": 1, "c": 0}},
    }
    graded, version = load_qrels(raw, tmp_path / "x.yaml")
    assert version == 2
    assert graded["q1"] == {"a": 2, "b": 1, "c": 0}
    assert validate_expected_chunks(raw, tmp_path / "x.yaml") == {"q1": ["a", "b"]}


@pytest.mark.parametrize(
    "raw",
    [
        {"version": 3, "labels": {}},
        {"version": 2, "labels": {"q1": {"a": 5}}},
        {"version": 2, "labels": {"q1": {"a": True}}},
        {"q1": {"a": 2}},
        {"version": 2, "labels": {"q1": {"": 1}}},
    ],
)
def test_malformed_labels_raise(raw, tmp_path):
    with pytest.raises(ValueError):
        load_qrels(raw, tmp_path / "x.yaml")


def test_grades_reach_the_question_and_the_metric(tmp_path, monkeypatch):
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.settings import settings

    qdir = tmp_path / "c" / "questions"
    qdir.mkdir(parents=True)
    (qdir / "tier1.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "q1",
                    "tier": 1,
                    "type": "factoid",
                    "hops": 1,
                    "question": "q?",
                    "ground_truth": {"answer": "a"},
                }
            ]
        )
    )
    (qdir / "expected_chunks.yaml").write_text(
        yaml.safe_dump(
            {"version": 2, "labels": {"q1": {"doc::best": 2, "doc::ok": 1, "doc::no": 0}}}
        )
    )
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    [q] = load_questions("c")

    assert sorted(q.expected_chunks) == ["doc::best", "doc::ok"]
    assert q.expected_grades == {"doc::best": 2, "doc::ok": 1}

    def _ranked(*ids):
        return [
            RetrievedChunk(chunk_id=c, doc_id="doc", content="t", rank=i + 1, source_strategy="x")
            for i, c in enumerate(ids)
        ]

    grades = {c: float(g) for c, g in q.expected_grades.items()}
    better = compute_all(
        _ranked("doc::best", "doc::ok"), set(q.expected_chunks), 2, expected_relevance=grades
    )
    worse = compute_all(
        _ranked("doc::ok", "doc::best"), set(q.expected_chunks), 2, expected_relevance=grades
    )
    assert better.ndcg_at_k > worse.ndcg_at_k, "the grade orders the two rankings"


def test_the_judge_output_parses_as_grades_or_as_a_bare_list():
    valid = {"a", "b", "c"}
    assert labeler._parse_grades('Sure: {"a": 2, "b": 1, "zzz": 2, "c": 0}', valid) == {
        "a": 2,
        "b": 1,
        "c": 0,
    }
    assert labeler._parse_grades('```json\n["a", "zzz"]\n```', valid) == {"a": 1}
    assert labeler._parse_grades("no json here", valid) == {}
    assert labeler._parse_grades('{"a": true}', valid) == {}


def test_the_writer_records_the_version_and_the_pool(tmp_path):
    path = tmp_path / "expected_chunks.yaml"
    labeler._write_expected_chunks(
        path, {"q1": {"a": 2}}, {"retrievers": ["bm25", "naive_vector"], "n_random": 10}
    )

    raw = yaml.safe_load(path.read_text())
    assert raw["version"] == 2
    assert raw["labels"] == {"q1": {"a": 2}}
    assert raw["pool"]["retrievers"] == ["bm25", "naive_vector"]
    graded, version = load_qrels(raw, path)
    assert version == 2 and graded["q1"] == {"a": 2}


@pytest.mark.asyncio
async def test_the_pool_adds_random_chunks_the_retrievers_missed():
    chunk_ids = [f"doc::s{i}" for i in range(30)]

    class _BM25:
        name = "bm25"
        _chunk_ids = chunk_ids
        _corpus_texts = [f"text {i}" for i in range(30)]

        async def query(self, question, top_k, corpus=""):
            retrieved = [
                RetrievedChunk(
                    chunk_id=c, doc_id="doc", content="t", rank=i + 1, source_strategy="bm25"
                )
                for i, c in enumerate(chunk_ids[:3])
            ]
            return SimpleNamespace(retrieval=SimpleNamespace(retrieved=retrieved))

    seen_prompt: dict = {}

    class _LLM:
        async def extract(self, text, system_prompt=""):
            seen_prompt["text"] = text
            return SimpleNamespace(text='{"doc::s0": 2}', cost_usd=0.0)

    grades, cost = await labeler.label_one_question(
        "q?", _BM25(), _LLM(), "c", n_candidates=3, n_random=5
    )

    assert grades == {"doc::s0": 2}
    shown = [line for line in seen_prompt["text"].splitlines() if line.startswith("[doc::")]
    assert len(shown) == 8, "three retrieved plus five random chunks reached the judge"
    assert len({line for line in shown}) == 8


def test_a_truncated_judge_object_raises_instead_of_storing_an_empty_label():
    grades, parsed = labeler._parse_grades('{"a": 2, "b":', {"a", "b"}, report=True)
    assert (grades, parsed) == ({}, False)
    listed, parsed_list = labeler._parse_grades('["a"]', {"a"}, report=True)
    assert (listed, parsed_list) == ({"a": 1}, True), "the list branch reports too"

    class _BM25:
        name = "bm25"
        _chunk_ids: list[str] = []
        _corpus_texts: list[str] = []

        async def query(self, question, top_k, corpus=""):
            from kb_arena.models.retrieval import RetrievedChunk

            return SimpleNamespace(
                retrieval=SimpleNamespace(
                    retrieved=[
                        RetrievedChunk(
                            chunk_id="doc::a",
                            doc_id="doc",
                            content="t",
                            rank=1,
                            source_strategy="bm25",
                        )
                    ]
                )
            )

    class _Truncated:
        async def extract(self, text, system_prompt=""):
            return SimpleNamespace(text='{"doc::a": 2, "doc::b":', cost_usd=0.0)

    with pytest.raises(labeler.JudgeParseError, match="did not parse as grades"):
        asyncio.run(
            labeler.label_one_question("q?", _BM25(), _Truncated(), "c", n_candidates=1, n_random=0)
        )


def test_the_system_prompt_asks_for_grades():
    import inspect

    source = inspect.getsource(labeler.label_one_question)
    assert "JSON object literal mapping chunk_id to a grade" in source
    assert "JSON array literal" not in source


def test_judged_negatives_reach_bpref():
    """bpref counts only judged non-relevant chunks. Without the labels it
    treats every retrieved chunk as one, which penalises a run for ranking
    a chunk nobody judged."""
    from kb_arena.benchmark.ir_metrics import compute_all
    from kb_arena.models.retrieval import RetrievedChunk

    def _ranked(*ids):
        return [
            RetrievedChunk(chunk_id=c, doc_id="doc", content="t", rank=i + 1, source_strategy="x")
            for i, c in enumerate(ids)
        ]

    expected = {"doc::yes"}
    ranked = _ranked("doc::unjudged", "doc::yes")
    guessed = compute_all(ranked, expected, 2)
    judged = compute_all(ranked, expected, 2, judged_nonrelevant={"doc::elsewhere"})

    assert guessed.bpref < 1.0, "with no labels the unjudged chunk counts against the run"
    assert judged.bpref == 1.0, "a chunk nobody judged no longer counts against it"


def test_a_judged_negative_travels_from_the_file_to_bpref(tmp_path, monkeypatch):
    """The whole path, not `compute_all` on its own.

    An earlier version of this slice passed a hand-written set straight to
    `compute_all`, so it went green while `Question` carried no field for the
    negatives and the loader dropped every one of them.
    """
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.benchmark.retriever_lab import _negatives_of
    from kb_arena.settings import settings

    qdir = tmp_path / "c" / "questions"
    qdir.mkdir(parents=True)
    (qdir / "tier1.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "id": "q1",
                    "tier": 1,
                    "type": "factoid",
                    "hops": 1,
                    "question": "q?",
                    "ground_truth": {"answer": "a"},
                }
            ]
        )
    )
    (qdir / "expected_chunks.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "labels": {"q1": {"doc::best": 2, "doc::ok": 1, "doc::no": 0, "doc::no2": 0}},
            }
        )
    )
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    [q] = load_questions("c")

    assert q.judged_negatives == ["doc::no", "doc::no2"]
    assert _negatives_of(q) == {"doc::no", "doc::no2"}


def test_a_question_with_no_judged_negative_keeps_the_bpref_proxy():
    """An empty set turns the TREC proxy off and scores a bad ranking as perfect."""
    from kb_arena.benchmark.retriever_lab import _negatives_of

    assert _negatives_of(SimpleNamespace(judged_negatives=[])) is None
    assert _negatives_of(SimpleNamespace()) is None

    ranked = [
        RetrievedChunk(chunk_id=c, doc_id="doc", content="t", rank=i + 1, source_strategy="x")
        for i, c in enumerate(["doc::junk1", "doc::junk2", "doc::junk3", "doc::best", "doc::ok"])
    ]
    expected = {"doc::best", "doc::ok"}
    proxy_on = compute_all(ranked, expected, 5, judged_nonrelevant=None)
    proxy_off = compute_all(ranked, expected, 5, judged_nonrelevant=set())
    assert (
        proxy_on.bpref < proxy_off.bpref
    ), "three junk chunks above both relevant chunks must not score a perfect bpref"


@pytest.mark.parametrize(
    "answer",
    [
        '{"doc::a": "2", "doc::b": "1"}',
        '{"doc::A": 2, "doc::B": 1}',
        '{"doc::a": 2.5}',
        "{}",
    ],
)
def test_a_json_object_with_no_usable_grade_counts_as_unparsed(answer):
    """A wrapper key or a string grade decodes as JSON and grades nothing."""
    grades, parsed = labeler._parse_grades(answer, {"doc::a", "doc::b"}, report=True)
    assert grades == {}
    assert parsed is False, f"{answer} must not store an empty label"


@pytest.mark.parametrize(
    "text",
    [
        '{"note": "graded below"} {"doc::a": 2}',
        '{"grades": {"doc::a": 2}}',
        'Here are the grades:\n{"doc::a": 2}',
    ],
)
def test_the_search_moves_past_an_object_that_grades_nothing(text):
    """A wrapper key or a prose opener must not lose grades the judge did give."""
    grades, parsed = labeler._parse_grades(text, {"doc::a"}, report=True)
    assert grades == {"doc::a": 2}
    assert parsed is True


def test_a_versioned_file_refuses_a_bare_list(tmp_path):
    """A list in a version 2 file would read as every chunk at grade 1."""
    with pytest.raises(ValueError, match="not a list"):
        load_qrels({"version": 2, "pool": {}, "labels": {"q1": ["a", "b"]}}, tmp_path / "x.yaml")


def test_a_strategy_prefix_never_reaches_a_stored_label():
    """`ir_metrics` strips a prefix from a retrieved id, never from a label."""
    from kb_arena.benchmark.ir_metrics import canonical_chunk_id

    assert canonical_chunk_id("L1:doc::sec") == "doc::sec"
    assert canonical_chunk_id("qna:doc::sec") == "doc::sec"
    assert canonical_chunk_id("doc::sec") == "doc::sec"


def test_a_prefixed_label_is_unreachable_for_every_other_strategy():
    """The reason the labeler strips the prefix, stated as a measurement."""
    ranked = [
        RetrievedChunk(chunk_id="doc::sec", doc_id="doc", content="t", rank=1, source_strategy="x")
    ]
    clean = compute_all(ranked, {"doc::sec"}, 5)
    prefixed = compute_all(ranked, {"doc::sec", "L1:doc::sec"}, 5)
    assert clean.recall_at_k == 1.0
    assert prefixed.recall_at_k < 1.0


def test_a_failed_judgment_still_counts_its_cost():
    """A corpus whose every judgment fails must not report a spend of zero."""
    error = labeler.JudgeParseError("nothing parsed", cost_usd=0.004)
    assert error.cost_usd == 0.004


def test_the_judge_prompt_asks_for_every_candidate_and_shows_a_zero():
    """A prompt that says exclude produces no judged negative at all."""
    prompt = labeler.JUDGE_PROMPT
    assert "grade EVERY candidate" in prompt
    assert "exclude only chunks" not in prompt
    assert '"s3-overview::lifecycle": 0' in prompt


def test_the_fingerprint_tracks_the_labels_and_not_the_pool(tmp_path, monkeypatch):
    """A change to --n-candidates alone must not read as different ground truth."""
    from kb_arena.benchmark.manifest import qrels_fingerprint
    from kb_arena.settings import settings

    qdir = tmp_path / "c" / "questions"
    qdir.mkdir(parents=True)
    path = qdir / "expected_chunks.yaml"
    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))

    labels = {"q1": {"doc::a": 2}}
    path.write_text(yaml.safe_dump({"version": 2, "pool": {"n_candidates": 20}, "labels": labels}))
    first = qrels_fingerprint("c")
    path.write_text(yaml.safe_dump({"version": 2, "pool": {"n_candidates": 50}, "labels": labels}))
    second = qrels_fingerprint("c")
    other = {"q1": {"doc::b": 2}}
    path.write_text(yaml.safe_dump({"version": 2, "pool": {"n_candidates": 50}, "labels": other}))
    third = qrels_fingerprint("c")

    assert first == second, "the pool record describes the labels, it is not the labels"
    assert first != third, "a different label must move the fingerprint"


def test_an_empty_candidate_pool_returns_a_mapping_not_a_list():
    """A list here writes `labels: {q1: []}`, which reloads as a permanent empty label."""

    class _EmptyBM25:
        name = "bm25"
        _chunk_ids: list[str] = []
        _corpus_texts: list[str] = []

        async def query(self, *a, **k):
            return SimpleNamespace(retrieval=None, cost_usd=0.0)

    grades, cost = asyncio.run(
        labeler.label_one_question("q?", _EmptyBM25(), llm=None, corpus="c", n_random=0)
    )
    assert grades == {}
    assert isinstance(grades, dict)
    assert cost == 0.0


def test_a_negative_matches_the_way_a_positive_does():
    """A positive matched through the prefix strip, a negative by raw equality."""
    ranked = [
        RetrievedChunk(chunk_id=c, doc_id="doc", content="t", rank=i + 1, source_strategy="x")
        for i, c in enumerate(["L1:doc::no", "L1:doc::yes"])
    ]
    metrics = compute_all(ranked, {"doc::yes"}, 5, judged_nonrelevant={"doc::no"})
    assert metrics.bpref < 1.0, "a judged negative above the hit must cost bpref"


def test_a_question_the_judge_rejected_never_falls_back_to_the_document():
    """An all-zero label is ground truth, not an absence of it."""
    ranked = [
        RetrievedChunk(
            chunk_id="doc::other", doc_id="doc", content="t", rank=1, source_strategy="x"
        )
    ]
    judged = compute_all(ranked, set(), 5, expected_doc_ids={"doc"}, judged_nonrelevant={"doc::a"})
    unjudged = compute_all(ranked, set(), 5, expected_doc_ids={"doc"})
    assert unjudged.recall_at_k == 1.0, "with no chunk judgments the document match stands"
    assert judged.recall_at_k == 0.0, "the judge rejected every chunk, so nothing is relevant"


def test_the_optimizer_reads_the_grades_and_the_negatives():
    """The lab and the sweep must score the same evidence, or they disagree."""
    import inspect

    from kb_arena.benchmark import optimizer

    source = inspect.getsource(optimizer._score_trial)
    assert "expected_relevance=" in source
    assert "judged_nonrelevant=" in source


def test_a_candidate_with_no_chunk_in_the_corpus_never_becomes_a_label():
    """A cluster summary is reachable only by the strategy that made it."""
    import inspect

    source = inspect.getsource(labeler.label_one_question)
    assert "corpus_ids" in source
    assert "canonical_chunk_id(c.chunk_id) in corpus_ids" in source


def test_a_failed_forced_relabel_drops_the_stale_grade():
    """Republishing an old grade reads as a judgment the judge never made."""
    import inspect

    source = inspect.getsource(labeler.label_corpus)
    assert "del out_dict[q.id]" in source
