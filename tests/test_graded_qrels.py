"""Graded ground truth from every retriever plus random chunks, in a versioned file."""

from __future__ import annotations

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
