"""A result says how many of its questions a human checked, and refuses to call drafts evidence."""

from __future__ import annotations

import asyncio
import json

from kb_arena.benchmark import reporter
from kb_arena.benchmark.review import count_statuses, publication_blockers, review_summary
from kb_arena.chatbot import api
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult, Score
from kb_arena.settings import settings


def _record(status: str = "unspecified", qid: str = "q1"):
    return AnswerRecord(
        question_id=qid,
        strategy="bm25",
        answer="a",
        score=Score(accuracy=1.0),
        question_review_status=status,
    )


def test_counts_cover_every_status_and_treat_an_unknown_one_as_unspecified():
    counts = count_statuses(
        [
            _record("human-reviewed"),
            _record("machine-assisted-draft"),
            {"question_review_status": "made up"},
        ]
    )
    assert counts == {"human-reviewed": 1, "machine-assisted-draft": 1, "unspecified": 1}
    assert count_statuses([]) == {
        "human-reviewed": 0,
        "machine-assisted-draft": 0,
        "unspecified": 0,
    }


def test_only_a_fully_reviewed_result_is_publishable():
    reviewed = [_record("human-reviewed", f"q{i}") for i in range(3)]
    assert review_summary(reviewed)["publishable"] is True
    assert review_summary(reviewed)["reviewed_share"] == 1.0
    assert publication_blockers(reviewed) == []

    mixed = reviewed + [_record("machine-assisted-draft", "q4")]
    summary = review_summary(mixed)
    assert summary["publishable"] is False
    assert summary["reviewed_share"] == 0.75
    assert publication_blockers(mixed) == ["1 of 4 questions are machine-assisted drafts"]
    assert publication_blockers([]) == ["the result scored no questions"]
    assert "no review status" in publication_blockers([_record()])[0]


def test_the_report_says_a_draft_result_is_not_citable(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    bench = BenchmarkResult(
        corpus="c",
        strategy="bm25",
        run_id="r1",
        records=[_record("machine-assisted-draft"), _record("human-reviewed", "q2")],
        accuracy_by_tier={1: 0.9},
    )
    (tmp_path / "c_bm25.json").write_text(bench.model_dump_json())

    reporter.generate_report(corpus="c", output=str(tmp_path / "report.md"))

    text = (tmp_path / "report.md").read_text()
    assert "Not citable evidence" in text
    assert "1 of 2 questions are machine-assisted drafts" in text
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["review"]["publishable"] is False
    assert summary["review"]["counts"]["human-reviewed"] == 1


def test_the_report_says_so_when_every_question_is_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    bench = BenchmarkResult(
        corpus="c",
        strategy="bm25",
        run_id="r1",
        records=[_record("human-reviewed"), _record("human-reviewed", "q2")],
        accuracy_by_tier={1: 0.9},
    )
    (tmp_path / "c_bm25.json").write_text(bench.model_dump_json())

    reporter.generate_report(corpus="c", output=str(tmp_path / "report.md"))

    text = (tmp_path / "report.md").read_text()
    assert "Every scored question is human-reviewed" in text
    assert "Not citable" not in text


def test_a_leaderboard_row_carries_its_review_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    for run_id, status in (("r1", "human-reviewed"), ("r2", "machine-assisted-draft")):
        run_dir = tmp_path / f"run_{run_id}"
        run_dir.mkdir()
        bench = BenchmarkResult(
            corpus="c",
            strategy="bm25",
            run_id=run_id,
            records=[_record(status)],
            overall_accuracy=0.5,
        )
        (run_dir / "c_bm25.json").write_text(bench.model_dump_json())

    board = asyncio.run(api.leaderboard(None, corpus="c"))["leaderboard"]

    assert len(board) == 1
    review = board[0]["review"]
    assert review["questions"] == 2
    assert review["counts"] == {"human-reviewed": 1, "machine-assisted-draft": 1, "unspecified": 0}
    assert review["reviewed_share"] == 0.5
    assert review["publishable"] is False
