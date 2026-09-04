"""Arena ratings live inside one corpus and one rubric, and name who voted."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.arena.engine import INITIAL_ELO, ArenaEngine, ArenaState, Match, scope_key
from kb_arena.settings import settings


def _strategies():
    out = {}
    for name in ("alpha", "beta"):
        result = MagicMock()
        result.answer = f"{name} says"
        result.latency_ms = 10.0
        result.cost_usd = 0.0
        result.sources = []
        result.mock = (
            False  # a MagicMock attribute reads truthy, and the arena refuses a mock answer
        )
        s = AsyncMock()
        s.query = AsyncMock(return_value=result)
        out[name] = s
    return out


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    return ArenaEngine(_strategies())


def _vote_a(arena, corpus, rubric="default", voter="human"):
    match = Match(
        id=f"m-{corpus}-{rubric}-{arena.state.total_votes}",
        question="q",
        strategy_a="alpha",
        strategy_b="beta",
        answer_a="x",
        answer_b="y",
        corpus=corpus,
        rubric=rubric,
    )
    arena.state.matches.append(match)
    return arena.vote(match.id, "a", voter=voter)


def test_votes_on_one_corpus_never_move_another_corpus(arena):
    _vote_a(arena, "aws-compute")
    _vote_a(arena, "aws-compute")

    aws = arena.state.ratings("aws-compute")
    nist = arena.state.ratings("nist-800-171-r3")
    assert aws["alpha"] > INITIAL_ELO > aws["beta"]
    assert nist == {}
    assert [row["strategy"] for row in arena.leaderboard(corpus="aws-compute")] == ["alpha", "beta"]
    # the other corpus lists every strategy at the initial rating, with no match behind it
    untouched = arena.leaderboard(corpus="nist-800-171-r3")
    assert {row["elo"] for row in untouched} == {INITIAL_ELO}
    assert all(row["matches"] == 0 for row in untouched)


def test_a_rubric_is_its_own_scope_and_the_board_names_its_voters(arena):
    _vote_a(arena, "aws-compute", rubric="default", voter="human")
    _vote_a(arena, "aws-compute", rubric="citations", voter="reviewer-1")

    default = arena.leaderboard(corpus="aws-compute", rubric="default")
    citations = arena.leaderboard(corpus="aws-compute", rubric="citations")
    assert default[0]["matches"] == 1 and citations[0]["matches"] == 1
    assert default[0]["voters"] == ["human"]
    assert citations[0]["voters"] == ["reviewer-1"]
    assert default[0]["scope"] == {"corpus": "aws-compute", "rubric": "default"}


def test_a_vote_answers_with_its_scope_ratings_only(arena):
    result = _vote_a(arena, "aws-compute")

    assert result["corpus"] == "aws-compute" and result["rubric"] == "default"
    assert set(result["elo"]) == {"alpha", "beta"}
    assert arena.state.ratings("other") == {}


def test_the_vote_log_carries_the_scope_and_the_voter(arena, tmp_path):
    _vote_a(arena, "aws-compute", voter="reviewer-2")

    line = json.loads((tmp_path / "arena_votes.jsonl").read_text().splitlines()[0])
    assert line["corpus"] == "aws-compute"
    assert line["rubric"] == "default"
    assert line["voter"] == "reviewer-2"
    assert set(line["elo_snapshot"]) == {"alpha", "beta"}


def test_a_legacy_state_file_keeps_its_global_table_under_its_own_scope(tmp_path):
    path = tmp_path / "arena_state.json"
    path.write_text(
        json.dumps({"elo": {"alpha": 1300.0, "beta": 1100.0}, "total_votes": 7, "matches": []})
    )

    state = ArenaState.load(path)

    assert state.elo_by_scope == {scope_key("", "default"): {"alpha": 1300.0, "beta": 1100.0}}
    assert state.ratings("aws-compute") == {}, "an old global table is not a corpus's rating"
    state.save(path)
    saved = json.loads(path.read_text())
    assert saved["elo_by_scope"]["all|default"]["alpha"] == 1300.0


@pytest.mark.asyncio
async def test_a_match_carries_the_scope_it_was_made_in(arena):
    match = await arena.create_match("q", corpus="nist-800-171-r3", rubric="citations")

    assert match.corpus == "nist-800-171-r3"
    assert match.rubric == "citations"
    # the engine assigns a and b at random, so read which strategy sat at b
    winner = match.strategy_b
    arena.vote(match.id, "b")
    assert arena.state.ratings("nist-800-171-r3", "citations")[winner] > INITIAL_ELO
    assert arena.state.ratings("nist-800-171-r3", "default") == {}


def test_the_request_models_carry_rubric_and_voter():
    from kb_arena.models.api import ArenaMatchRequest, ArenaVoteRequest

    assert ArenaMatchRequest(question="q").rubric == "default"
    assert ArenaMatchRequest(question="q", rubric="citations").rubric == "citations"
    with pytest.raises(ValueError):
        ArenaMatchRequest(question="q", rubric="bad rubric!")
    assert ArenaVoteRequest(match_id="m", winner="a").voter == "human"
    with pytest.raises(ValueError):
        ArenaVoteRequest(match_id="m", winner="a", voter="")


def test_a_legacy_match_names_its_voter_as_legacy(tmp_path):
    path = tmp_path / "arena_state.json"
    path.write_text(
        json.dumps(
            {
                "elo": {"alpha": 1216.0, "beta": 1184.0},
                "total_votes": 1,
                "matches": [
                    {
                        "id": "m1",
                        "question": "q",
                        "strategy_a": "alpha",
                        "strategy_b": "beta",
                        "winner": "a",
                    }
                ],
            }
        )
    )
    state = ArenaState.load(path)
    assert state.matches[0].voter == "legacy"


@pytest.mark.asyncio
async def test_the_leaderboard_route_counts_each_match_once(arena):
    from types import SimpleNamespace

    from kb_arena.chatbot import api

    _vote_a(arena, "aws-compute")
    _vote_a(arena, "aws-compute")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arena=arena)))

    body = await api.arena_leaderboard(request, corpus="aws-compute")

    assert body["total_votes"] == 2
    assert body["scope"] == {"corpus": "aws-compute", "rubric": "default"}
    assert "aws-compute|default" in body["scopes"]
