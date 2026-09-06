"""Arena ratings live inside one corpus and one rubric, and name who voted."""

from __future__ import annotations

import asyncio
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


@pytest.mark.asyncio
async def test_the_route_rejects_a_bad_scope_and_reports_both_vote_counts(arena, monkeypatch):
    from types import SimpleNamespace

    import pytest as _pytest
    from fastapi import HTTPException

    from kb_arena.chatbot import api

    _vote_a(arena, "aws-compute")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arena=arena)), headers={})

    with _pytest.raises(HTTPException) as bad:
        await api.arena_leaderboard(request, corpus="../etc")
    assert bad.value.status_code == 400

    body = await api.arena_leaderboard(request, corpus="aws-compute")
    assert body["votes_in_history"] == 1
    assert body["total_votes"] == arena.state.total_votes


def test_a_named_voter_needs_the_reviewer_key():
    from kb_arena.settings import settings as live

    assert live.arena_reviewer_key == "", "a named voter is refused until an operator sets a key"


def test_a_corrupt_rating_is_dropped_not_published(tmp_path):
    path = tmp_path / "arena_state.json"
    path.write_text(
        json.dumps(
            {
                "elo": {"alpha": True, "beta": 1250.0, "gamma": "high"},
                "elo_by_scope": {"c|default": {"alpha": 1300.0, "beta": None}},
                "total_votes": 1,
                "matches": [],
            }
        )
    )

    state = ArenaState.load(path)

    assert state.elo == {"beta": 1250.0}
    assert state.elo_by_scope["c|default"] == {"alpha": 1300.0}


def test_a_named_voter_without_the_key_is_refused_not_recorded_as_human(arena):
    from types import SimpleNamespace

    import pytest as _pytest
    from fastapi import HTTPException

    from kb_arena.chatbot import api
    from kb_arena.models.api import ArenaVoteRequest

    match = Match(
        id="m-key", question="q", strategy_a="alpha", strategy_b="beta", answer_a="x", answer_b="y"
    )
    arena.state.matches.append(match)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arena=arena)), headers={})
    body = ArenaVoteRequest(match_id="m-key", winner="a", voter="reviewer-9")

    with _pytest.raises(HTTPException) as refused:
        asyncio.run(api.arena_vote(body, request))

    assert refused.value.status_code == 403
    assert match.winner is None, "the match stays open for the real reviewer"


def test_the_page_refreshes_the_matchs_own_board_after_a_vote():
    from pathlib import Path

    page = Path("web/app/arena/page.tsx").read_text()
    # The refreshed board is the match's own scope. The reply now carries that
    # scope in a name, because a reply landing after a corpus change is
    # dropped rather than read into the corpus now on screen.
    assert "const votedCorpus = data.corpus ?? corpus;" in page
    assert "fetchLeaderboard(votedCorpus)" in page


@pytest.mark.asyncio
async def test_the_leaderboard_route_answers_503_when_the_arena_is_down():
    """An empty board reads as a real result, so an outage must not return one."""
    from types import SimpleNamespace

    from kb_arena.chatbot import api

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arena=None)))

    response = await api.arena_leaderboard(request, corpus="aws-compute")

    assert response.status_code == 503
    assert b"arena_unavailable" in response.body


def test_a_corrupt_vote_total_never_counts_as_a_vote(tmp_path):
    """A JSON true adds like a 1, so the loader must refuse it."""
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"elo": {"alpha": 1200.0}, "total_votes": True}))

    state = ArenaState.load(path)

    assert state.total_votes == 0


def test_a_legacy_table_survives_a_file_that_already_holds_another_scope(tmp_path):
    """A file with one corpus scope still carries a global table that predates scoping."""
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps(
            {
                "elo": {"alpha": 1240.0, "beta": 1160.0},
                "total_votes": 4,
                "elo_by_scope": {"aws-compute|default": {"alpha": 1210.0}},
            }
        )
    )

    state = ArenaState.load(path)

    assert state.elo_by_scope["all|default"] == {"alpha": 1240.0, "beta": 1160.0}
    assert state.elo_by_scope["aws-compute|default"] == {"alpha": 1210.0}


def test_the_page_drops_a_stale_board_when_the_read_fails():
    """A board from an earlier corpus beside a new corpus name reads as that corpus."""
    from pathlib import Path

    page = Path("web/app/arena/page.tsx").read_text()
    assert "setBoardError(" in page
    assert "setLeaderboard([]);" in page
    assert "data.votes_in_history ?? data.total_votes ?? 0" in page


def test_the_packaged_bundle_carries_the_scoped_board():
    """`kb-arena serve` ships this bundle, so a stale one hides every page fix."""
    from pathlib import Path

    static = Path("kb_arena/static")
    if not static.exists():
        pytest.skip("no packaged bundle in this checkout")
    hits = [p for p in static.rglob("*.js") if "votes_in_history" in p.read_text(errors="ignore")]
    assert hits, "the packaged bundle predates the scoped leaderboard"


def test_the_packaged_bundle_matches_its_sources():
    """A field grep only catches the one stale bundle somebody thought to name.

    This catches any of them. The bundle carries the digest of the frontend
    sources it was built from, and a page edit without a rebuild breaks it.
    """
    from pathlib import Path

    from kb_arena.frontend_bundle import read_stamp, source_digest

    web, static = Path("web"), Path("kb_arena/static")
    if not web.is_dir() or not static.is_dir():
        pytest.skip("no frontend sources or no packaged bundle in this checkout")
    stamp = read_stamp(static)
    assert stamp, "the packaged bundle carries no source stamp; run scripts/sync_frontend_bundle.py"
    digest, count = source_digest(web)
    assert stamp["digest"] == digest, (
        f"the packaged bundle was built from {stamp['files']} source files with digest "
        f"{stamp['digest'][:12]}, and the tree now holds {count} files with digest "
        f"{digest[:12]}. Run `npx next build` in web/, then scripts/sync_frontend_bundle.py."
    )


def test_a_scope_table_of_the_wrong_type_never_takes_the_arena_down(tmp_path):
    """An uncaught load error leaves `arena` as None and answers 503 everywhere."""
    path = tmp_path / "arena.json"
    path.write_text(json.dumps({"elo": {"alpha": 1200.0}, "elo_by_scope": ["broken"]}))

    state = ArenaState.load(path)

    assert state.elo_by_scope == {"all|default": {"alpha": 1200.0}}


def test_a_scope_entry_of_the_wrong_type_is_dropped_not_kept(tmp_path):
    path = tmp_path / "arena.json"
    path.write_text(
        json.dumps({"elo_by_scope": {"aws|default": {"alpha": 1200.0}, "bad|default": "nope"}})
    )

    state = ArenaState.load(path)

    assert set(state.elo_by_scope) == {"aws|default"}


@pytest.mark.asyncio
async def test_a_new_rubric_stops_at_the_scope_cap(arena, monkeypatch):
    """Every distinct rubric makes a rating table the file keeps forever."""
    monkeypatch.setattr(settings, "arena_max_scopes", 2)
    arena.state.elo_by_scope = {"a|default": {}, "b|default": {}}

    with pytest.raises(ValueError, match="rating scopes"):
        await arena.create_match("q?", corpus="c", rubric="brand-new")

    # An existing scope still works, so the cap never blocks ordinary use.
    match = await arena.create_match("q?", corpus="a", rubric="default")
    assert match.rubric == "default"


@pytest.mark.asyncio
async def test_a_burst_of_matches_cannot_walk_past_the_scope_cap(arena, monkeypatch):
    """The table grows on the vote, so a check at match time alone bounds nothing.

    `ArenaState.ratings` creates the entry with `setdefault`, and only
    `_update_elo` calls it. Matches on new rubrics leave the table empty, so a
    cap read at match time sees room that later votes then consume.
    """
    monkeypatch.setattr(settings, "arena_max_scopes", 2)
    assert arena.state.elo_by_scope == {}

    matches = []
    for i in range(4):
        try:
            matches.append(await arena.create_match("q?", corpus="c", rubric=f"r{i}"))
        except ValueError:
            break

    errors = 0
    for match in matches:
        result = arena.vote(match.id, "a")
        if "error" in result:
            errors += 1

    assert len(arena.state.elo_by_scope) <= 2, "the vote path must respect the cap too"
    assert errors, "a vote past the cap must say so, not grow the file in silence"


@pytest.mark.asyncio
async def test_the_two_vote_counts_agree_about_what_a_vote_is(arena):
    """`votes_in_history` counted a stored winner the leaderboard rows ignore."""
    from types import SimpleNamespace

    from kb_arena.chatbot import api

    _vote_a(arena, "aws-compute")
    arena.state.matches[-1].winner = "invalid"
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arena=arena)))

    body = await api.arena_leaderboard(request, corpus="aws-compute")

    assert body["votes_in_history"] == 0
    assert sum(row["matches"] for row in body["leaderboard"]) == 0


def test_the_page_refuses_a_leaderboard_reply_of_the_wrong_shape():
    """A vote count that arrives as a string renders as a real number of votes."""
    from pathlib import Path

    page = Path("web/app/arena/page.tsx").read_text()
    assert "isLeaderboardResponse" in page
    assert (
        'typeof board.votes_in_history === "number"' in page
        or "counted(board.votes_in_history)" in page
    )


def test_the_sync_refuses_to_stamp_a_build_older_than_its_sources(tmp_path):
    """The stamp records the source digest at sync time, not at build time.

    A sync without a rebuild wrote a fresh digest over a stale build, so
    `test_the_packaged_bundle_matches_its_sources` passed while the packaged
    pages held the old code. A reviewer found exactly that on the decision-flow
    branch, and only a grep of the built chunks showed it.
    """
    import importlib.util
    import os
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "sync_bundle", root / "scripts" / "sync_frontend_bundle.py"
    )
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)

    web = tmp_path / "web"
    (web / "app").mkdir(parents=True)
    page = web / "app" / "page.tsx"
    page.write_text("export default function Page() { return null; }\n")
    out = web / "out"
    out.mkdir()
    built = out / "index.html"
    built.write_text("<html></html>")

    os.utime(built, (1000, 1000))
    os.utime(page, (2000, 2000))
    assert sync._newest_source_time(web) > sync._build_time(out)

    os.utime(built, (3000, 3000))
    assert sync._newest_source_time(web) < sync._build_time(out)
