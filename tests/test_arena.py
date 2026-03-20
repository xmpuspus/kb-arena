"""Tests for the Arena engine."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.arena.engine import INITIAL_ELO, ArenaEngine, ArenaState, Match


@pytest.fixture
def mock_strategies():
    strategies = {}
    for name in ["naive_vector", "knowledge_graph", "hybrid"]:
        strat = AsyncMock()
        result = MagicMock()
        result.answer = f"Answer from {name}"
        result.latency_ms = 500.0
        result.cost_usd = 0.01
        result.sources = ["doc1.md"]
        strat.query = AsyncMock(return_value=result)
        strategies[name] = strat
    return strategies


@pytest.fixture
def arena(mock_strategies, tmp_path):
    from kb_arena import settings

    settings.settings.results_path = str(tmp_path)
    return ArenaEngine(mock_strategies)


def test_initial_elo(arena):
    for name in arena.strategies:
        assert arena.state.elo[name] == INITIAL_ELO


@pytest.mark.asyncio
async def test_create_match(arena):
    match = await arena.create_match("What is Lambda?")
    assert match.id
    assert match.question == "What is Lambda?"
    assert match.strategy_a in arena.strategies
    assert match.strategy_b in arena.strategies
    assert match.strategy_a != match.strategy_b
    assert match.answer_a
    assert match.answer_b
    assert match.winner is None


def test_vote_a_wins(arena):
    match = Match(
        id="test1",
        question="q",
        strategy_a="naive_vector",
        strategy_b="knowledge_graph",
        answer_a="a",
        answer_b="b",
        timestamp=1.0,
    )
    arena.state.matches.append(match)

    result = arena.vote("test1", "a")
    assert result["winner"] == "a"
    assert result["strategy_a"] == "naive_vector"
    # Winner's ELO should increase
    assert arena.state.elo["naive_vector"] > INITIAL_ELO
    assert arena.state.elo["knowledge_graph"] < INITIAL_ELO


def test_vote_tie(arena):
    match = Match(
        id="test2",
        question="q",
        strategy_a="naive_vector",
        strategy_b="hybrid",
        answer_a="a",
        answer_b="b",
        timestamp=1.0,
    )
    arena.state.matches.append(match)

    result = arena.vote("test2", "tie")
    assert result["winner"] == "tie"
    # ELO should stay close to initial (equal-rated players, 0.5 vs expected 0.5)
    assert abs(arena.state.elo["naive_vector"] - INITIAL_ELO) < 1


def test_vote_invalid_winner(arena):
    match = Match(
        id="test3",
        question="q",
        strategy_a="naive_vector",
        strategy_b="hybrid",
        answer_a="a",
        answer_b="b",
        timestamp=1.0,
    )
    arena.state.matches.append(match)
    result = arena.vote("test3", "invalid")
    assert "error" in result


def test_vote_duplicate(arena):
    match = Match(
        id="test4",
        question="q",
        strategy_a="naive_vector",
        strategy_b="hybrid",
        answer_a="a",
        answer_b="b",
        winner="a",
        timestamp=1.0,
    )
    arena.state.matches.append(match)
    result = arena.vote("test4", "b")
    assert "error" in result


def test_vote_not_found(arena):
    result = arena.vote("nonexistent", "a")
    assert "error" in result


def test_leaderboard(arena):
    # Add some matches with votes
    for i in range(5):
        match = Match(
            id=f"m{i}",
            question="q",
            strategy_a="naive_vector",
            strategy_b="knowledge_graph",
            answer_a="a",
            answer_b="b",
            winner="a",
            timestamp=float(i),
        )
        arena.state.matches.append(match)
        arena._update_elo(match)

    board = arena.leaderboard()
    assert len(board) == 3  # 3 strategies
    assert board[0]["strategy"] == "naive_vector"  # most wins
    assert board[0]["wins"] == 5
    assert board[0]["elo"] > INITIAL_ELO


def test_state_persistence(arena, tmp_path):
    match = Match(
        id="persist1",
        question="q",
        strategy_a="naive_vector",
        strategy_b="hybrid",
        answer_a="a",
        answer_b="b",
        winner="a",
        timestamp=1.0,
    )
    arena.state.matches.append(match)
    arena.state.total_votes = 1
    arena._update_elo(match)
    arena.state.save(arena._state_path)

    # Load from disk
    loaded = ArenaState.load(arena._state_path)
    assert loaded.total_votes == 1
    assert len(loaded.matches) == 1
    assert loaded.matches[0].id == "persist1"
    assert loaded.elo["naive_vector"] > INITIAL_ELO
