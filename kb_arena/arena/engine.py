"""Arena engine - blind A/B strategy matchups with ELO rating."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from kb_arena.benchmark.atomic import append_jsonl, atomic_write_text
from kb_arena.exceptions import ArenaError
from kb_arena.settings import settings

log = logging.getLogger(__name__)

INITIAL_ELO = 1200.0
K_FACTOR = 32.0


@dataclass
class Match:
    """A single A/B matchup between two strategies."""

    id: str
    question: str
    strategy_a: str
    strategy_b: str
    answer_a: str
    answer_b: str
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    cost_a: float = 0.0
    cost_b: float = 0.0
    winner: str | None = None  # "a", "b", "tie", or None (pending)
    timestamp: float = 0.0
    sources_a: list[str] = field(default_factory=list)
    sources_b: list[str] = field(default_factory=list)
    # The scope a vote counts in. A rating on one corpus under one rubric
    # says nothing about another corpus, so ratings never cross a scope.
    corpus: str = ""
    rubric: str = "default"
    # Who voted: "human" from the arena page, or a named reviewer. A rating
    # built from one source of votes is labeled with it.
    voter: str = ""


def scope_key(corpus: str, rubric: str = "default") -> str:
    return f"{corpus or 'all'}|{rubric or 'default'}"


def _scope_table(raw) -> dict:
    """The scoped ratings from a state file, or an empty table.

    A JSON array reads as a Python list, and a list has no `.items()`. That
    error escaped the loader and took every arena route to 503.
    """
    if not isinstance(raw, dict):
        if raw:
            log.warning("Dropping arena scope table of type %s", type(raw).__name__)
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, dict)}


def _vote_count(raw) -> int:
    """A vote total from a state file, or 0 when the file holds something else.

    A JSON true reads as a Python bool and adds like a 1, so a corrupt file
    could otherwise report a vote nobody cast.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        log.warning("Dropping non-integer arena vote total: %r", raw)
        return 0
    return max(0, raw)


def _numeric_ratings(raw) -> dict[str, float]:
    """Ratings from a state file, with anything that is not a real number dropped.

    A JSON true reads as a Python bool, and round() accepts it, so a corrupt
    file could publish an ELO of 1.0 instead of being refused.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for name, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            log.warning("Dropping non-numeric arena rating for %s: %r", name, value)
            continue
        if not math.isfinite(float(value)):
            log.warning("Dropping non-finite arena rating for %s: %r", name, value)
            continue
        out[str(name)] = float(value)
    return out


@dataclass
class ArenaState:
    """Persistent arena state with ELO ratings and match history."""

    elo: dict[str, float] = field(default_factory=dict)  # legacy global view, kept for old readers
    matches: list[Match] = field(default_factory=list)
    total_votes: int = 0
    # Ratings per scope: scope_key -> strategy -> ELO. The legacy global
    # ratings load into the "all|default" scope, so an old state file keeps
    # its numbers under the scope it was in fact built from.
    elo_by_scope: dict[str, dict[str, float]] = field(default_factory=dict)

    def ratings(self, corpus: str = "", rubric: str = "default") -> dict[str, float]:
        return self.elo_by_scope.setdefault(scope_key(corpus, rubric), {})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "elo": self.elo,
            "elo_by_scope": self.elo_by_scope,
            "total_votes": self.total_votes,
            "matches": [
                {
                    "id": m.id,
                    "question": m.question,
                    "strategy_a": m.strategy_a,
                    "strategy_b": m.strategy_b,
                    "answer_a": m.answer_a[:500],  # truncate for storage
                    "answer_b": m.answer_b[:500],
                    "latency_a_ms": m.latency_a_ms,
                    "latency_b_ms": m.latency_b_ms,
                    "cost_a": m.cost_a,
                    "cost_b": m.cost_b,
                    "winner": m.winner,
                    "timestamp": m.timestamp,
                    "corpus": m.corpus,
                    "rubric": m.rubric,
                    "voter": m.voter,
                }
                for m in self.matches[-200:]  # keep last 200
            ],
        }
        atomic_write_text(path, json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ArenaState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            state = cls(
                elo=data.get("elo", {}),
                total_votes=_vote_count(data.get("total_votes", 0)),
                elo_by_scope=_scope_table(data.get("elo_by_scope")),
            )
            state.elo = _numeric_ratings(state.elo)
            state.elo_by_scope = {
                key: _numeric_ratings(ratings) for key, ratings in state.elo_by_scope.items()
            }
            if scope_key("", "default") not in state.elo_by_scope and state.elo:
                # An old file holds one global table. It was built from every
                # corpus at once, so it keeps that scope, not a corpus's own.
                state.elo_by_scope[scope_key("", "default")] = dict(state.elo)
            for m in data.get("matches", []):
                state.matches.append(
                    Match(
                        id=m["id"],
                        question=m["question"],
                        strategy_a=m["strategy_a"],
                        strategy_b=m["strategy_b"],
                        answer_a=m.get("answer_a", ""),
                        answer_b=m.get("answer_b", ""),
                        latency_a_ms=m.get("latency_a_ms", 0),
                        latency_b_ms=m.get("latency_b_ms", 0),
                        cost_a=m.get("cost_a", 0),
                        cost_b=m.get("cost_b", 0),
                        winner=m.get("winner"),
                        timestamp=m.get("timestamp", 0),
                        corpus=m.get("corpus", ""),
                        rubric=m.get("rubric", "default"),
                        # A match from before voters were recorded says so.
                        voter=m.get("voter") or ("legacy" if m.get("winner") else ""),
                    )
                )
            return state
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            # A corrupt file must cost the ratings, never the service. An
            # uncaught error here leaves `arena` as None and answers 503 on
            # every arena route.
            log.warning("Corrupt arena state, starting fresh", exc_info=True)
            return cls()


class ArenaEngine:
    """Manages blind A/B matches between retrieval strategies."""

    def __init__(self, strategies: dict) -> None:
        self.strategies = strategies
        self._state_path = Path(settings.results_path) / "arena_state.json"
        self.state = ArenaState.load(self._state_path)
        # Initialize ELO for new strategies
        for name in strategies:
            if name not in self.state.elo:
                self.state.elo[name] = INITIAL_ELO

    async def create_match(self, question: str, corpus: str = "", rubric: str = "default") -> Match:
        """Pick two random strategies, query both, return a blind match."""
        names = list(self.strategies.keys())
        if len(names) < 2:
            raise ValueError("Need at least 2 strategies for arena mode")
        # Every distinct rubric makes a rating table that the file keeps and
        # every leaderboard lists. Without a cap a caller grows both without
        # limit, one new name at a time.
        key = scope_key(corpus, rubric)
        cap = settings.arena_max_scopes
        if key not in self.state.elo_by_scope and len(self.state.elo_by_scope) >= cap:
            raise ValueError(
                f"The arena already holds {cap} rating scopes. Reuse one of them, "
                f"or raise KB_ARENA_ARENA_MAX_SCOPES."
            )
        a_name, b_name = random.sample(names, 2)
        selected_corpus = corpus or "all"

        result_a, result_b = await asyncio.gather(
            self.strategies[a_name].query(question, corpus=selected_corpus),
            self.strategies[b_name].query(question, corpus=selected_corpus),
        )

        # A mock answer reports an outage, not a retrieval design. Rating it would
        # write an infrastructure failure into the persistent leaderboard.
        mocked = [
            name
            for name, result in ((a_name, result_a), (b_name, result_b))
            if getattr(result, "mock", False)
        ]
        if mocked:
            raise ArenaError(f"Strategy unavailable, so the match cannot be rated: {mocked}")

        match = Match(
            id=uuid4().hex[:8],
            question=question,
            strategy_a=a_name,
            strategy_b=b_name,
            answer_a=result_a.answer,
            answer_b=result_b.answer,
            latency_a_ms=result_a.latency_ms,
            latency_b_ms=result_b.latency_ms,
            cost_a=result_a.cost_usd,
            cost_b=result_b.cost_usd,
            sources_a=result_a.sources,
            sources_b=result_b.sources,
            timestamp=time.time(),
            corpus=corpus,
            rubric=rubric or "default",
        )
        self.state.matches.append(match)
        return match

    def vote(self, match_id: str, winner: str, voter: str = "human") -> dict:
        """Record a vote and update the match's scope ratings. winner: 'a', 'b', or 'tie'."""
        if winner not in ("a", "b", "tie"):
            return {"error": f"Invalid winner: {winner}. Must be 'a', 'b', or 'tie'"}

        match = next((m for m in self.state.matches if m.id == match_id), None)
        if not match:
            return {"error": "Match not found"}
        if match.winner is not None:
            return {"error": "Match already voted on"}

        match.winner = winner
        match.voter = voter or "human"
        self.state.total_votes += 1
        self._update_elo(match)
        self.state.save(self._state_path)
        self._append_vote_jsonl(match)

        return {
            "strategy_a": match.strategy_a,
            "strategy_b": match.strategy_b,
            "winner": winner,
            "corpus": match.corpus or "all",
            "rubric": match.rubric,
            "elo": dict(self.state.ratings(match.corpus, match.rubric)),
            "total_votes": self.state.total_votes,
        }

    def _update_elo(self, match: Match) -> None:
        """Standard ELO rating update, inside the match's own scope."""
        ratings = self.state.ratings(match.corpus, match.rubric)
        ea = ratings.get(match.strategy_a, INITIAL_ELO)
        eb = ratings.get(match.strategy_b, INITIAL_ELO)
        expected_a = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))

        if match.winner == "a":
            score_a = 1.0
        elif match.winner == "b":
            score_a = 0.0
        else:  # tie
            score_a = 0.5

        ratings[match.strategy_a] = ea + K_FACTOR * (score_a - expected_a)
        ratings[match.strategy_b] = eb + K_FACTOR * ((1 - score_a) - (1 - expected_a))
        # The legacy global table mirrors the global scope only, for readers
        # of the old field. A corpus vote never touches it.
        if scope_key(match.corpus, match.rubric) == scope_key("", "default"):
            self.state.elo[match.strategy_a] = ratings[match.strategy_a]
            self.state.elo[match.strategy_b] = ratings[match.strategy_b]

    def leaderboard(self, corpus: str = "", rubric: str = "default") -> list[dict]:
        """Strategies sorted by ELO inside one scope. Votes from other scopes never count."""
        key = scope_key(corpus, rubric)
        ratings = self.state.elo_by_scope.get(key, {})
        scoped = [m for m in self.state.matches if scope_key(m.corpus, m.rubric) == key]
        # Every strategy the engine knows shows on the board, at the initial
        # rating when the scope holds no vote for it yet.
        names = list(self.strategies) + [n for n in ratings if n not in self.strategies]
        board = []
        for name in names:
            elo = ratings.get(name, INITIAL_ELO)
            wins = sum(
                1
                for m in scoped
                if m.winner
                and (
                    (m.strategy_a == name and m.winner == "a")
                    or (m.strategy_b == name and m.winner == "b")
                )
            )
            losses = sum(
                1
                for m in scoped
                if m.winner
                and (
                    (m.strategy_a == name and m.winner == "b")
                    or (m.strategy_b == name and m.winner == "a")
                )
            )
            ties = sum(
                1
                for m in scoped
                if m.winner == "tie" and (m.strategy_a == name or m.strategy_b == name)
            )
            voters = sorted(
                {
                    m.voter or "legacy"
                    for m in scoped
                    if m.winner and name in (m.strategy_a, m.strategy_b)
                }
            )
            board.append(
                {
                    "strategy": name,
                    "scope": {"corpus": corpus or "all", "rubric": rubric or "default"},
                    "voters": voters,
                    "elo": round(elo, 1),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "matches": wins + losses + ties,
                }
            )
        return sorted(board, key=lambda x: x["elo"], reverse=True)

    def _append_vote_jsonl(self, match: Match) -> None:
        """Append-only JSONL log of all votes (survives state resets)."""
        jsonl_path = self._state_path.parent / "arena_votes.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "match_id": match.id,
            "question": match.question[:200],
            "strategy_a": match.strategy_a,
            "strategy_b": match.strategy_b,
            "winner": match.winner,
            "latency_a_ms": round(match.latency_a_ms, 1),
            "latency_b_ms": round(match.latency_b_ms, 1),
            "cost_a": match.cost_a,
            "cost_b": match.cost_b,
            "timestamp": match.timestamp,
            "corpus": match.corpus or "all",
            "rubric": match.rubric,
            "voter": match.voter,
            "elo_snapshot": {
                k: round(v, 1) for k, v in self.state.ratings(match.corpus, match.rubric).items()
            },
        }
        append_jsonl(jsonl_path, record)

    def get_pending_match(self, match_id: str) -> Match | None:
        """Get a match by ID."""
        return next((m for m in self.state.matches if m.id == match_id), None)
