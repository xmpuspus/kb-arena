"""Arena engine - blind A/B strategy matchups with ELO rating."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

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


@dataclass
class ArenaState:
    """Persistent arena state with ELO ratings and match history."""

    elo: dict[str, float] = field(default_factory=dict)
    matches: list[Match] = field(default_factory=list)
    total_votes: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "elo": self.elo,
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
                }
                for m in self.matches[-200:]  # keep last 200
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ArenaState:
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            state = cls(
                elo=data.get("elo", {}),
                total_votes=data.get("total_votes", 0),
            )
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
                    )
                )
            return state
        except (json.JSONDecodeError, KeyError):
            log.warning("Corrupt arena state, starting fresh")
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

    async def create_match(self, question: str, corpus: str = "") -> Match:
        """Pick two random strategies, query both, return a blind match."""
        names = list(self.strategies.keys())
        if len(names) < 2:
            raise ValueError("Need at least 2 strategies for arena mode")
        a_name, b_name = random.sample(names, 2)

        result_a, result_b = await asyncio.gather(
            self.strategies[a_name].query(question),
            self.strategies[b_name].query(question),
        )

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
        )
        self.state.matches.append(match)
        return match

    def vote(self, match_id: str, winner: str) -> dict:
        """Record a vote and update ELO. winner: 'a', 'b', or 'tie'."""
        if winner not in ("a", "b", "tie"):
            return {"error": f"Invalid winner: {winner}. Must be 'a', 'b', or 'tie'"}

        match = next((m for m in self.state.matches if m.id == match_id), None)
        if not match:
            return {"error": "Match not found"}
        if match.winner is not None:
            return {"error": "Match already voted on"}

        match.winner = winner
        self.state.total_votes += 1
        self._update_elo(match)
        self.state.save(self._state_path)

        return {
            "strategy_a": match.strategy_a,
            "strategy_b": match.strategy_b,
            "winner": winner,
            "elo": dict(self.state.elo),
            "total_votes": self.state.total_votes,
        }

    def _update_elo(self, match: Match) -> None:
        """Standard ELO rating update."""
        ea = self.state.elo.get(match.strategy_a, INITIAL_ELO)
        eb = self.state.elo.get(match.strategy_b, INITIAL_ELO)
        expected_a = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))

        if match.winner == "a":
            score_a = 1.0
        elif match.winner == "b":
            score_a = 0.0
        else:  # tie
            score_a = 0.5

        self.state.elo[match.strategy_a] = ea + K_FACTOR * (score_a - expected_a)
        self.state.elo[match.strategy_b] = eb + K_FACTOR * ((1 - score_a) - (1 - expected_a))

    def leaderboard(self) -> list[dict]:
        """Return strategies sorted by ELO rating."""
        board = []
        for name, elo in self.state.elo.items():
            wins = sum(
                1
                for m in self.state.matches
                if m.winner
                and (
                    (m.strategy_a == name and m.winner == "a")
                    or (m.strategy_b == name and m.winner == "b")
                )
            )
            losses = sum(
                1
                for m in self.state.matches
                if m.winner
                and (
                    (m.strategy_a == name and m.winner == "b")
                    or (m.strategy_b == name and m.winner == "a")
                )
            )
            ties = sum(
                1
                for m in self.state.matches
                if m.winner == "tie" and (m.strategy_a == name or m.strategy_b == name)
            )
            board.append(
                {
                    "strategy": name,
                    "elo": round(elo, 1),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "matches": wins + losses + ties,
                }
            )
        return sorted(board, key=lambda x: x["elo"], reverse=True)

    def get_pending_match(self, match_id: str) -> Match | None:
        """Get a match by ID."""
        return next((m for m in self.state.matches if m.id == match_id), None)
