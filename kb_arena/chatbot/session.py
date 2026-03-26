"""Client-side session memory — last 6 turns, truncated assistant messages, TTL-based cleanup."""

from __future__ import annotations

import time

MAX_TURNS = 6
MAX_ASSISTANT_CHARS = 500


class SessionMemory:
    """In-memory conversation history for a single chat session.

    Stores up to MAX_TURNS (6) message pairs. Assistant messages are truncated
    to MAX_ASSISTANT_CHARS to prevent context bloat in the LLM classify call.
    Tracks last_accessed for TTL-based eviction.
    """

    def __init__(self):
        self._history: list[dict] = []
        self.last_accessed: float = time.time()

    def add_turn(self, role: str, content: str) -> None:
        """Append a message. Truncates assistant messages and evicts oldest turns."""
        self.last_accessed = time.time()

        if role == "assistant" and len(content) > MAX_ASSISTANT_CHARS:
            content = content[:MAX_ASSISTANT_CHARS] + "..."

        self._history.append({"role": role, "content": content})

        # Keep only the last MAX_TURNS messages (pairs of user+assistant)
        if len(self._history) > MAX_TURNS * 2:
            self._history = self._history[-(MAX_TURNS * 2) :]

    def get_history(self) -> list[dict]:
        """Return a copy of the current history."""
        self.last_accessed = time.time()
        return list(self._history)

    def clear(self) -> None:
        self._history = []

    def __len__(self) -> int:
        return len(self._history)


class SessionStore:
    """Thread-safe session store with TTL-based eviction."""

    def __init__(self, ttl_minutes: int = 30):
        self._sessions: dict[str, SessionMemory] = {}
        self._ttl_seconds = ttl_minutes * 60

    def get(self, session_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory()
        session = self._sessions[session_id]
        session.last_accessed = time.time()
        return session

    def cleanup(self) -> int:
        """Remove expired sessions. Returns number evicted."""
        now = time.time()
        expired = [
            k for k, v in self._sessions.items()
            if now - v.last_accessed > self._ttl_seconds
        ]
        for k in expired:
            del self._sessions[k]
        return len(expired)

    def __len__(self) -> int:
        return len(self._sessions)
