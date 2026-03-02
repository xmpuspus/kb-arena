"""Client-side session memory — last 6 turns, truncated assistant messages."""

from __future__ import annotations

MAX_TURNS = 6
MAX_ASSISTANT_CHARS = 500


class SessionMemory:
    """In-memory conversation history for a single chat session.

    Stores up to MAX_TURNS (6) message pairs. Assistant messages are truncated
    to MAX_ASSISTANT_CHARS to prevent context bloat in the LLM classify call.
    """

    def __init__(self):
        self._history: list[dict] = []

    def add_turn(self, role: str, content: str) -> None:
        """Append a message. Truncates assistant messages and evicts oldest turns."""
        if role == "assistant" and len(content) > MAX_ASSISTANT_CHARS:
            content = content[:MAX_ASSISTANT_CHARS] + "..."

        self._history.append({"role": role, "content": content})

        # Keep only the last MAX_TURNS messages (pairs of user+assistant)
        if len(self._history) > MAX_TURNS * 2:
            self._history = self._history[-(MAX_TURNS * 2):]

    def get_history(self) -> list[dict]:
        """Return a copy of the current history."""
        return list(self._history)

    def clear(self) -> None:
        self._history = []

    def __len__(self) -> int:
        return len(self._history)
