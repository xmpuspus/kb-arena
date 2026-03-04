"""Parser protocol — all parsers must implement this interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from kb_arena.models.document import Document


@runtime_checkable
class Parser(Protocol):
    """Protocol for document parsers.

    Every parser must implement parse(path, corpus) -> list[Document].
    The path is a local file; the corpus is the target corpus name.
    """

    def parse(self, path: Path, corpus: str) -> list[Document]: ...
