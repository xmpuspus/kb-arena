"""Retrieval strategies — 5 approaches to answering questions from documentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


async def build_vector_indexes(corpus: str = "all", strategy: str = "all") -> None:
    """Build vector indexes for the specified strategies.

    Called by the CLI build-vectors command.
    """
    raise NotImplementedError("Vector index building will be implemented by strategies agent")
