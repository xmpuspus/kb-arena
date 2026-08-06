"""Compare retrieval architectures and choose with reproducible evidence."""

import os as _os

# ChromaDB 0.5.x can emit failing telemetry callbacks from local-only clients.
# Set one value before any strategy constructs a shared Chroma system.
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from kb_arena.models.benchmark import BenchmarkResult, Question
from kb_arena.models.document import Document, Section
from kb_arena.models.graph import Entity, Relationship
from kb_arena.strategies.base import Strategy

__all__ = [
    "Document",
    "Section",
    "Entity",
    "Relationship",
    "Question",
    "BenchmarkResult",
    "Strategy",
]

try:
    from importlib.metadata import version

    __version__ = version("kb-arena")
except Exception:
    __version__ = "0.1.0"
