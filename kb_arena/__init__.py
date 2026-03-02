"""KB Arena — Benchmark knowledge graphs vs vector RAG on real documentation."""

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

__version__ = "0.1.0"
