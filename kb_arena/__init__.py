"""Compare retrieval architectures and choose with reproducible evidence."""

import os as _os
import tomllib as _tomllib
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
from pathlib import Path as _Path

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


def _resolve_version() -> str:
    try:
        return _distribution_version("kb-arena")
    except _PackageNotFoundError:
        manifest = _Path(__file__).resolve().parents[1] / "pyproject.toml"
        try:
            return _tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["version"]
        except (KeyError, OSError, _tomllib.TOMLDecodeError):
            return "unknown"


__version__ = _resolve_version()
