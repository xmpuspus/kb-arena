"""Ingestion pipeline orchestrator — stub for ingest agent."""

from __future__ import annotations


def run_ingest(path: str, corpus: str = "custom", format: str = "auto") -> None:
    """Parse raw documents and write JSONL to datasets/{corpus}/processed/."""
    raise NotImplementedError("Ingest pipeline will be implemented by ingest agent")
