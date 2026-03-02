"""Benchmark reporter — stub for benchmark agent."""

from __future__ import annotations


def generate_report(corpus: str = "all", output: str | None = None) -> None:
    """Generate markdown report from results JSON."""
    raise NotImplementedError("Reporter will be implemented by benchmark agent")
