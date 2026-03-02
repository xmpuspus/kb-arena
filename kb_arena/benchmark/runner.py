"""Benchmark runner — stub for benchmark agent."""

from __future__ import annotations


async def run_benchmark(
    corpus: str = "all", strategy: str = "all", tier: int = 0
) -> None:
    """Run benchmark questions against specified strategies."""
    raise NotImplementedError("Benchmark runner will be implemented by benchmark agent")
