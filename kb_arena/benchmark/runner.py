"""Benchmark runner — orchestrates strategy × question evaluation."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kb_arena.benchmark.evaluator import evaluate
from kb_arena.benchmark.questions import load_questions
from kb_arena.llm.client import LLMClient
from kb_arena.models.benchmark import AnswerRecord, BenchmarkResult
from kb_arena.settings import settings
from kb_arena.strategies.base import Strategy

console = Console()

STRATEGY_NAMES = ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]
CORPUS_NAMES = ["python-stdlib", "kubernetes", "sec-edgar"]


def _load_strategies(strategy_filter: str) -> list[Strategy]:
    from kb_arena.strategies import get_strategy

    names = STRATEGY_NAMES if strategy_filter == "all" else [strategy_filter]
    active = []
    for name in names:
        try:
            s = get_strategy(name)
            active.append(s)
        except Exception as e:
            console.print(f"[yellow]Skipping strategy {name}: {e}[/yellow]")
    return active


async def _run_one(
    strategy: Strategy,
    question_id: str,
    question_text: str,
    ground_truth,
    constraints,
    llm: LLMClient,
    semaphore: asyncio.Semaphore,
) -> AnswerRecord:
    async with semaphore:
        t0 = time.perf_counter()
        try:
            result = await strategy.query(question_text)
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = result.answer
            sources = result.sources
            tokens = result.tokens_used
            cost = result.cost_usd
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = f"[ERROR] {e}"
            sources = []
            tokens = 0
            cost = 0.0

        score = await evaluate(answer, ground_truth, constraints, llm=llm)

        return AnswerRecord(
            question_id=question_id,
            strategy=strategy.name,
            answer=answer,
            score=score,
            latency_ms=latency_ms,
            tokens_used=tokens,
            cost_usd=cost,
            sources=sources,
        )


def _aggregate(bench: BenchmarkResult) -> BenchmarkResult:
    """Compute per-tier accuracy, avg latency, cost_per_correct."""
    by_tier: dict[int, list[float]] = {}
    total_latency = 0.0
    correct = 0

    for rec in bench.records:
        # Parse tier from question_id like "py-t1-001" or "k8s-t3-012"
        try:
            tier = int(rec.question_id.split("-t")[1].split("-")[0])
        except (IndexError, ValueError):
            tier = 0
        by_tier.setdefault(tier, []).append(rec.score.accuracy)
        total_latency += rec.latency_ms
        bench.total_cost_usd += rec.cost_usd
        if rec.score.accuracy >= 0.7:
            correct += 1

    bench.accuracy_by_tier = {t: sum(v) / len(v) for t, v in by_tier.items()}
    bench.avg_latency_ms = total_latency / len(bench.records) if bench.records else 0.0
    bench.cost_per_correct = bench.total_cost_usd / correct if correct else 0.0
    bench.total_questions = len(bench.records)
    return bench


async def run_benchmark(
    corpus: str = "all",
    strategy: str = "all",
    tier: int = 0,
) -> None:
    """Run benchmark questions against specified strategies.

    Loads questions, calls each strategy × question concurrently (bounded by semaphore),
    evaluates with structural + LLM judge, writes results/{corpus}_{strategy}.json.
    """
    llm = LLMClient()
    semaphore = asyncio.Semaphore(settings.benchmark_max_concurrent)
    results_dir = Path(settings.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    corpora = CORPUS_NAMES if corpus == "all" else [corpus]
    strategies = _load_strategies(strategy)

    if not strategies:
        console.print("[red]No strategies available. Run build_vectors / build_graph first.[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for corp in corpora:
            try:
                questions = load_questions(corp, tier=tier)
            except FileNotFoundError:
                console.print(f"[yellow]No questions for corpus: {corp}[/yellow]")
                continue

            if not questions:
                continue

            total_tasks = len(strategies) * len(questions)
            task = progress.add_task(f"[cyan]{corp}", total=total_tasks)

            for strat in strategies:
                bench = BenchmarkResult(corpus=corp, strategy=strat.name)

                coros = [
                    _run_one(
                        strat, q.id, q.question, q.ground_truth, q.constraints, llm, semaphore
                    )
                    for q in questions
                ]

                for coro in asyncio.as_completed(coros):
                    rec = await coro
                    bench.records.append(rec)
                    progress.advance(task)

                bench = _aggregate(bench)

                out_path = results_dir / f"{corp}_{strat.name}.json"
                out_path.write_text(bench.model_dump_json(indent=2))
                console.print(f"[green]Wrote {out_path}[/green]")
