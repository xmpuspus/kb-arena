"""Benchmark runner — orchestrates strategy x question evaluation.

Enhanced with per-query timeouts, retry logic, latency percentiles,
and reliability tracking.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kb_arena.benchmark.evaluator import evaluate
from kb_arena.benchmark.questions import discover_corpora, load_questions
from kb_arena.llm.client import LLMClient
from kb_arena.models.benchmark import (
    AnswerRecord,
    BenchmarkResult,
    LatencyStats,
    ReliabilityStats,
)
from kb_arena.settings import settings
from kb_arena.strategies.base import Strategy

console = Console()

STRATEGY_NAMES = ["naive_vector", "contextual_vector", "qna_pairs", "knowledge_graph", "hybrid"]

RETRY_BACKOFF_S = 1.0


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
        attempt = 0
        last_error = ""
        max_retries = settings.benchmark_max_retries
        query_timeout = settings.benchmark_query_timeout_s

        while attempt <= max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    strategy.query(question_text),
                    timeout=query_timeout,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                answer = result.answer
                sources = result.sources
                tokens = result.tokens_used
                cost = result.cost_usd
                retrieval_latency_ms = result.retrieval_latency_ms
                generation_latency_ms = result.generation_latency_ms

                is_empty = not answer or not answer.strip()
                is_error = answer.startswith("[ERROR]") if answer else True

                score = await evaluate(
                    answer,
                    ground_truth,
                    constraints,
                    sources=sources,
                    llm=llm,
                )

                return AnswerRecord(
                    question_id=question_id,
                    strategy=strategy.name,
                    answer=answer,
                    score=score,
                    latency_ms=latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    tokens_used=tokens,
                    cost_usd=cost,
                    sources=sources,
                    is_error=is_error,
                    is_empty=is_empty,
                    attempt_count=attempt,
                    response_length=len(answer) if answer else 0,
                )

            except TimeoutError:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = f"Timeout after {query_timeout}s"
                if attempt <= max_retries:
                    await asyncio.sleep(RETRY_BACKOFF_S * attempt)
                    continue

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = str(e)
                if attempt <= max_retries:
                    await asyncio.sleep(RETRY_BACKOFF_S * attempt)
                    continue

        # All retries exhausted
        from kb_arena.models.benchmark import Score

        error_score = Score(accuracy=0.0, completeness=0.0, faithfulness=0.0)

        return AnswerRecord(
            question_id=question_id,
            strategy=strategy.name,
            answer=f"[ERROR] {last_error}",
            score=error_score,
            latency_ms=latency_ms,
            is_error=True,
            is_empty=True,
            error_message=last_error,
            attempt_count=attempt,
            response_length=0,
        )


def _aggregate(
    bench: BenchmarkResult,
    questions_map: dict[str, str],
) -> BenchmarkResult:
    """Compute all aggregate metrics from individual records.

    Pure computation — no mutation of running totals.
    """
    if not bench.records:
        return bench

    # Buckets for per-tier and per-type aggregation
    accuracy_by_tier: dict[int, list[float]] = {}
    completeness_by_tier: dict[int, list[float]] = {}
    faithfulness_by_tier: dict[int, list[float]] = {}
    latency_by_tier: dict[int, list[float]] = {}
    accuracy_by_type: dict[str, list[float]] = {}

    all_latencies: list[float] = []
    total_cost = 0.0
    correct = 0
    error_count = 0
    empty_count = 0
    timeout_count = 0
    faithfulness_values: list[float] = []
    source_attr_values: list[float] = []
    entity_cov_values: list[float] = []
    response_lengths: list[int] = []

    for rec in bench.records:
        try:
            tier = int(rec.question_id.split("-t")[1].split("-")[0])
        except (IndexError, ValueError):
            tier = 0
        qtype = questions_map.get(rec.question_id, "unknown")

        # Per-tier accuracy
        accuracy_by_tier.setdefault(tier, []).append(rec.score.accuracy)
        completeness_by_tier.setdefault(tier, []).append(rec.score.completeness)
        faithfulness_by_tier.setdefault(tier, []).append(rec.score.faithfulness)
        latency_by_tier.setdefault(tier, []).append(rec.latency_ms)

        # Per-type accuracy
        accuracy_by_type.setdefault(qtype, []).append(rec.score.accuracy)

        all_latencies.append(rec.latency_ms)
        total_cost += rec.cost_usd

        if rec.score.accuracy >= 0.7:
            correct += 1

        if rec.is_error:
            error_count += 1
        if rec.is_empty:
            empty_count += 1
        if "Timeout" in rec.error_message:
            timeout_count += 1

        faithfulness_values.append(rec.score.faithfulness)
        source_attr_values.append(rec.score.source_attribution)
        entity_cov_values.append(rec.score.entity_coverage)
        response_lengths.append(rec.response_length)

    n = len(bench.records)
    successful = n - error_count

    # Accuracy by tier
    bench.accuracy_by_tier = {t: sum(v) / len(v) for t, v in accuracy_by_tier.items()}
    bench.completeness_by_tier = {t: sum(v) / len(v) for t, v in completeness_by_tier.items()}
    bench.faithfulness_by_tier = {t: sum(v) / len(v) for t, v in faithfulness_by_tier.items()}
    bench.accuracy_by_type = {t: sum(v) / len(v) for t, v in accuracy_by_type.items()}

    # Latency
    bench.latency = LatencyStats.from_values(all_latencies)
    bench.avg_latency_ms = bench.latency.avg_ms
    bench.latency_by_tier = {t: LatencyStats.from_values(v) for t, v in latency_by_tier.items()}

    # Reliability
    bench.reliability = ReliabilityStats(
        total_queries=n,
        successful_queries=successful,
        error_count=error_count,
        empty_count=empty_count,
        timeout_count=timeout_count,
        error_rate=error_count / n if n else 0.0,
        empty_rate=empty_count / n if n else 0.0,
        success_rate=successful / n if n else 0.0,
        avg_faithfulness=sum(faithfulness_values) / n if n else 0.0,
        avg_source_attribution=sum(source_attr_values) / n if n else 0.0,
        avg_entity_coverage=sum(entity_cov_values) / n if n else 0.0,
        avg_response_length=sum(response_lengths) / n if n else 0.0,
    )

    # Cost
    bench.total_cost_usd = total_cost
    bench.cost_per_correct = total_cost / correct if correct else 0.0
    bench.total_questions = n

    return bench


async def run_benchmark(
    corpus: str = "all",
    strategy: str = "all",
    tier: int = 0,
) -> None:
    """Run benchmark questions against specified strategies.

    Loads questions, calls each strategy x question concurrently (bounded by semaphore),
    evaluates with structural + entity coverage + source attribution + LLM judge,
    writes results/{corpus}_{strategy}.json.
    """
    llm = LLMClient()
    semaphore = asyncio.Semaphore(settings.benchmark_max_concurrent)
    results_dir = Path(settings.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    corpora = discover_corpora() if corpus == "all" else [corpus]
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

            # Build question type map for aggregation
            questions_map = {q.id: q.type for q in questions}

            total_tasks = len(strategies) * len(questions)
            task = progress.add_task(f"[cyan]{corp}", total=total_tasks)

            for strat in strategies:
                bench = BenchmarkResult(corpus=corp, strategy=strat.name)

                coros = [
                    _run_one(strat, q.id, q.question, q.ground_truth, q.constraints, llm, semaphore)
                    for q in questions
                ]

                for coro in asyncio.as_completed(coros):
                    rec = await coro
                    bench.records.append(rec)
                    progress.advance(task)

                bench = _aggregate(bench, questions_map)

                out_path = results_dir / f"{corp}_{strat.name}.json"
                out_path.write_text(bench.model_dump_json(indent=2))
                console.print(f"[green]Wrote {out_path}[/green]")
