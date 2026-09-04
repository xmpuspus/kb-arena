"""Orchestrate strategy by question benchmark evaluation.

Enhanced with per-query timeouts, retry logic, latency percentiles,
and reliability tracking.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import AsyncIterator, Awaitable, Iterable
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kb_arena.benchmark.evaluator import evaluate
from kb_arena.benchmark.ir_metrics import compute_all as compute_ir_metrics
from kb_arena.benchmark.manifest import SCHEMA_VERSION, build_manifest, judge_provider_of
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
from kb_arena.strategies.catalog import default_strategy_names

console = Console()

STRATEGY_NAMES = list(default_strategy_names())
# Optional rerank_vector and sqr strategies are intentionally outside the
# default "all" set. Explicit requests receive a clear install hint when their
# extra is missing.

RETRY_BASE_S = 1.0  # base for exponential backoff: 1s, 2s, 4s, ...


class BenchmarkExecutionError(RuntimeError):
    """A benchmark could not produce a complete, trustworthy result."""


class BenchmarkIncompleteError(BenchmarkExecutionError):
    """A benchmark stopped at its budget boundary before all queries ran."""


_T = TypeVar("_T")


async def _as_completed_bounded(
    awaitables: Iterable[Awaitable[_T]], limit: int
) -> AsyncIterator[_T]:
    """Yield results as they complete without scheduling the full input at once."""
    if limit < 1:
        raise ValueError("concurrency limit must be at least 1")

    iterator = iter(awaitables)
    pending: set[asyncio.Future[_T]] = set()
    done: set[asyncio.Future[_T]] = set()

    def _fill() -> None:
        while len(pending) < limit:
            try:
                awaitable = next(iterator)
            except StopIteration:
                break
            pending.add(asyncio.ensure_future(awaitable))

    _fill()
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                yield task.result()
            _fill()
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, *done, return_exceptions=True)


def _is_retryable(exc: BaseException) -> bool:
    """Distinguish transient errors (rate limit, network, server 5xx, timeout)
    from configuration errors (auth, validation, missing model). Retrying the
    latter just burns wall-clock and credits without ever succeeding.
    """
    name = type(exc).__name__
    msg = str(exc).lower()

    if isinstance(exc, TimeoutError | asyncio.TimeoutError):
        return True

    # Match Anthropic, OpenAI, and generic SDK exception class names to
    # avoid hard-importing every SDK at runtime.
    transient = {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIError",  # ambiguous; status_code check below narrows it
        "InternalServerError",
        "ServiceUnavailableError",
    }
    permanent = {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
        "ValueError",
        "KeyError",
        "TypeError",
        "AttributeError",
    }
    if name in permanent:
        return False
    if name in transient:
        return True

    # Status code embedded in message (httpx common pattern).
    if any(code in msg for code in ("429", "500", "502", "503", "504", "timeout", "rate limit")):
        return True
    if any(code in msg for code in ("401", "403", "404", "400")):
        return False
    return False  # Fail fast on unknown errors instead of retrying forever.


def _classify_error(exc_or_message) -> str:
    """Categorize an error for stat aggregation. Returns a stable enum-like string."""
    if isinstance(exc_or_message, BaseException):
        if isinstance(exc_or_message, TimeoutError | asyncio.TimeoutError):
            return "timeout"
        msg = str(exc_or_message).lower()
        name = type(exc_or_message).__name__
    else:
        msg = str(exc_or_message).lower()
        name = ""
    if "timeout" in msg or "timed out" in msg or name in ("TimeoutError", "APITimeoutError"):
        return "timeout"
    if "rate limit" in msg or "429" in msg or name == "RateLimitError":
        return "rate_limit"
    if any(code in msg for code in ("connection", "connect", "dns", "resolve")):
        return "connection"
    if "auth" in msg or "401" in msg or "403" in msg or name == "AuthenticationError":
        return "auth"
    return "other"


def _load_strategies(strategy_filter: str) -> list[Strategy]:
    """Resolve a strategy filter into instantiated strategies.

    Accepts "all", a single name, or a comma-separated list ("naive_vector,qiss").
    Every requested strategy must initialize so the run cannot silently omit evidence.
    """
    from kb_arena.strategies import get_strategy

    if strategy_filter == "all":
        names = list(STRATEGY_NAMES)
    else:
        names = [n.strip() for n in strategy_filter.split(",") if n.strip()]

    active: list[Strategy] = []
    failures: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        try:
            active.append(get_strategy(name))
        except Exception as e:
            failures.append(f"{name}: {e}")
    if failures:
        raise BenchmarkExecutionError(
            "Requested strategies could not initialize: " + "; ".join(failures)
        )
    return active


async def _run_one(
    strategy: Strategy,
    question_id: str,
    question_text: str,
    ground_truth,
    constraints,
    expected_chunks: list[str],
    llm: LLMClient,
    semaphore: asyncio.Semaphore,
    top_k: int = 5,
    reference_free: bool = False,
    corpus: str = "all",
    review_status: str = "unspecified",
    reviewed_by: str = "",
) -> AnswerRecord:
    async with semaphore:
        attempt = 0
        last_error = ""
        last_exception: BaseException | None = None
        max_retries = settings.benchmark_max_retries
        query_timeout = settings.benchmark_query_timeout_s

        while attempt <= max_retries:
            attempt += 1
            t0 = time.perf_counter()
            try:
                query_call = (
                    strategy.query(question_text, top_k=top_k)
                    if corpus == "all"
                    else strategy.query(question_text, top_k=top_k, corpus=corpus)
                )
                result = await asyncio.wait_for(
                    query_call,
                    timeout=query_timeout,
                )
                if result.mock:
                    raise BenchmarkExecutionError(
                        f"{strategy.name}/{question_id} returned a mock result"
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
                    question_text=question_text,
                    context_chunks=(
                        [chunk.content for chunk in result.retrieval.retrieved if chunk.content]
                        if result.retrieval
                        else []
                    ),
                    reference_free=reference_free,
                )

                ir_metrics = None
                if result.retrieval and result.retrieval.retrieved:
                    ir_metrics = compute_ir_metrics(
                        retrieved=result.retrieval.retrieved,
                        expected_ids=set(expected_chunks),
                        k=top_k,
                        expected_doc_ids=set(ground_truth.source_refs),
                    )

                return AnswerRecord(
                    question_id=question_id,
                    question_review_status=review_status,
                    question_reviewed_by=reviewed_by,
                    strategy=strategy.name,
                    answer=answer,
                    score=score,
                    latency_ms=latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    tokens_used=tokens,
                    cost_usd=cost + score.evaluation_cost_usd,
                    generation_cost_usd=cost,
                    evaluation_cost_usd=score.evaluation_cost_usd,
                    sources=sources,
                    is_error=is_error,
                    is_empty=is_empty,
                    attempt_count=attempt,
                    response_length=len(answer) if answer else 0,
                    retrieval_metrics=ir_metrics,
                )

            except TimeoutError as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = f"Timeout after {query_timeout}s"
                last_exception = e
                if attempt <= max_retries:
                    delay = RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                _ = e  # keep reference for typing

            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                last_error = str(e)
                last_exception = e
                # Don't retry permanent errors (bad API key, missing model, etc.)
                # Burning 7 minutes per benchmark run on a typo is worse than failing fast.
                if attempt <= max_retries and _is_retryable(e):
                    delay = RETRY_BASE_S * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

        raise BenchmarkExecutionError(
            f"{strategy.name}/{question_id} failed after {attempt} attempt(s): {last_error}"
        ) from last_exception


def _aggregate(
    bench: BenchmarkResult,
    questions_map: dict[str, str | tuple[str, int]],
) -> BenchmarkResult:
    if not bench.records:
        return bench

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
        question_meta = questions_map.get(rec.question_id)
        if isinstance(question_meta, tuple):
            qtype, tier = question_meta
        else:
            qtype = question_meta or rec.question_type
            if rec.question_tier > 0:
                tier = rec.question_tier
            else:
                try:
                    tier = int(rec.question_id.split("-t")[1].split("-")[0])
                except (IndexError, ValueError):
                    tier = 0
        rec.question_tier = tier
        rec.question_type = qtype

        accuracy_by_tier.setdefault(tier, []).append(rec.score.accuracy)
        completeness_by_tier.setdefault(tier, []).append(rec.score.completeness)
        faithfulness_by_tier.setdefault(tier, []).append(rec.score.faithfulness)
        latency_by_tier.setdefault(tier, []).append(rec.latency_ms)

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

    bench.accuracy_by_tier = {t: sum(v) / len(v) for t, v in accuracy_by_tier.items()}
    bench.completeness_by_tier = {t: sum(v) / len(v) for t, v in completeness_by_tier.items()}
    bench.faithfulness_by_tier = {t: sum(v) / len(v) for t, v in faithfulness_by_tier.items()}
    bench.accuracy_by_type = {t: sum(v) / len(v) for t, v in accuracy_by_type.items()}

    bench.latency = LatencyStats.from_values(all_latencies)
    bench.avg_latency_ms = bench.latency.avg_ms
    bench.latency_by_tier = {t: LatencyStats.from_values(v) for t, v in latency_by_tier.items()}

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

    bench.total_cost_usd = total_cost
    bench.cost_per_correct = total_cost / correct if correct else 0.0
    bench.total_questions = n

    ir_records = [r for r in bench.records if r.retrieval_metrics]
    if ir_records:
        bench.ir_top_k = ir_records[0].retrieval_metrics.k
        denom = len(ir_records)
        bench.mean_recall_at_k = sum(r.retrieval_metrics.recall_at_k for r in ir_records) / denom
        bench.mean_precision_at_k = (
            sum(r.retrieval_metrics.precision_at_k for r in ir_records) / denom
        )
        bench.mean_hit_at_k = sum(r.retrieval_metrics.hit_at_k for r in ir_records) / denom
        bench.mean_mrr = sum(r.retrieval_metrics.mrr for r in ir_records) / denom
        bench.mean_ndcg_at_k = sum(r.retrieval_metrics.ndcg_at_k for r in ir_records) / denom

    return bench


class CostCapExceededError(Exception):
    """Raised when cumulative benchmark cost exceeds the configured cap."""


async def run_benchmark(
    corpus: str = "all",
    strategy: str = "all",
    tier: int = 0,
    split: str = "",
    parallel: bool = True,
    reference_free: bool = False,
    top_k: int = 5,
) -> None:
    """Run benchmark questions against specified strategies.

    Loads questions, calls each strategy x question concurrently (bounded by semaphore),
    evaluates with structural + entity coverage + source attribution + LLM judge,
    writes results/{corpus}_{strategy}.json.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    run_id = uuid4().hex[:8]
    timestamp = datetime.now(UTC).isoformat()
    cost_cap = settings.benchmark_cost_cap_usd
    if not math.isfinite(cost_cap) or cost_cap < 0:
        raise BenchmarkExecutionError("Benchmark cost cap must be finite and non-negative")
    config_snap = {
        "llm_provider": settings.llm_provider,
        "generate_model": settings.generate_model,
        "max_concurrent": settings.benchmark_max_concurrent,
        "query_timeout_s": settings.benchmark_query_timeout_s,
        "top_k": top_k,
        "question_split": split or "all",
        "reference_free": reference_free,
        "ragas_enabled": settings.benchmark_enable_ragas,
        "cost_cap_usd": cost_cap,
        "execution_mode": (
            "cost_capped_serial" if cost_cap > 0 else "parallel" if parallel else "serial"
        ),
    }

    llm = LLMClient()
    semaphore = asyncio.Semaphore(settings.benchmark_max_concurrent)
    results_dir = Path(settings.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    corpora = discover_corpora() if corpus == "all" else [corpus]
    strategies = _load_strategies(strategy)

    if not strategies:
        raise BenchmarkExecutionError(
            "No strategies available. Run build_vectors / build_graph first."
        )

    cumulative_total_cost = 0.0
    selected_questions = False

    console.print(f"[dim]Run ID: {run_id}[/dim]")
    if cost_cap > 0:
        console.print(f"[dim]Cost cap: ${cost_cap:.2f}[/dim]")
        if parallel:
            console.print(
                "[dim]Capped runs launch one query at a time so queued work stops at the cap.[/dim]"
            )

    for corp in corpora:
        try:
            questions = load_questions(corp, tier=tier, split=split)
        except FileNotFoundError:
            if corpus != "all":
                raise BenchmarkExecutionError(f"No questions for corpus: {corp}") from None
            console.print(f"[yellow]No questions for corpus: {corp}[/yellow]")
            continue

        if not questions:
            if corpus != "all":
                selected = f" for split {split!r}" if split else ""
                raise BenchmarkExecutionError(f"No questions selected for corpus {corp}{selected}")
            continue
        selected_questions = True

        questions_map = {q.id: (q.type, q.tier) for q in questions}
        manifest = build_manifest(
            corp, questions, top_k=top_k, split=split, reference_free=reference_free
        )

        def _write_result(bench: BenchmarkResult) -> None:
            # Latest (backward compat)
            latest_path = results_dir / f"{bench.corpus}_{bench.strategy}.json"
            latest_path.write_text(bench.model_dump_json(indent=2))
            # Timestamped run copy
            run_dir = results_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_path = run_dir / f"{bench.corpus}_{bench.strategy}.json"
            run_path.write_text(bench.model_dump_json(indent=2))

        if parallel and len(strategies) > 1 and cost_cap <= 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_ids: dict[str, object] = {}
                for strat in strategies:
                    tid = progress.add_task(f"[cyan]{strat.name}", total=len(questions))
                    task_ids[strat.name] = tid

                async def _run_strategy_parallel(strat: Strategy) -> BenchmarkResult:
                    bench = BenchmarkResult(
                        corpus=corp,
                        strategy=strat.name,
                        run_id=run_id,
                        timestamp=timestamp,
                        config_snapshot=config_snap,
                        schema_version=SCHEMA_VERSION,
                        judge_provider=judge_provider_of(manifest),
                        manifest=manifest,
                    )
                    coros = (
                        _run_one(
                            strat,
                            q.id,
                            q.question,
                            q.ground_truth,
                            q.constraints,
                            q.expected_chunks,
                            llm,
                            semaphore,
                            top_k=top_k,
                            reference_free=reference_free,
                            corpus=corp,
                            review_status=q.review_status,
                            reviewed_by=q.reviewed_by,
                        )
                        for q in questions
                    )
                    async for rec in _as_completed_bounded(
                        coros, settings.benchmark_max_concurrent
                    ):
                        bench.records.append(rec)
                        progress.advance(task_ids[strat.name])
                    bench = _aggregate(bench, questions_map)
                    return bench

                results_list = await asyncio.gather(
                    *[_run_strategy_parallel(s) for s in strategies]
                )

            cumulative_cost = 0.0
            for bench in results_list:
                _write_result(bench)
                overall_acc = (
                    sum(bench.accuracy_by_tier.values()) / len(bench.accuracy_by_tier)
                    if bench.accuracy_by_tier
                    else 0.0
                )
                cumulative_cost += bench.total_cost_usd
                cumulative_total_cost += bench.total_cost_usd
                console.print(
                    f"  {bench.strategy}: {len(bench.records)} questions, "
                    f"acc={overall_acc:.1%}, "
                    f"${bench.total_cost_usd:.4f}, "
                    f"avg {bench.avg_latency_ms:.0f}ms"
                )
            if cost_cap > 0 and cumulative_total_cost >= cost_cap:
                console.print(
                    f"[red]Cost cap exceeded: ${cumulative_total_cost:.4f} >= "
                    f"${cost_cap:.2f}. Halting benchmark.[/red]"
                )
                raise BenchmarkIncompleteError(
                    f"Cost cap reached at ${cumulative_total_cost:.4f} before the run completed"
                )
        else:
            # Sequential path
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                total_tasks = len(strategies) * len(questions)
                task = progress.add_task(f"[cyan]{corp}", total=total_tasks)
                cumulative_cost = 0.0

                for strat in strategies:
                    bench = BenchmarkResult(
                        corpus=corp,
                        strategy=strat.name,
                        run_id=run_id,
                        timestamp=timestamp,
                        config_snapshot=config_snap,
                        schema_version=SCHEMA_VERSION,
                        judge_provider=judge_provider_of(manifest),
                        manifest=manifest,
                    )

                    async def _run_question(q):
                        return await _run_one(
                            strat,
                            q.id,
                            q.question,
                            q.ground_truth,
                            q.constraints,
                            q.expected_chunks,
                            llm,
                            semaphore,
                            top_k=top_k,
                            reference_free=reference_free,
                            corpus=corp,
                            review_status=q.review_status,
                            reviewed_by=q.reviewed_by,
                        )

                    if cost_cap > 0:
                        records = []
                        for question in questions:
                            records.append(await _run_question(question))
                            if cumulative_total_cost + sum(r.cost_usd for r in records) >= cost_cap:
                                break
                    else:
                        records = []
                        coros = (_run_question(question) for question in questions)
                        async for record in _as_completed_bounded(
                            coros, settings.benchmark_max_concurrent
                        ):
                            records.append(record)

                    for rec in records:
                        bench.records.append(rec)
                        cumulative_cost += rec.cost_usd
                        cumulative_total_cost += rec.cost_usd
                        progress.update(
                            task,
                            description=f"[cyan]{corp} [dim]${cumulative_cost:.4f}[/dim]",
                        )
                        progress.advance(task)

                        if cost_cap > 0 and cumulative_total_cost >= cost_cap:
                            console.print(
                                f"\n[red]Cost cap reached: ${cumulative_total_cost:.4f} >= "
                                f"${cost_cap:.2f}. Halting benchmark.[/red]"
                            )
                            bench.stopped_by_cost_cap = True
                            bench = _aggregate(bench, questions_map)
                            _write_result(bench)
                            raise BenchmarkIncompleteError(
                                f"Cost cap reached at ${cumulative_total_cost:.4f} "
                                "before the run completed"
                            )

                    bench = _aggregate(bench, questions_map)
                    _write_result(bench)

                    overall_acc = (
                        sum(bench.accuracy_by_tier.values()) / len(bench.accuracy_by_tier)
                        if bench.accuracy_by_tier
                        else 0.0
                    )
                    console.print(
                        f"  {strat.name}: {len(bench.records)} questions, "
                        f"acc={overall_acc:.1%}, "
                        f"${bench.total_cost_usd:.4f}, "
                        f"avg {bench.avg_latency_ms:.0f}ms"
                    )

        console.print(
            f"[green]Done {corp}:[/green] {len(strategies)} strategies, "
            f"${cumulative_cost:.4f} total"
        )

    if not selected_questions:
        raise BenchmarkExecutionError("No benchmark questions were selected")
