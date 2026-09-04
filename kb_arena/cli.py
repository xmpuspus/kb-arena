"""KB Arena CLI for a multi-stage retrieval comparison pipeline.

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="kb-arena",
    help="Compare retrieval architectures on your documentation with reproducible evidence.",
    no_args_is_help=True,
)
console = Console()


def _print_version(value: bool) -> None:
    if value:
        from kb_arena import __version__

        console.print(f"kb-arena {__version__}")
        raise typer.Exit()


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    version: bool = typer.Option(  # noqa: ARG001 - handled by callback
        False,
        "--version",
        "-V",
        help="Print kb-arena version and exit.",
        callback=_print_version,
        is_eager=True,
    ),
) -> None:
    from kb_arena.logging_config import configure_logging, warn_unknown_env
    from kb_arena.settings import settings

    level = logging.DEBUG if verbose else settings.log_level
    handler = RichHandler(rich_tracebacks=True, show_path=verbose)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    configure_logging(level, handler=handler)
    warn_unknown_env()


# Pipeline: init-corpus -> ingest -> build-graph/build-vectors ->
# generate-questions -> benchmark -> report -> serve
_PIPELINE_NEXT: dict[str, str] = {
    "ingest": "kb-arena build-graph --corpus {corpus} && kb-arena build-vectors --corpus {corpus}",
    "build_graph": "kb-arena build-vectors --corpus {corpus}",
    "build_vectors": "kb-arena generate-questions --corpus {corpus} --count 50",
    "generate_questions": "kb-arena benchmark --corpus {corpus}",
    "benchmark": "kb-arena report --corpus {corpus}",
    "report": "kb-arena serve",
}


def _next_step(command: str, corpus: str = "") -> None:
    hint = _PIPELINE_NEXT.get(command)
    if hint:
        console.print(f"\nNext: [bold]{hint.format(corpus=corpus)}[/bold]")


def _cli_error(code: str, message: str, fmt: str = "rich") -> None:
    if fmt == "json":
        import json
        import sys

        sys.stderr.write(json.dumps({"error": {"code": code, "message": message}}) + "\n")
    else:
        console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


def _validate_unit_interval(value: float, option: str) -> None:
    import math

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        _cli_error(
            "BAD_THRESHOLD",
            f"{option} must be a finite number between 0 and 1.",
        )


def _preflight(
    needs_llm: bool = False,
    needs_embeddings: bool = False,
    needs_neo4j: bool = False,
) -> None:
    """Verify that the credentials this command actually needs are configured.

    Generation and embedding providers are independent. Local Ollama and BGE
    providers need no API key.
    """
    from kb_arena.settings import settings

    errors: list[str] = []
    if needs_llm:
        llm_provider = settings.llm_provider.lower()
        if llm_provider == "anthropic" and not (settings.llm_api_key or settings.anthropic_api_key):
            errors.append(
                "Anthropic API key required for generation. Set "
                "KB_ARENA_ANTHROPIC_API_KEY, or use KB_ARENA_LLM_PROVIDER=ollama."
            )
        elif llm_provider == "openai" and not (settings.llm_api_key or settings.openai_api_key):
            errors.append(
                "OpenAI API key required for generation. Set KB_ARENA_OPENAI_API_KEY, "
                "or use KB_ARENA_LLM_PROVIDER=ollama."
            )
        elif llm_provider not in {"anthropic", "openai", "ollama"}:
            errors.append(
                f"Unknown KB_ARENA_LLM_PROVIDER={llm_provider!r}. "
                "Valid: anthropic, ollama, openai."
            )

    if needs_embeddings:
        embedding_provider = settings.embedding_provider.lower()
        embedding_keys = {
            "openai": settings.openai_api_key,
            "voyage": settings.voyage_api_key,
            "cohere": settings.cohere_api_key,
            "gemini": settings.gemini_api_key,
        }
        if embedding_provider in embedding_keys and not embedding_keys[embedding_provider]:
            env_name = f"KB_ARENA_{embedding_provider.upper()}_API_KEY"
            errors.append(
                f"{embedding_provider.title()} API key required for embeddings. "
                f"Set {env_name}, or use KB_ARENA_EMBEDDING_PROVIDER=bge or ollama."
            )
        elif embedding_provider not in {*embedding_keys, "bge", "ollama"}:
            errors.append(
                f"Unknown KB_ARENA_EMBEDDING_PROVIDER={embedding_provider!r}. "
                "Valid: bge, cohere, gemini, ollama, openai, voyage."
            )
    if errors:
        for e in errors:
            console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@app.command()
def ingest(
    path: str = typer.Argument(
        ...,
        help="Path, URL, or github:owner/repo to ingest",
    ),
    corpus: str = typer.Option("custom", help="Corpus name (e.g. aws-compute, my-docs)"),
    format: str = typer.Option(
        "auto",
        help="Parser: auto, markdown, html, sec-edgar, pdf, docx, plaintext, web, csv, github",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be ingested"),
):
    """Stage 1: Parse raw documents into unified Document model.

    Supports local files/dirs, URLs (auto-detected), and github:owner/repo.
    Writes JSONL to datasets/{corpus}/processed/
    """
    from collections import Counter
    from pathlib import Path

    from kb_arena.ingest.pipeline import _EXT_MAP, is_http_url
    from kb_arena.settings import settings

    detected_format = format
    if format == "auto":
        if is_http_url(path):
            detected_format = "web"
        elif path.startswith("github:"):
            detected_format = "github"

    if dry_run:
        console.print(f"[bold]Dry run: ingest {path} --corpus {corpus}[/bold]\n")
        if detected_format in ("web", "github"):
            console.print(f"  Source type: {detected_format}")
            console.print("  Dry run not supported for web/github sources")
            return
        src = Path(path)
        if not src.exists():
            console.print(f"[red]  Path does not exist: {src}[/red]")
            raise typer.Exit(1)
        supported_exts = set(_EXT_MAP.keys())
        if src.is_file():
            files = [src]
        else:
            files = [
                f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in supported_exts
            ]
        ext_counts = Counter(f.suffix.lower() for f in files)
        console.print(f"  Files found: {len(files)}")
        for ext, count in sorted(ext_counts.items()):
            parser = _EXT_MAP.get(ext, "unknown")
            console.print(f"    {ext}: {count} ({parser} parser)")
        out_path = Path(settings.datasets_path) / corpus / "processed" / "documents.jsonl"
        console.print(f"  Output: {out_path}")
        console.print("\n  Remove --dry-run to execute.")
        return

    if detected_format in ("web", "github"):
        from kb_arena.ingest.pipeline import run_ingest_special

        ingested = run_ingest_special(source=path, corpus=corpus, format=detected_format)
    else:
        from kb_arena.ingest.pipeline import run_ingest

        ingested = run_ingest(path=path, corpus=corpus, format=format)

    if ingested <= 0:
        console.print("[red]Ingestion produced no documents.[/red]")
        raise typer.Exit(1)

    _next_step("ingest", corpus)


@app.command()
def build_graph(
    corpus: str = typer.Option(..., help="Corpus to build graph for"),
    schema: str = typer.Option("auto", help="Schema: auto"),
):
    """Stage 2: Extract entities/relationships, build Neo4j graph.

    Requires: ingest completed. Writes to Neo4j.
    """
    import asyncio

    _preflight(needs_llm=True)

    from kb_arena.graph.extractor import run_extraction

    asyncio.run(run_extraction(corpus=corpus, schema=schema))

    _next_step("build_graph", corpus)


@app.command()
def migrate_graph_schema(
    database: str = typer.Option(
        ...,
        "--database",
        help="Exact Neo4j database to migrate; also set KB_ARENA_NEO4J_DATABASE to this value.",
    ),
    confirm_dedicated_database: bool = typer.Option(
        False,
        "--confirm-dedicated-database",
        help="Confirm that the target Neo4j database is dedicated to KB Arena.",
    ),
):
    """Replace pre-0.10 graph constraints after explicit operator confirmation."""
    import asyncio

    if not database.strip():
        console.print("[red]Migration not run. --database must not be empty.[/red]")
        raise typer.Exit(1)
    if not confirm_dedicated_database:
        console.print(
            "[red]Migration not run. Use a Neo4j database dedicated to KB Arena, "
            "then pass --confirm-dedicated-database.[/red]"
        )
        raise typer.Exit(1)

    from kb_arena.graph.extractor import migrate_legacy_graph_schema

    dropped = asyncio.run(migrate_legacy_graph_schema(database.strip()))
    if dropped:
        console.print(f"[green]Migrated graph schema.[/green] Dropped: {', '.join(dropped)}")
    else:
        console.print("[green]Graph schema is current.[/green] No legacy constraints found.")


@app.command()
def build_vectors(
    corpus: str = typer.Option(..., help="Corpus to build vectors for"),
    strategy: str = typer.Option(
        "all", help="Strategy: all, naive_vector, contextual_vector, qna_pairs, raptor, pageindex"
    ),
):
    """Stage 3: Build indexes for vector strategies and PageIndex tree.

    Requires: ingest completed. Writes to ChromaDB (vector) or JSON (pageindex).
    """
    import asyncio

    llm_build_strategies = {"all", "contextual_vector", "qna_pairs", "raptor", "pageindex"}
    embedding_build_strategies = {
        "all",
        "naive_vector",
        "contextual_vector",
        "qna_pairs",
        "raptor",
        "qiss",
        "sqr",
    }
    _preflight(
        needs_llm=strategy in llm_build_strategies,
        needs_embeddings=strategy in embedding_build_strategies,
    )

    from kb_arena.strategies import build_vector_indexes

    asyncio.run(build_vector_indexes(corpus=corpus, strategy=strategy))

    _next_step("build_vectors", corpus)


@app.command()
def benchmark(
    corpus: str = typer.Option("all", help="Corpus name, or 'all' to run all discovered corpora"),
    strategy: str = typer.Option(
        "all",
        help="Strategy name or 'all'. Options: naive_vector, contextual_vector, "
        "qna_pairs, knowledge_graph, hybrid, raptor, pageindex",
    ),
    tier: int = typer.Option(0, help="Tier filter (0 = all tiers)"),
    split: str = typer.Option(
        "", "--split", help="Question split: development, validation, holdout, or all"
    ),
    parallel: bool = typer.Option(
        True, "--parallel/--no-parallel", help="Run strategies in parallel"
    ),
    fail_below: float = typer.Option(
        0.0, "--fail-below", help="Exit code 1 if accuracy below threshold (0.0-1.0)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be benchmarked"),
    reference_free: bool = typer.Option(
        False,
        "--reference-free",
        help="Evaluate on faithfulness + relevancy only (no ground truth)",
    ),
    ragas: bool = typer.Option(
        False, "--ragas", help="Enable RAGAS metrics (faithfulness, precision, recall, relevancy)"
    ),
    strategy_module: str = typer.Option(
        "",
        "--strategy-module",
        help="Import path for a custom Strategy plugin (e.g. my_pkg.my_strat)",
    ),
    top_k: int = typer.Option(5, "--top-k", help="Top-k chunks per query (drives IR metrics)"),
):
    """Stage 4: Run benchmark questions against specified strategies.

    Writes results to results/{corpus}.json
    """
    import asyncio

    _validate_unit_interval(fail_below, "--fail-below")

    if strategy_module:
        from kb_arena.strategies import register_plugin_strategy

        register_plugin_strategy(strategy_module)

    if dry_run:
        from kb_arena.benchmark.questions import discover_corpora, load_questions
        from kb_arena.benchmark.runner import STRATEGY_NAMES
        from kb_arena.settings import settings

        corpora = discover_corpora() if corpus == "all" else [corpus]
        strategy_names = STRATEGY_NAMES if strategy == "all" else [strategy]

        console.print("[bold]Dry run: benchmark[/bold]\n")
        total_queries = 0
        for corp in corpora:
            try:
                questions = load_questions(corp, tier=tier, split=split)
            except FileNotFoundError:
                console.print(f"  [yellow]{corp}: no questions found[/yellow]")
                continue
            n = len(questions)
            corp_queries = n * len(strategy_names)
            total_queries += corp_queries
            console.print(
                f"  {corp}: {n} questions x {len(strategy_names)} "
                f"strategies = {corp_queries} queries"
            )
        console.print(f"\n  Strategies: {', '.join(strategy_names)}")
        console.print(f"  Total queries: {total_queries}")
        console.print(f"  Max concurrency: {settings.benchmark_max_concurrent}")
        console.print(f"  Timeout per query: {settings.benchmark_query_timeout_s}s")

        # Cost/time estimates
        est_cost_per_query = 0.003  # ~$0.003 per query (Haiku eval + Sonnet gen avg)
        est_judge_cost = 0.005  # ~$0.005 per LLM judge call (Opus)
        est_cost = total_queries * (est_cost_per_query + est_judge_cost)
        avg_seconds_per_query = 4.5
        est_parallel = settings.benchmark_max_concurrent
        est_time_s = (total_queries / est_parallel) * avg_seconds_per_query
        est_minutes = est_time_s / 60

        console.print(f"\n  [bold]Estimated cost:[/bold] ~${est_cost:.2f}")
        console.print(f"  [bold]Estimated time:[/bold] ~{est_minutes:.0f} min")
        console.print(
            "  [dim](estimates assume Anthropic provider, actual cost varies by strategy)[/dim]"
        )
        console.print("\n  Remove --dry-run to execute.")
        return

    _preflight(needs_llm=True, needs_embeddings=True)

    if ragas or reference_free:
        from kb_arena.settings import settings as _settings

        _settings.benchmark_enable_ragas = True

    from kb_arena.benchmark.runner import BenchmarkExecutionError, run_benchmark

    try:
        asyncio.run(
            run_benchmark(
                corpus=corpus,
                strategy=strategy,
                tier=tier,
                split=split,
                parallel=parallel,
                reference_free=reference_free,
                top_k=top_k,
            )
        )
    except BenchmarkExecutionError as exc:
        console.print(f"[red]Benchmark failed: {exc}[/red]")
        raise typer.Exit(1) from None

    if fail_below > 0:
        from kb_arena.benchmark.reporter import _load_results

        all_results = _load_results(corpus if corpus != "all" else None)
        failed = False
        for r in all_results:
            if r.accuracy_by_tier:
                avg = sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
                if avg < fail_below:
                    console.print(
                        f"[red]FAIL: {r.strategy} accuracy {avg:.1%} < {fail_below:.1%}[/red]"
                    )
                    failed = True
        if failed:
            raise typer.Exit(1)
        console.print(f"[green]PASS: All strategies above {fail_below:.1%}[/green]")

    _next_step("benchmark", corpus)


_REPORT_FORMATS = ("rich", "json", "csv", "html", "markdown")


@app.command()
def report(
    corpus: str = typer.Option("all", help="Corpus to generate report for"),
    output: str | None = typer.Option(None, help="Output file path"),  # noqa: UP045
    format: str = typer.Option("rich", help="Output format: rich, json, csv, html, markdown"),
):
    """Generate benchmark report from results JSON."""
    if format not in _REPORT_FORMATS:
        _cli_error(
            "UNKNOWN_FORMAT",
            f"Unknown format '{format}'. Choose one of: {', '.join(_REPORT_FORMATS)}.",
        )

    if format == "markdown":
        from kb_arena.benchmark.reporter import generate_report

        generate_report(corpus=corpus, output=output)
        _next_step("report")
        return

    if format == "json":
        import json
        import sys

        from kb_arena.benchmark.reporter import _build_summary, _load_results

        results = _load_results(corpus)
        if not results:
            _cli_error("NO_RESULTS", "No results found. Run benchmark first.", fmt="json")
        summary = _build_summary(results)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return

    if format == "csv":
        from pathlib import Path

        from kb_arena.benchmark.reporter import _build_csv, _load_results
        from kb_arena.settings import settings

        results = _load_results(corpus)
        if not results:
            _cli_error("NO_RESULTS", "No results found. Run benchmark first.")
        csv_text = _build_csv(results)
        if output:
            Path(output).write_text(csv_text)
            console.print(f"CSV written to {output}")
        else:
            print(csv_text)
        return

    if format == "html":
        from pathlib import Path

        from kb_arena.benchmark.reporter import _build_html, _load_results
        from kb_arena.settings import settings

        results = _load_results(corpus)
        if not results:
            _cli_error("NO_RESULTS", "No results found. Run benchmark first.")
        html_text = _build_html(results, corpus)
        out = Path(output) if output else Path(settings.results_path) / "report.html"
        out.write_text(html_text)
        console.print(f"HTML report written to {out}")
        return

    from kb_arena.benchmark.reporter import generate_report

    generate_report(corpus=corpus, output=output)

    _next_step("report")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Stage 5: Launch side-by-side chatbot demo.

    Requires: at least one strategy built.
    """
    import uvicorn

    uvicorn.run(
        "kb_arena.chatbot.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def init_corpus(
    name: str = typer.Argument(..., help="Name for the new corpus (e.g. my-docs)"),
):
    """Scaffold a new corpus directory structure.

    Creates datasets/{name}/ with raw/, processed/, questions/ subdirectories.
    """
    import re
    from pathlib import Path

    from kb_arena.settings import settings

    if not re.match(r"^[a-zA-Z0-9_-]+$", name):
        console.print(
            f"[red]Invalid corpus name '{name}'. "
            "Use only letters, digits, hyphens, underscores.[/red]"
        )
        raise typer.Exit(1)

    base = Path(settings.datasets_path) / name
    if base.exists():
        console.print(f"[yellow]Corpus directory already exists: {base}[/yellow]")
        return

    for subdir in ["raw", "processed", "questions"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    # Drop a sample question YAML so users can see the schema without reading docs.
    # The example must validate as a Question; the loader rejects anything else.
    sample_q = base / "questions" / "tier1_factoid.yaml.example"
    sample_q.write_text(
        "# Rename to tier1_factoid.yaml to activate. One YAML file per tier.\n"
        "# Fields: id, tier (1-5), type, hops, split, review_status, reviewed_by,\n"
        "# rationale, source_anchors, question, ground_truth, constraints.\n"
        "# review_status is one of machine-assisted-draft, human-reviewed, unspecified.\n"
        f"- id: {name}-t1-001\n"
        "  tier: 1\n"
        "  type: factoid\n"
        "  hops: 1\n"
        "  split: development\n"
        "  review_status: unspecified\n"
        '  reviewed_by: ""\n'
        '  rationale: ""\n'
        "  source_anchors: []\n"
        '  question: "What is X?"\n'
        "  ground_truth:\n"
        '    answer: "X is ..."\n'
        "    source_refs: []\n"
        "    required_entities: []\n"
        "  constraints:\n"
        '    must_mention: ["X"]\n'
        "    must_not_claim: []\n",
        encoding="utf-8",
    )

    console.print(f"[green]Created corpus scaffold:[/green] {base}/")
    console.print("  raw/         <- drop your documents here")
    console.print("  processed/   <- ingest output goes here")
    console.print("  questions/   <- benchmark questions (YAML, see sample)")
    console.print()
    console.print(
        f"Next: [bold]kb-arena run --corpus {name}[/bold] "
        "(orchestrates ingest -> build-graph -> build-vectors -> benchmark)"
    )


@app.command()
def run(
    corpus: str = typer.Option(..., help="Corpus to run end-to-end"),
    docs: str | None = typer.Option(  # noqa: UP045
        None,
        "--docs",
        help="Optional path/URL/github: spec to ingest before building. "
        "If omitted, files in datasets/{corpus}/raw/ are ingested automatically.",
    ),
    skip_graph: bool = typer.Option(
        False, "--skip-graph", help="Skip build-graph (useful when Neo4j is unavailable)"
    ),
    questions: int = typer.Option(50, help="Auto-generate this many questions if none exist"),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Skip stages whose outputs already exist"
    ),
):
    """One-shot pipeline: ingest -> build-graph -> build-vectors -> generate-questions -> benchmark.

    Each stage writes a checkpoint to datasets/{corpus}/.pipeline_state.json so a
    subsequent run --resume picks up where the last one stopped. Stages whose
    output is already on disk are skipped automatically.
    """
    import asyncio
    import json
    import re
    from pathlib import Path

    from rich.panel import Panel

    from kb_arena.settings import settings

    if not re.match(r"^[a-zA-Z0-9_-]+$", corpus):
        console.print(f"[red]Invalid corpus name '{corpus}'.[/red]")
        raise typer.Exit(1)

    base = Path(settings.datasets_path) / corpus
    if not base.exists():
        console.print(
            f"[red]Corpus directory not found: {base}\n" f"Run: kb-arena init-corpus {corpus}[/red]"
        )
        raise typer.Exit(1)

    state_path = base / ".pipeline_state.json"
    state: dict = {}
    if resume and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}

    def _save_state() -> None:
        state_path.write_text(json.dumps(state, indent=2))

    def _stage(name: str, action, checkpoint: dict | None = None) -> None:
        checkpoint = checkpoint or {}
        checkpoint_matches = all(state.get(key) == value for key, value in checkpoint.items())
        if resume and state.get(name) == "done" and checkpoint_matches:
            console.print(f"[dim][skip][/dim] {name} already complete")
            return
        console.print(Panel.fit(f"[bold]{name}[/bold]", style="cyan"))
        if action() is False:
            return
        state[name] = "done"
        state.update(checkpoint)
        _save_state()

    from kb_arena.strategies import load_documents

    def _has_documents() -> bool:
        return bool(load_documents(corpus))

    def _has_questions() -> bool:
        from kb_arena.benchmark.questions import load_questions

        try:
            return bool(load_questions(corpus))
        except FileNotFoundError:
            return False

    has_documents = _has_documents()
    has_questions = _has_questions()
    regenerate_questions = False
    state_changed = False
    if resume and not has_documents:
        for stage_name in ("ingest", "build_graph", "build_vectors", "benchmark"):
            state_changed = state.pop(stage_name, None) is not None or state_changed
    if resume and not has_questions:
        for stage_name in ("generate_questions", "benchmark"):
            state_changed = state.pop(stage_name, None) is not None or state_changed
    if "benchmark" not in state:
        state_changed = state.pop("benchmark_strategies", None) is not None or state_changed

    if docs is not None and (state.get("ingest") != "done" or state.get("ingest_source") != docs):
        regenerate_questions = state.get("generate_questions") == "done"
        for stage_name in (
            "ingest",
            "ingest_source",
            "build_graph",
            "build_vectors",
            "generate_questions",
            "benchmark",
            "benchmark_strategies",
        ):
            state_changed = state.pop(stage_name, None) is not None or state_changed

    # 1. ingest explicit input, or automatically use a populated raw directory.
    ingest_source = docs
    if not ingest_source:
        raw = base / "raw"
        from kb_arena.ingest.pipeline import SUPPORTED_EXTENSIONS

        has_raw = raw.exists() and any(
            path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            for path in raw.rglob("*")
        )
        if not has_documents and has_raw:
            ingest_source = str(raw)
        elif not has_documents:
            console.print(
                "[red]No documents found to process.[/red]\n"
                f"Drop files into {raw}/, pass --docs PATH, or run kb-arena ingest separately."
            )
            raise typer.Exit(1)

    needs_llm = any(
        (
            not skip_graph and not (resume and state.get("build_graph") == "done"),
            not (resume and state.get("build_vectors") == "done"),
            (not has_questions or regenerate_questions)
            and not (resume and state.get("generate_questions") == "done"),
        )
    )
    needs_embeddings = not (resume and state.get("build_vectors") == "done")
    if needs_llm or needs_embeddings:
        _preflight(needs_llm=needs_llm, needs_embeddings=needs_embeddings)
    if state_changed:
        _save_state()

    if ingest_source:

        def _ingest():
            from urllib.parse import urlsplit

            source_scheme = urlsplit(ingest_source).scheme.lower()
            if source_scheme in {"http", "https"}:
                from kb_arena.ingest.pipeline import run_ingest_special

                ingested = run_ingest_special(source=ingest_source, corpus=corpus, format="web")
            elif source_scheme == "github":
                from kb_arena.ingest.pipeline import run_ingest_special

                ingested = run_ingest_special(source=ingest_source, corpus=corpus, format="github")
            else:
                from kb_arena.ingest.pipeline import run_ingest

                ingested = run_ingest(path=ingest_source, corpus=corpus, format="auto")
            if ingested <= 0:
                console.print("[red]Ingestion produced no documents; pipeline stopped.[/red]")
                raise typer.Exit(1)

        _stage("ingest", _ingest, checkpoint={"ingest_source": ingest_source})

    # 2. build-graph (skippable when Neo4j is unavailable)
    graph_available = bool(resume and state.get("build_graph") == "done" and not skip_graph)
    if not skip_graph:
        from neo4j.exceptions import ServiceUnavailable

        from kb_arena.exceptions import GraphError
        from kb_arena.graph.extractor import run_extraction

        def _graph() -> bool | None:
            nonlocal graph_available
            try:
                asyncio.run(run_extraction(corpus=corpus))
            except (GraphError, OSError, ConnectionError, ServiceUnavailable) as exc:
                graph_available = False
                console.print(
                    f"[yellow]build-graph failed: {exc}\n"
                    "Continuing with vector strategies. "
                    "Re-run with Neo4j running to enable knowledge_graph + hybrid.[/yellow]"
                )
                # Graceful degradation: don't mark stage done so future --resume retries.
                return False
            graph_available = True
            return None

        _stage("build_graph", _graph)
    else:
        console.print("[dim][skip][/dim] build_graph (--skip-graph)")

    # 3. build-vectors (always)
    from kb_arena.strategies import build_vector_indexes

    def _vectors():
        asyncio.run(build_vector_indexes(corpus=corpus))

    _stage("build_vectors", _vectors)

    # 4. generate-questions (only if no questions exist)
    if not has_questions or regenerate_questions:
        from kb_arena.benchmark.question_gen import run_question_generation

        def _questions():
            asyncio.run(run_question_generation(corpus=corpus, count=questions))
            if not _has_questions():
                console.print(
                    "[red]Question generation produced no questions; pipeline stopped.[/red]"
                )
                raise typer.Exit(1)

        _stage("generate_questions", _questions)
    else:
        console.print("[dim][skip][/dim] generate_questions (questions already exist)")

    # 5. benchmark
    from kb_arena.benchmark.runner import run_benchmark
    from kb_arena.strategies.catalog import default_strategy_names

    benchmark_strategies = "all"
    if not graph_available:
        benchmark_strategies = ",".join(
            name for name in default_strategy_names() if name not in {"knowledge_graph", "hybrid"}
        )

    benchmark_checkpoint = {"benchmark_strategies": benchmark_strategies}
    benchmark_complete = (
        resume
        and state.get("benchmark") == "done"
        and state.get("benchmark_strategies") == benchmark_strategies
    )
    if not benchmark_complete and (not needs_llm or not needs_embeddings):
        _preflight(
            needs_llm=not needs_llm,
            needs_embeddings=not needs_embeddings,
        )

    def _bench():
        asyncio.run(run_benchmark(corpus=corpus, strategy=benchmark_strategies))

    _stage("benchmark", _bench, checkpoint=benchmark_checkpoint)

    console.print()
    console.print(
        Panel.fit(
            "[bold green]Pipeline complete[/bold green]\n\n"
            f"Run [bold]kb-arena serve[/bold] to explore results in the dashboard,\n"
            f"or [bold]kb-arena report --corpus {corpus}[/bold] for a Markdown summary.",
            style="green",
        )
    )


@app.command()
def generate_questions(
    corpus: str = typer.Option(..., help="Corpus to generate questions for"),
    count: int = typer.Option(50, help="Total questions to generate (distributed across tiers)"),
):
    """Auto-generate benchmark questions from ingested documents using LLM.

    Reads processed JSONL, generates questions per tier, writes YAML.
    """
    import asyncio

    _preflight(needs_llm=True)

    from kb_arena.benchmark.question_gen import run_question_generation

    asyncio.run(run_question_generation(corpus=corpus, count=count))

    _next_step("generate_questions", corpus)


@app.command()
def demo(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Launch the demo with pre-computed aws-compute benchmark results.

    No API keys, Docker service, or setup is needed to explore the checked results.
    """
    import os
    import webbrowser
    from pathlib import Path
    from threading import Timer

    from kb_arena.settings import settings

    results_dir = Path("results")
    result_files = list(results_dir.glob("aws-compute_*.json")) if results_dir.exists() else []

    if not result_files:
        # Seed from bundled package data so `pip install kb-arena && kb-arena demo` just works
        import importlib.resources

        try:
            bundled = importlib.resources.files("kb_arena") / "data"
            bundled_files = [f for f in bundled.iterdir() if f.name.startswith("aws-compute_")]
            if bundled_files:
                results_dir.mkdir(exist_ok=True)
                for f in bundled_files:
                    dest = results_dir / f.name
                    if not dest.exists():
                        dest.write_bytes(f.read_bytes())
                result_files = list(results_dir.glob("aws-compute_*.json"))
                n = len(result_files)
                console.print(f"[dim]Loaded {n} bundled result(s) into ./results/[/dim]")
        except Exception:
            pass

    if not result_files:
        console.print(
            "[red]No aws-compute results found.[/red]\n"
            "The demo requires pre-computed benchmark results in results/.\n"
            "Clone the full repo: [bold]git clone https://github.com/xmpuspus/kb-arena[/bold]"
        )
        raise typer.Exit(1)

    console.print(f"[green]Found {len(result_files)} benchmark result(s)[/green]")

    import socket

    actual_port = port
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", candidate)) != 0:
                actual_port = candidate
                break
    else:
        console.print(f"[red]No available port in range {port}-{port + 19}[/red]")
        raise typer.Exit(1)

    if actual_port != port:
        console.print(f"[yellow]Port {port} in use, using {actual_port}[/yellow]")

    console.print(f"Starting API server on http://localhost:{actual_port}")
    console.print(f"API docs at http://localhost:{actual_port}/docs\n")

    def open_browser():
        import importlib.resources

        static_dir = importlib.resources.files("kb_arena") / "static"
        if hasattr(static_dir, "is_dir") and static_dir.is_dir():
            # Bundled frontend - open the dashboard directly
            console.print("[green]Serving bundled frontend dashboard[/green]")
            webbrowser.open(f"http://localhost:{actual_port}/benchmark/")
        else:
            # No bundled frontend - check for dev server
            for fe_port in (3000, 3001):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("localhost", fe_port)) == 0:
                        console.print(f"[green]Frontend detected on port {fe_port}[/green]")
                        webbrowser.open(f"http://localhost:{fe_port}/benchmark")
                        return
            webbrowser.open(f"http://localhost:{actual_port}/docs")

    Timer(1.5, open_browser).start()

    import uvicorn

    previous_demo_environment = os.environ.get("KB_ARENA_DEMO_MODE")
    previous_demo_setting = settings.demo_mode
    os.environ["KB_ARENA_DEMO_MODE"] = "true"
    settings.demo_mode = True
    try:
        uvicorn.run("kb_arena.chatbot.api:app", host=host, port=actual_port)
    finally:
        settings.demo_mode = previous_demo_setting
        if previous_demo_environment is None:
            os.environ.pop("KB_ARENA_DEMO_MODE", None)
        else:
            os.environ["KB_ARENA_DEMO_MODE"] = previous_demo_environment


@app.command()
def generate_qa(
    corpus: str = typer.Option(..., help="Corpus to generate Q&A pairs for"),
    output: str | None = typer.Option(None, help="Output JSONL path"),  # noqa: UP045
):
    """Generate Q&A pairs from your documentation.

    Reads processed JSONL, generates 3-5 Q&A pairs per section using LLM,
    writes results as JSONL. Only needs Anthropic key (no embeddings).
    """
    import asyncio

    _preflight(needs_llm=True)

    from kb_arena.generate.cli_runner import run_generate_qa

    asyncio.run(run_generate_qa(corpus=corpus, output=output))


@app.command()
def audit(
    corpus: str = typer.Option(..., help="Corpus to audit"),
    output: str | None = typer.Option(None, help="Output JSON path"),  # noqa: UP045
    max_sections: int = typer.Option(50, help="Max sections to audit"),
):
    """Find gaps in your documentation.

    Generates Q&A pairs per section, self-evaluates them, and classifies
    sections as strong (>=70%), weak (30-70%), or gap (<30%).
    """
    import asyncio

    _preflight(needs_llm=True)

    from kb_arena.audit.analyzer import run_audit
    from kb_arena.audit.display import display_audit_report

    report = asyncio.run(run_audit(corpus=corpus, max_sections=max_sections))
    display_audit_report(report, output=output)


@app.command()
def fix(
    corpus: str = typer.Option(..., help="Corpus to fix"),
    max_fixes: int = typer.Option(10, help="Max fix recommendations"),
    output: str | None = typer.Option(None, help="Output markdown path"),  # noqa: UP045
):
    """Generate fix recommendations for weak documentation.

    Runs audit internally, then generates actionable recommendations
    with draft content for sections scoring below 70%.
    """
    import asyncio

    _preflight(needs_llm=True)

    from kb_arena.audit.analyzer import run_audit
    from kb_arena.audit.display import display_fix_report
    from kb_arena.audit.fixer import generate_fixes
    from kb_arena.llm.client import LLMClient
    from kb_arena.strategies import load_documents

    async def _run():
        report = await run_audit(corpus=corpus)
        documents = load_documents(corpus)
        llm = LLMClient()
        return await generate_fixes(report, documents, llm, max_fixes=max_fixes)

    fix_report = asyncio.run(_run())
    display_fix_report(fix_report, output=output)


@app.command()
def health(
    format: str = typer.Option("rich", help="Output format: rich, json"),
):
    """Report pipeline progress, service connectivity, and API key status by corpus."""
    import asyncio
    from pathlib import Path

    from kb_arena.settings import settings

    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(settings.openai_api_key)

    async def check_neo4j():
        driver = None
        try:
            import neo4j

            driver = neo4j.AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            async with driver.session(database=settings.neo4j_database) as session:
                result = await session.run("RETURN 1")
                await result.consume()
            return True
        except Exception:
            return False
        finally:
            if driver is not None:
                await driver.close()

    neo4j_ok = asyncio.run(check_neo4j())

    chroma_collections = 0
    collections = []
    try:
        import chromadb

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        collections = chroma.list_collections()
        chroma_collections = len(collections)
    except Exception:
        pass

    datasets_dir = Path(settings.datasets_path)
    results_dir = Path(settings.results_path)
    corpora_data: dict[str, dict] = {}

    if datasets_dir.exists():
        corpus_dirs = sorted(
            d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        for d in corpus_dirs:
            name = d.name
            raw_count = (
                sum(1 for _ in (d / "raw").glob("*") if _.is_file() and _.name != ".gitkeep")
                if (d / "raw").is_dir()
                else 0
            )
            has_processed = (d / "processed").is_dir() and any((d / "processed").glob("*.jsonl"))
            question_count = 0
            if (d / "questions").is_dir():
                for qf in (d / "questions").glob("*.yaml"):
                    try:
                        question_count += qf.read_text().count("- id:")
                    except OSError:
                        pass
            has_vectors = False
            try:
                if collections:
                    has_vectors = any(name in c.name for c in collections)
            except Exception:
                pass
            result_count = (
                len(list(results_dir.glob(f"{name}_*.json"))) if results_dir.exists() else 0
            )
            qa_pairs_path = d / "qa-pairs" / "qa_pairs.jsonl"
            qa_pair_count = 0
            if qa_pairs_path.exists():
                lines = qa_pairs_path.read_text().splitlines()
                qa_pair_count = sum(1 for line in lines if line.strip())

            corpora_data[name] = {
                "raw_docs": raw_count,
                "processed": has_processed,
                "vectors": has_vectors,
                "graph": neo4j_ok,
                "questions": question_count,
                "results": result_count,
                "qa_pairs": qa_pair_count,
            }

    if format == "json":
        import json
        import sys

        health_data = {
            "api_keys": {"anthropic": has_anthropic, "openai": has_openai},
            "services": {
                "neo4j": neo4j_ok,
                "chromadb": chroma_collections,
            },
            "corpora": corpora_data,
        }
        sys.stdout.write(json.dumps(health_data, indent=2) + "\n")
        return

    console.print("[bold]KB Arena Health Check[/bold]\n")

    console.print("  API Keys:")
    ant_status = "[green]set[/green]" if has_anthropic else "[red]missing[/red]"
    oai_status = "[green]set[/green]" if has_openai else "[red]missing[/red]"
    console.print(f"    Anthropic: {ant_status}")
    if not has_anthropic:
        console.print(
            "               [dim]Set KB_ARENA_ANTHROPIC_API_KEY in .env"
            " (needed for graph, question gen, benchmark)[/dim]"
        )
    console.print(f"    OpenAI:    {oai_status}")
    if not has_openai:
        console.print(
            "               [dim]Set KB_ARENA_OPENAI_API_KEY in .env"
            " (needed for embeddings / vector strategies)[/dim]"
        )
    console.print()

    console.print("  Services:")
    neo_status = "[green]connected[/green]" if neo4j_ok else "[yellow]unavailable[/yellow]"
    console.print(f"    Neo4j:    {neo_status}")
    if not neo4j_ok:
        console.print(
            "              [dim]Run: docker compose up neo4j -d"
            " (needed for graph + hybrid strategies)[/dim]"
        )
    if chroma_collections > 0:
        console.print(f"    ChromaDB: [green]{chroma_collections} collection(s)[/green]")
    else:
        console.print("    ChromaDB: [yellow]unavailable[/yellow]")
    console.print()

    if not corpora_data:
        console.print("  No corpora found. Run: [bold]kb-arena init-corpus my-docs[/bold]\n")
        return

    console.print("  Corpora:")
    for name, data in corpora_data.items():
        raw_s = (
            f"[green]{data['raw_docs']} doc(s)[/green]" if data["raw_docs"] else "[dim]empty[/dim]"
        )
        proc_s = "[green]yes[/green]" if data["processed"] else "[dim]no[/dim]"
        vec_s = "[green]yes[/green]" if data["vectors"] else "[dim]no[/dim]"
        graph_s = "[green]yes[/green]" if data["graph"] else "[dim]no[/dim]"
        q_s = f"[green]{data['questions']}[/green]" if data["questions"] else "[dim]0[/dim]"
        r_s = (
            f"[green]{data['results']} strategy(ies)[/green]"
            if data["results"]
            else "[dim]none[/dim]"
        )
        qa_s = f"[green]{data['qa_pairs']} pairs[/green]" if data["qa_pairs"] else "[dim]none[/dim]"

        console.print(f"    [bold]{name}[/bold]")
        console.print(f"      raw: {raw_s}  processed: {proc_s}")
        console.print(f"      vectors: {vec_s}  graph: {graph_s}")
        console.print(f"      questions: {q_s}  results: {r_s}  qa-pairs: {qa_s}")
    console.print()


@app.command()
def eval(
    corpus: str = typer.Option("all", help="Corpus to evaluate"),
    ci: bool = typer.Option(False, "--ci", help="CI mode: exit non-zero on regression"),
    threshold: list[str] = typer.Option(
        [],
        "--threshold",
        help="Metric thresholds as metric=value (e.g. accuracy=0.7 faithfulness=0.8)",
    ),
    format: str = typer.Option("rich", help="Output format: rich, json"),
):
    """Evaluate latest benchmark results against thresholds.

    CI/CD mode: exits non-zero if any metric falls below its threshold.
    Use with --ci --threshold accuracy=0.7 --threshold faithfulness=0.8
    """
    from kb_arena.benchmark.reporter import _load_results

    parsed_thresholds: dict[str, float] = {}
    for t in threshold:
        if "=" not in t:
            _cli_error("BAD_THRESHOLD", f"Invalid threshold format: {t}. Use metric=value")
        metric, val = t.split("=", 1)
        try:
            parsed_thresholds[metric.strip()] = float(val.strip())
        except ValueError:
            _cli_error("BAD_THRESHOLD", f"Invalid threshold value: {val}")
        _validate_unit_interval(parsed_thresholds[metric.strip()], "--threshold")

    results = _load_results(corpus if corpus != "all" else None)
    if not results:
        _cli_error("NO_RESULTS", "No results found. Run benchmark first.", fmt=format)

    failed = False
    for r in results:
        if not r.accuracy_by_tier:
            continue
        avg_acc = sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
        metrics = {
            "accuracy": avg_acc,
            "faithfulness": r.reliability.avg_faithfulness if r.reliability else 0.0,
        }

        for metric, thresh in parsed_thresholds.items():
            actual = metrics.get(metric, 0.0)
            if actual < thresh:
                console.print(f"[red]FAIL: {r.strategy} {metric}={actual:.3f} < {thresh}[/red]")
                failed = True
            elif ci:
                console.print(
                    f"[green]PASS: {r.strategy} {metric}={actual:.3f} >= {thresh}[/green]"
                )

    if format == "json":
        import json as _json
        import sys

        summary = {
            "strategies": [
                {
                    "strategy": r.strategy,
                    "accuracy": (
                        sum(r.accuracy_by_tier.values()) / len(r.accuracy_by_tier)
                        if r.accuracy_by_tier
                        else 0.0
                    ),
                    "cost": r.total_cost_usd,
                }
                for r in results
            ],
            "passed": not failed,
        }
        sys.stdout.write(_json.dumps(summary, indent=2) + "\n")

    if ci and failed:
        raise typer.Exit(1)
    if ci and not failed:
        console.print("[green]All thresholds passed.[/green]")


@app.command()
def compare(
    a: str = typer.Option(..., "--a", help="Strategy A, the baseline"),
    b: str = typer.Option(..., "--b", help="Strategy B, the candidate"),
    corpus: str = typer.Option("aws-compute", "--corpus", help="Corpus both results belong to"),
    run_a: str = typer.Option("", "--run-a", help="Run id for A. Default: the latest result file"),
    run_b: str = typer.Option("", "--run-b", help="Run id for B. Default: the latest result file"),
    metric: str = typer.Option(
        "accuracy", "--metric", help="Score field, or latency_ms, or cost_usd"
    ),
    lab: str = typer.Option(
        "", "--lab", help="A retriever-lab JSON. Compare two strategies inside it"
    ),
    out: str = typer.Option("", "--out", help="Where to write the JSON artifact"),
):
    """Pair two strategies question by question: deltas, CI, effect size, win/tie/loss."""
    import json as _json
    from pathlib import Path as _Path

    from rich.table import Table

    from kb_arena.benchmark.compare import (
        METRIC_NAME,
        SAFE_ID,
        compare_lab,
        compare_result_files,
        resolve_result_path,
    )
    from kb_arena.settings import settings

    results_dir = _Path(settings.results_path)
    for name, value in (("--a", a), ("--b", b), ("--corpus", corpus)):
        if not SAFE_ID.fullmatch(value):
            console.print(f"[red]{name} must be letters, digits, dot, dash, or underscore[/red]")
            raise typer.Exit(1)
    if not METRIC_NAME.fullmatch(metric):
        console.print("[red]--metric must be a metric name, letters, digits, and underscores[/red]")
        raise typer.Exit(1)
    if lab and (run_a or run_b):
        console.print(
            "[red]--lab compares two strategies inside one file. Drop --run-a and --run-b.[/red]"
        )
        raise typer.Exit(1)
    try:
        if lab:
            result = compare_lab(_Path(lab), a, b, metric=metric)
        else:
            path_a = resolve_result_path(results_dir, corpus, a, run_a or None)
            path_b = resolve_result_path(results_dir, corpus, b, run_b or None)
            for path in (path_a, path_b):
                if not path.exists():
                    console.print(
                        f"[red]No result file at {path}. Run kb-arena benchmark first.[/red]"
                    )
                    raise typer.Exit(1)
            result = compare_result_files(path_a, path_b, metric=metric)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    meta = result["meta"]
    if meta["comparable"]:
        console.print(f"[green]Comparable:[/green] {result['n_paired']} paired questions")
    else:
        console.print("[yellow]Not a clean comparison:[/yellow] " + "; ".join(meta["reasons"]))
    low, high = result["delta_ci_95"]
    p_value = result["wilcoxon_p"]
    table = Table(title=f"{result['b']} minus {result['a']} on {result['metric']}")
    for col in (
        "n",
        f"mean {result['a']}",
        f"mean {result['b']}",
        "delta",
        "95% CI",
        "p",
        "d",
        "W/T/L",
    ):
        table.add_column(col, justify="right")
    table.add_row(
        str(result["n_paired"]),
        f"{result['mean_a']:.4f}",
        f"{result['mean_b']:.4f}",
        f"{result['mean_delta']:+.4f}",
        f"[{low:+.4f}, {high:+.4f}]",
        "n/a" if p_value is None else f"{p_value:.3f}",
        f"{result['effect_size_d']:+.2f}",
        f"{result['wins']}/{result['ties']}/{result['losses']}",
    )
    console.print(table)
    if result["ci_excludes_zero"] and result["significant"]:
        console.print("[green]The CI excludes zero and p < 0.05.[/green]")
    else:
        console.print("[dim]No significant difference at this sample size.[/dim]")

    if out:
        out_path = _Path(out)
    elif lab:
        out_path = _Path(lab).parent / f"compare_lab_{a}_vs_{b}_{metric}.json"
    else:
        tag_a, tag_b = run_a or "latest", run_b or "latest"
        out_path = results_dir / f"compare_{corpus}_{a}@{tag_a}_vs_{b}@{tag_b}_{metric}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_json.dumps(result, indent=2))
    console.print(f"[dim]Wrote {out_path}[/dim]")


@app.command(name="retriever-lab")
def retriever_lab(
    corpus: str = typer.Option("all", help="Corpus to evaluate"),
    top_k: int = typer.Option(5, "--top-k", help="Top-k chunks per query"),
    strategies: str = typer.Option("all", help="Strategy filter (or 'all')"),
    split: str = typer.Option(
        "", "--split", help="Question split: development, validation, holdout, or all"
    ),
    min_recall: float = typer.Option(
        0.30,
        "--min-recall",
        help="Exit non-zero if any strategy's mean Recall@k drops below this",
    ),
    ceiling_k: int = typer.Option(
        0,
        "--ceiling-k",
        help="Deeper cutoff for the retrieval-ceiling diagnostic (0 = top_k*4). "
        "Reports base-retriever Recall@top_k vs Recall@ceiling_k = ranking headroom. "
        "The diagnostic runs on its own only when a strategy ranks over the naive_vector "
        "pool; pass this flag to force it for any run.",
    ),
):
    """Run retrieval-only benchmark with classical IR metrics. ~10x cheaper than `benchmark`."""
    import asyncio as _asyncio

    from kb_arena.benchmark.retriever_lab import run_retriever_lab

    _validate_unit_interval(min_recall, "--min-recall")
    _preflight(needs_embeddings=True)
    exit_code = _asyncio.run(
        run_retriever_lab(corpus, strategies, top_k, min_recall, ceiling_k or None, split=split)
    )
    if exit_code:
        raise typer.Exit(exit_code)


@app.command(name="quantum-diagnostics")
def quantum_diagnostics(
    corpus: str = typer.Option("aws-compute", help="Corpus to profile"),
    sample_questions: int = typer.Option(5, "--sample-questions", help="Questions to time SQR on"),
):
    """Honest caveats for the quantum strategies: PCA variance loss per qubit count,
    SWAP-test error vs shots, and SQR's wall-clock overhead over naive_vector.

    Needs the optional [quantum] extra (pip install 'kb-arena[quantum]')."""
    import asyncio as _asyncio
    import importlib.util
    import json as _json
    from pathlib import Path

    from rich.console import Console as _Console
    from rich.table import Table as _Table

    if importlib.util.find_spec("qiskit") is None:
        _Console().print(
            "[red]The [quantum] extra is required.[/red] "
            "Install with: pip install 'kb-arena[quantum]'"
        )
        raise typer.Exit(1)

    from kb_arena.settings import settings
    from kb_arena.strategies.quantum.diagnostics import run_quantum_diagnostics

    _preflight(needs_embeddings=True)
    console = _Console()
    diag = _asyncio.run(run_quantum_diagnostics(corpus, sample_questions=sample_questions))

    pca_t = _Table(title=f"PCA variance retained: {corpus} ({diag.n_embedding_samples} samples)")
    pca_t.add_column("n_qubits", justify="right")
    pca_t.add_column("encoded dim", justify="right")
    pca_t.add_column("variance explained", justify="right")
    for p in diag.pca_variance_curve:
        marker = " [dim](sqr default)[/dim]" if p.n_qubits == settings.sqr_n_qubits else ""
        pca_t.add_row(f"{p.n_qubits}{marker}", str(p.encoded_dim), f"{p.variance_explained:.3f}")
    console.print(pca_t)

    shot_t = _Table(title="SWAP-test error vs shots (statevector = exact)")
    shot_t.add_column("shots", justify="right")
    shot_t.add_column("mean abs fidelity error", justify="right")
    for s in diag.shot_error_curve:
        shot_t.add_row(str(s.shots), f"{s.mean_abs_error:.4f}")
    console.print(shot_t)

    console.print(
        f"[bold]Quantum overhead[/bold]: naive coarse {diag.naive_retrieval_ms:.1f}ms -> "
        f"SQR total {diag.sqr_total_ms:.1f}ms = "
        f"[bold]+{diag.mean_quantum_overhead_ms:.1f}ms[/bold] "
        f"(mean over {diag.sample_questions} questions)"
    )

    out_dir = Path(settings.results_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quantum_diagnostics_{corpus}.json"
    out_path.write_text(_json.dumps(diag.model_dump(), indent=2))
    console.print(f"[green]Written {out_path}[/green]")


@app.command(name="label-chunks")
def label_chunks(
    corpus: str = typer.Option(..., help="Corpus to label"),
    force: bool = typer.Option(False, "--force", help="Re-label even if labels exist"),
    n_candidates: int = typer.Option(20, "--n-candidates", help="BM25 candidates per question"),
):
    """Generate datasets/{corpus}/questions/expected_chunks.yaml via BM25 + Haiku judge.

    Cost-capped by KB_ARENA_COST_CAP_USD. Idempotent: skips already-labeled
    questions unless --force.
    """
    import asyncio as _asyncio

    from kb_arena.benchmark.expected_chunks import label_corpus

    _preflight(needs_llm=True, needs_embeddings=True)
    result = _asyncio.run(label_corpus(corpus, force=force, n_candidates=n_candidates))
    note = " (halted by cost cap)" if result.get("halted_by_cost_cap") else ""
    console.print(
        f"[green]Labeled {result['labeled']}, skipped {result['skipped']} "
        f"of {result['total_questions']} (cost ${result['cost_usd']:.4f}{note})[/green]"
    )
    console.print(f"Saved to {result['path']}")


@app.command(name="optimize")
def optimize(
    corpus: str = typer.Option(..., help="Corpus to optimize against"),
    strategies: str = typer.Option("all", help="Strategy filter: 'all' or comma-separated names"),
    split: str = typer.Option(
        "auto",
        "--split",
        help=(
            "Question split: auto (development when labeled), development, validation, "
            "holdout, or all"
        ),
    ),
    top_ks: str = typer.Option("3,5,10", "--top-ks", help="Comma-separated top-k values to sweep"),
    chunk_sizes: str = typer.Option(
        "", "--chunk-sizes", help="Comma-separated chunk-token sizes (chunking strategies only)"
    ),
    embedding_providers: str = typer.Option(
        "",
        "--embedding-providers",
        help="Comma-separated embedding providers (openai,voyage,cohere,bge,ollama,gemini)",
    ),
    reranker_backends: str = typer.Option(
        "", "--reranker-backends", help="Comma-separated reranker backends (rerank_vector only)"
    ),
    metric: str = typer.Option(
        "ndcg", "--metric", help="Metric to optimize: ndcg|recall|mrr|hit|precision"
    ),
    method: str = typer.Option("grid", "--method", help="Search method: grid|random"),
    max_trials: int = typer.Option(0, "--max-trials", help="Cap trials per strategy (0 = no cap)"),
    seed: int = typer.Option(0, "--seed", help="RNG seed for --method random"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the trial plan and cost preview, then exit"
    ),
):
    """Automated retrieval-strategy hyperparameter search.

    Sweeps chunk size, top-k, embedding provider and reranker backend per
    strategy, scores each configuration on a retrieval IR metric (retrieval-only,
    ~10x cheaper than `benchmark`), and reports the tuned optimum and its delta
    versus the current defaults. QnA Pairs and RAPTOR reuse their prebuilt indexes
    and sweep top-k only, so optimization never regenerates LLM-built artifacts.
    `--dry-run` needs no API keys.
    """
    import asyncio as _asyncio

    from kb_arena.benchmark.optimizer import run_optimize, validate_optimize_inputs

    def _ints(s: str, option: str) -> list[int]:
        try:
            return [int(x) for x in s.split(",") if x.strip()]
        except ValueError:
            raise ValueError(f"Invalid integer list for {option}.") from None

    def _strs(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    try:
        parsed_top_ks = _ints(top_ks, "--top-ks")
        parsed_chunk_sizes = _ints(chunk_sizes, "--chunk-sizes")
        validate_optimize_inputs(
            strategies,
            top_ks=parsed_top_ks,
            chunk_sizes=parsed_chunk_sizes,
            method=method,
            max_trials=max_trials,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    if not dry_run:
        _preflight(needs_embeddings=True)

    exit_code = _asyncio.run(
        run_optimize(
            corpus,
            strategies,
            top_ks=parsed_top_ks,
            chunk_sizes=parsed_chunk_sizes,
            embedding_providers=_strs(embedding_providers),
            reranker_backends=_strs(reranker_backends),
            metric=metric,
            method=method,
            max_trials=max_trials,
            seed=seed,
            dry_run=dry_run,
            split=split,
        )
    )
    if exit_code:
        raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
