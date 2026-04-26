"""KB Arena CLI — multi-stage pipeline (cloudwright Typer + Rich pattern).

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="kb-arena",
    help="Benchmark retrieval strategies (vector, graph, hybrid) on your documentation.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _setup(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )


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


def _preflight(
    needs_anthropic: bool = False,
    needs_openai: bool = False,
    needs_neo4j: bool = False,
) -> None:
    from kb_arena.settings import settings

    errors: list[str] = []
    if needs_anthropic and not settings.anthropic_api_key:
        errors.append("Anthropic API key required. Set KB_ARENA_ANTHROPIC_API_KEY in .env")
    if needs_openai and not settings.openai_api_key:
        errors.append(
            "OpenAI API key required (for embeddings). Set KB_ARENA_OPENAI_API_KEY in .env"
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

    from kb_arena.ingest.pipeline import _EXT_MAP

    detected_format = format
    if format == "auto":
        if path.startswith(("http://", "https://")):
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
        out_path = Path("datasets") / corpus / "processed" / "documents.jsonl"
        console.print(f"  Output: {out_path}")
        console.print("\n  Remove --dry-run to execute.")
        return

    if detected_format in ("web", "github"):
        from kb_arena.ingest.pipeline import run_ingest_special

        run_ingest_special(source=path, corpus=corpus, format=detected_format)
    else:
        from kb_arena.ingest.pipeline import run_ingest

        run_ingest(path=path, corpus=corpus, format=format)

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

    _preflight(needs_anthropic=True)

    from kb_arena.graph.extractor import run_extraction

    asyncio.run(run_extraction(corpus=corpus, schema=schema))

    _next_step("build_graph", corpus)


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

    _preflight(needs_openai=True)

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

    _preflight(needs_anthropic=True, needs_openai=True)

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
                questions = load_questions(corp, tier=tier)
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

    if ragas:
        from kb_arena.settings import settings as _settings

        _settings.benchmark_enable_ragas = True

    from kb_arena.benchmark.runner import run_benchmark

    asyncio.run(
        run_benchmark(
            corpus=corpus,
            strategy=strategy,
            tier=tier,
            parallel=parallel,
            reference_free=reference_free,
            top_k=top_k,
        )
    )

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


@app.command()
def report(
    corpus: str = typer.Option("all", help="Corpus to generate report for"),
    output: str | None = typer.Option(None, help="Output file path"),  # noqa: UP045
    format: str = typer.Option("rich", help="Output format: rich, json, csv, html"),
):
    """Generate benchmark report from results JSON."""
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
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
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
    from pathlib import Path

    base = Path("datasets") / name
    if base.exists():
        console.print(f"[yellow]Corpus directory already exists: {base}[/yellow]")
        return

    for subdir in ["raw", "processed", "questions"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)

    console.print(f"[green]Created corpus scaffold:[/green] {base}/")
    console.print("  raw/         ← drop your documents here")
    console.print("  processed/   ← ingest output goes here")
    console.print("  questions/   ← benchmark questions (YAML)")
    console.print()
    console.print(f"Next: [bold]kb-arena ingest {base}/raw/ --corpus {name}[/bold]")


@app.command()
def generate_questions(
    corpus: str = typer.Option(..., help="Corpus to generate questions for"),
    count: int = typer.Option(50, help="Total questions to generate (distributed across tiers)"),
):
    """Auto-generate benchmark questions from ingested documents using LLM.

    Reads processed JSONL, generates questions per tier, writes YAML.
    """
    import asyncio

    _preflight(needs_anthropic=True)

    from kb_arena.benchmark.question_gen import run_question_generation

    asyncio.run(run_question_generation(corpus=corpus, count=count))

    _next_step("generate_questions", corpus)


@app.command()
def demo(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to listen on"),
):
    """Launch the demo with pre-computed aws-compute benchmark results.

    No API keys, no Docker, no setup needed — just explore real results.
    """
    import webbrowser
    from pathlib import Path
    from threading import Timer

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

    uvicorn.run("kb_arena.chatbot.api:app", host=host, port=actual_port)


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

    _preflight(needs_anthropic=True)

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

    _preflight(needs_anthropic=True)

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

    _preflight(needs_anthropic=True)

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
    """Pipeline status — per-corpus progress, service connectivity, API keys."""
    import asyncio
    from pathlib import Path

    from kb_arena.settings import settings

    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(settings.openai_api_key)

    async def check_neo4j():
        try:
            import neo4j

            driver = neo4j.AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            await driver.verify_connectivity()
            await driver.close()
            return True
        except Exception:
            return False

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


@app.command(name="retriever-lab")
def retriever_lab(
    corpus: str = typer.Option("all", help="Corpus to evaluate"),
    top_k: int = typer.Option(5, "--top-k", help="Top-k chunks per query"),
    strategies: str = typer.Option("all", help="Strategy filter (or 'all')"),
    min_recall: float = typer.Option(
        0.30,
        "--min-recall",
        help="Exit non-zero if any strategy's mean Recall@k drops below this",
    ),
):
    """Run retrieval-only benchmark with classical IR metrics. ~10x cheaper than `benchmark`."""
    import asyncio as _asyncio

    from kb_arena.benchmark.retriever_lab import run_retriever_lab

    _preflight(needs_openai=True)
    exit_code = _asyncio.run(run_retriever_lab(corpus, strategies, top_k, min_recall))
    if exit_code:
        raise typer.Exit(exit_code)


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

    _preflight(needs_anthropic=True, needs_openai=True)
    result = _asyncio.run(label_corpus(corpus, force=force, n_candidates=n_candidates))
    note = " (halted by cost cap)" if result.get("halted_by_cost_cap") else ""
    console.print(
        f"[green]Labeled {result['labeled']}, skipped {result['skipped']} "
        f"of {result['total_questions']} (cost ${result['cost_usd']:.4f}{note})[/green]"
    )
    console.print(f"Saved to {result['path']}")


if __name__ == "__main__":
    app()
