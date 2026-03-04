"""KB Arena CLI — multi-stage pipeline (cloudwright Typer + Rich pattern).

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="kb-arena",
    help="Benchmark retrieval strategies (vector, graph, hybrid) on your documentation.",
    no_args_is_help=True,
)
console = Console()


def _preflight(
    needs_anthropic: bool = False,
    needs_openai: bool = False,
    needs_neo4j: bool = False,
) -> None:
    """Check required services/keys upfront with actionable error messages."""
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
):
    """Stage 1: Parse raw documents into unified Document model.

    Supports local files/dirs, URLs (auto-detected), and github:owner/repo.
    Writes JSONL to datasets/{corpus}/processed/
    """
    # Auto-detect source type
    detected_format = format
    if format == "auto":
        if path.startswith(("http://", "https://")):
            detected_format = "web"
        elif path.startswith("github:"):
            detected_format = "github"

    if detected_format in ("web", "github"):
        from kb_arena.ingest.pipeline import run_ingest_special

        run_ingest_special(source=path, corpus=corpus, format=detected_format)
    else:
        from kb_arena.ingest.pipeline import run_ingest

        run_ingest(path=path, corpus=corpus, format=format)


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


@app.command()
def build_vectors(
    corpus: str = typer.Option(..., help="Corpus to build vectors for"),
    strategy: str = typer.Option("all", help="Strategy: all, naive, contextual, qna"),
):
    """Stage 3: Build vector indexes for strategies 1-3.

    Requires: ingest completed. Writes to ChromaDB.
    """
    import asyncio

    _preflight(needs_openai=True)

    from kb_arena.strategies import build_vector_indexes

    asyncio.run(build_vector_indexes(corpus=corpus, strategy=strategy))


@app.command()
def benchmark(
    corpus: str = typer.Option("all", help="Corpus name, or 'all' to run all discovered corpora"),
    strategy: str = typer.Option(
        "all",
        help="Strategy: all, naive_vector, contextual_vector, qna_pairs, knowledge_graph, hybrid",
    ),
    tier: int = typer.Option(0, help="Tier filter (0 = all tiers)"),
):
    """Stage 4: Run benchmark questions against specified strategies.

    Writes results to results/{corpus}.json
    """
    import asyncio

    _preflight(needs_anthropic=True, needs_openai=True)

    from kb_arena.benchmark.runner import run_benchmark

    asyncio.run(run_benchmark(corpus=corpus, strategy=strategy, tier=tier))


@app.command()
def report(
    corpus: str = typer.Option("all", help="Corpus to generate report for"),
    output: Optional[str] = typer.Option(None, help="Output file path"),  # noqa: UP045
):
    """Generate benchmark report from results JSON."""
    from kb_arena.benchmark.reporter import generate_report

    generate_report(corpus=corpus, output=output)


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
        console.print(
            "[red]No aws-compute results found.[/red]\n"
            "The demo requires pre-computed benchmark results in results/.\n"
            "Clone the full repo: [bold]git clone https://github.com/your-org/kb-arena[/bold]"
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

    # Open the frontend if it's running, otherwise open API docs
    def open_browser():
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
    output: Optional[str] = typer.Option(None, help="Output JSONL path"),  # noqa: UP045
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
    output: Optional[str] = typer.Option(None, help="Output JSON path"),  # noqa: UP045
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
    output: Optional[str] = typer.Option(None, help="Output markdown path"),  # noqa: UP045
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
def health():
    """Pipeline status — per-corpus progress, service connectivity, API keys."""
    import asyncio
    from pathlib import Path

    from kb_arena.settings import settings

    console.print("[bold]KB Arena Health Check[/bold]\n")

    # API keys
    has_anthropic = bool(settings.anthropic_api_key)
    has_openai = bool(settings.openai_api_key)
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

    # Services
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
    console.print("  Services:")
    neo_status = "[green]connected[/green]" if neo4j_ok else "[yellow]unavailable[/yellow]"
    console.print(f"    Neo4j:    {neo_status}")
    if not neo4j_ok:
        console.print(
            "              [dim]Run: docker compose up neo4j -d"
            " (needed for graph + hybrid strategies)[/dim]"
        )

    try:
        import chromadb

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        collections = chroma.list_collections()
        console.print(f"    ChromaDB: [green]{len(collections)} collection(s)[/green]")
    except Exception:
        console.print("    ChromaDB: [yellow]unavailable[/yellow]")
    console.print()

    # Per-corpus pipeline status
    datasets_dir = Path(settings.datasets_path)
    results_dir = Path(settings.results_path)

    if not datasets_dir.exists():
        console.print("  No datasets/ directory found.\n")
        return

    corpus_dirs = sorted(
        d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    )

    if not corpus_dirs:
        console.print("  No corpora found. Run: [bold]kb-arena init-corpus my-docs[/bold]\n")
        return

    console.print("  Corpora:")
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
        result_count = len(list(results_dir.glob(f"{name}_*.json"))) if results_dir.exists() else 0
        qa_pairs_path = d / "qa-pairs" / "qa_pairs.jsonl"
        qa_pair_count = 0
        if qa_pairs_path.exists():
            lines = qa_pairs_path.read_text().splitlines()
            qa_pair_count = sum(1 for line in lines if line.strip())

        raw_s = f"[green]{raw_count} doc(s)[/green]" if raw_count else "[dim]empty[/dim]"
        proc_s = "[green]yes[/green]" if has_processed else "[dim]no[/dim]"
        vec_s = "[green]yes[/green]" if has_vectors else "[dim]no[/dim]"
        graph_s = "[green]yes[/green]" if neo4j_ok else "[dim]no[/dim]"
        q_s = f"[green]{question_count}[/green]" if question_count else "[dim]0[/dim]"
        r_s = f"[green]{result_count} strategy(ies)[/green]" if result_count else "[dim]none[/dim]"
        qa_s = f"[green]{qa_pair_count} pairs[/green]" if qa_pair_count else "[dim]none[/dim]"

        console.print(f"    [bold]{name}[/bold]")
        console.print(f"      raw: {raw_s}  processed: {proc_s}")
        console.print(f"      vectors: {vec_s}  graph: {graph_s}")
        console.print(f"      questions: {q_s}  results: {r_s}  qa-pairs: {qa_s}")
    console.print()


if __name__ == "__main__":
    app()
