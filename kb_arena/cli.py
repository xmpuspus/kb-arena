"""KB Arena CLI — multi-stage pipeline (cloudwright Typer + Rich pattern).

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="kb-arena",
    help="Benchmark retrieval strategies (vector, graph, hybrid) on your documentation.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to raw documents directory"),
    corpus: str = typer.Option("custom", help="Corpus name (e.g. aws-compute, my-docs)"),
    format: str = typer.Option("auto", help="Parser format: auto, markdown, html, sec-edgar"),
):
    """Stage 1: Parse raw documents into unified Document model.

    Writes JSONL to datasets/{corpus}/processed/
    """
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

    from kb_arena.benchmark.runner import run_benchmark

    asyncio.run(run_benchmark(corpus=corpus, strategy=strategy, tier=tier))


@app.command()
def report(
    corpus: str = typer.Option("all", help="Corpus to generate report for"),
    output: str | None = typer.Option(None, help="Output file path"),
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

    from kb_arena.benchmark.question_gen import run_question_generation

    asyncio.run(run_question_generation(corpus=corpus, count=count))


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
        console.print("               (needed for graph, question gen, benchmark)")
    console.print(f"    OpenAI:    {oai_status}")
    if not has_openai:
        console.print("               (needed for embeddings / vector strategies)")
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
        console.print("              (needed for graph + hybrid strategies)")

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

        raw_s = f"[green]{raw_count} doc(s)[/green]" if raw_count else "[dim]empty[/dim]"
        proc_s = "[green]yes[/green]" if has_processed else "[dim]no[/dim]"
        vec_s = "[green]yes[/green]" if has_vectors else "[dim]no[/dim]"
        graph_s = "[green]yes[/green]" if neo4j_ok else "[dim]no[/dim]"
        q_s = f"[green]{question_count}[/green]" if question_count else "[dim]0[/dim]"
        r_s = f"[green]{result_count} strategy(ies)[/green]" if result_count else "[dim]none[/dim]"

        console.print(f"    [bold]{name}[/bold]")
        console.print(f"      raw: {raw_s}  processed: {proc_s}")
        console.print(f"      vectors: {vec_s}  graph: {graph_s}")
        console.print(f"      questions: {q_s}  results: {r_s}")
    console.print()


if __name__ == "__main__":
    app()
