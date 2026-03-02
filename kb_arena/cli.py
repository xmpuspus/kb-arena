"""KB Arena CLI — multi-stage pipeline (cloudwright Typer + Rich pattern).

Each command is independently runnable and re-runnable.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="kb-arena",
    help="Benchmark knowledge graphs vs vector RAG on real documentation.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to raw documents directory"),
    corpus: str = typer.Option(
        "custom", help="Corpus name: python-stdlib, kubernetes, sec-edgar, custom"
    ),
    format: str = typer.Option("auto", help="Parser format: auto, markdown, html, sec-edgar"),
):
    """Stage 1: Parse raw documents into unified Document model.

    Writes JSONL to datasets/{corpus}/processed/
    """
    from kb_arena.ingest.pipeline import run_ingest

    run_ingest(path=path, corpus=corpus, format=format)


@app.command()
def build_graph(
    corpus: str = typer.Option("python-stdlib", help="Corpus to build graph for"),
    schema: str = typer.Option("auto", help="Schema: auto, python, kubernetes, sec"),
):
    """Stage 2: Extract entities/relationships, build Neo4j graph.

    Requires: ingest completed. Writes to Neo4j.
    """
    import asyncio

    from kb_arena.graph.extractor import run_extraction

    asyncio.run(run_extraction(corpus=corpus, schema=schema))


@app.command()
def build_vectors(
    corpus: str = typer.Option("python-stdlib", help="Corpus to build vectors for"),
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
    corpus: str = typer.Option("all", help="Corpus: all, python-stdlib, kubernetes, sec-edgar"),
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
def download(
    corpus: str = typer.Argument(
        ..., help="Corpus to download: python-stdlib, kubernetes, sec-edgar"
    ),
):
    """Download raw dataset files for a corpus."""
    console.print(f"[bold]Downloading {corpus} dataset...[/bold]")
    console.print("[yellow]Dataset download not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
