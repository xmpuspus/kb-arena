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
        "custom", help="Corpus name: aws-compute, aws-storage, aws-networking, custom"
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
    corpus: str = typer.Option("aws-compute", help="Corpus to build graph for"),
    schema: str = typer.Option("auto", help="Schema: auto, aws"),
):
    """Stage 2: Extract entities/relationships, build Neo4j graph.

    Requires: ingest completed. Writes to Neo4j.
    """
    import asyncio

    from kb_arena.graph.extractor import run_extraction

    asyncio.run(run_extraction(corpus=corpus, schema=schema))


@app.command()
def build_vectors(
    corpus: str = typer.Option("aws-compute", help="Corpus to build vectors for"),
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
    corpus: str = typer.Option("all", help="Corpus: all, aws-compute, aws-storage, aws-networking"),
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
        ..., help="Corpus to download: aws-compute, aws-storage, aws-networking"
    ),
):
    """Download raw dataset files for a corpus."""
    console.print(f"[bold]Downloading {corpus} dataset...[/bold]")
    console.print("[yellow]Dataset download not yet implemented[/yellow]")


@app.command()
def health():
    """Quick check — Neo4j connectivity, ChromaDB collections, question counts."""
    import asyncio

    from kb_arena.benchmark.questions import load_all_questions
    from kb_arena.settings import settings

    console.print("[bold]KB Arena Health Check[/bold]\n")

    # Questions
    try:
        all_q = load_all_questions()
        console.print(f"  Questions loaded: [green]{len(all_q)}[/green]")
    except Exception as exc:
        console.print(f"  Questions: [red]error — {exc}[/red]")

    # ChromaDB
    try:
        import chromadb

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        collections = chroma.list_collections()
        console.print(f"  ChromaDB collections: [green]{len(collections)}[/green]")
    except Exception as exc:
        console.print(f"  ChromaDB: [red]unavailable — {exc}[/red]")

    # Neo4j
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
    status = "[green]connected[/green]" if neo4j_ok else "[yellow]unavailable[/yellow]"
    console.print(f"  Neo4j: {status}")
    console.print()


if __name__ == "__main__":
    app()
