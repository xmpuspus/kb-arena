"""LLM-based entity/relationship extraction with schema-constrained output."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from kb_arena.exceptions import GraphError
from kb_arena.graph.neo4j_store import Neo4jStore
from kb_arena.graph.resolver import resolve_entities
from kb_arena.graph.schema import (
    get_schema,
    node_type_values,
    rel_type_values,
    valid_node_type,
    valid_rel_type,
)
from kb_arena.llm.client import LLMClient
from kb_arena.models.document import Document, Section
from kb_arena.models.graph import Entity, ExtractionResult, Relationship
from kb_arena.settings import settings

logger = logging.getLogger(__name__)
_console = Console()

_SYSTEM_PROMPT_TEMPLATE = """You are a knowledge graph extraction engine for {corpus} documentation.

Extract entities and relationships from the provided text section.

ALLOWED NODE TYPES (use exactly these values):
{node_types}

ALLOWED RELATIONSHIP TYPES (use exactly these values):
{rel_types}

Output ONLY valid JSON matching this schema:
{{
  "entities": [
    {{
      "id": "<unique_id>",
      "name": "<display name>",
      "fqn": "<fully qualified name, dot-separated, e.g. aws.lambda or react.usestate>",
      "type": "<one of the allowed node types>",
      "description": "<one sentence>",
      "properties": {{}},
      "aliases": []
    }}
  ],
  "relationships": [
    {{
      "source_fqn": "<fqn of source entity>",
      "target_fqn": "<fqn of target entity>",
      "type": "<one of the allowed relationship types>",
      "properties": {{}}
    }}
  ]
}}

Rules:
- Use ONLY the allowed types listed above. Any other type will be rejected.
- fqn must be globally unique and dot-separated
- Omit entities with no clear type match
- Omit relationships where either endpoint fqn is not in the entity list
"""


def _build_system_prompt(corpus: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        corpus=corpus,
        node_types="\n".join(f"  - {v}" for v in node_type_values(corpus)),
        rel_types="\n".join(f"  - {v}" for v in rel_type_values(corpus)),
    )


def _section_text(section: Section) -> str:
    parts = [f"# {section.title}", section.content]
    for cb in section.code_blocks:
        parts.append(f"```{cb.language}\n{cb.code}\n```")
    for table in section.tables:
        if table.headers:
            parts.append(" | ".join(table.headers))
        for row in table.rows:
            parts.append(" | ".join(row))
    return "\n\n".join(parts)


def _validate_result(raw: dict, corpus: str, section_id: str) -> ExtractionResult:
    """Parse LLM JSON output and reject anything with unknown types."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    seen_fqns: set[str] = set()

    for e in raw.get("entities", []):
        if not valid_node_type(corpus, e.get("type", "")):
            logger.debug("Rejected entity type '%s' (not in schema)", e.get("type"))
            continue
        entity = Entity(
            id=e.get("id", e.get("fqn", "")),
            name=e.get("name", ""),
            fqn=e.get("fqn", ""),
            type=e["type"],
            description=e.get("description", ""),
            properties=e.get("properties", {}),
            aliases=e.get("aliases", []),
            source_section_id=section_id,
        )
        entities.append(entity)
        seen_fqns.add(entity.fqn)

    for r in raw.get("relationships", []):
        if not valid_rel_type(corpus, r.get("type", "")):
            logger.debug("Rejected rel type '%s' (not in schema)", r.get("type"))
            continue
        # Drop edges referencing entities not in this extraction batch
        if r.get("source_fqn") not in seen_fqns or r.get("target_fqn") not in seen_fqns:
            continue
        relationships.append(
            Relationship(
                source_fqn=r["source_fqn"],
                target_fqn=r["target_fqn"],
                type=r["type"],
                properties=r.get("properties", {}),
                source_section_id=section_id,
            )
        )

    return ExtractionResult(entities=entities, relationships=relationships, section_id=section_id)


async def _extract_section(
    section: Section, corpus: str, llm: LLMClient, system_prompt: str
) -> ExtractionResult:
    text = _section_text(section)
    try:
        resp = await llm.extract(text=text, system_prompt=system_prompt)
        # Strip markdown fences if present (LLM often wraps JSON in ```json ... ```)
        cleaned = re.sub(r"```(?:json)?\s*", "", resp.text).strip().rstrip("`").strip()
        raw = json.loads(cleaned)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.warning("Extraction failed for section %s: %s", section.id, exc)
        return ExtractionResult(section_id=section.id)
    except Exception as exc:
        raise GraphError(f"Unexpected extraction error for section {section.id}") from exc

    return _validate_result(raw, corpus, section.id)


_EXTRACTION_SEMAPHORE = asyncio.Semaphore(5)


async def extract_document(
    doc: Document, llm: LLMClient, system_prompt: str, event_callback=None
) -> ExtractionResult:
    """Extract all entities/relationships from a document's sections.

    Runs up to 5 section extractions concurrently.
    """
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    async def _bounded(section):
        async with _EXTRACTION_SEMAPHORE:
            return await _extract_section(section, doc.corpus, llm, system_prompt)

    results = await asyncio.gather(*[_bounded(s) for s in doc.sections])

    for result in results:
        all_entities.extend(result.entities)
        all_relationships.extend(result.relationships)

    if event_callback:
        for result in results:
            for entity in result.entities:
                await event_callback(
                    {
                        "type": "entity",
                        "data": {"id": entity.id, "name": entity.name, "type": entity.type},
                    }
                )
            for rel in result.relationships:
                await event_callback(
                    {
                        "type": "relationship",
                        "data": {
                            "source": rel.source_fqn,
                            "target": rel.target_fqn,
                            "type": rel.type,
                        },
                    }
                )

    merged_pairs, review_queue = resolve_entities(all_entities)
    if review_queue:
        logger.info("Document %s: %d entity pairs queued for review", doc.id, len(review_queue))

    absorbed_ids = {pair[1] for pair in merged_pairs if pair[1] != pair[2]}
    deduped = [e for e in all_entities if e.id not in absorbed_ids]

    return ExtractionResult(
        entities=deduped,
        relationships=all_relationships,
        document_id=doc.id,
    )


async def run_extraction(corpus: str = "custom", schema: str = "auto", event_callback=None) -> None:
    """Orchestrate: load processed JSONL → extract → resolve → load to Neo4j."""
    get_schema(corpus)

    processed_dir = Path(settings.datasets_path) / corpus / "processed"
    jsonl_files = list(processed_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No processed JSONL files found in %s", processed_dir)
        return

    llm = LLMClient()
    system_prompt = _build_system_prompt(corpus)
    store = await Neo4jStore.connect()

    # Load schema DDL — corpus-specific if available, otherwise default
    cypher_dir = Path("cypher")
    schema_file = cypher_dir / f"schema_{corpus}.cypher"
    if not schema_file.exists():
        schema_file = cypher_dir / "schema_default.cypher"
    if schema_file.exists():
        await store.load_schema(schema_file)

    node_enum, rel_enum = get_schema(corpus)
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    # Pre-scan to count total sections for progress
    docs_to_process: list[Document] = []
    for jsonl_path in jsonl_files:
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            docs_to_process.append(Document.model_validate_json(line))

    total_sections = sum(len(d.sections) for d in docs_to_process)

    if event_callback:
        await event_callback(
            {
                "type": "started",
                "data": {"corpus": corpus, "total_sections": total_sections},
            }
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} sections"),
        TimeElapsedColumn(),
        console=_console,
    ) as progress:
        extract_task = progress.add_task(f"Extracting [bold]{corpus}[/bold]", total=total_sections)

        for doc in docs_to_process:
            result = await extract_document(doc, llm, system_prompt, event_callback=event_callback)
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
            progress.advance(extract_task, advance=len(doc.sections))
            if event_callback:
                await event_callback(
                    {
                        "type": "section_done",
                        "data": {
                            "doc_id": doc.id,
                            "entities_count": len(result.entities),
                            "rels_count": len(result.relationships),
                        },
                    }
                )

    # Group by type for batch loading — nodes first, then edges
    from collections import defaultdict

    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for e in all_entities:
        nodes_by_type[e.type].append(e.model_dump(exclude={"embedding"}))

    edges_by_type: dict[str, list[dict]] = defaultdict(list)
    for r in all_relationships:
        edges_by_type[r.type].append(r.model_dump())

    total_loads = len(nodes_by_type) + len(edges_by_type)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        console=_console,
    ) as progress:
        load_task = progress.add_task("Loading to Neo4j", total=total_loads)

        for node_type_val, records in nodes_by_type.items():
            try:
                label = node_enum(node_type_val)
                created = await store.load_nodes(records, label)
                logger.info("Loaded %d %s nodes", created, node_type_val)
            except ValueError:
                logger.warning("Skipping unknown node type '%s'", node_type_val)
            progress.advance(load_task)

        for rel_type_val, records in edges_by_type.items():
            try:
                rel = rel_enum(rel_type_val)
                created = await store.load_edges(records, rel)
                logger.info("Loaded %d %s edges", created, rel_type_val)
            except ValueError:
                logger.warning("Skipping unknown rel type '%s'", rel_type_val)
            progress.advance(load_task)

    if event_callback:
        await event_callback(
            {
                "type": "complete",
                "data": {
                    "total_entities": len(all_entities),
                    "total_relationships": len(all_relationships),
                },
            }
        )

    await store.close()
    _console.print(
        f"[green]Done.[/green] {len(all_entities)} entities, "
        f"{len(all_relationships)} relationships extracted for [bold]{corpus}[/bold]"
    )
