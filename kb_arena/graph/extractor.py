"""LLM-based entity/relationship extraction with schema-constrained output."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from importlib.resources import as_file, files
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
        # NOTE: do not drop edges referencing entities outside this section batch.
        # Cross-section edges are essential for multi-hop graph queries and are
        # validated globally after all sections have been extracted (see run_extraction).
        if not r.get("source_fqn") or not r.get("target_fqn"):
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
    except Exception as exc:
        raise GraphError(f"Unexpected extraction error for section {section.id}") from exc

    try:
        # Strip markdown fences if present (LLM often wraps JSON in ```json ... ```)
        cleaned = re.sub(r"```(?:json)?\s*", "", resp.text).strip().rstrip("`").strip()
        raw = json.loads(cleaned)
        if not isinstance(raw, dict):
            raise TypeError("extraction response must be a JSON object")
        entities = raw.get("entities", [])
        relationships = raw.get("relationships", [])
        if not isinstance(entities, list) or not isinstance(relationships, list):
            raise TypeError("entities and relationships must be arrays")
        if not all(isinstance(item, dict) for item in [*entities, *relationships]):
            raise TypeError("entities and relationships must contain objects")
        return _validate_result(raw, corpus, section.id)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as exc:
        raise GraphError(f"Invalid extraction response for section {section.id}") from exc


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

    results: list[ExtractionResult] = []
    for start in range(0, len(doc.sections), 5):
        tasks = [
            asyncio.create_task(_bounded(section)) for section in doc.sections[start : start + 5]
        ]
        try:
            results.extend(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

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

    # Stamp the source document id so graph retrieval can map an entity back to
    # its "doc::section" identity (section-level IR ground truth matching).
    # _validate_result runs per-section and has no doc context; this does.
    for e in deduped:
        e.source_doc_id = doc.id

    return ExtractionResult(
        entities=deduped,
        relationships=all_relationships,
        document_id=doc.id,
    )


async def _load_schema(store: Neo4jStore, corpus: str) -> None:
    """Load corpus-specific DDL from the installed package, falling back to the default."""
    legacy_constraints = await store.legacy_constraint_names()
    if legacy_constraints:
        names = ", ".join(legacy_constraints)
        raise GraphError(
            "Legacy graph constraints require an explicit migration before rebuilding "
            f"({names}). Use a dedicated Neo4j database, then run: "
            "kb-arena migrate-graph-schema --database <name> "
            "--confirm-dedicated-database"
        )

    package = files("kb_arena.cypher")
    schema_resource = package.joinpath(f"schema_{corpus}.cypher")
    if not schema_resource.is_file():
        schema_resource = package.joinpath("schema_default.cypher")
    if not schema_resource.is_file():
        raise GraphError("The installed KB Arena package does not contain a graph schema")

    with as_file(schema_resource) as schema_path:
        await store.load_schema(schema_path)


async def migrate_legacy_graph_schema(database: str) -> list[str]:
    """Install the 0.10 schema and explicitly remove known legacy constraints."""
    store = await Neo4jStore.connect(database=database)
    try:
        package = files("kb_arena.cypher")
        schema_resource = package.joinpath("schema_default.cypher")
        if not schema_resource.is_file():
            raise GraphError("The installed KB Arena package does not contain a graph schema")
        with as_file(schema_resource) as schema_path:
            await store.load_schema(schema_path)
        return await store.drop_legacy_constraints()
    finally:
        await store.close()


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

    store = await Neo4jStore.connect()
    try:
        await _load_schema(store, corpus)
    except BaseException:
        await store.close()
        raise

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} sections"),
            TimeElapsedColumn(),
            console=_console,
        ) as progress:
            extract_task = progress.add_task(
                f"Extracting [bold]{corpus}[/bold]", total=total_sections
            )

            for doc in docs_to_process:
                result = await extract_document(
                    doc, llm, system_prompt, event_callback=event_callback
                )
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
    except BaseException:
        await store.close()
        raise

    # Cross-section edge validation: keep only edges whose endpoints exist in the
    # global entity set. Entities are extracted per-section but documentation
    # routinely contains cross-section relationships (e.g. Lambda -> API Gateway
    # in different sections). Validating against the union preserves these.
    global_fqns: set[str] = {e.fqn for e in all_entities if e.fqn}
    valid_relationships: list[Relationship] = []
    dropped = 0
    for r in all_relationships:
        if r.source_fqn in global_fqns and r.target_fqn in global_fqns:
            valid_relationships.append(r)
        else:
            dropped += 1
    if dropped:
        logger.info(
            "Dropped %d edges referencing entities not in global FQN set "
            "(out of %d cross-section edges)",
            dropped,
            len(all_relationships),
        )
    all_relationships = valid_relationships

    # Group by type for batch loading — nodes first, then edges
    from collections import defaultdict

    nodes_by_type: dict[str, list[dict]] = defaultdict(list)
    for e in all_entities:
        record = e.model_dump(exclude={"embedding"})
        record["corpus"] = corpus
        record["entity_id"] = f"{corpus}::{e.fqn}"
        nodes_by_type[e.type].append(record)

    edges_by_type: dict[str, list[dict]] = defaultdict(list)
    for r in all_relationships:
        record = r.model_dump()
        record["corpus"] = corpus
        record["source_entity_id"] = f"{corpus}::{r.source_fqn}"
        record["target_entity_id"] = f"{corpus}::{r.target_fqn}"
        edges_by_type[r.type].append(record)

    total_loads = len(nodes_by_type) + len(edges_by_type)
    try:
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
    finally:
        await store.close()

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

    _console.print(
        f"[green]Done.[/green] {len(all_entities)} entities, "
        f"{len(all_relationships)} relationships extracted for [bold]{corpus}[/bold]"
    )
