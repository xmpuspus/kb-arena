"""Strategy: Metadata Filtered — metadata and access-aware dense retrieval.

A caller passes an `AccessFilter` describing what it may see: an allowed tag
set, an allowed owner set, a maximum classification level, and an allow-list
of document IDs. The filter is applied while Chroma searches, not after a
fixed top_k is already decided, so a restricted caller never loses a real hit
to a chunk it was never allowed to see in the first place.

Classification, owner, and doc ID are scalar per chunk, so they go straight
into the Chroma `where` clause alongside the existing activation/corpus
clause (see `chroma_index.index_where`). Tags are multi-valued, and Chroma
metadata values must be scalars, so tags are stored as a joined string and
checked in Python against an over-fetched candidate pool, then the result is
cut to top_k. Both paths filter before the top_k cut, never after.

Fail-closed by construction: a chunk with no recorded classification does not
match a classification `$in` clause, and a chunk with no recorded tags never
intersects a requested tag set. An unknown classification level cannot be
turned into a `$in` clause at all, so the strategy raises instead of guessing
whether it means "allow everything" or "allow nothing".
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import chromadb

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import (
    MAX_RETRIEVAL_CANDIDATES,
    AnswerResult,
    Strategy,
    validate_top_k,
)
from kb_arena.strategies.chroma_index import (
    INIT_LOCK,
    index_build_lock,
    new_generation,
    parse_query_result,
    publish_collection_build,
    query_in_thread,
    run_to_completion,
)
from kb_arena.strategies.embeddings import get_embedding_function
from kb_arena.strategies.naive_vector import _chunk_text

COLLECTION_NAME = "metadata_filtered"

# Lowest to highest sensitivity. A document with no declared classification
# defaults to the highest level, so an unclassified document is never handed
# out to a caller who only asked for "public".
CLASSIFICATION_LEVELS = ("public", "internal", "confidential", "restricted")

# Over-fetch this many times top_k before the tag filter runs, so tag
# exclusion shrinks the candidate pool instead of a result that is already
# capped at top_k.
TAG_FANOUT = 8

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


@dataclass(frozen=True, slots=True)
class AccessFilter:
    """One caller's view. A `None` field means that dimension is not restricted."""

    allowed_tags: frozenset[str] | None = None
    allowed_owners: frozenset[str] | None = None
    max_classification: str | None = None
    allowed_doc_ids: frozenset[str] | None = None


def _classification(doc: Document) -> str:
    value = doc.metadata.get("classification")
    return value if value in CLASSIFICATION_LEVELS else "restricted"


def _tags(doc: Document) -> str:
    return ",".join(sorted(str(t) for t in doc.metadata.get("tags", [])))


def _document_metadata(doc: Document) -> dict[str, Any]:
    return {
        "source_id": doc.id,
        "corpus": doc.corpus,
        "owner": str(doc.metadata.get("owner", "")),
        "classification": _classification(doc),
        "tags_csv": _tags(doc),
    }


def _allowed_classifications(max_classification: str) -> list[str]:
    """Every level at or below the requested ceiling, or raise if unknown.

    An unknown level cannot be mapped to a ceiling, so the caller gets a clear
    error instead of a filter that silently admits or denies everything.
    """
    if max_classification not in CLASSIFICATION_LEVELS:
        raise ValueError(
            f"Unknown classification level {max_classification!r}. "
            f"Known levels: {', '.join(CLASSIFICATION_LEVELS)}."
        )
    rank = CLASSIFICATION_LEVELS.index(max_classification)
    return list(CLASSIFICATION_LEVELS[: rank + 1])


def _excludes_everything(access_filter: AccessFilter) -> bool:
    """An explicit empty allow-list means nobody is allowed, not unrestricted."""
    return any(
        field is not None and not field
        for field in (
            access_filter.allowed_tags,
            access_filter.allowed_owners,
            access_filter.allowed_doc_ids,
        )
    )


def _scalar_where(access_filter: AccessFilter) -> dict[str, Any] | None:
    """Build the Chroma `where` clause for the scalar-valued filter dimensions."""
    clauses: list[dict[str, Any]] = []
    if access_filter.max_classification is not None:
        clauses.append(
            {"classification": {"$in": _allowed_classifications(access_filter.max_classification)}}
        )
    if access_filter.allowed_owners is not None:
        clauses.append({"owner": {"$in": list(access_filter.allowed_owners)}})
    if access_filter.allowed_doc_ids is not None:
        clauses.append({"source_id": {"$in": list(access_filter.allowed_doc_ids)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _matches_tags(metadata: dict[str, Any], allowed_tags: frozenset[str]) -> bool:
    doc_tags = set(metadata.get("tags_csv", "").split(","))
    return bool(doc_tags & allowed_tags)


class MetadataFilteredStrategy(Strategy):
    """Naive-vector chunking and embedding, with an access filter pushed into retrieval."""

    name = "metadata_filtered"

    def __init__(self, chroma_client=None):
        super().__init__()
        self._client = chroma_client
        self._collection = None
        self._llm = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self):
        with INIT_LOCK:
            if self._collection is not None:
                return self._collection
            ef = get_embedding_function()
            self._collection = self._get_client().get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """Chunk sections and upsert with owner, classification, and tag metadata."""
        collection = await asyncio.to_thread(self._get_collection)
        ids, texts, metadatas = [], [], []

        for doc in documents:
            meta = _document_metadata(doc)
            for section in doc.sections:
                chunks = _chunk_text(section.content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.id}::{section.id}::{i}"
                    ids.append(f"{doc.corpus}::{chunk_id}")
                    texts.append(chunk)
                    metadatas.append({**meta, "chunk_id": chunk_id})

        corpora = list(dict.fromkeys(doc.corpus for doc in documents))
        if not corpora:
            return
        generation = new_generation()
        async with index_build_lock():
            await run_to_completion(
                publish_collection_build,
                collection,
                COLLECTION_NAME,
                corpora,
                generation,
                ids,
                texts,
                metadatas,
            )

    async def query(
        self,
        question: str,
        top_k: int = 5,
        corpus: str = "all",
        access_filter: AccessFilter | None = None,
    ) -> AnswerResult:
        """Similarity search with an access filter applied during retrieval."""
        validate_top_k(top_k)
        start = self._start_timer()
        access_filter = access_filter or AccessFilter()

        retrieval_start = time.perf_counter()
        retrieved_chunks: list[RetrievedChunk] = []
        if not _excludes_everything(access_filter):
            collection = await asyncio.to_thread(self._get_collection)
            where = _scalar_where(access_filter)
            candidate_k = min(max(top_k * TAG_FANOUT, top_k), MAX_RETRIEVAL_CANDIDATES)
            query_kwargs: dict[str, Any] = {
                "query_texts": [question],
                "n_results": candidate_k,
                "include": ["documents", "metadatas", "distances"],
            }
            results = await query_in_thread(
                collection, COLLECTION_NAME, corpus, query_kwargs, where
            )
            ids, chunks, metas, distances = parse_query_result(results)

            if access_filter.allowed_tags is not None:
                allowed_tags = access_filter.allowed_tags
                keep = [i for i in range(len(chunks)) if _matches_tags(metas[i], allowed_tags)]
            else:
                keep = list(range(len(chunks)))
            keep = keep[:top_k]

            retrieved_chunks = [
                RetrievedChunk(
                    chunk_id=metas[i].get("chunk_id") or ids[i],
                    doc_id=metas[i].get("source_id", "") or "",
                    content=chunks[i],
                    score=1.0 - distances[i],
                    rank=rank + 1,
                    source_strategy=self.name,
                    metadata=dict(metas[i]),
                )
                for rank, i in enumerate(keep)
            ]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        trace = RetrievalTrace(
            query=question, retrieved=retrieved_chunks, latency_ms=retrieval_ms, top_k=top_k
        )
        sources = list(dict.fromkeys(c.doc_id for c in retrieved_chunks if c.doc_id))
        context = "\n\n---\n\n".join(c.content for c in retrieved_chunks)

        llm = self._get_llm()
        gen_start = time.perf_counter()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=SYSTEM_PROMPT,
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency_ms = self._record_metrics(
            start, tokens=resp.total_tokens, cost=resp.cost_usd, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
