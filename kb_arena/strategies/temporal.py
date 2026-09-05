"""Strategy: Temporal — version-aware dense retrieval with an "as of" query.

Documents can carry `metadata.document_family`, `metadata.version`, and
`metadata.effective_date` (ISO 8601, e.g. "2026-01-01"). Documents that
declare no family are their own family of one, so untitled corpora behave
exactly like naive_vector.

By default the strategy prefers the highest version of each family. An
`as_of` date restricts each family to versions that were already effective on
that date, so a caller can ask what the corpus looked like at a past point in
time.

Once a newer eligible version of a family exists among the retrieved
candidates, every chunk from an older version of that family is dropped, not
only the chunks whose section happens to still exist in the new version. A
narrower or reorganized new version would otherwise leave a same-numbered old
chunk with no direct replacement, and that chunk would wrongly survive. The
selection runs on an over-fetched candidate pool before the top_k cut, so
version filtering never shrinks an already-final top_k.

An `as_of` value that is not a valid ISO date cannot be compared against any
effective_date, so the strategy raises rather than guessing "treat it as no
constraint" or "treat it as excluding everything".
"""

from __future__ import annotations

import asyncio
import time
from datetime import date
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

COLLECTION_NAME = "temporal"

# Over-fetch this many times top_k before version selection runs, so a family
# with several versions in the candidate pool still leaves top_k winners.
VERSION_FANOUT = 6

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


def _document_metadata(doc: Document) -> dict[str, Any]:
    return {
        "source_id": doc.id,
        "corpus": doc.corpus,
        "document_family": str(doc.metadata.get("document_family", doc.id)),
        "version": int(doc.metadata.get("version", 1)),
        # Sorts lexicographically like a real ISO date. An undated document is
        # always eligible, so it never disappears from an `as_of` query.
        "effective_date": str(doc.metadata.get("effective_date", "")),
    }


def _validate_as_of(as_of: str) -> str:
    """The date, normalised to `YYYY-MM-DD` for the comparison below.

    The comparison is lexicographic against stored `effective_date` strings,
    which are hyphenated. `date.fromisoformat` also accepts the compact form,
    and returning that text unchanged compared `20240601` against `2024-01-01`
    and picked the wrong version. Normalising makes both forms answer alike.
    """
    try:
        parsed = date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}.") from exc
    return parsed.isoformat()


def _current_versions(metas: list[dict[str, Any]], as_of: str | None) -> dict[str, tuple[int, str]]:
    """Return the winning (version, effective_date) per document family.

    A row is eligible when it has no effective_date or its effective_date is
    on or before `as_of`. The highest eligible version wins its family.
    """
    winners: dict[str, tuple[int, str]] = {}
    for meta in metas:
        effective_date = meta.get("effective_date", "")
        if as_of is not None and effective_date and effective_date > as_of:
            continue
        family = meta.get("document_family", "")
        candidate = (meta.get("version", 1), effective_date)
        if family not in winners or candidate > winners[family]:
            winners[family] = candidate
    return winners


class TemporalStrategy(Strategy):
    """Naive-vector chunking and embedding, with version selection pushed into retrieval."""

    name = "temporal"

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
        """Chunk sections and upsert with document family, version, and effective date."""
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
        as_of: str | None = None,
    ) -> AnswerResult:
        """Similarity search that keeps only each document family's current version."""
        validate_top_k(top_k)
        start = self._start_timer()
        if as_of is not None:
            as_of = _validate_as_of(as_of)

        collection = await asyncio.to_thread(self._get_collection)

        retrieval_start = time.perf_counter()
        candidate_k = min(max(top_k * VERSION_FANOUT, top_k), MAX_RETRIEVAL_CANDIDATES)
        query_kwargs: dict[str, Any] = {
            "query_texts": [question],
            "n_results": candidate_k,
            "include": ["documents", "metadatas", "distances"],
        }
        results = await query_in_thread(collection, COLLECTION_NAME, corpus, query_kwargs)
        ids, chunks, metas, distances = parse_query_result(results)

        winners = _current_versions(metas, as_of)
        keep = [
            i
            for i in range(len(chunks))
            if (metas[i].get("version", 1), metas[i].get("effective_date", ""))
            == winners.get(metas[i].get("document_family", ""))
        ][:top_k]
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

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
