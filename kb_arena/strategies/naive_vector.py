"""Strategy 1: Naive Vector RAG — the baseline everyone uses.

512-token chunks, 50-token overlap, minimal metadata (source_id only).
This is what happens when you "dump Confluence into a vector DB."
Deliberately simple — this is the strawman all others beat.
"""

from __future__ import annotations

import asyncio
import time

import chromadb

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
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
from kb_arena.tokenizer import detokenize, tokenize

# Back-compat constants — kept for callers that import them directly. The live
# defaults are KB_ARENA_CHUNK_TOKENS / KB_ARENA_CHUNK_OVERLAP_TOKENS in Settings;
# _chunk_text() resolves None args from there so `kb-arena optimize` can sweep.
CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
COLLECTION_NAME = "naive_vector"

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


def _chunk_text(
    text: str, chunk_tokens: int | None = None, overlap_tokens: int | None = None
) -> list[str]:
    """Split text into overlapping chunks by BPE token count.

    chunk_tokens / overlap_tokens default to the live Settings values when not
    passed explicitly, so a per-trial settings override (the optimizer) takes
    effect without touching call sites.
    """
    if chunk_tokens is None:
        chunk_tokens = settings.chunk_tokens
    if overlap_tokens is None:
        overlap_tokens = settings.chunk_overlap_tokens
    if chunk_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("chunk overlap must satisfy 0 <= overlap_tokens < chunk_tokens")
    tokens = tokenize(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunks.append(detokenize(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap_tokens

    return chunks


class NaiveVectorStrategy(Strategy):
    """Minimal vector RAG — ChromaDB + text-embedding-3-small, no metadata enrichment."""

    name = "naive_vector"

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
        """Chunk all sections and upsert into ChromaDB. Minimal metadata: source_id only."""
        collection = await asyncio.to_thread(self._get_collection)
        ids, texts, metadatas = [], [], []

        for doc in documents:
            for section in doc.sections:
                chunks = _chunk_text(section.content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.id}::{section.id}::{i}"
                    ids.append(f"{doc.corpus}::{chunk_id}")
                    texts.append(chunk)
                    metadatas.append(
                        {
                            "source_id": doc.id,
                            "corpus": doc.corpus,
                            "chunk_id": chunk_id,
                        }
                    )

        corpora = list(dict.fromkeys(doc.corpus for doc in documents))
        if not corpora:
            return
        generation = new_generation()
        async with index_build_lock():
            # Embedding and upsert run on the calling thread inside Chroma,
            # so they go to a worker thread to keep the loop free.
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

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """Top-k cosine similarity → concatenate chunks → Sonnet."""
        validate_top_k(top_k)
        start = self._start_timer()
        collection = await asyncio.to_thread(self._get_collection)

        retrieval_start = time.perf_counter()
        query_kwargs = {
            "query_texts": [question],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        results = await query_in_thread(collection, COLLECTION_NAME, corpus, query_kwargs)
        ids, chunks, metas, distances = parse_query_result(results)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=(
                    metas[i].get("chunk_id")
                    if i < len(metas) and metas[i].get("chunk_id")
                    else ids[i]
                    if i < len(ids)
                    else f"unknown-{i}"
                ),
                doc_id=(metas[i].get("source_id") if i < len(metas) else "") or "",
                content=chunks[i] if i < len(chunks) else "",
                score=1.0 - distances[i],
                rank=i + 1,
                source_strategy=self.name,
                metadata=dict(metas[i]) if i < len(metas) else {},
            )
            for i in range(len(chunks))
        ]
        trace = RetrievalTrace(
            query=question, retrieved=retrieved_chunks, latency_ms=retrieval_ms, top_k=top_k
        )

        sources = list({m.get("source_id", "") for m in metas if m.get("source_id")})
        context = "\n\n---\n\n".join(chunks)

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
