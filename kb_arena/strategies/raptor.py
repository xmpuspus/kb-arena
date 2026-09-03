"""Strategy 6: RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Sarthi et al. 2024. Builds a hierarchical tree of LLM-generated cluster summaries
over the corpus. Leaf nodes (L0) = raw chunks. Higher levels = cluster summaries.
Query-time search across all levels simultaneously gives Tier 4/5 (integration,
architecture) questions access to broad topic synthesis that flat vector search misses.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import chromadb
import numpy as np

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
from kb_arena.strategies.chroma_index import (
    INIT_LOCK,
    activate_generations,
    discard_staged_ids,
    index_activation_lock,
    index_build_lock,
    index_read_lock,
    index_where,
    new_generation,
    parse_query_result,
    prune_collection,
    run_to_completion,
    staged_where,
    upsert_staged_records,
)
from kb_arena.strategies.embeddings import get_embedding_function
from kb_arena.tokenizer import detokenize, tokenize

logger = logging.getLogger(__name__)

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
COLLECTION_NAMES = tuple(f"raptor_l{level}" for level in range(3))

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "The context includes both detailed passages and higher-level summaries.\n"
    "Use the most specific accurate information available. Be concise and accurate."
)

_SUMMARIZE_SYSTEM = (
    "You are a technical documentation analyst. Synthesize the following passages into a "
    "concise summary covering key concepts, entities, technical relationships, and "
    "configuration details. Write a single coherent paragraph."
)


def _chunk_text(
    text: str, chunk_tokens: int | None = None, overlap_tokens: int | None = None
) -> list[str]:
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


def _cosine_kmeans(embeddings: np.ndarray, k: int, max_iter: int = 15) -> list[int]:
    """K-means on L2-normalized embeddings. Returns assignment list."""
    n = len(embeddings)
    if n <= k:
        return list(range(n))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    idx = np.linspace(0, n - 1, k, dtype=int)
    centroids = normed[idx].copy()
    assignments = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        sims = normed @ centroids.T  # (n, k)
        new_assignments = np.argmax(sims, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for ci in range(k):
            mask = assignments == ci
            if mask.any():
                mean = normed[mask].mean(axis=0)
                norm = np.linalg.norm(mean)
                centroids[ci] = mean / (norm + 1e-8)
    return assignments.tolist()


class RaptorStrategy(Strategy):
    """RAPTOR hierarchical retrieval — L0 chunks + LLM cluster summaries at L1/L2."""

    name = "raptor"

    def __init__(self, chroma_client=None):
        super().__init__()
        self._client = chroma_client
        self._collections: dict[int, Any] = {}
        self._llm = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self, level: int):
        # get_or_create_collection reaches the sqlite system store. Resolving
        # it once per level keeps five concurrent searches from opening
        # fifteen write transactions on the pool.
        with INIT_LOCK:
            cached = self._collections.get(level)
            if cached is not None:
                return cached
            ef = get_embedding_function()
            collection = self._get_client().get_or_create_collection(
                name=f"raptor_l{level}",
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[level] = collection
            return collection

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def _summarize_cluster(self, texts: list[str]) -> str:
        joined = "\n\n---\n\n".join(texts[:20])  # cap at 20 chunks
        resp = await self._get_llm().generate(
            query="Synthesize these passages into a single summary.",
            context=joined,
            system_prompt=_SUMMARIZE_SYSTEM,
            max_tokens=512,
        )
        return resp.text.strip()

    async def _build_level(
        self,
        source_collection,
        target_collection,
        level_tag: str,
        corpus: str,
        generation: str,
    ) -> list[str]:
        """Cluster source_collection and upsert summaries to target_collection."""
        data = await asyncio.to_thread(
            source_collection.get,
            where=staged_where(corpus, generation),
            include=["embeddings", "documents"],
        )
        ids_list = data.get("ids") or []
        embeddings_raw = data.get("embeddings")
        embeddings = embeddings_raw if embeddings_raw is not None else []
        docs = data.get("documents") or []

        if not ids_list:
            return []

        emb_array = np.array(embeddings, dtype=np.float32)
        k = max(1, len(ids_list) // 5)
        assignments = await asyncio.to_thread(_cosine_kmeans, emb_array, k)

        clusters: dict[int, list[str]] = {}
        for ci, doc in zip(assignments, docs):
            clusters.setdefault(ci, []).append(doc)

        summary_ids, summary_texts, summary_metas = [], [], []
        for ci, texts in clusters.items():
            summary = await self._summarize_cluster(texts)
            summary_ids.append(f"{corpus}::{level_tag}_cluster_{ci}")
            summary_texts.append(summary)
            summary_metas.append(
                {
                    "source_id": f"cluster_{ci}",
                    "corpus": corpus,
                    "chunk_id": f"{level_tag}_cluster_{ci}",
                    "level": int(level_tag[-1]),
                }
            )

        return await run_to_completion(
            upsert_staged_records,
            target_collection,
            generation,
            summary_ids,
            summary_texts,
            summary_metas,
        )

    async def build_index(self, documents: list[Document]) -> None:
        """Chunk all sections → L0. Cluster L0 → L1 summaries. Optionally L1 → L2."""
        # get_or_create_collection and the embedding function set up on the
        # calling thread, so the cold start goes to a worker thread too.
        l0 = await asyncio.to_thread(self._get_collection, 0)
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
                            "level": 0,
                        }
                    )

        corpora = list(dict.fromkeys(doc.corpus for doc in documents))
        if not corpora:
            return
        l1 = await asyncio.to_thread(self._get_collection, 1)
        l2 = await asyncio.to_thread(self._get_collection, 2)

        generation = new_generation()
        collections = (l0, l1, l2)
        staged_by_level: list[list[str]] = [[], [], []]
        total_l1 = 0

        def _activate_and_prune() -> None:
            # The activation lock is a file lock and prune walks the whole
            # collection, so both stay off the event loop.
            with index_activation_lock():
                activate_generations(
                    {
                        collection_name: {corpus: generation for corpus in corpora}
                        for collection_name in COLLECTION_NAMES
                    }
                )
                for collection, collection_name in zip(collections, COLLECTION_NAMES):
                    try:
                        prune_collection(collection, collection_name, corpora)
                    except Exception as exc:
                        logger.warning(
                            "Could not prune inactive %s records: %s", collection_name, exc
                        )

        async with index_build_lock():
            try:
                staged_by_level[0] = await run_to_completion(
                    upsert_staged_records, l0, generation, ids, texts, metadatas
                )
                for corpus in corpora:
                    l1_ids = await self._build_level(l0, l1, "l1", corpus, generation)
                    staged_by_level[1].extend(l1_ids)
                    total_l1 += len(l1_ids)

                    l2_ids = (
                        await self._build_level(l1, l2, "l2", corpus, generation)
                        if len(l1_ids) >= 10
                        else []
                    )
                    staged_by_level[2].extend(l2_ids)
                    if l2_ids:
                        logger.info("RAPTOR: built %d L2 summaries for %s", len(l2_ids), corpus)

                await run_to_completion(_activate_and_prune)
            except Exception:
                for collection, staged_ids in zip(collections, staged_by_level):
                    await run_to_completion(discard_staged_ids, collection, staged_ids)
                raise

        logger.info("RAPTOR: built %d L0 chunks, %d L1 summaries", len(ids), total_l1)

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """Search L0, L1, L2 simultaneously → fuse context → Sonnet."""
        validate_top_k(top_k)
        start = self._start_timer()

        retrieval_start = time.perf_counter()
        all_chunks: list[str] = []
        all_sources: set[str] = set()
        retrieved_chunks: list[RetrievedChunk] = []

        def _search_levels():
            # Every level's count and query run on the calling thread inside
            # Chroma, so the whole locked search moves to a worker thread.
            found = []
            with index_read_lock():
                for level in (0, 1, 2):
                    coll = self._get_collection(level)
                    count = coll.count()
                    if count == 0:
                        continue
                    n = min(top_k, count)
                    query_kwargs = {
                        "query_texts": [question],
                        "n_results": n,
                        "include": ["documents", "metadatas", "distances"],
                    }
                    query_kwargs["where"] = index_where(COLLECTION_NAMES[level], corpus)
                    found.append((level, coll.query(**query_kwargs)))
            return found

        for level, results in await asyncio.to_thread(_search_levels):
            ids, chunks, metas, distances = parse_query_result(results)
            all_chunks.extend(chunks)
            for i, ch_text in enumerate(chunks):
                src = (metas[i].get("source_id") if i < len(metas) else "") or ""
                if src:
                    all_sources.add(src)
                raw_id = (
                    metas[i].get("chunk_id")
                    if i < len(metas) and metas[i].get("chunk_id")
                    else ids[i]
                    if i < len(ids)
                    else f"unknown-{i}"
                )
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=f"L{level}:{raw_id}",
                        doc_id=src,
                        content=ch_text,
                        score=1.0 - distances[i],
                        rank=len(retrieved_chunks) + 1,
                        source_strategy=self.name,
                        metadata={
                            "level": level,
                            **(dict(metas[i]) if i < len(metas) else {}),
                        },
                    )
                )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        trace = RetrievalTrace(
            query=question, retrieved=retrieved_chunks, latency_ms=retrieval_ms, top_k=top_k
        )

        if not all_chunks:
            latency_ms = self._record_metrics(start)
            return AnswerResult(
                answer="No indexed content found. Run build-vectors --strategy raptor first.",
                sources=[],
                retrieval=trace,
                strategy=self.name,
                latency_ms=latency_ms,
            )

        context = "\n\n---\n\n".join(all_chunks)
        llm = self._get_llm()
        gen_start = time.perf_counter()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=SYSTEM_PROMPT,
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        sources = list(all_sources)
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
