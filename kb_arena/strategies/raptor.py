"""Strategy 6: RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Sarthi et al. 2024. Builds a hierarchical tree of LLM-generated cluster summaries
over the corpus. Leaf nodes (L0) = raw chunks. Higher levels = cluster summaries.
Query-time search across all levels simultaneously gives Tier 4/5 (integration,
architecture) questions access to broad topic synthesis that flat vector search misses.
"""

from __future__ import annotations

import logging
import time

import chromadb
import numpy as np

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.embeddings import OpenAIEmbedding
from kb_arena.tokenizer import detokenize, tokenize

logger = logging.getLogger(__name__)

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50

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
    text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[str]:
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
        self._llm = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self, level: int):
        ef = OpenAIEmbedding()
        return self._get_client().get_or_create_collection(
            name=f"raptor_l{level}",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

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

    async def _build_level(self, source_collection, target_collection, level_tag: str) -> int:
        """Cluster source_collection and upsert summaries to target_collection."""
        data = source_collection.get(include=["embeddings", "documents"])
        ids_list = data.get("ids") or []
        embeddings_raw = data.get("embeddings")
        embeddings = embeddings_raw if embeddings_raw is not None else []
        docs = data.get("documents") or []

        if not ids_list:
            return 0

        emb_array = np.array(embeddings, dtype=np.float32)
        k = max(1, len(ids_list) // 5)
        assignments = _cosine_kmeans(emb_array, k)

        clusters: dict[int, list[str]] = {}
        for ci, doc in zip(assignments, docs):
            clusters.setdefault(ci, []).append(doc)

        summary_ids, summary_texts, summary_metas = [], [], []
        for ci, texts in clusters.items():
            summary = await self._summarize_cluster(texts)
            summary_ids.append(f"{level_tag}_cluster_{ci}")
            summary_texts.append(summary)
            summary_metas.append({"source_id": f"cluster_{ci}", "level": int(level_tag[-1])})

        if summary_ids:
            batch = 500
            for start in range(0, len(summary_ids), batch):
                target_collection.upsert(
                    ids=summary_ids[start : start + batch],
                    documents=summary_texts[start : start + batch],
                    metadatas=summary_metas[start : start + batch],
                )

        return len(summary_ids)

    async def build_index(self, documents: list[Document]) -> None:
        """Chunk all sections → L0. Cluster L0 → L1 summaries. Optionally L1 → L2."""
        l0 = self._get_collection(0)
        ids, texts, metadatas = [], [], []

        for doc in documents:
            for section in doc.sections:
                chunks = _chunk_text(section.content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.id}::{section.id}::{i}"
                    ids.append(chunk_id)
                    texts.append(chunk)
                    metadatas.append({"source_id": doc.id, "level": 0})

        if ids:
            batch = 500
            for start in range(0, len(ids), batch):
                l0.upsert(
                    ids=ids[start : start + batch],
                    documents=texts[start : start + batch],
                    metadatas=metadatas[start : start + batch],
                )

        l1 = self._get_collection(1)
        n_l1 = await self._build_level(l0, l1, "l1")
        logger.info("RAPTOR: built %d L0 chunks, %d L1 summaries", len(ids), n_l1)

        if n_l1 >= 10:
            l2 = self._get_collection(2)
            n_l2 = await self._build_level(l1, l2, "l2")
            logger.info("RAPTOR: built %d L2 summaries", n_l2)

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Search L0, L1, L2 simultaneously → fuse context → Sonnet."""
        start = self._start_timer()

        retrieval_start = time.perf_counter()
        all_chunks: list[str] = []
        all_sources: set[str] = set()

        for level in (0, 1, 2):
            try:
                coll = self._get_collection(level)
                count = coll.count()
                if count == 0:
                    continue
                n = min(top_k, count)
                results = coll.query(query_texts=[question], n_results=n)
                chunks = results["documents"][0] if results["documents"] else []
                metas = results["metadatas"][0] if results["metadatas"] else []
                all_chunks.extend(chunks)
                for m in metas:
                    src = m.get("source_id", "")
                    if src:
                        all_sources.add(src)
            except Exception as exc:
                logger.debug("RAPTOR: skipping level %d - %s", level, exc)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not all_chunks:
            latency_ms = self._record_metrics(start)
            return AnswerResult(
                answer="No indexed content found. Run build-vectors --strategy raptor first.",
                sources=[],
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
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
