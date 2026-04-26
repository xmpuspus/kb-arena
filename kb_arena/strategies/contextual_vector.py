"""Strategy 2: Contextual Vector RAG — same chunking with enriched metadata.

Prepends heading_path to each chunk (Anthropic's contextual retrieval technique).
Adds rich metadata for ChromaDB `where` filtering before similarity search.
The "best practice" vector approach — shows how much metadata helps.
"""

from __future__ import annotations

import time

import chromadb

from kb_arena.models.document import Document, Section
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.embeddings import OpenAIEmbedding
from kb_arena.strategies.naive_vector import CHUNK_TOKENS, OVERLAP_TOKENS, _chunk_text

COLLECTION_NAME = "contextual_vector"

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "The context includes section headings to help you understand where each passage comes from.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


def _heading_prefix(section: Section) -> str:
    """Build heading path prefix like '## Lambda > Invoke'."""
    if not section.heading_path:
        return section.title
    return " > ".join(section.heading_path)


def _enrich_chunk(chunk: str, section: Section) -> str:
    """Prepend heading path to chunk (contextual retrieval — Anthropic pattern)."""
    prefix = _heading_prefix(section)
    return f"## {prefix}\n\n{chunk}"


def _section_metadata(doc: Document, section: Section) -> dict:
    """Rich metadata for ChromaDB where-filtering."""
    return {
        "source_id": doc.id,
        "source_doc": doc.source,
        "section_path": " > ".join(section.heading_path) if section.heading_path else section.title,
        "module": section.heading_path[0] if section.heading_path else "",
        "has_code": len(section.code_blocks) > 0,
        "has_table": len(section.tables) > 0,
    }


class ContextualVectorStrategy(Strategy):
    """Vector RAG with heading-path context prepended and rich metadata filtering."""

    name = "contextual_vector"

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
        if self._collection is None:
            ef = OpenAIEmbedding()
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
        """Chunk sections, prepend heading path, upsert with rich metadata."""
        collection = self._get_collection()
        ids, texts, metadatas = [], [], []

        for doc in documents:
            for section in doc.sections:
                raw_chunks = _chunk_text(section.content, CHUNK_TOKENS, OVERLAP_TOKENS)
                meta = _section_metadata(doc, section)
                for i, chunk in enumerate(raw_chunks):
                    chunk_id = f"{doc.id}::{section.id}::{i}"
                    enriched = _enrich_chunk(chunk, section)
                    ids.append(chunk_id)
                    texts.append(enriched)
                    metadatas.append(meta)

        if ids:
            batch = 500
            for start in range(0, len(ids), batch):
                collection.upsert(
                    ids=ids[start : start + batch],
                    documents=texts[start : start + batch],
                    metadatas=metadatas[start : start + batch],
                )

    async def query(self, question: str, top_k: int = 5, where: dict | None = None) -> AnswerResult:
        """Similarity search with optional metadata pre-filter."""
        start = self._start_timer()
        collection = self._get_collection()

        retrieval_start = time.perf_counter()
        query_kwargs: dict = {
            "query_texts": [question],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)
        chunks = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        ids = results["ids"][0] if results.get("ids") else []
        distances = results["distances"][0] if results.get("distances") else []
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=ids[i] if i < len(ids) else f"unknown-{i}",
                doc_id=(metas[i].get("source_id") if i < len(metas) else "") or "",
                content=chunks[i] if i < len(chunks) else "",
                score=1.0 - (distances[i] if i < len(distances) else 0.0),
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
