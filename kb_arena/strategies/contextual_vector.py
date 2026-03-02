"""Strategy 2: Contextual Vector RAG — same chunking with enriched metadata.

Prepends heading_path to each chunk (Anthropic's contextual retrieval technique).
Adds rich metadata for ChromaDB `where` filtering before similarity search.
The "best practice" vector approach — shows how much metadata helps.
"""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from kb_arena.models.document import Document, Section
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.naive_vector import CHUNK_TOKENS, OVERLAP_TOKENS, _chunk_text

COLLECTION_NAME = "contextual_vector"

SYSTEM_PROMPT = """You are a documentation assistant. Answer the question using ONLY the provided context.
The context includes section headings to help you understand where each passage comes from.
If the context doesn't contain enough information, say so. Be concise and accurate."""


def _heading_prefix(section: Section) -> str:
    """Build heading path prefix like '## json > json.loads'."""
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
        self._client = chroma_client
        self._collection = None
        self._llm = None

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            ef = OpenAIEmbeddingFunction(
                api_key=settings.anthropic_api_key or "sk-placeholder",
                model_name=settings.embedding_model,
            )
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

        query_kwargs: dict = {"query_texts": [question], "n_results": top_k}
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)
        chunks = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        sources = list({m.get("source_id", "") for m in metas if m.get("source_id")})
        context = "\n\n---\n\n".join(chunks)

        llm = self._get_llm()
        answer = await llm.generate(
            query=question,
            context=context,
            system_prompt=SYSTEM_PROMPT,
        )

        latency_ms = self._record_metrics(start, sources=sources)
        return AnswerResult(
            answer=answer,
            sources=sources,
            strategy=self.name,
            latency_ms=latency_ms,
        )
