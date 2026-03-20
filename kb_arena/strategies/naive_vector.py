"""Strategy 1: Naive Vector RAG — the baseline everyone uses.

512-token chunks, 50-token overlap, minimal metadata (source_id only).
This is what happens when you "dump Confluence into a vector DB."
Deliberately simple — this is the strawman all others beat.
"""

from __future__ import annotations

import time

import chromadb

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.embeddings import OpenAIEmbedding
from kb_arena.tokenizer import detokenize, tokenize

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 50
COLLECTION_NAME = "naive_vector"

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the provided context.\n"
    "If the context doesn't contain enough information, say so. Be concise and accurate."
)


def _chunk_text(
    text: str, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = OVERLAP_TOKENS
) -> list[str]:
    """Split text into overlapping chunks by BPE token count."""
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
        """Chunk all sections and upsert into ChromaDB. Minimal metadata: source_id only."""
        collection = self._get_collection()
        ids, texts, metadatas = [], [], []

        for doc in documents:
            for section in doc.sections:
                chunks = _chunk_text(section.content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc.id}::{section.id}::{i}"
                    ids.append(chunk_id)
                    texts.append(chunk)
                    metadatas.append({"source_id": doc.id})

        if ids:
            # ChromaDB upsert in batches of 500 to avoid payload limits
            batch = 500
            for start in range(0, len(ids), batch):
                collection.upsert(
                    ids=ids[start : start + batch],
                    documents=texts[start : start + batch],
                    metadatas=metadatas[start : start + batch],
                )

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Top-k cosine similarity → concatenate chunks → Sonnet."""
        start = self._start_timer()
        collection = self._get_collection()

        retrieval_start = time.perf_counter()
        results = collection.query(query_texts=[question], n_results=top_k)
        chunks = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

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
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
