"""Strategy: BM25 - traditional keyword matching baseline.

Uses BM25Okapi for lexical retrieval, then LLM for answer generation.
The "pre-neural" baseline showing whether embeddings add value for your docs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the "
    "provided context passages. If the context does not contain the answer, "
    "say so. Be concise and accurate."
)


class BM25Strategy(Strategy):
    """BM25 keyword matching - the lexical retrieval baseline."""

    name = "bm25"

    def __init__(self) -> None:
        super().__init__()
        self._bm25: BM25Okapi | None = None
        self._corpus_texts: list[str] = []
        self._corpus_sources: list[str] = []
        self._chunk_ids: list[str] = []
        self._index_path: Path | None = None
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """Tokenize all sections for BM25 scoring."""
        texts: list[str] = []
        sources: list[str] = []
        chunk_ids: list[str] = []
        for doc in documents:
            for section in doc.sections:
                content = section.content.strip()
                if content:
                    texts.append(content)
                    sources.append(doc.id)
                    chunk_ids.append(f"{doc.id}::{section.id}")

        if not texts:
            log.warning("No text content found for BM25 index")
            return

        self._corpus_texts = texts
        self._corpus_sources = sources
        self._chunk_ids = chunk_ids

        # BM25 uses whitespace tokenization intentionally - it's the lexical baseline
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

        # Persist for query-time loading
        corpus_name = documents[0].metadata.get("corpus", "default") if documents else "default"
        index_dir = Path(settings.datasets_path) / corpus_name / "processed"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "bm25_index.json"
        index_path.write_text(
            json.dumps(
                {"texts": texts, "sources": sources, "chunk_ids": chunk_ids},
                ensure_ascii=False,
            )
        )
        self._index_path = index_path
        log.info("BM25 index built: %d passages", len(texts))

    def _ensure_index(self, corpus: str = "") -> bool:
        """Load BM25 index from disk if not already loaded."""
        if self._bm25 is not None:
            return True

        # Try to find the index
        search_paths = []
        if corpus:
            search_paths.append(
                Path(settings.datasets_path) / corpus / "processed" / "bm25_index.json"
            )
        # Glob for any corpus
        datasets = Path(settings.datasets_path)
        if datasets.exists():
            for p in datasets.glob("*/processed/bm25_index.json"):
                search_paths.append(p)

        for path in search_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    self._corpus_texts = data["texts"]
                    self._corpus_sources = data["sources"]
                    # chunk_ids was added in v0.5.0; fall back to synthesized IDs
                    # for indexes built by older versions so the index keeps working.
                    self._chunk_ids = data.get("chunk_ids") or [
                        f"{src}::passage-{i}" for i, src in enumerate(self._corpus_sources)
                    ]
                    tokenized = [t.lower().split() for t in self._corpus_texts]
                    self._bm25 = BM25Okapi(tokenized)
                    self._index_path = path
                    return True
                except (json.JSONDecodeError, KeyError):
                    log.warning("Corrupt BM25 index at %s", path)
                    continue
        return False

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        start = self._start_timer()

        if not self._ensure_index():
            return AnswerResult(
                answer="BM25 index not built. Run: kb-arena build-vectors --strategy bm25",
                sources=[],
                strategy=self.name,
            )

        # Retrieval phase
        retrieval_start = time.perf_counter()
        query_tokens = question.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        passages = [self._corpus_texts[i] for i in top_indices]
        sources = list({self._corpus_sources[i] for i in top_indices})
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=self._chunk_ids[idx],
                doc_id=self._corpus_sources[idx],
                content=self._corpus_texts[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                source_strategy=self.name,
            )
            for rank, idx in enumerate(top_indices)
        ]
        trace = RetrievalTrace(
            query=question, retrieved=retrieved_chunks, latency_ms=retrieval_ms, top_k=top_k
        )

        context = "\n\n---\n\n".join(passages)

        # Generation phase
        gen_start = time.perf_counter()
        llm = self._get_llm()
        resp = await llm.generate(query=question, context=context, system_prompt=SYSTEM_PROMPT)
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
