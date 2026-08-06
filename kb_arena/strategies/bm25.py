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
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k

log = logging.getLogger(__name__)
BM25_INDEX_FORMAT_VERSION = 2

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
        self._loaded_corpus = ""
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """Build and persist one BM25 index per corpus."""
        passages_by_corpus: dict[str, dict[str, list[str]]] = {}
        for doc in documents:
            passages = passages_by_corpus.setdefault(
                doc.corpus,
                {"texts": [], "sources": [], "chunk_ids": []},
            )
            for section in doc.sections:
                content = section.content.strip()
                if content:
                    passages["texts"].append(content)
                    passages["sources"].append(doc.id)
                    passages["chunk_ids"].append(f"{doc.id}::{section.id}")

        passages_by_corpus = {
            corpus: passages for corpus, passages in passages_by_corpus.items() if passages["texts"]
        }
        if not passages_by_corpus:
            log.warning("No text content found for BM25 index")
            return

        for corpus, passages in passages_by_corpus.items():
            index_dir = Path(settings.datasets_path) / corpus / "processed"
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path = index_dir / "bm25_index.json"
            payload = {
                "format_version": BM25_INDEX_FORMAT_VERSION,
                "corpus": corpus,
                **passages,
            }
            index_path.write_text(
                json.dumps(payload, ensure_ascii=False),
            )
            log.info("BM25 index built for %s: %d passages", corpus, len(passages["texts"]))

        combined = {"texts": [], "sources": [], "chunk_ids": []}
        for passages in passages_by_corpus.values():
            for key in combined:
                combined[key].extend(passages[key])
        loaded_corpus = "all" if len(passages_by_corpus) > 1 else next(iter(passages_by_corpus))
        self._activate_index(combined, loaded_corpus)

    def _activate_index(
        self,
        data: dict[str, list[str]],
        corpus: str,
        path: Path | None = None,
    ) -> None:
        self._corpus_texts = data["texts"]
        self._corpus_sources = data["sources"]
        self._chunk_ids = data.get("chunk_ids") or [
            f"{src}::passage-{i}" for i, src in enumerate(self._corpus_sources)
        ]
        tokenized = [text.lower().split() for text in self._corpus_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._index_path = path
        self._loaded_corpus = corpus

    def _ensure_index(self, corpus: str = "") -> bool:
        """Load BM25 index from disk if not already loaded."""
        requested_corpus = corpus or "all"
        if self._bm25 is not None and (
            requested_corpus == self._loaded_corpus
            or (requested_corpus == "all" and not self._loaded_corpus)
        ):
            return True

        if corpus:
            search_paths = [Path(settings.datasets_path) / corpus / "processed" / "bm25_index.json"]
        else:
            search_paths = sorted(Path(settings.datasets_path).glob("*/processed/bm25_index.json"))

        combined = {"texts": [], "sources": [], "chunk_ids": []}
        loaded_paths: list[Path] = []
        for path in search_paths:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    stored_corpus = data.get("corpus")
                    if (
                        data.get("format_version") != BM25_INDEX_FORMAT_VERSION
                        or stored_corpus != path.parent.parent.name
                    ):
                        log.warning("Skipping legacy or mismatched BM25 index at %s", path)
                        continue
                    # Format 2 always writes real section IDs. A present-but-empty list
                    # means the index is corrupt, and synthesising passage IDs here would
                    # score fabricated provenance against the expected chunks.
                    chunk_ids = data["chunk_ids"]
                    if not isinstance(chunk_ids, list) or len(chunk_ids) != len(data["sources"]):
                        log.warning("Corrupt BM25 index at %s: chunk_ids mismatch sources", path)
                        continue
                    combined["texts"].extend(data["texts"])
                    combined["sources"].extend(data["sources"])
                    combined["chunk_ids"].extend(chunk_ids)
                    loaded_paths.append(path)
                except (json.JSONDecodeError, KeyError):
                    log.warning("Corrupt BM25 index at %s", path)
                    continue

        if not loaded_paths:
            return False
        self._activate_index(
            combined,
            requested_corpus,
            loaded_paths[0] if len(loaded_paths) == 1 else None,
        )
        return True

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        validate_top_k(top_k)
        start = self._start_timer()

        selected_corpus = "" if corpus == "all" else corpus
        if not self._ensure_index(selected_corpus):
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
