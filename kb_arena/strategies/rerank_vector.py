"""Strategy 9: Cross-encoder Reranker over Naive Vector.

Wraps `naive_vector` retrieval at top_k * `RERANK_FANOUT` and rescores the
candidates with a cross-encoder. Two backends are supported:

* `bge`     — BAAI/bge-reranker-v2-m3 via sentence-transformers.CrossEncoder
              (free, fully on-prem, CPU-friendly).
* `cohere`  — Cohere Rerank v4 / v3.5 via the cohere SDK.
* `voyage`  — Voyage Rerank 2.5.
* `jina`    — Jina Reranker v3 via HTTP.

Pick the backend via `KB_ARENA_RERANKER_BACKEND` (default `bge` — no key needed).

Why a separate strategy: KB Arena's North Star is architecture-vs-architecture
benchmarking. The reranker layer is widely considered the highest-leverage
production accuracy lever in 2026; users need an apples-to-apples comparison
between "naive_vector" and "naive_vector + BGE rerank".
"""

from __future__ import annotations

import math
import time
from typing import Any

from kb_arena.exceptions import RerankerError
from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import (
    MAX_RETRIEVAL_CANDIDATES,
    AnswerResult,
    Strategy,
    validate_top_k,
)
from kb_arena.strategies.naive_vector import NaiveVectorStrategy

RERANK_FANOUT = 4  # retrieve top_k * 4, rerank, keep top_k


class _Reranker:
    """Backend protocol — `score(query, passages) -> list[float]`."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        raise NotImplementedError


class BGECrossEncoder(_Reranker):
    """BAAI/bge-reranker-v2-m3 — local, free, ~600 MB model, CPU-runnable."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for the bge reranker. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._ce = CrossEncoder(model)

    def score(self, query: str, passages: list[str]) -> list[float]:
        pairs = [(query, p) for p in passages]
        return [float(s) for s in self._ce.predict(pairs)]


class CohereReranker(_Reranker):
    """Cohere Rerank v4 / v3.5."""

    def __init__(self, api_key: str | None = None, model: str = "rerank-v3.5") -> None:
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise ImportError("cohere is required for the cohere reranker.") from exc
        from kb_arena.settings import settings

        self._client = cohere.Client(api_key=api_key or settings.cohere_api_key or None)
        self._model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        resp = self._client.rerank(
            query=query, documents=passages, model=self._model, top_n=len(passages)
        )
        # Cohere returns results sorted by relevance; map back to input order.
        scores = [0.0] * len(passages)
        for r in resp.results:
            scores[r.index] = float(r.relevance_score)
        return scores


class VoyageReranker(_Reranker):
    """Voyage Rerank 2.5."""

    def __init__(self, api_key: str | None = None, model: str = "rerank-2.5") -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("voyageai is required for the voyage reranker.") from exc
        from kb_arena.settings import settings

        self._client = voyageai.Client(api_key=api_key or settings.voyage_api_key or None)
        self._model = model

    def score(self, query: str, passages: list[str]) -> list[float]:
        resp = self._client.rerank(query=query, documents=passages, model=self._model)
        scores = [0.0] * len(passages)
        for r in resp.results:
            scores[r.index] = float(r.relevance_score)
        return scores


def _make_reranker() -> _Reranker:
    from kb_arena.settings import settings

    backend = (settings.reranker_backend or "bge").lower()
    if backend == "bge":
        return BGECrossEncoder(model=settings.reranker_model or "BAAI/bge-reranker-v2-m3")
    if backend == "cohere":
        return CohereReranker(model=settings.reranker_model or "rerank-v3.5")
    if backend == "voyage":
        return VoyageReranker(model=settings.reranker_model or "rerank-2.5")
    raise ValueError(f"Unknown KB_ARENA_RERANKER_BACKEND={backend!r}. Valid: bge, cohere, voyage.")


class RerankVectorStrategy(Strategy):
    """Naive Vector + cross-encoder reranking. Configurable backend."""

    name = "rerank_vector"

    def __init__(self, chroma_client: Any = None, llm_client: Any = None) -> None:
        super().__init__()
        self._base = NaiveVectorStrategy(chroma_client=chroma_client)
        self._reranker: _Reranker | None = None
        self._llm = llm_client

    def _get_reranker(self) -> _Reranker:
        if self._reranker is None:
            self._reranker = _make_reranker()
        return self._reranker

    async def build_index(self, documents: list[Document]) -> None:
        await self._base.build_index(documents)

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        start = self._start_timer()
        validate_top_k(top_k)
        # Retrieve a wider pool first.
        candidate_k = min(
            max(top_k * RERANK_FANOUT, top_k + 5),
            MAX_RETRIEVAL_CANDIDATES,
        )

        retrieve_t0 = time.perf_counter()
        candidate = await self._base.query(
            question,
            top_k=candidate_k,
            corpus=corpus,
        )
        retrieve_ms = (time.perf_counter() - retrieve_t0) * 1000

        chunks: list[RetrievedChunk] = (
            list(candidate.retrieval.retrieved) if candidate.retrieval else []
        )
        if not chunks:
            # Nothing to rerank; pass through the base answer unchanged.
            return candidate

        passages = [c.content or "" for c in chunks]
        try:
            scores = self._get_reranker().score(question, passages)
        except Exception as exc:
            raise RerankerError(f"Reranker backend failed: {exc}") from exc

        # A NaN compares False against everything, so it can hold rank 1 and silently
        # push the genuinely relevant chunk out of the generated context.
        for position, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise RerankerError(f"Reranker returned a non-numeric score at position {position}")
            if not math.isfinite(score):
                raise RerankerError(f"Reranker returned a non-finite score at position {position}")

        ranked: list[tuple[float, RetrievedChunk]] = sorted(
            zip(scores, chunks, strict=True), key=lambda x: x[0], reverse=True
        )
        kept = ranked[:top_k]
        for i, (s, c) in enumerate(kept):
            c.rank = i + 1
            c.score = float(s)

        # Generate a fresh answer over the reranked context — the base answer was
        # generated from the wider candidate pool, but we want the final answer
        # produced over the post-rerank top-k for a fair comparison.
        from kb_arena.llm.client import LLMClient

        llm = self._llm or LLMClient()
        context = "\n\n---\n\n".join(c.content or "" for _, c in kept)
        gen_t0 = time.perf_counter()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=(
                "You are a documentation assistant. Answer using only the context below."
            ),
        )
        gen_ms = (time.perf_counter() - gen_t0) * 1000

        sources = list(dict.fromkeys(c.doc_id for _, c in kept if c.doc_id))
        trace = RetrievalTrace(
            query=question,
            retrieved=[c for _, c in kept],
            latency_ms=retrieve_ms,
            top_k=top_k,
        )
        total_tokens = (candidate.tokens_used or 0) + resp.total_tokens
        total_cost = (candidate.cost_usd or 0.0) + resp.cost_usd

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieve_ms,
            generation_latency_ms=gen_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
