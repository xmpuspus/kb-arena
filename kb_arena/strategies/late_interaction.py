"""Strategy 12: Late Interaction, a ColBERT-style reranker over naive_vector.

A pooled embedding reduces a whole passage to one vector, so a query token with
no single strong match anywhere in the passage still scores through the average.
Late interaction keeps one embedding per token instead, and scores a query and a
passage by MaxSim: for every query token, take its best cosine match among the
passage tokens, then average those best matches. A passage only scores well when
every query token, not just the passage on the whole, finds real support.

This reranks the same `naive_vector` candidates that `rerank_vector` and the
quantum strategies rerank, so it inherits the corpus, ground truth, and IR
metrics without new plumbing. Needs the optional `[late-interaction]` extra
(transformers, torch), lazy-imported inside the token encoder so the core
package and CI stay light.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import (
    MAX_RETRIEVAL_CANDIDATES,
    AnswerResult,
    Strategy,
    validate_top_k,
)
from kb_arena.strategies.naive_vector import NaiveVectorStrategy

# ---------------------------------------------------------------------------
# Pure MaxSim math, no I/O, fully unit-testable without a model.
# ---------------------------------------------------------------------------


def unit_rows(mat: Any) -> np.ndarray:
    """Return each row of `mat` normalized to unit length; a zero row is left as is."""
    arr = np.atleast_2d(np.asarray(mat, dtype=np.float64))
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def maxsim(query_tokens: Any, doc_tokens: Any) -> float:
    """Average, over every query token, of its best cosine match among doc tokens.

    `query_tokens` and `doc_tokens` are (num_tokens, dim) matrices, one row per
    token. Rows do not need to already be unit length; this function normalizes
    them. Returns 0.0 when either side has no tokens, so an empty passage never
    wins by comparing against nothing.
    """
    q = unit_rows(query_tokens)
    d = unit_rows(doc_tokens)
    if q.shape[0] == 0 or d.shape[0] == 0:
        return 0.0
    sims = q @ d.T
    return float(sims.max(axis=1).mean())


def late_interaction_scores(query_tokens: Any, doc_token_list: list[Any]) -> np.ndarray:
    """Return one MaxSim score per document in `doc_token_list`.

    Document token counts vary, so each document is scored on its own rather
    than through one batched matrix multiply.
    """
    return np.array([maxsim(query_tokens, doc_tokens) for doc_tokens in doc_token_list])


# ---------------------------------------------------------------------------
# Token encoder, backend protocol so a future model swap needs no strategy change.
# ---------------------------------------------------------------------------


class _TokenEncoder:
    """Wraps a transformer encoder to emit one embedding per input token."""

    def __init__(self, model: str) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the late_interaction "
                "strategy. Install with: pip install 'kb-arena[late-interaction]'"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModel.from_pretrained(model)
        self._model.eval()

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        """Return one (num_tokens, dim) matrix of token embeddings per text."""
        out = []
        for text in texts:
            batch = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with self._torch.no_grad():
                hidden = self._model(**batch).last_hidden_state[0]
            out.append(hidden.numpy())
        return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class LateInteractionStrategy(Strategy):
    """ColBERT-style MaxSim reranker over naive_vector candidates."""

    name = "late_interaction"

    def __init__(self, chroma_client: Any = None, llm_client: Any = None) -> None:
        super().__init__()
        self._base = NaiveVectorStrategy(chroma_client=chroma_client)
        self._llm = llm_client
        self._encoder: _TokenEncoder | None = None

    def _get_encoder(self) -> _TokenEncoder:
        if self._encoder is None:
            from kb_arena.settings import settings

            self._encoder = _TokenEncoder(model=settings.late_interaction_model)
        return self._encoder

    async def build_index(self, documents: list[Document]) -> None:
        await self._base.build_index(documents)

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        from kb_arena.settings import settings

        start = self._start_timer()
        validate_top_k(top_k)
        fanout = max(int(settings.late_interaction_fanout), 1)
        candidate_k = min(
            max(top_k * fanout, top_k + 5),
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
            return candidate

        encoder = self._get_encoder()
        query_tokens = encoder.encode([question])[0]
        doc_token_list = encoder.encode([c.content or "" for c in chunks])
        scores = late_interaction_scores(query_tokens, doc_token_list)

        ranked = sorted(zip(scores.tolist(), chunks, strict=True), key=lambda x: x[0], reverse=True)
        kept = ranked[:top_k]
        for i, (s, c) in enumerate(kept):
            c.rank = i + 1
            c.score = float(s)
            c.metadata = {**c.metadata, "late_interaction_maxsim": float(s)}

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
