"""Strategy 12: HyDE — Hypothetical Document Embeddings, over naive_vector.

Asks the LLM for a hypothetical passage that would answer the question, then
retrieves naive_vector's index with that passage instead of the question text
(Gao et al., 2022). A plausible answer sits closer, in embedding space, to a
real answer chunk than the bare question does. The final answer still
responds to the real question, over the chunks the rewrite found.

Reuses naive_vector's own `query()` for the actual Chroma call, the same way
qiss and hybrid reuse a sub-strategy's full query and pay for, then discard,
its answer. The rewrite step and the final generation are the two calls that
make HyDE cost more than naive_vector per query.
"""

from __future__ import annotations

import time

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
from kb_arena.strategies.naive_vector import SYSTEM_PROMPT, NaiveVectorStrategy

HYDE_SYSTEM_PROMPT = (
    "Write a short passage that plausibly answers the question below, in a neutral "
    "documentation tone. State the likely facts directly. Do not say you cannot answer."
)


class HydeStrategy(Strategy):
    """Rewrites the query into a hypothetical answer before retrieving over naive_vector."""

    name = "hyde"

    def __init__(self, chroma_client=None, llm_client=None):
        super().__init__()
        self._base = NaiveVectorStrategy(chroma_client=chroma_client)
        self._llm = llm_client

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        await self._base.build_index(documents)

    async def _hypothetical_document(self, question: str) -> tuple[str, int, float]:
        """Ask the LLM for a hypothetical answer to embed instead of the question.

        Raises when the LLM call fails or returns nothing. Retrieving with an
        empty rewrite would still return chunks that look like a real result.
        """
        resp = await self._get_llm().generate(
            query=question, context="", system_prompt=HYDE_SYSTEM_PROMPT
        )
        if not resp.text.strip():
            raise ValueError("HyDE rewrite returned an empty hypothetical document")
        return resp.text, resp.total_tokens, resp.cost_usd

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        validate_top_k(top_k)
        start = self._start_timer()
        hypothetical, rewrite_tokens, rewrite_cost = await self._hypothetical_document(question)

        retrieval_start = time.perf_counter()
        candidate = await self._base.query(hypothetical, top_k=top_k, corpus=corpus)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        chunks: list[RetrievedChunk] = (
            list(candidate.retrieval.retrieved) if candidate.retrieval else []
        )
        sources = list(dict.fromkeys(c.doc_id for c in chunks if c.doc_id))
        context = "\n\n---\n\n".join(c.content for c in chunks if c.content)

        gen_start = time.perf_counter()
        resp = await self._get_llm().generate(
            query=question, context=context, system_prompt=SYSTEM_PROMPT
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        trace = RetrievalTrace(
            query=question, retrieved=chunks, latency_ms=retrieval_ms, top_k=top_k
        )
        total_tokens = rewrite_tokens + candidate.tokens_used + resp.total_tokens
        total_cost = rewrite_cost + candidate.cost_usd + resp.cost_usd

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
