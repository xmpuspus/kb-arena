"""Strategy 13: Multi-Query — sub-query decomposition fused with RRF, over naive_vector.

Asks the LLM for several independent search queries for one question, retrieves
naive_vector's index once per sub-query, and fuses the ranked chunk lists with
the same Reciprocal Rank Fusion hybrid.py uses for its procedural branch
(Cormack et al., 2009). One phrasing can miss a chunk that only matches
another phrasing; RRF lets any sub-query surface a chunk into the final
context.

Each sub-query call reuses naive_vector's full `query()`, so it also pays for,
then discards, a generated answer to that sub-query, the same trade-off qiss
and hybrid make for reuse over a second retrieval-only code path. The cost of
every discarded answer is tracked in the result, not dropped.
"""

from __future__ import annotations

import asyncio
import time

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
from kb_arena.strategies.hybrid import _reciprocal_rank_fusion
from kb_arena.strategies.naive_vector import SYSTEM_PROMPT, NaiveVectorStrategy

# The ceiling on `KB_ARENA_MULTI_QUERY_N`. One question already costs one
# rewrite call plus this many retrievals and generations.
MAX_SUB_QUERIES = 10

MULTI_QUERY_SYSTEM_PROMPT = (
    "Write {n} different search queries that would help find the answer to the question "
    "below. One per line, no numbering, no extra text."
)


class MultiQueryStrategy(Strategy):
    """Decomposes the query into sub-queries, retrieves each, and fuses with RRF."""

    name = "multi_query"

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

    async def _sub_queries(self, question: str) -> tuple[list[str], int, float]:
        """Ask the LLM for N sub-queries.

        Raises when none come back usable. A silent fall-back to the original
        question would run naive_vector once and call the result a fusion.
        """
        # Every sub-query becomes one concurrent retrieval and one LLM call, so
        # this number IS the fan-out. Only a floor was enforced, and a setting
        # of 10000 bought ten thousand concurrent generations from one question.
        n = min(max(int(settings.multi_query_n), 1), MAX_SUB_QUERIES)
        resp = await self._get_llm().generate(
            query=question,
            context="",
            system_prompt=MULTI_QUERY_SYSTEM_PROMPT.format(n=n),
        )
        lines = [ln.strip("-• \t") for ln in (resp.text or "").splitlines() if ln.strip()]
        subs = [ln for ln in lines if len(ln) > 3][:n]
        if not subs:
            raise ValueError("multi_query rewrite returned no usable sub-queries")
        return subs, resp.total_tokens, resp.cost_usd

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        validate_top_k(top_k)
        start = self._start_timer()
        subqueries, rewrite_tokens, rewrite_cost = await self._sub_queries(question)

        retrieval_start = time.perf_counter()
        sub_results = await asyncio.gather(
            *[self._base.query(sq, top_k=top_k, corpus=corpus) for sq in subqueries]
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        ranked_lists: list[list[RetrievedChunk]] = [
            list(r.retrieval.retrieved) for r in sub_results if r.retrieval
        ]
        fused = _reciprocal_rank_fusion(ranked_lists)[:top_k]

        sources = list(dict.fromkeys(c.doc_id for c in fused if c.doc_id))
        context = "\n\n---\n\n".join(c.content for c in fused if c.content)

        gen_start = time.perf_counter()
        resp = await self._get_llm().generate(
            query=question, context=context, system_prompt=SYSTEM_PROMPT
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        trace = RetrievalTrace(
            query=question, retrieved=fused, latency_ms=retrieval_ms, top_k=top_k
        )
        sub_tokens = sum(r.tokens_used for r in sub_results)
        sub_cost = sum(r.cost_usd for r in sub_results)
        total_tokens = rewrite_tokens + sub_tokens + resp.total_tokens
        total_cost = rewrite_cost + sub_cost + resp.cost_usd

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
