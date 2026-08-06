"""Strategy 5: Hybrid Graph + Vector — intent-routed, RRF-fused.

Three-stage intent classification (via IntentRouter or keyword fallback):
- factoid / exploratory  -> contextual_vector (primary)
- comparison / relational -> knowledge_graph (primary)
- procedural             -> both retrieved, fused via Reciprocal Rank Fusion

For procedural questions, RRF combines the ranked passage lists from each
sub-strategy and the top-k fused chunks become the actual context for the final
Sonnet generation. Earlier versions reranked answer strings — that was wrong.
This version reranks passages, which is what the README has always claimed.
"""

from __future__ import annotations

import asyncio
import logging
import time

from kb_arena.models.document import Document
from kb_arena.models.graph import GraphContext
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult, Strategy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a documentation assistant with access to both a knowledge graph and a vector index.\n"
    "The context below comes from the most relevant sources across both retrieval methods.\n"
    "Answer accurately and completely. Cite where you found the information when useful."
)

# Standard RRF constant — Cormack et al. 2009 recommend 60.
_RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = _RRF_K
) -> list[RetrievedChunk]:
    """Fuse multiple ranked chunk lists into one. Higher RRF = better."""
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}
    for ranking in ranked_lists:
        for chunk in ranking:
            cid = chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + chunk.rank)
            # Keep the highest-ranked instance of each chunk for content/metadata.
            existing = chunk_by_id.get(cid)
            if existing is None or chunk.rank < existing.rank:
                chunk_by_id[cid] = chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    result: list[RetrievedChunk] = []
    for i, (cid, fused_score) in enumerate(fused):
        c = chunk_by_id[cid].model_copy()
        c.rank = i + 1
        c.score = fused_score
        result.append(c)
    return result


def _merge_sources(*source_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    merged = []
    for sl in source_lists:
        for s in sl:
            if s and s not in seen:
                seen.add(s)
                merged.append(s)
    return merged


class HybridStrategy(Strategy):
    """Intent-routed retrieval: vector for simple, graph for complex, RRF-fused for procedural."""

    name = "hybrid"

    def __init__(self, neo4j_driver=None, chroma_client=None, router=None, llm=None):
        super().__init__()
        self._neo4j = neo4j_driver
        self._chroma = chroma_client
        self._router = router
        self._llm = llm
        self._vector_strategy = None
        self._graph_strategy = None

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    def _get_vector(self):
        if self._vector_strategy is None:
            from kb_arena.strategies.contextual_vector import ContextualVectorStrategy

            self._vector_strategy = ContextualVectorStrategy(chroma_client=self._chroma)
        return self._vector_strategy

    def _get_graph(self):
        if self._graph_strategy is None:
            from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy

            self._graph_strategy = KnowledgeGraphStrategy(neo4j_driver=self._neo4j)
        return self._graph_strategy

    async def _classify(self, question: str, history: list[dict] | None = None) -> str:
        """Use the IntentRouter when available; otherwise keyword fallback."""
        if self._router is not None:
            try:
                intent = await self._router.classify(question, history)
                return intent.value if hasattr(intent, "value") else str(intent)
            except Exception as exc:
                logger.warning("Intent router failed, using keyword fallback: %s", exc)

        q = question.lower()
        if any(kw in q for kw in ["compare", "vs", "difference", "versus"]):
            return "comparison"
        if any(kw in q for kw in ["depend", "require", "affect", "downstream"]):
            return "relational"
        if any(
            kw in q
            for kw in ["how do", "how can", "how to", "steps", "setup", "configure", "implement"]
        ):
            return "procedural"
        return "factoid"

    async def build_index(self, documents: list[Document]) -> None:
        """Delegate to both sub-strategies."""
        await self._get_vector().build_index(documents)
        await self._get_graph().build_index(documents)

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """Route by intent. Procedural fuses passages via RRF, then generates one final answer."""
        start = self._start_timer()
        intent = await self._classify(question)
        llm = self._get_llm()

        sources: list[str] = []
        graph_ctx: GraphContext | None = None
        total_tokens = 0
        total_cost = 0.0
        retrieval_ms = 0.0
        gen_ms = 0.0
        sub_traces: list[RetrievalTrace] = []

        if intent in ("comparison", "relational"):
            retrieval_start = time.perf_counter()
            graph_result = await self._get_graph().query(
                question,
                top_k=top_k,
                corpus=corpus,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            answer = graph_result.answer
            sources = graph_result.sources
            graph_ctx = graph_result.graph_context
            total_tokens = graph_result.tokens_used
            total_cost = graph_result.cost_usd
            gen_ms = graph_result.generation_latency_ms
            if graph_result.retrieval:
                sub_traces.append(graph_result.retrieval)

        elif intent in ("factoid", "exploratory"):
            retrieval_start = time.perf_counter()
            vector_result = await self._get_vector().query(
                question,
                top_k=top_k,
                corpus=corpus,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            answer = vector_result.answer
            sources = vector_result.sources
            total_tokens = vector_result.tokens_used
            total_cost = vector_result.cost_usd
            gen_ms = vector_result.generation_latency_ms
            if vector_result.retrieval:
                sub_traces.append(vector_result.retrieval)

        else:
            # Procedural — RRF fuse passages, generate one final answer over fused context.
            retrieval_start = time.perf_counter()
            vector_result, graph_result = await asyncio.gather(
                self._get_vector().query(question, top_k=top_k * 2, corpus=corpus),
                self._get_graph().query(question, top_k=top_k * 2, corpus=corpus),
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

            # Sub-strategies generated their own answers; we discard those and
            # pay only for the fusion generation. Track their costs anyway since
            # they were spent on retrieval+generation upstream — for cost honesty.
            total_tokens = vector_result.tokens_used + graph_result.tokens_used
            total_cost = vector_result.cost_usd + graph_result.cost_usd

            vec_chunks: list[RetrievedChunk] = (
                list(vector_result.retrieval.retrieved) if vector_result.retrieval else []
            )
            graph_chunks: list[RetrievedChunk] = (
                list(graph_result.retrieval.retrieved) if graph_result.retrieval else []
            )

            if vector_result.retrieval:
                sub_traces.append(vector_result.retrieval)
            if graph_result.retrieval:
                sub_traces.append(graph_result.retrieval)

            fused = _reciprocal_rank_fusion([vec_chunks, graph_chunks])[:top_k]

            # Build the context from the actual passage content of the fused chunks.
            context_parts: list[str] = []
            for ch in fused:
                if ch.content:
                    context_parts.append(ch.content)
            # Add graph node descriptions when chunks didn't carry content.
            if graph_result.graph_context and not context_parts:
                for node in graph_result.graph_context.nodes[:top_k]:
                    desc = node.get("description") or ""
                    if desc:
                        context_parts.append(f"{node.get('name', '')}: {desc}")

            context = "\n\n---\n\n".join(context_parts)
            gen_start = time.perf_counter()
            resp = await llm.generate(
                query=question,
                context=context,
                system_prompt=SYSTEM_PROMPT,
            )
            gen_ms = (time.perf_counter() - gen_start) * 1000
            answer = resp.text
            total_tokens += resp.total_tokens
            total_cost += resp.cost_usd
            sources = _merge_sources(vector_result.sources, graph_result.sources)
            graph_ctx = graph_result.graph_context

        # Build the trace returned to the caller. For procedural we already have
        # the fused list; for the routed branches we merge sub-traces by best rank.
        if intent == "procedural":
            merged_chunks = fused
        else:
            merged_chunks = []
            seen_ids: set[str] = set()
            for sub in sub_traces:
                for ch in sub.retrieved:
                    if ch.chunk_id in seen_ids:
                        continue
                    seen_ids.add(ch.chunk_id)
                    merged_chunks.append(ch)
            merged_chunks.sort(key=lambda c: c.score, reverse=True)
            for i, ch in enumerate(merged_chunks[:top_k]):
                ch.rank = i + 1
            merged_chunks = merged_chunks[:top_k]

        trace = RetrievalTrace(
            query=question,
            retrieved=merged_chunks,
            latency_ms=retrieval_ms,
            top_k=top_k,
        )

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources, graph_context=graph_ctx
        )
        return AnswerResult(
            answer=answer,
            sources=sources,
            graph_context=graph_ctx,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )

    async def stream_answer(
        self,
        question: str,
        history: list[dict] | None = None,
        corpus: str = "all",
    ):
        """Stream from the routed sub-strategy. Procedural falls back to vector for latency."""
        intent = await self._classify(question, history)
        primary = (
            self._get_graph() if intent in ("comparison", "relational") else self._get_vector()
        )
        async for token in primary.stream_answer(question, history, corpus=corpus):
            yield token
        # The sub-strategy's stream_answer already yields a meta packet at the end;
        # api.py consumes the LAST one it sees, so no extra emission here.
        # But if the sub-strategy didn't yield a meta packet (legacy plugin),
        # build one from a non-streaming query() so the SSE consumer gets accurate meta.
        # We can't easily detect that mid-stream — accept the cost of one extra query() only
        # when no meta arrived; for now rely on default base behaviour which always emits.
        return
