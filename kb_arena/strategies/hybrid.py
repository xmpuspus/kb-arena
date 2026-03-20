"""Strategy 5: Hybrid Graph + Vector — intent-routed, result-fused.

Three-stage intent classification determines routing:
- factoid / exploratory → contextual_vector
- comparison / relational → knowledge_graph
- procedural → both, results fused, deduplicated, re-ranked by Sonnet

Re-ranking: Sonnet scores each chunk 0-1, top 5 passed to final generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from kb_arena.models.document import Document
from kb_arena.models.graph import GraphContext
from kb_arena.strategies.base import AnswerResult, Strategy

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a documentation assistant with access to both a knowledge graph and a vector index.\n"
    "The context below comes from the most relevant sources across both retrieval methods.\n"
    "Answer accurately and completely. Cite where you found the information when useful."
)

RERANK_PROMPT = """Rate how relevant this passage is to the question on a scale of 0.0 to 1.0.
Return ONLY a JSON object: {"score": 0.8}

Question: {question}
Passage: {passage}"""


async def _score_passage(llm, question: str, passage: str) -> tuple[str, float]:
    """Score a single passage — runs concurrently with asyncio.gather."""
    try:
        resp = await llm.classify(
            query="",
            system_prompt=RERANK_PROMPT.format(question=question, passage=passage[:1000]),
            max_tokens=50,
        )
        data = json.loads(resp.strip())
        score = float(data.get("score", 0.5))
    except Exception:
        score = 0.5
    return passage, score


async def _rerank_passages(llm, question: str, passages: list[str]) -> list[tuple[str, float]]:
    """Score all passages concurrently with Haiku and return sorted (passage, score) pairs."""
    results = await asyncio.gather(*[_score_passage(llm, question, p) for p in passages])
    scored = list(results)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _deduplicate_passages(passages: list[str], threshold: int = 50) -> list[str]:
    """Remove near-duplicate passages by leading token overlap."""
    seen_prefixes: set[str] = set()
    unique = []
    for p in passages:
        prefix = " ".join(p.split()[:threshold])
        if prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            unique.append(p)
    return unique


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
    """Intent-routed retrieval: vector for simple, graph for complex, both for procedural."""

    name = "hybrid"

    def __init__(self, neo4j_driver=None, chroma_client=None, router=None):
        super().__init__()
        self._neo4j = neo4j_driver
        self._chroma = chroma_client
        self._router = router
        self._llm = None
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
        """Use the IntentRouter if available, fall back to keyword rules."""
        if self._router is not None:
            try:
                intent = await self._router.classify(question, history)
                return intent.value
            except Exception as exc:
                logger.warning("Intent router failed, using keyword fallback: %s", exc)

        # Inline keyword fallback (mirrors router logic)
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

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Route by intent, fuse results for procedural, re-rank top passages."""
        start = self._start_timer()
        intent = await self._classify(question)
        llm = self._get_llm()

        sources: list[str] = []
        graph_ctx: GraphContext | None = None
        passages: list[str] = []
        total_tokens = 0
        total_cost = 0.0
        retrieval_ms = 0.0
        gen_ms = 0.0

        if intent in ("comparison", "relational"):
            # Graph-primary
            retrieval_start = time.perf_counter()
            graph_result = await self._get_graph().query(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            answer = graph_result.answer
            sources = graph_result.sources
            graph_ctx = graph_result.graph_context
            total_tokens = graph_result.tokens_used
            total_cost = graph_result.cost_usd
            gen_ms = graph_result.generation_latency_ms

        elif intent in ("factoid", "exploratory"):
            # Vector-primary
            retrieval_start = time.perf_counter()
            vector_result = await self._get_vector().query(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            answer = vector_result.answer
            sources = vector_result.sources
            total_tokens = vector_result.tokens_used
            total_cost = vector_result.cost_usd
            gen_ms = vector_result.generation_latency_ms

        else:
            # Procedural — fuse both
            retrieval_start = time.perf_counter()
            vector_result = await self._get_vector().query(question, top_k=top_k)
            graph_result = await self._get_graph().query(question, top_k=top_k)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            total_tokens = vector_result.tokens_used + graph_result.tokens_used
            total_cost = vector_result.cost_usd + graph_result.cost_usd

            # Collect raw context chunks for re-ranking
            raw_passages = []
            if vector_result.answer:
                raw_passages.append(vector_result.answer)
            if graph_result.answer and graph_result.answer != vector_result.answer:
                raw_passages.append(graph_result.answer)

            # Add any additional short passages from graph context
            if graph_result.graph_context:
                for node in graph_result.graph_context.nodes[:5]:
                    desc = node.get("description", "")
                    if desc:
                        raw_passages.append(f"{node.get('name', '')}: {desc}")

            raw_passages = _deduplicate_passages(raw_passages)

            if len(raw_passages) > 1:
                scored = await _rerank_passages(llm, question, raw_passages[:10])
                passages = [p for p, _ in scored[:top_k]]
            else:
                passages = raw_passages

            context = "\n\n---\n\n".join(passages)
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

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources, graph_context=graph_ctx
        )
        return AnswerResult(
            answer=answer,
            sources=sources,
            graph_context=graph_ctx,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )

    async def stream_answer(self, question: str, history: list[dict] | None = None):
        """Route to appropriate strategy and stream its answer."""
        intent = await self._classify(question, history)

        if intent in ("comparison", "relational"):
            async for token in self._get_graph().stream_answer(question, history):
                yield token
            self.last_sources = self._get_graph().last_sources
            self.last_graph_context = self._get_graph().last_graph_context
            self.last_latency_ms = self._get_graph().last_latency_ms
        else:
            # For factoid/exploratory/procedural, stream from vector
            # (procedural full fusion is too slow for streaming; use vector as primary)
            async for token in self._get_vector().stream_answer(question, history):
                yield token
            self.last_sources = self._get_vector().last_sources
            self.last_latency_ms = self._get_vector().last_latency_ms
