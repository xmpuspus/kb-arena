"""Strategy 12: Agentic — iterative retrieve-judge-refine loop over naive_vector.

Each round retrieves from the same Chroma collection naive_vector builds, then
asks the LLM whether the gathered context already answers the question. When it
does not, the LLM proposes a refined query and the loop retrieves again.

Both the number of retrieval rounds and the number of LLM calls are hard caps
set at construction time. The loop always stops at whichever cap it reaches
first, even when the judge keeps asking for another round, because an
unbounded agentic loop turns one question into an open-ended API bill.

This strategy makes several LLM calls per question (one judge call per round
plus the final answer), so `default_benchmark=False` in catalog.py keeps it
out of the standard `all` benchmark. Running it across a 75-question benchmark
is a deliberate, budgeted choice, not something `all` should do by default.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import chromadb

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k
from kb_arena.strategies.chroma_index import INIT_LOCK, parse_query_result, query_in_thread
from kb_arena.strategies.embeddings import get_embedding_function
from kb_arena.strategies.naive_vector import COLLECTION_NAME, NaiveVectorStrategy

# Conservative defaults: two retrieval rounds, three LLM calls total (two judge
# calls plus the final answer). Raise both together for a wider search.
DEFAULT_MAX_ITERATIONS = 2
DEFAULT_MAX_LLM_CALLS = 3

JUDGE_SYSTEM_PROMPT = (
    "You judge whether the context below already answers the question completely.\n"
    'Reply with JSON only, no other text: {"enough": true or false, '
    '"refined_query": "a better search query, or empty if enough is true"}'
)

ANSWER_SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the context "
    "gathered across the retrieval rounds below. If it is not enough, say so."
)


def _parse_judge_decision(text: str) -> tuple[bool, str]:
    """Parse the judge's JSON verdict; treat anything unparseable as "enough".

    A malformed verdict must not buy another retrieval round for free — that
    would spend the budget on guessing instead of on a real signal, so the
    loop stops and answers with whatever context it already has.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return True, ""
    if not isinstance(data, dict):
        return True, ""
    enough = bool(data.get("enough", True))
    refined_query = str(data.get("refined_query") or "")
    return enough, refined_query


class AgenticStrategy(Strategy):
    """Retrieve, judge, refine, repeat — under a hard iteration and call budget."""

    name = "agentic"

    def __init__(
        self,
        chroma_client: Any = None,
        llm_client: Any = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    ) -> None:
        super().__init__()
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_llm_calls < 1:
            raise ValueError("max_llm_calls must be at least 1")
        self._base = NaiveVectorStrategy(chroma_client=chroma_client)
        self._client = chroma_client
        self._collection = None
        self._llm = llm_client
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        # What the last query() call actually spent, so a caller can tell a
        # cheap answer from one that ran the loop to its cap.
        self.last_budget: dict[str, Any] = {}

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self):
        # Reads the same collection naive_vector builds — agentic has no index
        # of its own to build or keep in sync.
        with INIT_LOCK:
            if self._collection is not None:
                return self._collection
            ef = get_embedding_function()
            self._collection = self._get_client().get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    async def build_index(self, documents: list[Document]) -> None:
        await self._base.build_index(documents)

    async def _retrieve(self, query_text: str, top_k: int, corpus: str) -> list[RetrievedChunk]:
        collection = await asyncio.to_thread(self._get_collection)
        results = await query_in_thread(
            collection,
            COLLECTION_NAME,
            corpus,
            {
                "query_texts": [query_text],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            },
        )
        ids, chunks, metas, distances = parse_query_result(results)
        return [
            RetrievedChunk(
                chunk_id=metas[i].get("chunk_id") or ids[i],
                doc_id=metas[i].get("source_id") or "",
                content=chunks[i],
                score=1.0 - distances[i],
                rank=i + 1,
                source_strategy=self.name,
                metadata=dict(metas[i]),
            )
            for i in range(len(chunks))
        ]

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        """Retrieve, judge, refine — up to max_iterations rounds, max_llm_calls total."""
        validate_top_k(top_k)
        start = self._start_timer()
        llm = self._get_llm()

        query_text = question
        collected: dict[str, RetrievedChunk] = {}
        iterations_used = 0
        llm_calls_used = 0
        stop_reason = "judge_satisfied"
        retrieval_ms = 0.0

        while True:
            retrieve_t0 = time.perf_counter()
            for chunk in await self._retrieve(query_text, top_k, corpus):
                collected.setdefault(chunk.chunk_id, chunk)
            retrieval_ms += (time.perf_counter() - retrieve_t0) * 1000
            iterations_used += 1

            # One call must always stay in reserve for the final answer below,
            # so the judge never spends the last slot in the budget.
            if llm_calls_used >= self.max_llm_calls - 1:
                stop_reason = "llm_call_budget"
                break

            context = "\n\n---\n\n".join(c.content for c in collected.values())
            judge_resp = await llm.generate(
                query=question, context=context, system_prompt=JUDGE_SYSTEM_PROMPT
            )
            llm_calls_used += 1
            enough, refined_query = _parse_judge_decision(judge_resp.text)
            if enough:
                stop_reason = "judge_satisfied"
                break
            if iterations_used >= self.max_iterations:
                stop_reason = "iteration_budget"
                break
            query_text = refined_query or query_text

        ranked = sorted(collected.values(), key=lambda c: c.score, reverse=True)[:top_k]
        for i, chunk in enumerate(ranked):
            chunk.rank = i + 1
        trace = RetrievalTrace(
            query=question, retrieved=ranked, latency_ms=retrieval_ms, top_k=top_k
        )
        sources = list(dict.fromkeys(c.doc_id for c in ranked if c.doc_id))

        context = "\n\n---\n\n".join(c.content for c in ranked)
        gen_t0 = time.perf_counter()
        resp = await llm.generate(
            query=question, context=context, system_prompt=ANSWER_SYSTEM_PROMPT
        )
        llm_calls_used += 1
        gen_ms = (time.perf_counter() - gen_t0) * 1000

        self.last_budget = {
            "iterations_used": iterations_used,
            "llm_calls_used": llm_calls_used,
            "max_iterations": self.max_iterations,
            "max_llm_calls": self.max_llm_calls,
            "stop_reason": stop_reason,
        }

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
