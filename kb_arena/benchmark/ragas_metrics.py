"""RAGAS-compatible evaluation metrics.

Implements four RAGAS-style metrics using LLM-as-judge:
- Faithfulness: Does the answer contain only claims supported by the context?
- Context Precision: How much of the retrieved context is relevant to the question?
- Context Recall: Does the retrieved context contain all info needed for the reference answer?
- Answer Relevancy: How relevant is the answer to the question asked?

These run alongside (not replacing) the existing LLM judge accuracy metric.
"""

from __future__ import annotations

import json
import logging
import re

from kb_arena.llm.client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_FAITHFULNESS_PROMPT = """You evaluate whether an answer is faithful to the given context.

Given the context (retrieved chunks) and the answer, identify claims in the answer
that are NOT supported by the context.

Return JSON:
{
  "supported_claims": <int>,
  "total_claims": <int>,
  "faithfulness": <float 0.0-1.0>
}

A faithfulness of 1.0 means every claim in the answer is supported by the context.
0.0 means none are supported. If the answer has no factual claims, return 1.0."""

_CONTEXT_PRECISION_PROMPT = """You evaluate how much of the retrieved context is relevant.

Given the question and the retrieved context chunks, determine what fraction
of the context is actually useful for answering the question.

Return JSON:
{
  "relevant_chunks": <int>,
  "total_chunks": <int>,
  "context_precision": <float 0.0-1.0>
}

1.0 means all retrieved context is relevant. 0.0 means none is relevant."""

_CONTEXT_RECALL_PROMPT = """You evaluate whether the retrieved context contains enough information.

Given the reference answer and the retrieved context, determine what fraction
of the key facts in the reference answer can be found in the context.

Return JSON:
{
  "facts_found": <int>,
  "total_facts": <int>,
  "context_recall": <float 0.0-1.0>
}

1.0 means the context contains all info needed. 0.0 means none."""

_ANSWER_RELEVANCY_PROMPT = """You evaluate how relevant an answer is to the question.

Given the question and the answer, score how directly and completely the answer
addresses the question asked.

Return JSON:
{
  "relevancy": <float 0.0-1.0>
}

1.0 = perfectly addresses the question. 0.5 = partially relevant. 0.0 = irrelevant."""


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, tolerating markdown fences."""
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {}


async def compute_faithfulness(
    answer: str,
    context_chunks: list[str],
    llm: LLMClient,
) -> float:
    """Score how faithful the answer is to the retrieved context."""
    if not answer.strip() or not context_chunks:
        return 0.0

    context = "\n---\n".join(context_chunks[:10])  # limit context size
    user = f"Context:\n{context}\n\nAnswer:\n{answer}"

    try:
        resp: LLMResponse = await llm.judge(
            answer=user,
            reference="",
            system_prompt=_FAITHFULNESS_PROMPT,
            max_tokens=200,
        )
        parsed = _parse_json_response(resp.text)
        return float(parsed.get("faithfulness", 0.0))
    except Exception as exc:
        logger.warning("Faithfulness eval failed: %s", exc)
        return 0.0


async def compute_context_precision(
    question: str,
    context_chunks: list[str],
    llm: LLMClient,
) -> float:
    """Score how much of the retrieved context is relevant to the question."""
    if not context_chunks:
        return 0.0

    context = "\n---\n".join(f"Chunk {i + 1}: {c}" for i, c in enumerate(context_chunks[:10]))
    user = f"Question: {question}\n\nRetrieved context:\n{context}"

    try:
        resp = await llm.judge(
            answer=user,
            reference="",
            system_prompt=_CONTEXT_PRECISION_PROMPT,
            max_tokens=200,
        )
        parsed = _parse_json_response(resp.text)
        return float(parsed.get("context_precision", 0.0))
    except Exception as exc:
        logger.warning("Context precision eval failed: %s", exc)
        return 0.0


async def compute_context_recall(
    reference_answer: str,
    context_chunks: list[str],
    llm: LLMClient,
) -> float:
    """Score how much of the reference answer is supported by retrieved context."""
    if not reference_answer.strip() or not context_chunks:
        return 0.0

    context = "\n---\n".join(context_chunks[:10])
    user = f"Reference answer:\n{reference_answer}\n\nRetrieved context:\n{context}"

    try:
        resp = await llm.judge(
            answer=user,
            reference="",
            system_prompt=_CONTEXT_RECALL_PROMPT,
            max_tokens=200,
        )
        parsed = _parse_json_response(resp.text)
        return float(parsed.get("context_recall", 0.0))
    except Exception as exc:
        logger.warning("Context recall eval failed: %s", exc)
        return 0.0


async def compute_answer_relevancy(
    question: str,
    answer: str,
    llm: LLMClient,
) -> float:
    """Score how relevant the answer is to the question (no reference needed)."""
    if not answer.strip():
        return 0.0

    user = f"Question: {question}\n\nAnswer:\n{answer}"

    try:
        resp = await llm.judge(
            answer=user,
            reference="",
            system_prompt=_ANSWER_RELEVANCY_PROMPT,
            max_tokens=200,
        )
        parsed = _parse_json_response(resp.text)
        return float(parsed.get("relevancy", 0.0))
    except Exception as exc:
        logger.warning("Answer relevancy eval failed: %s", exc)
        return 0.0
