"""Multi-pass evaluator: structural checks, entity coverage, source attribution,
LLM-as-judge, and optional RAGAS metrics.

Includes evaluation memoization to avoid re-scoring identical answer+reference pairs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import weakref
from collections import OrderedDict

from kb_arena.llm.client import LLMClient
from kb_arena.models.benchmark import Constraints, GroundTruth, Score
from kb_arena.settings import settings
from kb_arena.telemetry import traced_span

logger = logging.getLogger(__name__)

# Memoization cache keyed by every input that can change an evaluation.
_EVAL_CACHE_MAX_ENTRIES = 1024


class _ClientIdentity:
    """Hash an evaluator client by object identity, even if the client is unhashable."""

    __slots__ = ("client",)

    def __init__(self, client: object | None):
        self.client = client

    def __hash__(self) -> int:
        return id(self.client)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ClientIdentity) and self.client is other.client


_EvalCacheKey = tuple[_ClientIdentity, str]
_eval_cache: OrderedDict[_EvalCacheKey, Score] = OrderedDict()
_eval_inflight: dict[
    tuple[asyncio.AbstractEventLoop, _ClientIdentity, str], asyncio.Task[Score]
] = {}
# The judge cost belongs to whoever receives the result, not to whoever started the
# task. asyncio.shield keeps a shared evaluation running after its creator is
# cancelled, so tying the cost to creation would charge real spend to nobody. The
# weak keys drop themselves when the task goes away, including when every waiter is
# cancelled and no one claims the cost.
_eval_cost_unclaimed: weakref.WeakKeyDictionary[asyncio.Task[Score], bool] = (
    weakref.WeakKeyDictionary()
)


class EvaluationExecutionError(RuntimeError):
    """The evaluator could not produce a trustworthy score."""


class _CostTrackingLLM:
    """Track judge spend while preserving the LLM client's interface."""

    def __init__(self, delegate: LLMClient):
        self._delegate = delegate
        self.cost_usd = 0.0

    async def judge(self, *args, **kwargs):
        response = await self._delegate.judge(*args, **kwargs)
        self.cost_usd += response.cost_usd
        return response


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a retrieval benchmark.

Given the question, a reference answer, and a candidate answer, score the candidate on three
dimensions. Judge the candidate against the question. The reference shows what a correct
answer contains, but a correct answer can use different words.
Return ONLY valid JSON with these exact keys:
{
  "accuracy": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "rationale": "<one sentence naming the main reason for the scores>"
}

Scoring guidance:
- accuracy: Does the candidate answer the question correctly? 1.0 = fully correct, 0.0 = wrong
- completeness: Does it cover all key points in the reference? 1.0 = complete, 0.5 = partial
- faithfulness: Does it avoid hallucination/fabrication? 1.0 = no fabrication, 0.0 = makes things up

Be strict. A partially correct answer scores 0.5-0.7, not 0.9."""


# The verdict text a record keeps. Enough for a reader to see the JSON and
# its rationale, small enough that a results file stays readable.
_JUDGE_RAW_LIMIT = 1200


def judge_prompt_hash() -> str:
    """One hash over the system prompt and the user template, so either moving shows."""
    from kb_arena.llm.client import JUDGE_USER_TEMPLATE

    template = "\n".join(JUDGE_USER_TEMPLATE[k] for k in ("question", "reference", "candidate"))
    return _hash_text(JUDGE_SYSTEM_PROMPT + "\n" + template)


def _parse_verdict(text: str) -> dict:
    """The first JSON object in the judge's reply.

    The rationale is prose, so a brace or a quote inside it must not end the
    object early. A real decoder reads from each opening brace until one
    parses, instead of a regex that stops at the first closing brace.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(parsed, dict):
            return parsed
        start = text.find("{", start + 1)
    raise ValueError("judge returned no JSON score")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _structural_check(answer: str, constraints: Constraints) -> Score:
    """Pass 1: check must_mention and must_not_claim without LLM.

    Returns a Score. If false_claims found, accuracy is forced to 0.0.
    structural_pass=False means Pass 2 (LLM judge) should be skipped.
    """
    answer_lower = answer.lower()

    mentions_found = [
        term
        for term in constraints.must_mention
        if re.search(re.escape(term.lower()), answer_lower)
    ]
    false_claims = [
        term
        for term in constraints.must_not_claim
        if re.search(re.escape(term.lower()), answer_lower)
    ]

    mention_ratio = (
        len(mentions_found) / len(constraints.must_mention) if constraints.must_mention else 1.0
    )

    if false_claims:
        return Score(
            accuracy=0.0,
            completeness=mention_ratio,
            faithfulness=0.0,
            structural_pass=False,
            mentions_found=mentions_found,
            false_claims=false_claims,
        )

    return Score(
        accuracy=mention_ratio,  # provisional — will be refined by LLM judge
        completeness=mention_ratio,
        faithfulness=1.0,
        structural_pass=True,
        mentions_found=mentions_found,
        false_claims=[],
    )


def _check_entity_coverage(
    answer: str,
    required_entities: list[str],
) -> tuple[float, list[str]]:
    """Check how many required entities appear in the answer."""
    if not required_entities:
        return 1.0, []
    answer_lower = answer.lower()
    found = [ent for ent in required_entities if re.search(re.escape(ent.lower()), answer_lower)]
    ratio = len(found) / len(required_entities)
    return ratio, found


def _check_source_attribution(
    returned_sources: list[str],
    expected_refs: list[str],
) -> float:
    """Score how well the returned sources match expected source refs.

    Uses substring matching — a returned source "json.html#json.JSONDecodeError"
    matches expected ref "json.html#json.JSONDecodeError".
    """
    normalized_refs = [ref.strip().lower() for ref in expected_refs if ref.strip()]
    if not normalized_refs:
        return 1.0  # no expected refs = pass
    normalized_sources = [source.strip().lower() for source in returned_sources if source.strip()]
    if not normalized_sources:
        return 0.0  # expected refs but nothing returned

    matched = 0
    for ref in normalized_refs:
        if any(ref in source or source in ref for source in normalized_sources):
            matched += 1

    return matched / len(normalized_refs)


async def _evaluate_uncached(
    answer: str,
    ground_truth: GroundTruth,
    constraints: Constraints,
    sources: list[str] | None = None,
    llm: LLMClient | None = None,
    question_text: str = "",
    context_chunks: list[str] | None = None,
    reference_free: bool = False,
    strategy: str = "",
    corpus: str = "",
    top_k: int | None = None,
) -> Score:
    """Multi-pass evaluation.

    Pass 1: structural check (must_mention, must_not_claim) — <1ms
    Pass 2: entity coverage (required_entities) — <1ms
    Pass 3: source attribution (source_refs vs returned sources) — <1ms
    Pass 4: LLM-as-judge (accuracy, completeness, faithfulness) — ~500ms
    Pass 5: RAGAS metrics (faithfulness, context_precision, context_recall, answer_relevancy)

    reference_free=True skips passes that need ground_truth.answer and evaluates
    on faithfulness + answer relevancy only.
    """
    if reference_free:
        # Reference-free mode: skip structural checks that need ground truth
        score = Score(accuracy=0.0, completeness=0.0, faithfulness=1.0, structural_pass=True)
    else:
        score = _structural_check(answer, constraints)

        # Entity coverage
        entity_ratio, entities_found = _check_entity_coverage(
            answer, ground_truth.required_entities
        )
        score.entity_coverage = entity_ratio
        score.entities_found = entities_found

        # Source attribution
        score.source_attribution = _check_source_attribution(
            sources or [], ground_truth.source_refs
        )

    if not score.structural_pass or llm is None:
        return score

    tracked_llm = _CostTrackingLLM(llm)

    # Pass 4: LLM-as-judge (skip in reference-free mode)
    if not reference_free:
        try:
            with traced_span("kb_arena.judge", strategy=strategy, corpus=corpus, top_k=top_k):
                resp = await tracked_llm.judge(
                    answer=answer,
                    reference=ground_truth.answer,
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    question=question_text,
                )
            parsed = _parse_verdict(resp.text)
            required = {"accuracy", "completeness", "faithfulness"}
            missing = required - parsed.keys()
            if missing:
                raise ValueError(f"judge score missing fields: {sorted(missing)}")
            if any(
                isinstance(parsed[field], bool) or not isinstance(parsed[field], int | float)
                for field in required
            ):
                raise ValueError("judge scores must be JSON numbers between 0 and 1")
            judge_scores = {
                field: float(parsed[field])
                for field in ("accuracy", "completeness", "faithfulness")
            }
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in judge_scores.values()
            ):
                raise ValueError("judge scores must be finite numbers between 0 and 1")
            score.accuracy = judge_scores["accuracy"]
            score.completeness = judge_scores["completeness"]
            score.faithfulness = judge_scores["faithfulness"]
            score.judge_provider = resp.provider
            score.judge_model = resp.model
            score.judge_prompt_hash = judge_prompt_hash()
            score.judge_raw = resp.text[:_JUDGE_RAW_LIMIT]
            rationale = parsed.get("rationale")
            score.judge_rationale = rationale.strip() if isinstance(rationale, str) else ""
        except Exception as exc:
            raise EvaluationExecutionError(f"LLM judge failed: {exc}") from exc

    # Pass 5: RAGAS metrics (if enabled)
    if settings.benchmark_enable_ragas and llm is not None:
        from kb_arena.benchmark.ragas_metrics import (
            compute_answer_relevancy,
            compute_context_precision,
            compute_context_recall,
            compute_faithfulness,
        )

        chunks = context_chunks or []
        try:
            score.ragas_answer_relevancy = await compute_answer_relevancy(
                question_text, answer, tracked_llm
            )
            if chunks:
                score.ragas_faithfulness = await compute_faithfulness(answer, chunks, tracked_llm)
                score.ragas_context_precision = await compute_context_precision(
                    question_text, chunks, tracked_llm
                )
            if not reference_free and ground_truth.answer:
                score.ragas_context_recall = await compute_context_recall(
                    ground_truth.answer, chunks, tracked_llm
                )
        except Exception as exc:
            raise EvaluationExecutionError(f"RAGAS metrics failed: {exc}") from exc

    score.evaluation_cost_usd = tracked_llm.cost_usd
    return score


async def evaluate(
    answer: str,
    ground_truth: GroundTruth,
    constraints: Constraints,
    sources: list[str] | None = None,
    llm: LLMClient | None = None,
    question_text: str = "",
    context_chunks: list[str] | None = None,
    reference_free: bool = False,
    strategy: str = "",
    corpus: str = "",
    top_k: int | None = None,
) -> Score:
    """Evaluate once per unique input and share concurrent judge work.

    strategy/corpus/top_k only label the judge span for tracing; they never
    join cache_payload below, so two strategies that produce the same
    answer for the same question still share one judge call.
    """
    cache_payload = {
        "answer": answer,
        "ground_truth": ground_truth.model_dump(mode="json"),
        "constraints": constraints.model_dump(mode="json"),
        "sources": sources or [],
        "question_text": question_text,
        "context_chunks": context_chunks or [],
        "reference_free": reference_free,
        "ragas_enabled": settings.benchmark_enable_ragas,
        "llm_enabled": llm is not None,
        "llm_provider": settings.llm_provider,
        "judge_models": [
            settings.judge_model,
            settings.openai_judge_model,
            settings.ollama_judge_model,
        ],
    }
    cache_key = _hash_text(json.dumps(cache_payload, sort_keys=True, separators=(",", ":")))
    client_identity = _ClientIdentity(llm)
    scoped_cache_key = (client_identity, cache_key)
    cached = _eval_cache.get(scoped_cache_key)
    if cached is not None:
        _eval_cache.move_to_end(scoped_cache_key)
        logger.debug("Eval cache hit for %s", cache_key)
        result = cached.model_copy()
        result.evaluation_cost_usd = 0.0
        return result

    loop = asyncio.get_running_loop()
    inflight_key = (loop, client_identity, cache_key)
    task = _eval_inflight.get(inflight_key)
    if task is None:
        task = asyncio.create_task(
            _evaluate_uncached(
                answer,
                ground_truth,
                constraints,
                sources=sources,
                llm=llm,
                question_text=question_text,
                context_chunks=context_chunks,
                reference_free=reference_free,
                strategy=strategy,
                corpus=corpus,
                top_k=top_k,
            )
        )
        _eval_inflight[inflight_key] = task
        _eval_cost_unclaimed[task] = True

        def _finish(completed: asyncio.Task[Score]) -> None:
            if _eval_inflight.get(inflight_key) is completed:
                _eval_inflight.pop(inflight_key, None)
            if completed.cancelled():
                return
            try:
                score = completed.result()
            except Exception:
                return
            _eval_cache[scoped_cache_key] = score.model_copy()
            _eval_cache.move_to_end(scoped_cache_key)
            while len(_eval_cache) > _EVAL_CACHE_MAX_ENTRIES:
                _eval_cache.popitem(last=False)

        task.add_done_callback(_finish)

    score = await asyncio.shield(task)
    if _eval_cost_unclaimed.pop(task, False):
        return score
    shared = score.model_copy()
    shared.evaluation_cost_usd = 0.0
    return shared
