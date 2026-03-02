"""Two-pass evaluator: structural checks (no LLM) then LLM-as-judge.

Enhanced with entity coverage and source attribution scoring.
"""

from __future__ import annotations

import json
import re

from kb_arena.llm.client import LLMClient
from kb_arena.models.benchmark import Constraints, GroundTruth, Score

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a retrieval benchmark.

Given a reference answer and a candidate answer, score the candidate on three dimensions.
Return ONLY valid JSON with these exact keys:
{
  "accuracy": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>
}

Scoring guidance:
- accuracy: Does the candidate answer the question correctly? 1.0 = fully correct, 0.0 = wrong
- completeness: Does it cover all key points in the reference? 1.0 = complete, 0.5 = partial
- faithfulness: Does it avoid hallucination/fabrication? 1.0 = no fabrication, 0.0 = makes things up

Be strict. A partially correct answer scores 0.5-0.7, not 0.9."""


def _structural_check(answer: str, constraints: Constraints) -> Score:
    """Pass 1: check must_mention and must_not_claim without LLM.

    Returns a Score. If false_claims found, accuracy is forced to 0.0.
    structural_pass=False means Pass 2 (LLM judge) should be skipped.
    """
    answer_lower = answer.lower()

    mentions_found = [
        term for term in constraints.must_mention
        if re.search(re.escape(term.lower()), answer_lower)
    ]
    false_claims = [
        term for term in constraints.must_not_claim
        if re.search(re.escape(term.lower()), answer_lower)
    ]

    mention_ratio = len(mentions_found) / len(constraints.must_mention) if constraints.must_mention else 1.0

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
    found = [
        ent for ent in required_entities
        if re.search(re.escape(ent.lower()), answer_lower)
    ]
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
    if not expected_refs:
        return 1.0  # no expected refs = pass
    if not returned_sources:
        return 0.0  # expected refs but nothing returned

    matched = 0
    returned_lower = [s.lower() for s in returned_sources]
    for ref in expected_refs:
        ref_lower = ref.lower()
        if any(ref_lower in src or src in ref_lower for src in returned_lower):
            matched += 1

    return matched / len(expected_refs)


async def evaluate(
    answer: str,
    ground_truth: GroundTruth,
    constraints: Constraints,
    sources: list[str] | None = None,
    llm: LLMClient | None = None,
) -> Score:
    """Multi-pass evaluation.

    Pass 1: structural check (must_mention, must_not_claim) — <1ms
    Pass 2: entity coverage (required_entities) — <1ms
    Pass 3: source attribution (source_refs vs returned sources) — <1ms
    Pass 4: LLM-as-judge (accuracy, completeness, faithfulness) — ~500ms
    """
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

    # Pass 4: LLM-as-judge
    try:
        raw = await llm.judge(
            answer=answer,
            reference=ground_truth.answer,
            system_prompt=JUDGE_SYSTEM_PROMPT,
        )
        # Extract JSON — tolerate markdown fences
        json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            score.accuracy = float(parsed.get("accuracy", score.accuracy))
            score.completeness = float(parsed.get("completeness", score.completeness))
            score.faithfulness = float(parsed.get("faithfulness", score.faithfulness))
    except Exception:
        # Structural score stands if judge fails
        pass

    return score
