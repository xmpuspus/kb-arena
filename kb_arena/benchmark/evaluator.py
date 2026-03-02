"""Two-pass evaluator: structural checks (no LLM) then LLM-as-judge."""

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


async def evaluate(
    answer: str,
    ground_truth: GroundTruth,
    constraints: Constraints,
    llm: LLMClient | None = None,
) -> Score:
    """Two-pass evaluation.

    Pass 1 is always run (structural, <1ms).
    Pass 2 (LLM judge) only runs if structural_pass=True and llm is provided.
    """
    score = _structural_check(answer, constraints)

    if not score.structural_pass or llm is None:
        return score

    # Pass 2: LLM-as-judge
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
