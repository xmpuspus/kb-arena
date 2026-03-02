"""Intent router — three-stage classification (Pattern 5 from cloudcare).

Stage 1: keyword scan, ~1ms, no LLM
Stage 2: Haiku LLM, ~20 tokens, <50ms
Stage 3: regex fallback — never fails

All 5 intents feed into hybrid strategy's routing logic.
"""

from __future__ import annotations

import re
from enum import Enum


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    COMPARISON = "comparison"
    RELATIONAL = "relational"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


CLASSIFY_SYSTEM = """You are a query classifier. Classify the user's question into exactly one of:
factoid, comparison, relational, procedural, exploratory

factoid: single fact lookup ("What is X?", "What does X return?")
comparison: comparing two or more things ("X vs Y", "difference between X and Y")
relational: dependency/impact questions ("What depends on X?", "What happens if X?")
procedural: how-to questions ("How do I...", "Steps to configure...")
exploratory: broad questions ("Tell me about X", "Explain X")

Return ONLY the intent word, nothing else."""


class IntentRouter:
    """Three-stage intent classifier: keyword scan → Haiku LLM → regex fallback."""

    # Stage 1: keyword patterns per intent (evaluated in order, first match wins)
    KEYWORD_PATTERNS: dict[QueryIntent, list[str]] = {
        QueryIntent.COMPARISON: [
            r"\bcompare\b",
            r"\bvs\.?\b",
            r"\bdifference(?:s)? between\b",
            r"\bwhich is better\b",
            r"\badvantage(?:s)? of .+ over\b",
            r"\bversus\b",
        ],
        QueryIntent.RELATIONAL: [
            r"\bdepend(?:s|encies)?\b",
            r"\brequire(?:s|ments)?\b",
            r"\baffect(?:s)?\b",
            r"\bdownstream\b",
            r"\bif I .+ what happens\b",
            r"\bcause(?:s)?\b",
            r"\bimplication(?:s)?\b",
        ],
        QueryIntent.PROCEDURAL: [
            r"\bhow (?:do|can|should|to)\b",
            r"\bsteps? to\b",
            r"\bsetup\b",
            r"\bconfigure\b",
            r"\bimplement\b",
            r"\binstall\b",
            r"\bcreate\b",
            r"\bbuild\b",
        ],
    }

    def __init__(self, llm=None):
        self._llm = llm

    def _keyword_scan(self, query: str) -> QueryIntent | None:
        """Stage 1: O(patterns) regex scan. Returns None if no match."""
        for intent, patterns in self.KEYWORD_PATTERNS.items():
            if any(re.search(p, query, re.IGNORECASE) for p in patterns):
                return intent
        return None

    async def _llm_classify(self, query: str, history: list[dict] | None = None) -> QueryIntent | None:
        """Stage 2: Haiku call. Returns None on any failure."""
        if self._llm is None:
            return None
        try:
            result = await self._llm.classify(
                query=query,
                system_prompt=CLASSIFY_SYSTEM,
                allowed_values=[e.value for e in QueryIntent],
                history=history,
            )
            return QueryIntent(result)
        except Exception:
            return None

    def _fallback_classify(self, query: str) -> QueryIntent:
        """Stage 3: broader regex patterns — never fails, always returns an intent."""
        q = query.lower()
        if re.search(r"\b(what|who|when|where|which|define|definition|meaning)\b", q):
            return QueryIntent.FACTOID
        if re.search(r"\b(explain|overview|tell me|describe|about)\b", q):
            return QueryIntent.EXPLORATORY
        if re.search(r"\b(and|or|both|either|neither)\b", q):
            return QueryIntent.COMPARISON
        if re.search(r"\b(if|when|after|before|then)\b", q):
            return QueryIntent.RELATIONAL
        return QueryIntent.FACTOID  # safest default

    async def classify(self, query: str, history: list[dict] | None = None) -> QueryIntent:
        """Three-stage classification. Returns intent in <1ms (keyword) or <50ms (LLM)."""
        intent = self._keyword_scan(query)
        if intent is not None:
            return intent

        intent = await self._llm_classify(query, history)
        if intent is not None:
            return intent

        return self._fallback_classify(query)
