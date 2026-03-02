"""Benchmark models — question definitions, evaluation results, scoring."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GroundTruth(BaseModel):
    """Human-verified ground truth for a benchmark question."""

    answer: str
    source_refs: list[str] = Field(default_factory=list)
    required_entities: list[str] = Field(default_factory=list)


class Constraints(BaseModel):
    """Structural evaluation constraints for a question."""

    must_mention: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    max_tokens: int = 500


class Question(BaseModel):
    """A benchmark question with ground truth and evaluation constraints."""

    id: str
    tier: int = Field(ge=1, le=5)
    type: str  # factoid, comparison, relational, temporal, causal
    hops: int = Field(ge=1, le=5)
    question: str
    ground_truth: GroundTruth
    constraints: Constraints = Field(default_factory=Constraints)


class Score(BaseModel):
    """Evaluation score for a single answer."""

    accuracy: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0, default=0.0)
    faithfulness: float = Field(ge=0.0, le=1.0, default=1.0)
    structural_pass: bool = True
    mentions_found: list[str] = Field(default_factory=list)
    false_claims: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    """Record of a single strategy answering a single question."""

    question_id: str
    strategy: str
    answer: str
    score: Score
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    sources: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    """Full benchmark results for a corpus."""

    corpus: str
    strategy: str
    total_questions: int = 0
    records: list[AnswerRecord] = Field(default_factory=list)
    accuracy_by_tier: dict[int, float] = Field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_correct: float = 0.0
