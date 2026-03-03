"""Benchmark models — question definitions, evaluation results, scoring."""

from __future__ import annotations

import statistics

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
    tier: int = Field(ge=1, le=10)
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
    source_attribution: float = Field(ge=0.0, le=1.0, default=0.0)
    entity_coverage: float = Field(ge=0.0, le=1.0, default=0.0)
    structural_pass: bool = True
    mentions_found: list[str] = Field(default_factory=list)
    false_claims: list[str] = Field(default_factory=list)
    entities_found: list[str] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    """Record of a single strategy answering a single question."""

    question_id: str
    strategy: str
    answer: str
    score: Score
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    sources: list[str] = Field(default_factory=list)
    is_error: bool = False
    is_empty: bool = False
    error_message: str = ""
    attempt_count: int = 1
    response_length: int = 0


class LatencyStats(BaseModel):
    """Latency distribution statistics."""

    avg_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    @classmethod
    def from_values(cls, values: list[float]) -> LatencyStats:
        if not values:
            return cls()
        sorted_v = sorted(values)
        n = len(sorted_v)
        return cls(
            avg_ms=statistics.mean(sorted_v),
            p50_ms=sorted_v[n // 2],
            p95_ms=sorted_v[int(n * 0.95)] if n >= 20 else sorted_v[-1],
            p99_ms=sorted_v[int(n * 0.99)] if n >= 100 else sorted_v[-1],
            min_ms=sorted_v[0],
            max_ms=sorted_v[-1],
        )


class ReliabilityStats(BaseModel):
    """Response reliability metrics."""

    total_queries: int = 0
    successful_queries: int = 0
    error_count: int = 0
    empty_count: int = 0
    timeout_count: int = 0
    error_rate: float = 0.0
    empty_rate: float = 0.0
    success_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_source_attribution: float = 0.0
    avg_entity_coverage: float = 0.0
    avg_response_length: float = 0.0


class BenchmarkResult(BaseModel):
    """Full benchmark results for a corpus × strategy."""

    corpus: str
    strategy: str
    total_questions: int = 0
    records: list[AnswerRecord] = Field(default_factory=list)

    # Accuracy dimensions
    accuracy_by_tier: dict[int, float] = Field(default_factory=dict)
    completeness_by_tier: dict[int, float] = Field(default_factory=dict)
    faithfulness_by_tier: dict[int, float] = Field(default_factory=dict)
    accuracy_by_type: dict[str, float] = Field(default_factory=dict)

    # Latency dimensions
    avg_latency_ms: float = 0.0
    latency: LatencyStats = Field(default_factory=LatencyStats)
    latency_by_tier: dict[int, LatencyStats] = Field(default_factory=dict)

    # Reliability dimensions
    reliability: ReliabilityStats = Field(default_factory=ReliabilityStats)

    # Cost
    total_cost_usd: float = 0.0
    cost_per_correct: float = 0.0
