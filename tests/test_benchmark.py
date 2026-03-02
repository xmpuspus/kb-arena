"""Tests for the enhanced benchmark system — evaluator, models, aggregation."""

from __future__ import annotations

import pytest

from kb_arena.models.benchmark import (
    AnswerRecord,
    BenchmarkResult,
    Constraints,
    GroundTruth,
    LatencyStats,
    Question,
    Score,
)

# --- LatencyStats ---


class TestLatencyStats:
    def test_from_empty(self):
        stats = LatencyStats.from_values([])
        assert stats.avg_ms == 0.0
        assert stats.p50_ms == 0.0

    def test_from_single(self):
        stats = LatencyStats.from_values([100.0])
        assert stats.avg_ms == 100.0
        assert stats.p50_ms == 100.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 100.0

    def test_from_many(self):
        values = list(range(1, 101))  # 1 to 100
        stats = LatencyStats.from_values([float(v) for v in values])
        assert stats.avg_ms == pytest.approx(50.5)
        assert stats.p50_ms == 51.0  # index 50 of 0-99 (sorted)
        assert stats.p95_ms == 96.0  # int(100 * 0.95) = 95th index
        assert stats.p99_ms == 100.0  # int(100 * 0.99) = 99th index
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0

    def test_p95_small_sample(self):
        """For <20 samples, p95 falls back to max."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = LatencyStats.from_values(values)
        assert stats.p95_ms == 50.0  # max, since n < 20


# --- Score model ---


class TestScoreModel:
    def test_defaults(self):
        s = Score(accuracy=0.8)
        assert s.completeness == 0.0
        assert s.faithfulness == 1.0
        assert s.source_attribution == 0.0
        assert s.entity_coverage == 0.0
        assert s.entities_found == []

    def test_full_score(self):
        s = Score(
            accuracy=1.0,
            completeness=0.9,
            faithfulness=1.0,
            source_attribution=0.5,
            entity_coverage=0.75,
            entities_found=["json.loads", "json.dumps"],
        )
        assert len(s.entities_found) == 2


# --- AnswerRecord model ---


class TestAnswerRecord:
    def test_defaults(self):
        rec = AnswerRecord(
            question_id="py-t1-001",
            strategy="naive_vector",
            answer="test answer",
            score=Score(accuracy=0.8),
        )
        assert rec.is_error is False
        assert rec.is_empty is False
        assert rec.attempt_count == 1
        assert rec.retrieval_latency_ms == 0.0
        assert rec.generation_latency_ms == 0.0
        assert rec.response_length == 0

    def test_error_record(self):
        rec = AnswerRecord(
            question_id="py-t1-001",
            strategy="naive_vector",
            answer="[ERROR] Timeout",
            score=Score(accuracy=0.0),
            is_error=True,
            error_message="Timeout after 120s",
            attempt_count=3,
        )
        assert rec.is_error
        assert rec.attempt_count == 3


# --- Evaluator ---


class TestEvaluator:
    @pytest.fixture
    def sample_constraints(self):
        return Constraints(
            must_mention=["JSONDecodeError", "ValueError"],
            must_not_claim=["TypeError", "SyntaxError"],
        )

    @pytest.fixture
    def sample_ground_truth(self):
        return GroundTruth(
            answer="json.loads() raises JSONDecodeError.",
            source_refs=["json.html#json.JSONDecodeError"],
            required_entities=["json.loads", "json.JSONDecodeError"],
        )

    def test_structural_pass(self, sample_constraints):
        from kb_arena.benchmark.evaluator import _structural_check

        answer = "It raises JSONDecodeError, which is a subclass of ValueError."
        score = _structural_check(answer, sample_constraints)
        assert score.structural_pass is True
        assert score.accuracy == 1.0
        assert len(score.mentions_found) == 2
        assert len(score.false_claims) == 0

    def test_structural_fail_false_claim(self, sample_constraints):
        from kb_arena.benchmark.evaluator import _structural_check

        answer = "It raises TypeError when given invalid JSON."
        score = _structural_check(answer, sample_constraints)
        assert score.structural_pass is False
        assert score.accuracy == 0.0
        assert "TypeError" in score.false_claims

    def test_structural_partial_mentions(self, sample_constraints):
        from kb_arena.benchmark.evaluator import _structural_check

        answer = "It raises JSONDecodeError."
        score = _structural_check(answer, sample_constraints)
        assert score.accuracy == 0.5  # 1 of 2 mentions
        assert score.structural_pass is True

    def test_entity_coverage_full(self, sample_ground_truth):
        from kb_arena.benchmark.evaluator import _check_entity_coverage

        answer = "json.loads raises json.JSONDecodeError for invalid input."
        ratio, found = _check_entity_coverage(answer, sample_ground_truth.required_entities)
        assert ratio == 1.0
        assert len(found) == 2

    def test_entity_coverage_partial(self, sample_ground_truth):
        from kb_arena.benchmark.evaluator import _check_entity_coverage

        answer = "The json.loads function throws an error."
        ratio, found = _check_entity_coverage(answer, sample_ground_truth.required_entities)
        assert ratio == 0.5
        assert "json.loads" in found

    def test_entity_coverage_empty_required(self):
        from kb_arena.benchmark.evaluator import _check_entity_coverage

        ratio, found = _check_entity_coverage("any answer", [])
        assert ratio == 1.0
        assert found == []

    def test_source_attribution_match(self):
        from kb_arena.benchmark.evaluator import _check_source_attribution

        returned = ["json.html#json.JSONDecodeError", "json.html#json.loads"]
        expected = ["json.html#json.JSONDecodeError"]
        score = _check_source_attribution(returned, expected)
        assert score == 1.0

    def test_source_attribution_no_match(self):
        from kb_arena.benchmark.evaluator import _check_source_attribution

        returned = ["os.html#os.getcwd"]
        expected = ["json.html#json.JSONDecodeError"]
        score = _check_source_attribution(returned, expected)
        assert score == 0.0

    def test_source_attribution_no_sources_returned(self):
        from kb_arena.benchmark.evaluator import _check_source_attribution

        score = _check_source_attribution([], ["json.html#json.JSONDecodeError"])
        assert score == 0.0

    def test_source_attribution_no_expected(self):
        from kb_arena.benchmark.evaluator import _check_source_attribution

        score = _check_source_attribution(["something"], [])
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_no_llm(self, sample_ground_truth, sample_constraints):
        from kb_arena.benchmark.evaluator import evaluate

        answer = "json.loads raises json.JSONDecodeError, a subclass of ValueError."
        score = await evaluate(
            answer,
            sample_ground_truth,
            sample_constraints,
            sources=["json.html#json.JSONDecodeError"],
            llm=None,
        )
        assert score.accuracy == 1.0
        assert score.entity_coverage == 1.0  # both json.loads and json.JSONDecodeError found
        assert score.source_attribution == 1.0


# --- Aggregation ---


class TestAggregation:
    def _make_record(
        self,
        qid: str,
        accuracy: float,
        latency: float,
        cost: float = 0.01,
        is_error: bool = False,
    ) -> AnswerRecord:
        return AnswerRecord(
            question_id=qid,
            strategy="test",
            answer="test" if not is_error else "[ERROR] fail",
            score=Score(
                accuracy=accuracy,
                completeness=accuracy * 0.9,
                faithfulness=1.0 if not is_error else 0.0,
                source_attribution=0.5,
                entity_coverage=0.8,
            ),
            latency_ms=latency,
            cost_usd=cost,
            is_error=is_error,
            is_empty=is_error,
            error_message="fail" if is_error else "",
            response_length=100 if not is_error else 0,
        )

    def test_aggregate_basic(self):
        from kb_arena.benchmark.runner import _aggregate

        bench = BenchmarkResult(corpus="test", strategy="test")
        bench.records = [
            self._make_record("py-t1-001", 1.0, 100.0),
            self._make_record("py-t1-002", 0.8, 200.0),
            self._make_record("py-t2-001", 0.5, 300.0),
        ]
        questions_map = {
            "py-t1-001": "factoid",
            "py-t1-002": "factoid",
            "py-t2-001": "comparison",
        }

        result = _aggregate(bench, questions_map)

        # Tier accuracy
        assert 1 in result.accuracy_by_tier
        assert 2 in result.accuracy_by_tier
        assert result.accuracy_by_tier[1] == pytest.approx(0.9)  # avg of 1.0 and 0.8
        assert result.accuracy_by_tier[2] == pytest.approx(0.5)

        # Type accuracy
        assert "factoid" in result.accuracy_by_type
        assert "comparison" in result.accuracy_by_type
        assert result.accuracy_by_type["factoid"] == pytest.approx(0.9)

        # Latency
        assert result.latency.avg_ms == pytest.approx(200.0)
        assert result.latency.min_ms == 100.0
        assert result.latency.max_ms == 300.0

        # Latency by tier
        assert 1 in result.latency_by_tier
        assert result.latency_by_tier[1].avg_ms == pytest.approx(150.0)

        # Cost
        assert result.total_cost_usd == pytest.approx(0.03)
        assert result.total_questions == 3

        # Reliability
        assert result.reliability.success_rate == 1.0
        assert result.reliability.error_count == 0

    def test_aggregate_with_errors(self):
        from kb_arena.benchmark.runner import _aggregate

        bench = BenchmarkResult(corpus="test", strategy="test")
        bench.records = [
            self._make_record("py-t1-001", 1.0, 100.0),
            self._make_record("py-t1-002", 0.0, 500.0, is_error=True),
        ]
        questions_map = {"py-t1-001": "factoid", "py-t1-002": "factoid"}

        result = _aggregate(bench, questions_map)

        assert result.reliability.error_count == 1
        assert result.reliability.error_rate == 0.5
        assert result.reliability.success_rate == 0.5

    def test_aggregate_empty(self):
        from kb_arena.benchmark.runner import _aggregate

        bench = BenchmarkResult(corpus="test", strategy="test")
        result = _aggregate(bench, {})
        assert result.total_questions == 0


# --- Question loading ---


class TestQuestionLoading:
    def test_question_model(self):
        q = Question(
            id="py-t1-001",
            tier=1,
            type="factoid",
            hops=1,
            question="What does json.loads do?",
            ground_truth=GroundTruth(
                answer="Parses JSON string.",
                source_refs=["json.html"],
                required_entities=["json.loads"],
            ),
            constraints=Constraints(
                must_mention=["JSON"],
                must_not_claim=["XML"],
            ),
        )
        assert q.tier == 1
        assert q.ground_truth.required_entities == ["json.loads"]

    def test_question_tier_validation(self):
        with pytest.raises(Exception):
            Question(
                id="bad",
                tier=0,
                type="factoid",
                hops=1,
                question="?",
                ground_truth=GroundTruth(answer="a"),
            )
        with pytest.raises(Exception):
            Question(
                id="bad",
                tier=6,
                type="factoid",
                hops=1,
                question="?",
                ground_truth=GroundTruth(answer="a"),
            )
