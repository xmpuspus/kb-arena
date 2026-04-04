"""Tests for v0.4.0 features: RAGAS metrics, memoization, cost cap,
plugin system, reference-free eval, corpus validation, /ready endpoint,
exponential backoff, embedding retry, debug endpoint.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

# --- 1. Corpus name validation ---


class TestCorpusValidation:
    def test_valid_corpus_names(self):
        from kb_arena.models.api import ChatRequest

        for name in ["aws-compute", "my_docs", "test123", "A-B_C"]:
            req = ChatRequest(query="test", corpus=name)
            assert req.corpus == name

    def test_invalid_corpus_names(self):
        from kb_arena.models.api import ChatRequest

        for name in ["../etc", "foo/bar", "bad name", "test\x00null", "a.b", ""]:
            with pytest.raises(ValidationError):
                ChatRequest(query="test", corpus=name)


# --- 2. RAGAS metrics ---


class TestRagasMetrics:
    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.judge = AsyncMock()
        return llm

    async def test_faithfulness(self, mock_llm):
        from kb_arena.benchmark.ragas_metrics import compute_faithfulness

        mock_llm.judge.return_value = MagicMock(
            text='{"supported_claims": 3, "total_claims": 4, "faithfulness": 0.75}'
        )
        score = await compute_faithfulness("some answer", ["chunk1", "chunk2"], mock_llm)
        assert 0.0 <= score <= 1.0
        assert score == 0.75

    async def test_answer_relevancy(self, mock_llm):
        from kb_arena.benchmark.ragas_metrics import compute_answer_relevancy

        mock_llm.judge.return_value = MagicMock(text='{"relevancy": 0.9}')
        score = await compute_answer_relevancy("what is X?", "X is a thing", mock_llm)
        assert score == 0.9

    async def test_context_precision(self, mock_llm):
        from kb_arena.benchmark.ragas_metrics import compute_context_precision

        mock_llm.judge.return_value = MagicMock(
            text='{"relevant_chunks": 2, "total_chunks": 3, "context_precision": 0.67}'
        )
        score = await compute_context_precision("q?", ["c1", "c2", "c3"], mock_llm)
        assert score == 0.67

    async def test_context_recall(self, mock_llm):
        from kb_arena.benchmark.ragas_metrics import compute_context_recall

        mock_llm.judge.return_value = MagicMock(
            text='{"facts_found": 5, "total_facts": 5, "context_recall": 1.0}'
        )
        score = await compute_context_recall("reference answer", ["ctx"], mock_llm)
        assert score == 1.0

    async def test_empty_input_returns_zero(self, mock_llm):
        from kb_arena.benchmark.ragas_metrics import (
            compute_answer_relevancy,
            compute_faithfulness,
        )

        assert await compute_faithfulness("", [], mock_llm) == 0.0
        assert await compute_answer_relevancy("q", "", mock_llm) == 0.0


# --- 3. Evaluator memoization ---


class TestEvalMemoization:
    async def test_cache_hit(self):
        from kb_arena.benchmark.evaluator import _eval_cache, evaluate
        from kb_arena.models.benchmark import Constraints, GroundTruth

        _eval_cache.clear()

        gt = GroundTruth(answer="test answer")
        constraints = Constraints(must_mention=["test"])

        score1 = await evaluate("test answer here", gt, constraints, llm=None)
        assert len(_eval_cache) == 1

        score2 = await evaluate("test answer here", gt, constraints, llm=None)
        assert score1.accuracy == score2.accuracy
        # Still 1 entry — was cached
        assert len(_eval_cache) == 1


# --- 4. Cost cap setting ---


class TestCostCap:
    def test_default_cost_cap(self):
        from kb_arena.settings import Settings

        s = Settings()
        assert s.benchmark_cost_cap_usd == 0.0

    def test_ragas_setting(self):
        from kb_arena.settings import Settings

        s = Settings()
        assert s.benchmark_enable_ragas is False


# --- 5. Plugin system ---


class TestPluginSystem:
    def test_register_plugin(self):
        import types

        from kb_arena.strategies import STRATEGY_REGISTRY, register_plugin_strategy
        from kb_arena.strategies.base import Strategy

        class TestStrategy(Strategy):
            name = "test_plugin"

            async def build_index(self, documents):
                pass

            async def query(self, question, top_k=5):
                pass

        # Create a real module object
        mock_module = types.ModuleType("my.plugin")
        mock_module.TestStrategy = TestStrategy

        with patch("importlib.import_module", return_value=mock_module):
            register_plugin_strategy("my.plugin")

        assert "test_plugin" in STRATEGY_REGISTRY
        del STRATEGY_REGISTRY["test_plugin"]


# --- 6. PageIndex caching ---


class TestPageIndexCaching:
    def test_trees_loaded_flag(self):
        from kb_arena.strategies.pageindex import PageIndexStrategy

        strat = PageIndexStrategy()
        assert strat._trees_loaded is False
        assert strat._trees == {}


# --- 7. Exponential backoff ---


class TestExponentialBackoff:
    def test_retry_base_constant(self):
        from kb_arena.benchmark.runner import RETRY_BASE_S

        assert RETRY_BASE_S == 1.0


# --- 8. Embedding retry ---


class TestEmbeddingRetry:
    def test_retry_constants(self):
        from kb_arena.strategies.embeddings import _MAX_RETRIES, _TIMEOUT_S

        assert _MAX_RETRIES == 3
        assert _TIMEOUT_S == 30


# --- 9. Score model RAGAS fields ---


class TestScoreRagasFields:
    def test_ragas_fields_exist(self):
        from kb_arena.models.benchmark import Score

        s = Score(accuracy=0.5)
        assert s.ragas_faithfulness == 0.0
        assert s.ragas_context_precision == 0.0
        assert s.ragas_context_recall == 0.0
        assert s.ragas_answer_relevancy == 0.0


# --- 10. CostCapExceededError ---


class TestCostCapError:
    def test_error_class_exists(self):
        from kb_arena.benchmark.runner import CostCapExceededError

        assert issubclass(CostCapExceededError, Exception)


# --- 11. Ready endpoint ---


class TestReadyEndpoint:
    def test_ready_route_exists(self):
        from kb_arena.chatbot.api import app

        routes = [r.path for r in app.routes]
        assert "/ready" in routes

    def test_health_route_exists(self):
        from kb_arena.chatbot.api import app

        routes = [r.path for r in app.routes]
        assert "/health" in routes


# --- 12. Debug endpoint ---


class TestDebugEndpoint:
    def test_debug_route_exists(self):
        from kb_arena.chatbot.api import app

        routes = [r.path for r in app.routes]
        assert "/api/debug/explain" in routes


# --- 13. API version ---


class TestAPIVersion:
    def test_api_version_040(self):
        from kb_arena.chatbot.api import app

        assert app.version == "0.4.0"
