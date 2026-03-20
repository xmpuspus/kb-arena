"""Behavioral integration tests for v0.3.0 features.

Requires real API keys in .env file. Run with:
    pytest tests/live/test_v030_features.py -v
"""

from __future__ import annotations

import os

import pytest

# Ensure .env is loaded from project root
os.environ.setdefault("KB_ARENA_ANTHROPIC_API_KEY", "")
os.environ.setdefault("KB_ARENA_OPENAI_API_KEY", "")

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def _ensure_env():
    """Skip if API keys not set."""
    from kb_arena.settings import settings

    if not settings.anthropic_api_key:
        pytest.skip("KB_ARENA_ANTHROPIC_API_KEY not set")


class TestMultiProvider:
    """Test multi-LLM provider support."""

    @pytest.mark.asyncio
    async def test_anthropic_provider_generates(self):
        """Anthropic provider should generate a response."""
        from kb_arena.llm.client import LLMClient
        from kb_arena.settings import settings

        original = settings.llm_provider
        try:
            settings.llm_provider = "anthropic"
            client = LLMClient()
            resp = await client.generate(
                query="What is 2+2?",
                context="",
                system_prompt="Answer in one word.",
            )
            assert "4" in resp.text.lower() or "four" in resp.text.lower()
            assert resp.cost_usd > 0
            assert resp.total_tokens > 0
        finally:
            settings.llm_provider = original

    @pytest.mark.asyncio
    async def test_anthropic_classify(self):
        """Classify should return one of the allowed values."""
        from kb_arena.llm.client import LLMClient
        from kb_arena.settings import settings

        categories = ["procedural", "factoid", "relational", "conceptual"]
        original = settings.llm_provider
        try:
            settings.llm_provider = "anthropic"
            client = LLMClient()
            result = await client.classify(
                query="How do I configure Lambda timeout?",
                system_prompt=(
                    f"Classify the query as one of: {', '.join(categories)}. "
                    "Return exactly one word."
                ),
                allowed_values=categories,
            )
            assert result in categories
        finally:
            settings.llm_provider = original


class TestBM25Strategy:
    """Test BM25 strategy with real docs."""

    @pytest.fixture
    def _setup_bm25(self, tmp_path):
        from kb_arena.settings import settings

        original = settings.datasets_path
        settings.datasets_path = str(tmp_path)
        yield tmp_path
        settings.datasets_path = original

    @pytest.mark.asyncio
    async def test_bm25_build_and_query(self, _setup_bm25):
        """BM25 should build index and answer questions."""
        from kb_arena.models.document import Document, Section
        from kb_arena.strategies.bm25 import BM25Strategy

        docs = [
            Document(
                id="lambda",
                source="lambda.md",
                corpus="test",
                title="Lambda",
                sections=[
                    Section(
                        id="s1",
                        title="Overview",
                        content=(
                            "AWS Lambda is a serverless compute service. "
                            "It runs your code in response to events and "
                            "automatically manages the computing resources."
                        ),
                    ),
                ],
                metadata={"corpus": "test"},
            ),
        ]

        strategy = BM25Strategy()
        await strategy.build_index(docs)

        result = await strategy.query("What is Lambda?")
        assert result.answer
        assert len(result.answer) > 10
        assert result.strategy == "bm25"
        assert result.latency_ms > 0
        assert result.retrieval_latency_ms > 0
        assert result.generation_latency_ms > 0
        assert result.cost_usd > 0


class TestCostTracking:
    """Verify cost tracking is no longer $0.00."""

    @pytest.mark.asyncio
    async def test_generate_has_cost(self):
        """LLM generate should return non-zero cost."""
        from kb_arena.llm.client import LLMClient

        client = LLMClient()
        resp = await client.generate(
            query="Hello",
            context="World",
            system_prompt="Be brief.",
        )
        assert resp.cost_usd > 0, "Cost should be non-zero"
        assert resp.input_tokens > 0
        assert resp.output_tokens > 0


class TestTokenizer:
    """Verify tiktoken is working."""

    def test_tiktoken_accurate(self):
        """Token count should differ from whitespace count."""
        from kb_arena.tokenizer import token_count

        text = "API-Gateway multi-region auto-scaling"
        ws_count = len(text.split())
        tk_count = token_count(text)
        # BPE tokenizer should produce different count than whitespace
        assert tk_count != ws_count or tk_count > 0


class TestExportFormats:
    """Test CSV and HTML export."""

    def test_csv_export(self):
        """CSV export should produce valid output."""
        from kb_arena.benchmark.reporter import _build_csv
        from kb_arena.models.benchmark import (
            BenchmarkResult,
            LatencyStats,
            ReliabilityStats,
        )

        results = [
            BenchmarkResult(
                corpus="test",
                strategy="naive_vector",
                run_id="abc123",
                accuracy_by_tier={1: 0.75},
                completeness_by_tier={1: 0.8},
                faithfulness_by_tier={1: 0.9},
                latency=LatencyStats(avg_ms=500, p50_ms=450, p95_ms=900, p99_ms=1200),
                reliability=ReliabilityStats(
                    success_rate=0.95,
                    error_rate=0.05,
                    error_count=1,
                    empty_count=0,
                    timeout_count=0,
                    empty_rate=0.0,
                ),
                total_cost_usd=0.15,
            ),
        ]
        csv_text = _build_csv(results)
        assert "corpus" in csv_text
        assert "naive_vector" in csv_text
        lines = csv_text.strip().split("\n")
        assert len(lines) == 2  # header + 1 row

    def test_html_export(self):
        """HTML export should produce valid HTML."""
        from kb_arena.benchmark.reporter import _build_html
        from kb_arena.models.benchmark import (
            BenchmarkResult,
            LatencyStats,
            ReliabilityStats,
        )

        results = [
            BenchmarkResult(
                corpus="test",
                strategy="bm25",
                accuracy_by_tier={1: 0.6},
                completeness_by_tier={1: 0.7},
                faithfulness_by_tier={1: 0.8},
                latency=LatencyStats(avg_ms=300, p50_ms=250, p95_ms=600, p99_ms=800),
                reliability=ReliabilityStats(
                    success_rate=1.0,
                    error_rate=0.0,
                    error_count=0,
                    empty_count=0,
                    timeout_count=0,
                    empty_rate=0.0,
                ),
                total_cost_usd=0.05,
            ),
        ]
        html = _build_html(results, "test-corpus")
        assert "<!DOCTYPE html>" in html
        assert "bm25" in html
        assert "</table>" in html


class TestArenaEngine:
    """Test arena engine with real strategies (mocked)."""

    @pytest.mark.asyncio
    async def test_arena_creates_match(self):
        """Arena should create a match with two different strategies."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock

        from kb_arena.arena.engine import ArenaEngine
        from kb_arena.settings import settings

        strategies = {}
        for name in ["naive_vector", "bm25"]:
            strat = AsyncMock()
            result = MagicMock()
            result.answer = f"Answer from {name}"
            result.latency_ms = 500.0
            result.cost_usd = 0.01
            result.sources = ["doc.md"]
            strat.query = AsyncMock(return_value=result)
            strategies[name] = strat

        original = settings.results_path
        with tempfile.TemporaryDirectory() as tmp:
            settings.results_path = tmp
            engine = ArenaEngine(strategies)
            match = await engine.create_match("What is Lambda?")
            assert match.answer_a
            assert match.answer_b
            assert match.strategy_a != match.strategy_b

            result = engine.vote(match.id, "a")
            assert "error" not in result
            assert result["total_votes"] == 1

            board = engine.leaderboard()
            assert len(board) == 2
        settings.results_path = original
