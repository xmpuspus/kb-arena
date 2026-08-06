"""Corpus isolation checks for shared retrieval backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.models.retrieval import RetrievalTrace
from kb_arena.strategies.base import AnswerResult


def _empty_vector_response() -> dict:
    return {
        "documents": [[]],
        "metadatas": [[]],
        "ids": [[]],
        "distances": [[]],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy_path",
    [
        "kb_arena.strategies.naive_vector.NaiveVectorStrategy",
        "kb_arena.strategies.contextual_vector.ContextualVectorStrategy",
        "kb_arena.strategies.qna_pairs.QnAPairStrategy",
    ],
)
async def test_chroma_strategies_filter_selected_corpus(strategy_path):
    from importlib import import_module

    module_name, class_name = strategy_path.rsplit(".", 1)
    strategy_class = getattr(import_module(module_name), class_name)
    collection = MagicMock()
    collection.query.return_value = _empty_vector_response()
    strategy = strategy_class(chroma_client=MagicMock())
    strategy._collection = collection
    strategy._llm = AsyncMock()
    strategy._llm.generate.return_value = LLMResponse(text="")

    await strategy.query("question", corpus="nist")

    assert collection.query.call_args.kwargs["where"] == {"corpus": "nist"}


@pytest.mark.asyncio
async def test_pageindex_loads_only_selected_corpus():
    from kb_arena.strategies.pageindex import PageIndexStrategy

    strategy = PageIndexStrategy()
    strategy._load_tree = MagicMock(return_value=None)
    strategy._load_all_trees = MagicMock(side_effect=AssertionError("loaded all corpora"))

    result = await strategy.query("question", corpus="nist")

    strategy._load_tree.assert_called_once_with("nist")
    assert result.retrieval.retrieved == []


@pytest.mark.asyncio
async def test_raptor_filters_each_tree_level_by_corpus():
    from kb_arena.strategies.raptor import RaptorStrategy

    collection = MagicMock()
    collection.count.return_value = 1
    collection.query.return_value = _empty_vector_response()
    strategy = RaptorStrategy(chroma_client=MagicMock())
    strategy._get_collection = MagicMock(return_value=collection)

    await strategy.query("question", corpus="nist")

    assert collection.query.call_count == 3
    assert all(
        call.kwargs["where"] == {"corpus": "nist"} for call in collection.query.call_args_list
    )


@pytest.mark.asyncio
async def test_reranker_passes_corpus_to_base_retriever():
    from kb_arena.strategies.rerank_vector import RerankVectorStrategy

    strategy = RerankVectorStrategy()
    strategy._base.query = AsyncMock(
        return_value=AnswerResult(
            answer="",
            strategy="naive_vector",
            retrieval=RetrievalTrace(query="question", retrieved=[]),
        )
    )

    await strategy.query("question", corpus="nist")

    strategy._base.query.assert_awaited_once_with(
        "question",
        top_k=20,
        corpus="nist",
    )


@pytest.mark.asyncio
async def test_graph_template_receives_selected_corpus():
    from kb_arena.strategies.knowledge_graph import KnowledgeGraphStrategy

    driver = MagicMock()
    strategy = KnowledgeGraphStrategy(neo4j_driver=driver)
    strategy._run_cypher = AsyncMock(return_value=[])
    strategy._llm = AsyncMock()
    strategy._llm.generate.return_value = LLMResponse(text="No graph evidence")

    await strategy.query("What is Lambda?", corpus="nist")

    params = strategy._run_cypher.await_args.args[1]
    assert params["corpus"] == "nist"
    assert strategy._llm.extract.await_count == 0
