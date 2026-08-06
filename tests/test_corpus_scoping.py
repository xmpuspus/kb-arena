"""Corpus isolation checks for shared retrieval backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.llm.client import LLMResponse
from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace
from kb_arena.strategies.base import AnswerResult
from kb_arena.strategies.chroma_index import INDEX_FORMAT_VERSION


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

    assert collection.query.call_args.kwargs["where"] == {
        "$and": [{"index_version": INDEX_FORMAT_VERSION}, {"corpus": "nist"}]
    }


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
async def test_pageindex_builds_separate_trees_for_mixed_corpora(tmp_path, monkeypatch):
    from kb_arena.settings import settings
    from kb_arena.strategies.pageindex import PageIndexStrategy

    monkeypatch.setattr(settings, "datasets_path", str(tmp_path))
    strategy = PageIndexStrategy()
    strategy._llm = AsyncMock()
    documents = [
        Document(id="alpha-doc", source="alpha.md", corpus="alpha", title="Alpha"),
        Document(id="beta-doc", source="beta.md", corpus="beta", title="Beta"),
    ]

    await strategy.build_index(documents)

    alpha_tree = strategy._load_tree("alpha")
    beta_tree = strategy._load_tree("beta")
    assert alpha_tree is not None
    assert beta_tree is not None
    assert [document.id for document in alpha_tree.documents] == ["alpha-doc"]
    assert [document.id for document in beta_tree.documents] == ["beta-doc"]


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
        call.kwargs["where"]
        == {"$and": [{"index_version": INDEX_FORMAT_VERSION}, {"corpus": "nist"}]}
        for call in collection.query.call_args_list
    )


@pytest.mark.asyncio
async def test_raptor_builds_higher_levels_for_each_corpus():
    from kb_arena.strategies.raptor import RaptorStrategy

    strategy = RaptorStrategy(chroma_client=MagicMock())
    strategy._get_collection = MagicMock(side_effect=[MagicMock(), MagicMock(), MagicMock()])
    strategy._build_level = AsyncMock(side_effect=[["alpha::l1"], ["beta::l1"]])
    documents = [
        Document(id="alpha-doc", source="alpha.md", corpus="alpha", title="Alpha"),
        Document(id="beta-doc", source="beta.md", corpus="beta", title="Beta"),
    ]

    await strategy.build_index(documents)

    assert [call.args[3] for call in strategy._build_level.await_args_list] == ["alpha", "beta"]


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
    assert "KBArenaEntity" in strategy._run_cypher.await_args.args[0]
    assert strategy._llm.extract.await_count == 0


@pytest.mark.asyncio
async def test_unscoped_generated_cypher_falls_back_to_owned_fulltext_query():
    from kb_arena.strategies.knowledge_graph import FULLTEXT_SEARCH, KnowledgeGraphStrategy

    strategy = KnowledgeGraphStrategy(neo4j_driver=MagicMock())
    strategy._llm = AsyncMock()
    strategy._llm.extract.return_value = LLMResponse(text="MATCH (n) RETURN n.name AS name")
    strategy._run_cypher = AsyncMock(return_value=[])

    records, cypher, _ = await strategy._generate_cypher("question")

    assert records == []
    assert cypher == FULLTEXT_SEARCH
    strategy._run_cypher.assert_awaited_once_with(
        FULLTEXT_SEARCH,
        {"query": "question", "corpus": "all"},
    )
