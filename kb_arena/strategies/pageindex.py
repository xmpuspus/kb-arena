"""Strategy 7: PageIndex — Vectorless, Reasoning-Based Retrieval.

Inspired by VectifyAI/PageIndex. Builds a hierarchical tree index from the
document's natural section structure (headings, subheadings). At query time,
an LLM traverses the tree via beam search — reading titles and summaries at
each level to decide which branches are relevant — then generates an answer
from the selected leaf sections.

No embeddings, no vector database, no chunking. Pure LLM reasoning on
document structure.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from kb_arena.models.document import Document
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = (
    "You are a technical documentation analyst. Write a concise 1-2 sentence summary "
    "of the following section content. Focus on what topics, components, or procedures "
    "it covers. Be specific — mention entity names, configuration options, and key "
    "relationships. Do not include filler."
)

_SUMMARIZE_PARENT_SYSTEM = (
    "You are a technical documentation analyst. Given the titles and summaries of "
    "child sections, write a concise 1-2 sentence summary of the parent section. "
    "Focus on the overall theme and what subtopics it contains."
)

_SELECT_SYSTEM = (
    "You are a documentation navigation assistant. Given a user question and a list "
    "of document sections (each with a number, title, and summary), select the sections "
    "most likely to contain the answer.\n\n"
    "Respond with ONLY a comma-separated list of section numbers (e.g. '1,3,5'). "
    "Select up to {beam_width} sections. If none seem relevant, pick the closest matches."
)

_ANSWER_SYSTEM = (
    "You are a documentation assistant. Answer the question using ONLY the provided "
    "context sections. Each section includes its location in the document hierarchy. "
    "Be concise and accurate. Cite section titles when referencing specific information."
)


class TreeNode(BaseModel):
    """A node in the PageIndex tree. Leaves have content; internal nodes have summaries."""

    id: str
    title: str
    summary: str = ""
    level: int = 0
    content: str | None = None
    source_doc: str = ""
    children: list[TreeNode] = Field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def leaf_count(self) -> int:
        if self.is_leaf():
            return 1
        return sum(c.leaf_count() for c in self.children)


class CorpusTree(BaseModel):
    """Serializable tree index for an entire corpus."""

    corpus: str
    built_at: str
    documents: list[TreeNode] = Field(default_factory=list)


def _build_doc_tree(doc: Document) -> TreeNode:
    """Reconstruct a tree from a Document's flat sections using heading_path and level.

    parent_id/children fields are not populated by parsers, so we reconstruct
    the hierarchy from level (heading depth) and document order.
    """
    root = TreeNode(
        id=doc.id,
        title=doc.title,
        level=0,
        source_doc=doc.source,
    )

    if not doc.sections:
        root.content = ""
        return root

    # Stack tracks the current path from root to the deepest open node.
    # stack[i] is the node at depth i.
    stack: list[TreeNode] = [root]

    for section in doc.sections:
        node = TreeNode(
            id=section.id,
            title=section.title,
            level=section.level,
            source_doc=doc.id,
            # Leaf determination happens after all sections are added
            content=section.content,
        )

        target_depth = section.level
        # Pop back to parent level
        while len(stack) > target_depth:
            stack.pop()

        # Handle level jumps (e.g. h1 -> h3 with no h2)
        while len(stack) < target_depth:
            synthetic = TreeNode(
                id=f"{doc.id}::synthetic_l{len(stack)}",
                title="",
                level=len(stack),
                source_doc=doc.id,
            )
            stack[-1].children.append(synthetic)
            stack.append(synthetic)

        stack[-1].children.append(node)
        stack.append(node)

    # Mark internal nodes: clear content, keep only on leaves
    _strip_internal_content(root)
    return root


def _strip_internal_content(node: TreeNode) -> None:
    """Remove content from non-leaf nodes (they'll get summaries instead)."""
    if node.children:
        node.content = None
        for child in node.children:
            _strip_internal_content(child)


async def _generate_summaries(root: TreeNode, llm) -> int:
    """Bottom-up summary generation. Returns total LLM calls made."""
    calls = 0

    async def _summarize(node: TreeNode) -> None:
        nonlocal calls
        # Recurse children first (bottom-up)
        for child in node.children:
            await _summarize(child)

        if node.is_leaf():
            # Leaf: summarize content directly
            if node.content and len(node.content) > 100:
                resp = await llm.generate(
                    query=f"Summarize this section titled '{node.title}':",
                    context=node.content[:4000],
                    system_prompt=_SUMMARIZE_SYSTEM,
                    max_tokens=150,
                )
                node.summary = resp.text.strip()
                calls += 1
            else:
                node.summary = node.content or f"Section: {node.title}"
        else:
            # Internal node: summarize from children's summaries
            child_descriptions = "\n".join(
                f"- {c.title}: {c.summary}" for c in node.children if c.title
            )
            if child_descriptions:
                resp = await llm.generate(
                    query=f"Summarize the parent section '{node.title}' from its children:",
                    context=child_descriptions,
                    system_prompt=_SUMMARIZE_PARENT_SYSTEM,
                    max_tokens=150,
                )
                node.summary = resp.text.strip()
                calls += 1
            else:
                node.summary = f"Container section: {node.title}"

    await _summarize(root)
    return calls


async def _beam_traverse(
    candidates: list[TreeNode],
    question: str,
    llm,
    beam_width: int,
    max_depth: int,
    _depth: int = 0,
) -> tuple[list[TreeNode], float]:
    """Recursive beam search: LLM selects promising branches at each level.

    Returns (selected_leaves, total_llm_cost_usd).
    """
    if not candidates or _depth >= max_depth:
        return candidates, 0.0

    # All candidates are leaves — return them
    if all(c.is_leaf() for c in candidates):
        return candidates, 0.0

    # Present candidates to LLM for selection
    listing = "\n".join(f"{i + 1}. [{c.title}] - {c.summary}" for i, c in enumerate(candidates))
    prompt = (
        f"Question: {question}\n\n"
        f"Available sections:\n{listing}\n\n"
        f"Which sections (by number) are most relevant to answering the question?"
    )
    system = _SELECT_SYSTEM.format(beam_width=beam_width)

    resp = await llm.generate(
        query=prompt,
        context="",
        system_prompt=system,
        max_tokens=50,
    )
    level_cost = resp.cost_usd

    # Parse selection (comma-separated numbers)
    selected_indices = _parse_selection(resp.text, len(candidates), beam_width)
    selected = [candidates[i] for i in selected_indices]

    # Collect children from selected non-leaf nodes, keep leaf nodes as-is
    leaves: list[TreeNode] = []
    next_candidates: list[TreeNode] = []
    for node in selected:
        if node.is_leaf():
            leaves.append(node)
        else:
            next_candidates.extend(node.children)

    # Recurse into children
    deeper_cost = 0.0
    if next_candidates:
        deeper_leaves, deeper_cost = await _beam_traverse(
            next_candidates, question, llm, beam_width, max_depth, _depth + 1
        )
        leaves.extend(deeper_leaves)

    return leaves, level_cost + deeper_cost


def _parse_selection(text: str, total: int, beam_width: int) -> list[int]:
    """Parse LLM output like '1,3,5' into 0-based indices."""
    indices = []
    for token in text.replace(" ", "").split(","):
        token = token.strip().rstrip(".")
        try:
            idx = int(token) - 1  # 1-based to 0-based
            if 0 <= idx < total:
                indices.append(idx)
        except ValueError:
            continue

    if not indices:
        # Fallback: select first beam_width candidates
        indices = list(range(min(beam_width, total)))

    return indices[:beam_width]


def _tree_path(corpus: str) -> Path:
    return Path(settings.datasets_path) / corpus / "processed" / "pageindex_tree.json"


class PageIndexStrategy(Strategy):
    """PageIndex: vectorless retrieval via LLM-driven tree traversal."""

    name = "pageindex"

    def __init__(self):
        super().__init__()
        self._llm = None
        self._trees: dict[str, CorpusTree] = {}

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    def _load_tree(self, corpus: str) -> CorpusTree | None:
        """Load a corpus tree from JSON, with in-memory caching."""
        if corpus in self._trees:
            return self._trees[corpus]

        path = _tree_path(corpus)
        if not path.exists():
            return None

        tree = CorpusTree.model_validate_json(path.read_text())
        self._trees[corpus] = tree
        return tree

    def _load_all_trees(self) -> list[CorpusTree]:
        """Load all available corpus trees (for chatbot path where corpus is unknown)."""
        if self._trees:
            return list(self._trees.values())

        base = Path(settings.datasets_path)
        for tree_file in base.glob("*/processed/pageindex_tree.json"):
            corpus_name = tree_file.parent.parent.name
            if corpus_name not in self._trees:
                try:
                    tree = CorpusTree.model_validate_json(tree_file.read_text())
                    self._trees[corpus_name] = tree
                except Exception as exc:
                    logger.warning("Failed to load tree for %s: %s", corpus_name, exc)

        return list(self._trees.values())

    async def build_index(self, documents: list[Document]) -> None:
        """Build hierarchical tree index from documents and generate summaries."""
        if not documents:
            return

        corpus = documents[0].corpus
        llm = self._get_llm()
        doc_trees: list[TreeNode] = []
        total_calls = 0

        for doc in documents:
            root = _build_doc_tree(doc)
            calls = await _generate_summaries(root, llm)
            total_calls += calls
            doc_trees.append(root)

        tree = CorpusTree(
            corpus=corpus,
            built_at=datetime.now(UTC).isoformat(),
            documents=doc_trees,
        )

        path = _tree_path(corpus)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tree.model_dump_json(indent=2))
        self._trees[corpus] = tree

        total_sections = sum(d.leaf_count() for d in doc_trees)
        logger.info(
            "PageIndex: built tree for %s — %d docs, %d leaf sections, %d LLM calls",
            corpus,
            len(doc_trees),
            total_sections,
            total_calls,
        )

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Answer by LLM-driven tree traversal — no vectors, no embeddings."""
        start = self._start_timer()
        llm = self._get_llm()

        # Load all available trees
        trees = self._load_all_trees()
        if not trees:
            latency_ms = self._record_metrics(start)
            return AnswerResult(
                answer="No PageIndex tree found. Run build-vectors --strategy pageindex first.",
                sources=[],
                strategy=self.name,
                latency_ms=latency_ms,
            )

        # Collect all document roots across corpora
        all_roots: list[TreeNode] = []
        for tree in trees:
            all_roots.extend(tree.documents)

        beam_width = settings.pageindex_beam_width
        max_depth = settings.pageindex_max_depth

        # Beam search: traverse tree to find relevant leaf sections
        retrieval_start = time.perf_counter()
        selected_leaves, traversal_cost = await _beam_traverse(
            all_roots, question, llm, beam_width, max_depth
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if not selected_leaves:
            latency_ms = self._record_metrics(start)
            return AnswerResult(
                answer="Tree traversal found no relevant sections.",
                sources=[],
                strategy=self.name,
                latency_ms=latency_ms,
            )

        # Collect context from selected leaves
        context_parts = []
        sources: list[str] = []
        for leaf in selected_leaves[:top_k]:
            if leaf.content:
                context_parts.append(f"[{leaf.title}]\n{leaf.content}")
                sources.append(leaf.source_doc)

        context = "\n\n---\n\n".join(context_parts)

        # Generate final answer with Sonnet
        gen_start = time.perf_counter()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=_ANSWER_SYSTEM,
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        total_cost = resp.cost_usd + traversal_cost
        unique_sources = list(dict.fromkeys(sources))
        latency_ms = self._record_metrics(
            start, tokens=resp.total_tokens, cost=total_cost, sources=unique_sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=unique_sources,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=total_cost,
        )
