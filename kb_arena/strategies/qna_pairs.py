"""Strategy 3: QnA Pairs — pre-generated answers indexed by question embeddings.

Build time: LLM (Sonnet) generates 3-5 QnA pairs per section.
Query time: match question embeddings → return pre-generated answer (no LLM call).
Higher upfront cost, near-zero runtime cost once built.
"""

from __future__ import annotations

import chromadb

from kb_arena.generate.qna import generate_pairs_for_section
from kb_arena.models.document import Document, Section
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.embeddings import OpenAIEmbedding

COLLECTION_NAME = "qna_pairs"

ANSWER_PROMPT = (
    "You are a documentation assistant. A user asked a question"
    " and retrieved a pre-generated answer.\n"
    "Lightly rephrase the answer to directly address the user's phrasing."
    " Keep it factual and concise."
)


class QnAPairStrategy(Strategy):
    """Pre-generated QnA pairs indexed by question embeddings.

    Build is expensive (LLM per section). Queries are fast (embedding lookup only).
    """

    name = "qna_pairs"

    def __init__(self, chroma_client=None, llm_client=None):
        self._client = chroma_client
        self._collection = None
        self._llm = llm_client

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=settings.chroma_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            ef = OpenAIEmbedding()
            self._collection = self._get_client().get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def _generate_pairs(self, section: Section, doc_id: str) -> list[dict]:
        """Ask Sonnet to generate 3-5 QnA pairs for a section."""
        return await generate_pairs_for_section(section, doc_id, self._get_llm())

    async def build_index(self, documents: list[Document]) -> None:
        """Generate QnA pairs for every section, embed questions, store answers as metadata."""
        collection = self._get_collection()
        ids, questions, metadatas = [], [], []
        pair_counter = 0

        for doc in documents:
            for section in doc.sections:
                if not section.content.strip():
                    continue
                try:
                    pairs = await self._generate_pairs(section, doc.id)
                except Exception:
                    # On generation failure, skip section rather than crash build
                    continue

                for pair in pairs:
                    q = pair.get("question", "").strip()
                    a = pair.get("answer", "").strip()
                    if not q or not a:
                        continue

                    pair_id = f"qna::{doc.id}::{section.id}::{pair_counter}"
                    pair_counter += 1
                    ids.append(pair_id)
                    questions.append(q)
                    metadatas.append(
                        {
                            "answer": a[:2000],  # ChromaDB metadata value limit
                            "source_id": doc.id,
                            "section_id": section.id,
                            "section_ref": pair.get("section_ref", ""),
                        }
                    )

        if ids:
            batch = 500
            for start in range(0, len(ids), batch):
                collection.upsert(
                    ids=ids[start : start + batch],
                    documents=questions[start : start + batch],
                    metadatas=metadatas[start : start + batch],
                )

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Match question embedding → retrieve pre-generated answer.

        No LLM call for the answer itself — just embedding lookup.
        """
        start = self._start_timer()
        collection = self._get_collection()

        results = collection.query(query_texts=[question], n_results=top_k)
        metas = results["metadatas"][0] if results["metadatas"] else []
        matched_questions = results["documents"][0] if results["documents"] else []

        if not metas:
            return AnswerResult(
                answer="No relevant QnA pairs found for this question.",
                sources=[],
                strategy=self.name,
                latency_ms=self._record_metrics(start),
            )

        # Best match is first result
        best_meta = metas[0]
        best_answer = best_meta.get("answer", "")
        sources = list({m.get("source_id", "") for m in metas if m.get("source_id")})

        # If the matched question aligns well, return pre-generated answer directly
        # Otherwise do a light rephrase to address the user's phrasing
        matched_q = matched_questions[0] if matched_questions else ""
        answer = best_answer
        total_tokens = 0
        total_cost = 0.0
        if best_answer and matched_q and matched_q.lower() != question.lower():
            llm = self._get_llm()
            context = (
                f"User question: {question}\nMatched question: {matched_q}"
                f"\nPre-generated answer: {best_answer}"
            )
            resp = await llm.generate(
                query=question,
                context=context,
                system_prompt=ANSWER_PROMPT,
                max_tokens=500,
            )
            answer = resp.text
            total_tokens = resp.total_tokens
            total_cost = resp.cost_usd

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources
        )
        return AnswerResult(
            answer=answer,
            sources=sources,
            strategy=self.name,
            latency_ms=latency_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
