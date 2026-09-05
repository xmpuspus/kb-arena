"""Strategy 13: SPLADE, learned sparse retrieval over a term-weight index.

BM25 weighs the terms a passage actually contains. SPLADE expands a passage or
a query into a weighted set of vocabulary terms, including terms the text never
states but a masked-language model considers implied, then scores a query
against a passage by the dot product of their term weights. The index this
strategy builds and reads is the sparse analogue of `bm25_index.json`: one
term-weight map per passage instead of one token list.

Needs the optional `[splade]` extra (transformers, torch), lazy-imported inside
the term encoder so the core package and CI stay light. Building the index
needs the encoder up front, unlike a naive_vector reranker, so a plain install
gets a clear ImportError only when `build_index` actually runs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.base import AnswerResult, Strategy, validate_top_k

log = logging.getLogger(__name__)
SPLADE_INDEX_FORMAT_VERSION = 1

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the question using ONLY the "
    "provided context passages. If the context does not contain the answer, "
    "say so. Be concise and accurate."
)

# ---------------------------------------------------------------------------
# Pure sparse-vector math, no I/O, fully unit-testable without a model.
# ---------------------------------------------------------------------------


def splade_weights_from_logits(logits: Any, attention_mask: Any) -> np.ndarray:
    """Turn masked-language-model logits into one SPLADE weight vector.

    `logits` is (seq_len, vocab_size), `attention_mask` is (seq_len,) with 1 for
    a real token and 0 for padding. Applies log(1+relu(x)) per position, zeroes
    out padding positions first so they cannot win the max, then takes the max
    over the sequence. The result has one nonnegative weight per vocabulary term.
    """
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(attention_mask, dtype=np.float64)
    activated = np.log1p(np.clip(logits, 0, None))
    activated = activated * mask[:, None]
    return activated.max(axis=0)


def sparse_vector_to_terms(weights: Any, top_n: int | None = None) -> dict[int, float]:
    """Return `{term_id: weight}` for every nonzero weight, largest kept first.

    A full vocabulary-sized vector is mostly zero after the ReLU; keeping only
    the nonzero (and, when `top_n` is set, only the strongest) entries is what
    makes the index a sparse index rather than a dense one written as JSON.
    """
    weights = np.asarray(weights, dtype=np.float64)
    nonzero = np.nonzero(weights)[0]
    terms = {int(i): float(weights[i]) for i in nonzero}
    if top_n is not None and len(terms) > top_n:
        strongest = sorted(terms.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        terms = dict(strongest)
    return terms


def sparse_dot(query_terms: dict[int, float], doc_terms: dict[int, float]) -> float:
    """Dot product of two term-weight maps, summed over the terms they share."""
    if len(query_terms) > len(doc_terms):
        query_terms, doc_terms = doc_terms, query_terms
    return sum(
        weight * doc_terms[term] for term, weight in query_terms.items() if term in doc_terms
    )


# ---------------------------------------------------------------------------
# Term encoder, backend protocol so a future model swap needs no strategy change.
# ---------------------------------------------------------------------------


class _SpladeEncoder:
    """Wraps a masked-language-model encoder to emit one sparse term-weight map per text."""

    def __init__(self, model: str, top_n: int) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the splade strategy. "
                "Install with: pip install 'kb-arena[splade]'"
            ) from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForMaskedLM.from_pretrained(model)
        self._model.eval()
        self._top_n = top_n

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        out = []
        for text in texts:
            batch = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with self._torch.no_grad():
                logits = self._model(**batch).logits[0]
            weights = splade_weights_from_logits(logits.numpy(), batch["attention_mask"][0].numpy())
            out.append(sparse_vector_to_terms(weights, top_n=self._top_n))
        return out


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class SPLADEStrategy(Strategy):
    """Learned sparse retrieval, term-weight expansion scored against a sparse index."""

    name = "splade"

    def __init__(self) -> None:
        super().__init__()
        self._encoder: _SpladeEncoder | None = None
        self._corpus_texts: list[str] = []
        self._corpus_sources: list[str] = []
        self._chunk_ids: list[str] = []
        self._passage_terms: list[dict[int, float]] = []
        self._loaded_corpus = ""
        self._llm = None

    def _get_encoder(self) -> _SpladeEncoder:
        if self._encoder is None:
            self._encoder = _SpladeEncoder(
                model=settings.splade_model, top_n=settings.splade_top_terms
            )
        return self._encoder

    def _get_llm(self):
        if self._llm is None:
            from kb_arena.llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm

    async def build_index(self, documents: list[Document]) -> None:
        """Build and persist one SPLADE index per corpus."""
        passages_by_corpus: dict[str, dict[str, list]] = {}
        for doc in documents:
            passages = passages_by_corpus.setdefault(
                doc.corpus,
                {"texts": [], "sources": [], "chunk_ids": []},
            )
            for section in doc.sections:
                content = section.content.strip()
                if content:
                    passages["texts"].append(content)
                    passages["sources"].append(doc.id)
                    passages["chunk_ids"].append(f"{doc.id}::{section.id}")

        passages_by_corpus = {
            corpus: passages for corpus, passages in passages_by_corpus.items() if passages["texts"]
        }
        if not passages_by_corpus:
            log.warning("No text content found for SPLADE index")
            return

        encoder = self._get_encoder()
        for corpus, passages in passages_by_corpus.items():
            terms = encoder.encode(passages["texts"])
            passages["terms"] = terms
            index_dir = Path(settings.datasets_path) / corpus / "processed"
            index_dir.mkdir(parents=True, exist_ok=True)
            index_path = index_dir / "splade_index.json"
            payload = {
                "format_version": SPLADE_INDEX_FORMAT_VERSION,
                "corpus": corpus,
                "texts": passages["texts"],
                "sources": passages["sources"],
                "chunk_ids": passages["chunk_ids"],
                # JSON object keys are always strings, so term ids round-trip
                # through str() on write and int() on read.
                "terms": [{str(term): w for term, w in tv.items()} for tv in terms],
            }
            index_path.write_text(json.dumps(payload, ensure_ascii=False))
            log.info("SPLADE index built for %s: %d passages", corpus, len(passages["texts"]))

        combined_texts: list[str] = []
        combined_sources: list[str] = []
        combined_chunk_ids: list[str] = []
        combined_terms: list[dict[int, float]] = []
        for passages in passages_by_corpus.values():
            combined_texts.extend(passages["texts"])
            combined_sources.extend(passages["sources"])
            combined_chunk_ids.extend(passages["chunk_ids"])
            combined_terms.extend(passages["terms"])
        loaded_corpus = "all" if len(passages_by_corpus) > 1 else next(iter(passages_by_corpus))
        self._activate_index(
            combined_texts, combined_sources, combined_chunk_ids, combined_terms, loaded_corpus
        )

    def _activate_index(
        self,
        texts: list[str],
        sources: list[str],
        chunk_ids: list[str],
        terms: list[dict[int, float]],
        corpus: str,
    ) -> None:
        self._corpus_texts = texts
        self._corpus_sources = sources
        self._chunk_ids = chunk_ids
        self._passage_terms = terms
        self._loaded_corpus = corpus

    def _ensure_index(self, corpus: str = "") -> bool:
        """Load the SPLADE index from disk if not already loaded."""
        requested_corpus = corpus or "all"
        if self._passage_terms and (
            requested_corpus == self._loaded_corpus
            or (requested_corpus == "all" and not self._loaded_corpus)
        ):
            return True

        if corpus:
            search_paths = [
                Path(settings.datasets_path) / corpus / "processed" / "splade_index.json"
            ]
        else:
            base = Path(settings.datasets_path)
            search_paths = sorted(base.glob("*/processed/splade_index.json"))

        texts: list[str] = []
        sources: list[str] = []
        chunk_ids: list[str] = []
        terms: list[dict[int, float]] = []
        loaded_any = False
        for path in search_paths:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                if (
                    data.get("format_version") != SPLADE_INDEX_FORMAT_VERSION
                    or data.get("corpus") != path.parent.parent.name
                ):
                    log.warning("Skipping legacy or mismatched SPLADE index at %s", path)
                    continue
                raw_terms = data["terms"]
                if not isinstance(raw_terms, list) or len(raw_terms) != len(data["sources"]):
                    log.warning("Corrupt SPLADE index at %s: terms mismatch sources", path)
                    continue
                texts.extend(data["texts"])
                sources.extend(data["sources"])
                chunk_ids.extend(data["chunk_ids"])
                terms.extend({int(term): w for term, w in tv.items()} for tv in raw_terms)
                loaded_any = True
            except (json.JSONDecodeError, KeyError):
                log.warning("Corrupt SPLADE index at %s", path)
                continue

        if not loaded_any:
            return False
        self._activate_index(texts, sources, chunk_ids, terms, requested_corpus)
        return True

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        validate_top_k(top_k)
        start = self._start_timer()

        selected_corpus = "" if corpus == "all" else corpus
        if not self._ensure_index(selected_corpus):
            return AnswerResult(
                answer="SPLADE index not built. Run: kb-arena build-vectors --strategy splade",
                sources=[],
                strategy=self.name,
            )

        retrieval_start = time.perf_counter()
        encoder = self._get_encoder()
        query_terms = encoder.encode([question])[0]
        scores = [sparse_dot(query_terms, doc_terms) for doc_terms in self._passage_terms]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        passages = [self._corpus_texts[i] for i in top_indices]
        sources = list({self._corpus_sources[i] for i in top_indices})
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=self._chunk_ids[idx],
                doc_id=self._corpus_sources[idx],
                content=self._corpus_texts[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                source_strategy=self.name,
            )
            for rank, idx in enumerate(top_indices)
        ]
        trace = RetrievalTrace(
            query=question, retrieved=retrieved_chunks, latency_ms=retrieval_ms, top_k=top_k
        )

        context = "\n\n---\n\n".join(passages)

        gen_start = time.perf_counter()
        llm = self._get_llm()
        resp = await llm.generate(query=question, context=context, system_prompt=SYSTEM_PROMPT)
        gen_ms = (time.perf_counter() - gen_start) * 1000

        latency_ms = self._record_metrics(
            start, tokens=resp.total_tokens, cost=resp.cost_usd, sources=sources
        )

        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            tokens_used=resp.total_tokens,
            cost_usd=resp.cost_usd,
        )
