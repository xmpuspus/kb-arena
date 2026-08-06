"""Strategy 10: QISS — Quantum-Inspired Semantic Similarity (pure NumPy).

Reranks `naive_vector` candidates by quantum state fidelity

    F(q, d) = Tr(ρ_q · ρ_d) = |⟨q|d⟩|² = cos²(q, d)

where ρ = |v⟩⟨v| is the density matrix of the unit-normalized embedding. For a
single query this is exactly the squared cosine over the SAME embeddings
`naive_vector` uses, so QISS re-ranks the identical vectors apples-to-apples
(the contract verified by the invariant unit test).

Multi-query subspace projection (config-gated by `KB_ARENA_QISS_DECOMPOSE`): a
multi-hop question is decomposed into sub-queries q₁..qₘ by the LLM, and the
query becomes a *subspace* — the span of {question, q₁..qₘ}. A document is scored
by its projection onto that subspace,

    S(d) = Tr(P_span · ρ_d) = ‖P_span · d̂‖²

where P_span is the orthonormal projector onto the span. This is the rank-r
mixed-state generalization of single-query fidelity: a rank-1 subspace (just the
question) is exactly cos² (the contract above), and each extra sub-query
direction lets a chunk that answers one hop — even one orthogonal to the others —
still score. It replaces the v0.9.0 *coherent* superposition |Q⟩ = Σαᵢ|qᵢ⟩, which
scored by (Σcos)² — i.e. cosine to the sub-query *mean*, and that mean is ≈ the
holistic question (cosine ~0.77 on aws-compute), so it could never differ from
naive. The retained `superposition_state` / `classical_mixture_scores` functions
are kept as the comparison baselines that diagnosed this.

On aws-compute the subspace ties naive (Recall@5 ~0.71 vs 0.70 on 11 tier-4/5
questions): naive's top-20 already holds 92% of the relevant chunks, so this
corpus is too small to need decomposition. The operator is the correct one for
larger corpora where holistic retrieval misses a hop; the win is not claimed here.

Pure NumPy — no Qiskit, no optional install. Only `numpy` (already a transitive
dependency via chromadb) is used.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import MAX_RETRIEVAL_CANDIDATES, AnswerResult, Strategy
from kb_arena.strategies.embeddings import get_embedding_function
from kb_arena.strategies.naive_vector import NaiveVectorStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure fidelity math — the quantum-inspired core. No I/O, fully unit-testable.
# ---------------------------------------------------------------------------


def unit(v: Any) -> np.ndarray:
    """Return v / ‖v‖ as float64; a zero vector is returned unchanged."""
    arr = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def density_matrix(v: Any) -> np.ndarray:
    """ρ = |v⟩⟨v| — the rank-1 density matrix of the unit-normalized vector v."""
    u = unit(v)
    return np.outer(u, u)


def fidelity(q: Any, d: Any) -> float:
    """Tr(ρ_q · ρ_d) for two pure states, via their density matrices.

    Equals |⟨q|d⟩|² = cos²(q, d) for unit vectors. Computed explicitly through
    the density matrices (not a shortcut) so the single-query contract is the
    literal quantum expression.
    """
    rho_q = density_matrix(q)
    rho_d = density_matrix(d)
    # Tr(A·B) = Σ_ab A_ab B_ba; both density matrices are symmetric.
    return float(np.einsum("ab,ba->", rho_q, rho_d))


def fidelity_scores(query_state: Any, doc_matrix: Any) -> np.ndarray:
    """Vectorized Tr(ρ_q · ρ_dₙ) for every row dₙ of `doc_matrix`.

    Builds ρ_q = |q⟩⟨q| once and contracts it against each document's outer
    product with a single einsum. Document rows are unit-normalized internally.
    Returns scores in [0, 1].
    """
    q = unit(query_state)
    docs = np.asarray(doc_matrix, dtype=np.float64)
    if docs.ndim == 1:
        docs = docs[None, :]
    norms = np.linalg.norm(docs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    docs = docs / norms
    rho_q = np.outer(q, q)  # (dim, dim)
    # ⟨dₙ|ρ_q|dₙ⟩ = Σ_ab dₙ_b ρ_q_ba dₙ_a = (q·dₙ)²
    scores = np.einsum("nb,ba,na->n", docs, rho_q, docs)
    return np.clip(scores, 0.0, 1.0)


def superposition_state(sub_vectors: Any, weights: Any = None) -> np.ndarray:
    """|Q⟩ = Σᵢ αᵢ |qᵢ⟩, renormalized to a unit pure state.

    Default amplitudes are equal (αᵢ = 1/√m). Each sub-vector is unit-normalized
    before superposing so no single sub-query dominates purely by magnitude.
    """
    stacked = np.asarray([unit(v) for v in sub_vectors], dtype=np.float64)
    m = stacked.shape[0]
    if weights is None:
        w = np.full(m, 1.0 / np.sqrt(m))
    else:
        w = np.asarray(weights, dtype=np.float64)
    psi = (w[:, None] * stacked).sum(axis=0)
    return unit(psi)


def classical_mixture_scores(sub_matrix: Any, doc_matrix: Any, weights: Any = None) -> np.ndarray:
    """Diagonal-only (incoherent) fusion: Σᵢ αᵢ² Tr(ρ_qᵢ · ρ_d).

    This is the classical mixed-state score with NO interference cross-terms —
    the baseline that the coherent superposition is compared against.
    """
    subs = np.asarray(sub_matrix, dtype=np.float64)
    if subs.ndim == 1:
        subs = subs[None, :]
    m = subs.shape[0]
    if weights is None:
        w = np.full(m, 1.0 / np.sqrt(m))
    else:
        w = np.asarray(weights, dtype=np.float64)
    total = None
    for i in range(m):
        fi = fidelity_scores(subs[i], doc_matrix)
        total = (w[i] ** 2) * fi if total is None else total + (w[i] ** 2) * fi
    return np.clip(total, 0.0, 1.0) if total is not None else np.zeros(1)


def subspace_projection_scores(basis_matrix: Any, doc_matrix: Any) -> np.ndarray:
    """‖P_span · d‖² for each doc — projection onto the orthonormal span of the
    basis states (the holistic query plus its sub-queries).

    This is the rank-r projector / mixed-state generalization of single-query
    fidelity, and the multi-query operator QISS actually uses. A rank-1 basis
    (just the question) reduces *exactly* to cos² — the single-query contract —
    while extra sub-query directions extend the subspace so a chunk answering one
    hop, even one orthogonal to the others, still scores.

    It replaces the v0.9.0 coherent-centroid superposition. That operator scored
    by (Σᵢ cos(qᵢ,d))², which ranks by cosine to the sub-query *mean* — and the
    mean is ≈ the holistic question embedding (cosine ~0.77 on aws-compute), so it
    could never differ from naive retrieval. The subspace keeps every sub-query
    direction instead of collapsing them. Returns scores in [0, 1].
    """
    basis = np.asarray(basis_matrix, dtype=np.float64)
    if basis.ndim == 1:
        basis = basis[None, :]
    basis = basis / np.clip(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12, None)
    # Orthonormal basis of the span; SVD tolerates linearly-dependent sub-queries.
    u, s, _ = np.linalg.svd(basis.T, full_matrices=False)
    ortho = u[:, s > 1e-8]

    docs = np.asarray(doc_matrix, dtype=np.float64)
    if docs.ndim == 1:
        docs = docs[None, :]
    docs = docs / np.clip(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12, None)
    proj = docs @ ortho  # (n, k)
    return np.clip(np.sum(proj**2, axis=1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class QISSStrategy(Strategy):
    """Quantum-Inspired Semantic Similarity reranker over naive_vector."""

    name = "qiss"

    def __init__(self, chroma_client: Any = None, llm_client: Any = None) -> None:
        super().__init__()
        self._base = NaiveVectorStrategy(chroma_client=chroma_client)
        self._llm = llm_client
        self._embed_fn: Any = None

    def _get_embed_fn(self) -> Any:
        if self._embed_fn is None:
            self._embed_fn = get_embedding_function()
        return self._embed_fn

    async def build_index(self, documents: list[Document]) -> None:
        await self._base.build_index(documents)

    async def _maybe_decompose(self, question: str) -> list[str]:
        """Split a multi-hop question into sub-queries when decomposition is gated on.

        Returns `[question]` when disabled, when the LLM yields nothing (e.g. the
        retriever-lab LLM stub), or on any failure — so the default path is a
        clean single-query fidelity rerank.
        """
        from kb_arena.settings import settings

        if not settings.qiss_decompose:
            return [question]
        try:
            from kb_arena.llm.client import LLMClient

            llm = self._llm or LLMClient()
            # The instruction goes in system_prompt (NOT empty — the Anthropic
            # provider sets cache_control on the system block and 400s on an empty
            # one), the question goes in the query.
            resp = await llm.generate(
                query=question,
                context="",
                system_prompt=(
                    "Decompose the user's multi-hop question into 2-3 atomic "
                    "sub-questions, one per line, no numbering. If it is already "
                    "atomic, return it unchanged."
                ),
            )
            lines = [ln.strip("-• \t") for ln in (resp.text or "").splitlines() if ln.strip()]
            subs = [ln for ln in lines if len(ln) > 3][: settings.qiss_max_subqueries]
            return subs or [question]
        except Exception as exc:  # noqa: BLE001 — decomposition is best-effort
            logger.warning("QISS decomposition failed (%s) — using single query", exc)
            return [question]

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        from kb_arena.settings import settings

        start = self._start_timer()
        if not 1 <= top_k <= MAX_RETRIEVAL_CANDIDATES:
            raise ValueError(f"top_k must be between 1 and {MAX_RETRIEVAL_CANDIDATES}")
        fanout = max(int(settings.qiss_fanout), 1)
        candidate_k = min(
            max(top_k * fanout, top_k + 5),
            MAX_RETRIEVAL_CANDIDATES,
        )

        retrieve_t0 = time.perf_counter()
        candidate = await self._base.query(
            question,
            top_k=candidate_k,
            corpus=corpus,
        )
        retrieve_ms = (time.perf_counter() - retrieve_t0) * 1000

        chunks: list[RetrievedChunk] = (
            list(candidate.retrieval.retrieved) if candidate.retrieval else []
        )
        if not chunks:
            return candidate

        ef = self._get_embed_fn()
        # The query subspace: the holistic question always anchors it (rank-1 ->
        # cos², the single-query contract); decomposition adds sub-query
        # directions for multi-hop coverage.
        subqueries = await self._maybe_decompose(question)
        basis_texts = [question]
        if subqueries != [question]:
            basis_texts += subqueries
        basis_matrix = np.asarray(ef(basis_texts), dtype=np.float64)
        doc_matrix = np.asarray(ef([c.content or "" for c in chunks]), dtype=np.float64)

        scores = subspace_projection_scores(basis_matrix, doc_matrix)
        rank = basis_matrix.shape[0]
        if rank > 1:
            logger.info(
                "QISS subspace rerank: rank-%d span (question + %d sub-queries)", rank, rank - 1
            )

        ranked = sorted(zip(scores.tolist(), chunks, strict=True), key=lambda x: x[0], reverse=True)
        kept = ranked[:top_k]
        for i, (s, c) in enumerate(kept):
            c.rank = i + 1
            c.score = float(s)
            c.metadata = {**c.metadata, "qiss_fidelity": float(s), "qiss_subspace_rank": rank}

        from kb_arena.llm.client import LLMClient

        llm = self._llm or LLMClient()
        context = "\n\n---\n\n".join(c.content or "" for _, c in kept)
        gen_t0 = time.perf_counter()
        resp = await llm.generate(
            query=question,
            context=context,
            system_prompt=(
                "You are a documentation assistant. Answer using only the context below."
            ),
        )
        gen_ms = (time.perf_counter() - gen_t0) * 1000

        sources = list(dict.fromkeys(c.doc_id for _, c in kept if c.doc_id))
        trace = RetrievalTrace(
            query=question,
            retrieved=[c for _, c in kept],
            latency_ms=retrieve_ms,
            top_k=top_k,
        )
        total_tokens = (candidate.tokens_used or 0) + resp.total_tokens
        total_cost = (candidate.cost_usd or 0.0) + resp.cost_usd

        latency_ms = self._record_metrics(
            start, tokens=total_tokens, cost=total_cost, sources=sources
        )
        return AnswerResult(
            answer=resp.text,
            sources=sources,
            retrieval=trace,
            strategy=self.name,
            latency_ms=latency_ms,
            retrieval_latency_ms=retrieve_ms,
            generation_latency_ms=gen_ms,
            tokens_used=total_tokens,
            cost_usd=total_cost,
        )
