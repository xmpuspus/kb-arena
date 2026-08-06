"""Strategy 11: SQR — Simulated Quantum Reranker (Qiskit Aer SWAP test).

Reranks `naive_vector` candidates by a real SWAP-test circuit on the Qiskit Aer
simulator. For each candidate the embedding pair is PCA-reduced to 2ⁿ dims,
unit-normalized, amplitude-encoded with `qc.initialize()`, and compared by a
SWAP test whose ancilla statistics give

    P(0) = (1 + |⟨ψ_q|ψ_d⟩|²) / 2   ⇒   fidelity = 2·P(0) − 1.

Statevector mode (the benchmark default, `KB_ARENA_SQR_SHOTS=0`) reads the exact
ancilla probability — noiseless and reproducible. A positive shot count samples
the circuit instead, tracing the accuracy-vs-speed curve. All candidate circuits
for a query run in ONE batched Aer job (per-circuit dispatch is far slower).

Honest caveats this strategy surfaces (no cherry-picking):
* PCA into 2ⁿ amplitudes discards variance — the retained fraction is recorded
  per query (`sqr_pca_variance`) and curve-mapped corpus-wide in `diagnostics.py`.
* Quantum overhead = SQR rerank time beyond the naive_vector coarse retrieval is
  recorded per query (`sqr_overhead_ms`).

Needs the optional [quantum] extra (qiskit, qiskit-aer, scikit-learn). qiskit and
scikit-learn are lazy-imported so the core package and CI stay light.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from kb_arena.models.document import Document
from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.strategies.embeddings import get_embedding_function
from kb_arena.strategies.naive_vector import NaiveVectorStrategy
from kb_arena.strategies.quantum.circuits import build_swap_test_circuit
from kb_arena.strategies.quantum.utils import amplitude_encode, dim_for_qubits

logger = logging.getLogger(__name__)


def swap_test_fidelities(query_state: Any, doc_matrix: Any, *, shots: int = 0) -> np.ndarray:
    """|⟨ψ_q|ψ_dₙ⟩|² for each doc row, via batched SWAP-test circuits on Aer.

    Inputs must already be unit-normalized and power-of-two length. `shots=0`
    reads the exact ancilla probability from the statevector (default); `shots>0`
    samples. All circuits run in a single Aer job. Returns scores in [0, 1].
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    q = np.asarray(query_state, dtype=np.float64).ravel()
    docs = np.atleast_2d(np.asarray(doc_matrix, dtype=np.float64))
    if docs.shape[0] == 0:
        return np.zeros(0)

    if shots and shots > 0:
        sim = AerSimulator()
        circuits = [build_swap_test_circuit(q, d, measure=True) for d in docs]
        result = sim.run(transpile(circuits, sim), shots=int(shots)).result()
        p0 = []
        for i in range(len(circuits)):
            counts = result.get_counts(i)
            total = sum(counts.values()) or 1
            p0.append(counts.get("0", 0) / total)
    else:
        from qiskit.quantum_info import Statevector

        sim = AerSimulator(method="statevector")
        circuits = []
        for d in docs:
            circ = build_swap_test_circuit(q, d, measure=False)
            circ.save_statevector()
            circuits.append(circ)
        result = sim.run(transpile(circuits, sim)).result()
        p0 = [
            float(Statevector(result.get_statevector(i)).probabilities([0])[0])
            for i in range(len(circuits))
        ]

    fidelities = np.array([2.0 * p - 1.0 for p in p0])
    return np.clip(fidelities, 0.0, 1.0)


class SQRStrategy(Strategy):
    """Simulated Quantum Reranker — SWAP-test fidelity over naive_vector candidates."""

    name = "sqr"

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

    async def query(self, question: str, top_k: int = 5, corpus: str = "all") -> AnswerResult:
        from kb_arena.settings import settings

        start = self._start_timer()
        fanout = max(int(settings.sqr_fanout), 1)
        candidate_k = max(top_k * fanout, top_k + 5)
        target_dim = dim_for_qubits(int(settings.sqr_n_qubits))
        shots = int(settings.sqr_shots)

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

        rerank_t0 = time.perf_counter()
        ef = self._get_embed_fn()
        query_vec = np.asarray(ef([question])[0], dtype=np.float64)
        doc_vecs = np.asarray(ef([c.content or "" for c in chunks]), dtype=np.float64)

        # One PCA fit over query + docs so they share a subspace, then split.
        stacked = np.vstack([query_vec[None, :], doc_vecs])
        encoded, variance_explained = amplitude_encode(stacked, target_dim)
        q_enc, doc_enc = encoded[0], encoded[1:]

        scores = swap_test_fidelities(q_enc, doc_enc, shots=shots)
        rerank_ms = (time.perf_counter() - rerank_t0) * 1000

        ranked = sorted(zip(scores.tolist(), chunks, strict=True), key=lambda x: x[0], reverse=True)
        kept = ranked[:top_k]
        for i, (s, c) in enumerate(kept):
            c.rank = i + 1
            c.score = float(s)
            c.metadata = {
                **c.metadata,
                "sqr_fidelity": float(s),
                "sqr_pca_variance": variance_explained,
                "sqr_overhead_ms": rerank_ms,
                "sqr_n_qubits": int(settings.sqr_n_qubits),
                "sqr_mode": "shots" if shots > 0 else "statevector",
            }
        logger.info(
            "SQR: %d candidates, %d-dim amplitude encoding, PCA variance=%.3f, "
            "overhead=%.1fms, mode=%s",
            len(chunks),
            target_dim,
            variance_explained,
            rerank_ms,
            "shots" if shots > 0 else "statevector",
        )

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
