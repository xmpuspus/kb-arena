"""Honest-caveat diagnostics for the quantum strategies.

Computes the three numbers the project's North Star requires be reported without
cherry-picking:

1. PCA variance explained at each qubit count — the variance amplitude-encoding
   into 2ⁿ dims discards (run on the corpus embedding distribution, not a single
   low-rank candidate pool, so it reflects the true global cost).
2. SWAP-test fidelity error vs shot count — the accuracy/speed curve of sampled
   mode against the exact statevector default.
3. Quantum overhead ms — SQR end-to-end latency minus the naive_vector coarse
   retrieval it reranks. No cherry-picking: averaged over sample questions.

scikit-learn / qiskit are reached only through `reduce_pca` and
`swap_test_fidelities`, both lazy — but this orchestrator needs the [quantum]
extra to run end-to-end (it builds SWAP circuits and PCA fits).
"""

from __future__ import annotations

import logging

import numpy as np

from kb_arena.models.quantum import (
    PCAVariancePoint,
    QuantumDiagnostics,
    ShotErrorPoint,
)
from kb_arena.strategies.quantum.sqr import swap_test_fidelities
from kb_arena.strategies.quantum.utils import amplitude_encode, dim_for_qubits, reduce_pca

logger = logging.getLogger(__name__)


def pca_variance_curve(
    embeddings: np.ndarray | list, n_qubits_list: list[int]
) -> list[PCAVariancePoint]:
    """Variance retained when `embeddings` are reduced to 2ⁿ dims, for each n.

    Fit globally on the full embedding set so the number reflects the true cost
    of squeezing the corpus into 2ⁿ amplitudes, not the near-lossless reduction a
    tiny per-query candidate pool happens to allow.
    """
    points: list[PCAVariancePoint] = []
    for n in n_qubits_list:
        dim = dim_for_qubits(n)
        _, variance = reduce_pca(embeddings, dim)
        points.append(
            PCAVariancePoint(
                n_qubits=n,
                encoded_dim=dim,
                variance_explained=min(max(variance, 0.0), 1.0),
            )
        )
    return points


def swap_error_curve(
    query_state: np.ndarray, doc_matrix: np.ndarray, shots_list: list[int]
) -> list[ShotErrorPoint]:
    """Mean |sampled − exact| SWAP-test fidelity at each shot count."""
    exact = swap_test_fidelities(query_state, doc_matrix, shots=0)
    points: list[ShotErrorPoint] = []
    for shots in shots_list:
        approx = swap_test_fidelities(query_state, doc_matrix, shots=shots)
        err = float(np.mean(np.abs(approx - exact)))
        points.append(ShotErrorPoint(shots=shots, mean_abs_error=min(max(err, 0.0), 1.0)))
    return points


async def run_quantum_diagnostics(
    corpus: str = "aws-compute",
    n_qubits_list: list[int] | None = None,
    shots_list: list[int] | None = None,
    sample_questions: int = 5,
) -> QuantumDiagnostics:
    """Build the full diagnostics report from real corpus embeddings + runs."""
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.benchmark.retriever_lab import _PatchLLMClient
    from kb_arena.settings import settings
    from kb_arena.strategies import get_strategy, load_documents
    from kb_arena.strategies.embeddings import get_embedding_function
    from kb_arena.strategies.naive_vector import _chunk_text

    n_qubits_list = n_qubits_list or [2, 3, 4, 5, 6]
    shots_list = shots_list or [256, 1024, 4096, 16384]

    ef = get_embedding_function()
    documents = load_documents(corpus)
    chunk_texts = [
        ch for doc in documents for section in doc.sections for ch in _chunk_text(section.content)
    ]
    questions = load_questions(corpus)
    sample_q = questions[: max(sample_questions, 1)]

    chunk_vecs = np.asarray(ef(chunk_texts), dtype=np.float64)
    q_vecs = np.asarray(ef([q.question for q in sample_q]), dtype=np.float64)
    logger.info(
        "Diagnostics: %d corpus chunk embeddings (%dd)", chunk_vecs.shape[0], chunk_vecs.shape[1]
    )

    # PCA variance curve fits on the corpus chunk embeddings only, so the number
    # is deterministic and independent of how many sample questions are timed.
    pca_curve = pca_variance_curve(chunk_vecs, n_qubits_list)

    # Shot-error curve on one representative query + its candidate pool.
    target_dim = dim_for_qubits(int(settings.sqr_n_qubits))
    q0_vec = q_vecs[0]
    cos = (chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)) @ (
        q0_vec / np.linalg.norm(q0_vec)
    )
    cand_idx = np.argsort(-cos)[: max(int(settings.sqr_fanout) * 5, 10)]
    stacked = np.vstack([q0_vec[None, :], chunk_vecs[cand_idx]])
    encoded, _ = amplitude_encode(stacked, target_dim)
    shot_curve = swap_error_curve(encoded[0], encoded[1:], shots_list)

    # Quantum overhead: time SQR end-to-end vs its naive coarse retrieval.
    sqr = get_strategy("sqr")
    naive_ms: list[float] = []
    total_ms: list[float] = []
    patch = _PatchLLMClient()
    patch.__enter__()
    try:
        for q in sample_q:
            res = await sqr.query(q.question, top_k=5)
            naive_ms.append(res.retrieval_latency_ms)
            total_ms.append(res.latency_ms)
    finally:
        patch.__exit__(None, None, None)

    mean_naive = float(np.mean(naive_ms)) if naive_ms else 0.0
    mean_total = float(np.mean(total_ms)) if total_ms else 0.0
    overhead = max(mean_total - mean_naive, 0.0)

    return QuantumDiagnostics(
        corpus=corpus,
        n_embedding_samples=int(chunk_vecs.shape[0]),
        embedding_dim=int(chunk_vecs.shape[1]),
        pca_variance_curve=pca_curve,
        shot_error_curve=shot_curve,
        sample_questions=len(sample_q),
        mean_quantum_overhead_ms=overhead,
        naive_retrieval_ms=mean_naive,
        sqr_total_ms=mean_total,
    )
