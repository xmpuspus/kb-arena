"""Tests for Strategy 11: SQR — Simulated Quantum Reranker.

These exercise the real Qiskit Aer SWAP test and scikit-learn PCA, so the whole
module skips unless the optional [quantum] extra is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("sklearn")

from kb_arena.models.retrieval import RetrievalTrace, RetrievedChunk  # noqa: E402
from kb_arena.strategies.base import AnswerResult  # noqa: E402
from kb_arena.strategies.quantum.circuits import build_swap_test_circuit  # noqa: E402
from kb_arena.strategies.quantum.diagnostics import (  # noqa: E402
    pca_variance_curve,
    swap_error_curve,
)
from kb_arena.strategies.quantum.sqr import SQRStrategy, swap_test_fidelities  # noqa: E402
from kb_arena.strategies.quantum.utils import (  # noqa: E402
    amplitude_encode,
    reduce_pca,
    unit_rows,
)

# --- The contract: statevector SWAP test == |⟨ψ_q|ψ_d⟩|² ---


def test_swap_test_matches_inner_product_squared():
    q = unit_rows([0.2, 0.9, 0.3, 0.1])[0]
    docs = unit_rows([[0.2, 0.9, 0.3, 0.1], [0.7, 0.1, 0.1, 0.6], [-0.2, -0.9, -0.3, -0.1]])
    expected = [float(np.dot(q, d) ** 2) for d in docs]
    fids = swap_test_fidelities(q, docs, shots=0)
    assert np.allclose(fids, expected, atol=1e-6)
    assert fids[0] == pytest.approx(1.0, abs=1e-6)  # q vs q
    assert all(0.0 <= f <= 1.0 for f in fids)


def test_swap_test_shots_approximates_exact():
    q = unit_rows([0.5, 0.5, 0.5, 0.5])[0]
    docs = unit_rows([[0.6, 0.4, 0.5, 0.4]])
    exact = swap_test_fidelities(q, docs, shots=0)[0]
    approx = swap_test_fidelities(q, docs, shots=20000)[0]
    assert abs(approx - exact) < 0.05


def test_build_swap_test_circuit_shape():
    qc = build_swap_test_circuit([1, 0, 0, 0], [0, 1, 0, 0], measure=True)
    assert qc.num_qubits == 5  # 2 qubits/register * 2 + 1 ancilla
    assert qc.num_clbits == 1


def test_build_swap_test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        build_swap_test_circuit([1, 0, 0], [0, 1, 0])


# --- PCA reduction (non-centered) ---


def test_reduce_pca_shape_and_variance():
    rng = np.random.default_rng(0)
    x = rng.random((30, 128))
    reduced, var = reduce_pca(x, 16)
    assert reduced.shape == (30, 16)
    assert 0.0 <= var <= 1.0


def test_amplitude_encode_unit_norm():
    rng = np.random.default_rng(1)
    x = rng.random((10, 64))
    enc, var = amplitude_encode(x, 16)
    assert enc.shape == (10, 16)
    assert np.allclose(np.linalg.norm(enc, axis=1), 1.0, atol=1e-9)


def test_reduce_pca_small_pool_pads():
    rng = np.random.default_rng(3)
    x = rng.random((5, 64))  # fewer samples than 16 target dims
    reduced, _ = reduce_pca(x, 16)
    assert reduced.shape == (5, 16)  # padded out to the target dim


# --- Honest diagnostics curves ---


def test_pca_variance_curve_monotonic():
    rng = np.random.default_rng(2)
    x = rng.random((50, 256))
    curve = pca_variance_curve(x, [2, 3, 4, 5])
    vals = [p.variance_explained for p in curve]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert vals == sorted(vals)  # more qubits -> at least as much variance


def test_swap_error_curve_improves_with_shots():
    q = unit_rows([0.5, 0.5, 0.5, 0.5])[0]
    docs = unit_rows([[0.6, 0.4, 0.5, 0.4], [0.1, 0.9, 0.2, 0.3]])
    curve = swap_error_curve(q, docs, [256, 16384])
    assert all(0.0 <= p.mean_abs_error <= 1.0 for p in curve)
    assert curve[-1].mean_abs_error <= curve[0].mean_abs_error + 0.02


# --- Query path (mocked base + embeddings, no API) ---


@pytest.mark.asyncio
async def test_sqr_query_reranks_and_records_caveats(mock_chroma_client, mock_llm_client):
    from unittest.mock import AsyncMock

    strategy = SQRStrategy(chroma_client=mock_chroma_client, llm_client=mock_llm_client)

    contents = ["c0", "c1", "c2"]
    chunks = [
        RetrievedChunk(
            chunk_id=f"doc::sec::{i}",
            doc_id="doc",
            content=c,
            score=0.5,
            rank=i + 1,
            source_strategy="naive_vector",
        )
        for i, c in enumerate(contents)
    ]
    base = AnswerResult(
        answer="",
        sources=["doc"],
        retrieval=RetrievalTrace(query="Q", retrieved=chunks, latency_ms=1.0, top_k=12),
        strategy="naive_vector",
    )
    strategy._base.query = AsyncMock(return_value=base)

    table = {
        "Q": [1, 0, 0, 0, 0, 0, 0, 0],
        "c0": [0, 1, 0, 0, 0, 0, 0, 0],
        "c1": [1, 0, 0, 0, 0, 0, 0, 0],  # identical to Q -> highest fidelity
        "c2": [0.5, 0.5, 0, 0, 0, 0, 0, 0],
    }
    strategy._embed_fn = lambda texts: [table[t] for t in texts]

    result = await strategy.query("Q", top_k=2)
    assert result.strategy == "sqr"
    kept = result.retrieval.retrieved
    assert kept[0].content == "c1"
    assert kept[0].score >= kept[1].score
    md = kept[0].metadata
    assert "sqr_fidelity" in md
    assert "sqr_pca_variance" in md
    assert "sqr_overhead_ms" in md
    assert md["sqr_mode"] == "statevector"
