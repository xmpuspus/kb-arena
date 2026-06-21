"""Honest-caveat diagnostics for the quantum strategies (#10 qiss, #11 sqr).

These models carry the numbers the project's North Star demands be reported
without cherry-picking: how much variance amplitude-encoding discards at each
qubit count, how SWAP-test accuracy trades against shot count, and the real
wall-clock overhead the quantum rerank adds over the naive_vector baseline.

Every probability/fraction field is clamped to [0, 1] (the bpref-negative class
of bug the project guards against).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PCAVariancePoint(BaseModel):
    """Variance retained when corpus embeddings are reduced to 2ⁿ amplitudes."""

    n_qubits: int = Field(ge=1)
    encoded_dim: int = Field(ge=1)
    variance_explained: float = Field(ge=0.0, le=1.0)


class ShotErrorPoint(BaseModel):
    """Mean absolute SWAP-test fidelity error of sampled vs exact statevector mode."""

    shots: int = Field(ge=1)
    mean_abs_error: float = Field(ge=0.0, le=1.0)


class QuantumDiagnostics(BaseModel):
    """Reproducible honesty report for the SQR (and shared qiss) reduction layer."""

    corpus: str
    n_embedding_samples: int = Field(ge=0)
    embedding_dim: int = Field(ge=1)
    pca_variance_curve: list[PCAVariancePoint] = Field(default_factory=list)
    shot_error_curve: list[ShotErrorPoint] = Field(default_factory=list)
    sample_questions: int = Field(ge=0)
    mean_quantum_overhead_ms: float = Field(ge=0.0)
    naive_retrieval_ms: float = Field(ge=0.0)
    sqr_total_ms: float = Field(ge=0.0)
