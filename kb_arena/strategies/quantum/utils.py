"""Helpers for the SQR (Simulated Quantum Reranker) strategy.

PCA dimensionality reduction onto an amplitude-encodable size, unit
normalization for `qc.initialize()`, and qubit-count arithmetic. scikit-learn is
lazy-imported inside `reduce_pca` so importing this module never requires the
optional [quantum] extra.
"""

from __future__ import annotations

import numpy as np


def qubits_for_dim(dim: int) -> int:
    """Number of qubits whose 2ⁿ amplitudes can hold a `dim`-length state."""
    return max(1, int(np.ceil(np.log2(max(int(dim), 1)))))


def dim_for_qubits(n_qubits: int) -> int:
    """Amplitude-vector length for `n_qubits`: 2ⁿ (16 for the default 4 qubits)."""
    return 2 ** int(n_qubits)


def unit_rows(matrix: np.ndarray | list) -> np.ndarray:
    """Unit-normalize each row (L2). Zero rows are left as-is to avoid NaNs."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def reduce_pca(vectors: np.ndarray | list, target_dim: int) -> tuple[np.ndarray, float]:
    """PCA-reduce rows of `vectors` to `target_dim` columns (non-centered / LSA-style).

    Uses scikit-learn's TruncatedSVD rather than centered PCA on purpose: a
    reranker's candidate pool is tightly clustered around the query, so its mean
    points along the shared query-relevant direction. Centered PCA subtracts that
    mean and keeps only how candidates differ from *each other*, which destroys
    the relevance ordering (measured: Recall@5 collapses to ~0.18). Non-centered
    SVD preserves raw inner products, so the SWAP-test fidelity tracks cosine².

    When the sample count yields fewer than `target_dim` components (a small
    candidate pool), the result is zero-padded up to `target_dim` so the output
    is always amplitude-encodable into the same number of qubits.

    Returns (reduced, variance_explained) where variance_explained is the
    fraction of variance retained by the kept components — the honest cost of
    squeezing high-dim embeddings into 2ⁿ amplitudes.
    """
    from sklearn.decomposition import TruncatedSVD

    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    n_samples, n_features = x.shape
    # TruncatedSVD needs n_components < n_features and <= n_samples.
    n_components = max(1, min(int(target_dim), n_samples - 1, n_features - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    reduced = svd.fit_transform(x)
    variance_explained = float(svd.explained_variance_ratio_.sum())
    if reduced.shape[1] < target_dim:
        pad = np.zeros((reduced.shape[0], target_dim - reduced.shape[1]))
        reduced = np.hstack([reduced, pad])
    return reduced, variance_explained


def amplitude_encode(vectors: np.ndarray | list, target_dim: int) -> tuple[np.ndarray, float]:
    """PCA-reduce to `target_dim` then unit-normalize each row for amplitude encoding.

    Reduce query and documents in ONE PCA fit (pass them stacked) so they share a
    common subspace; split the rows back out afterwards. Returns
    (encoded_rows, variance_explained).
    """
    reduced, var = reduce_pca(vectors, target_dim)
    return unit_rows(reduced), var
