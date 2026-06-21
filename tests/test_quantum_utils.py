"""Pure (dependency-free) tests for the quantum layer: qubit math, normalization,
the diagnostics model's [0,1] clamps, and strategy registry wiring.

No qiskit / scikit-learn import here, so these run in core CI.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from pydantic import ValidationError

from kb_arena.benchmark.runner import STRATEGY_NAMES, _load_strategies
from kb_arena.models.quantum import PCAVariancePoint, QuantumDiagnostics, ShotErrorPoint
from kb_arena.strategies import STRATEGY_REGISTRY, get_strategy
from kb_arena.strategies.quantum.qiss import QISSStrategy
from kb_arena.strategies.quantum.sqr import SQRStrategy
from kb_arena.strategies.quantum.utils import dim_for_qubits, qubits_for_dim, unit_rows

# --- Qubit / dimension arithmetic ---


def test_dim_for_qubits():
    assert dim_for_qubits(4) == 16
    assert dim_for_qubits(2) == 4
    assert dim_for_qubits(8) == 256


def test_qubits_for_dim():
    assert qubits_for_dim(16) == 4
    assert qubits_for_dim(13) == 4  # rounds up to the next power of two
    assert qubits_for_dim(1) == 1


def test_unit_rows_normalizes_and_keeps_zero_rows():
    m = unit_rows([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    norms = np.linalg.norm(m, axis=1)
    assert norms[0] == pytest.approx(1.0)
    assert norms[1] == pytest.approx(0.0)  # zero row stays zero (no NaN)
    assert norms[2] == pytest.approx(1.0)


# --- Diagnostics model clamps (the bpref-negative class of guard) ---


def test_pca_variance_point_rejects_out_of_range():
    PCAVariancePoint(n_qubits=4, encoded_dim=16, variance_explained=0.65)  # ok
    with pytest.raises(ValidationError):
        PCAVariancePoint(n_qubits=4, encoded_dim=16, variance_explained=1.5)


def test_shot_error_point_rejects_negative():
    ShotErrorPoint(shots=256, mean_abs_error=0.04)  # ok
    with pytest.raises(ValidationError):
        ShotErrorPoint(shots=256, mean_abs_error=-0.1)


def test_quantum_diagnostics_overhead_non_negative():
    QuantumDiagnostics(
        corpus="aws-compute",
        n_embedding_samples=42,
        embedding_dim=3072,
        sample_questions=5,
        mean_quantum_overhead_ms=1021.4,
        naive_retrieval_ms=348.1,
        sqr_total_ms=1369.5,
    )
    with pytest.raises(ValidationError):
        QuantumDiagnostics(
            corpus="x",
            n_embedding_samples=1,
            embedding_dim=16,
            sample_questions=1,
            mean_quantum_overhead_ms=-1.0,
            naive_retrieval_ms=1.0,
            sqr_total_ms=1.0,
        )


# --- Registry / loader wiring ---


def test_qiss_and_sqr_registered():
    assert STRATEGY_REGISTRY["qiss"] is QISSStrategy
    assert STRATEGY_REGISTRY["sqr"] is SQRStrategy


def test_strategy_names_includes_qiss_excludes_sqr():
    # qiss is pure-NumPy core; sqr needs the optional extra so it is not in "all".
    assert "qiss" in STRATEGY_NAMES
    assert "sqr" not in STRATEGY_NAMES


def test_get_strategy_qiss():
    s = get_strategy("qiss")
    assert isinstance(s, QISSStrategy)
    assert s.name == "qiss"


def test_get_strategy_sqr_present_or_clear_error():
    if importlib.util.find_spec("qiskit") is not None:
        s = get_strategy("sqr")
        assert isinstance(s, SQRStrategy)
    else:
        with pytest.raises(ImportError, match="quantum"):
            get_strategy("sqr")


def test_load_strategies_splits_comma():
    loaded = [s.name for s in _load_strategies("naive_vector,qiss")]
    assert loaded == ["naive_vector", "qiss"]


def test_load_strategies_dedupes():
    loaded = [s.name for s in _load_strategies("qiss,qiss")]
    assert loaded == ["qiss"]
