"""SWAP-test circuit construction for the SQR strategy.

The SWAP test compares two amplitude-encoded states |ψ_q⟩, |ψ_d⟩ on separate
registers using one ancilla:

    ancilla ── H ──●── H ── (measure)
    query  ──────╳(register-wise controlled-SWAP with doc)──
    doc    ──────╳──────────────────────────────────────────

Measuring the ancilla gives P(0) = (1 + |⟨ψ_q|ψ_d⟩|²) / 2, so the state
fidelity is |⟨ψ_q|ψ_d⟩|² = 2·P(0) − 1.

qiskit is lazy-imported inside the builder so importing this module never
requires the optional [quantum] extra.
"""

from __future__ import annotations

import numpy as np


def build_swap_test_circuit(psi_q, psi_d, *, measure: bool = False):
    """Build a SWAP-test circuit comparing two unit, power-of-two-length states.

    Layout: qubit 0 is the ancilla, qubits [1..n] the query register, [n+1..2n]
    the doc register, where n = log2(len(psi_q)). `measure=True` adds a classical
    bit and measures the ancilla (shots mode); `measure=False` leaves the circuit
    unmeasured for exact statevector readout.
    """
    from qiskit import QuantumCircuit

    psi_q = np.asarray(psi_q, dtype=np.float64).ravel()
    psi_d = np.asarray(psi_d, dtype=np.float64).ravel()
    if len(psi_q) != len(psi_d):
        raise ValueError(f"SWAP test needs equal-length states; got {len(psi_q)}, {len(psi_d)}")
    n = int(round(np.log2(len(psi_q))))
    if 2**n != len(psi_q):
        raise ValueError(f"State length must be a power of two; got {len(psi_q)}")

    qc = QuantumCircuit(1 + 2 * n, 1 if measure else 0)
    ancilla = 0
    qreg = list(range(1, 1 + n))
    dreg = list(range(1 + n, 1 + 2 * n))

    qc.initialize(psi_q, qreg)
    qc.initialize(psi_d, dreg)
    qc.h(ancilla)
    for a, b in zip(qreg, dreg, strict=True):
        qc.cswap(ancilla, a, b)
    qc.h(ancilla)
    if measure:
        qc.measure(ancilla, 0)
    return qc
