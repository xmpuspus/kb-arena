"""Quantum and quantum-inspired retrieval strategies.

Two rerankers that sit on top of `naive_vector` coarse retrieval (the same
pattern as `rerank_vector`), so they inherit the corpus, ground truth, IR
metrics and the optimizer's statistical layer with zero new metrics code:

* `qiss` — Quantum-Inspired Semantic Similarity. Pure NumPy. Scores candidates
  by state fidelity Tr(ρ_q·ρ_d) = |⟨q|d⟩|² over the SAME embeddings naive_vector
  uses, plus an optional multi-query *superposition* fusion with genuine
  interference cross-terms. No Qiskit, no extra install.
* `sqr` — Simulated Quantum Reranker. A real SWAP-test circuit on the Qiskit
  Aer statevector simulator (exact by default; shots is an accuracy/speed knob).
  Needs the optional `[quantum]` extra (qiskit, qiskit-aer, scikit-learn).
"""

from __future__ import annotations

from kb_arena.strategies.quantum.qiss import QISSStrategy
from kb_arena.strategies.quantum.sqr import SQRStrategy

# SQRStrategy lazy-imports qiskit/qiskit-aer/scikit-learn (the optional [quantum]
# extra) inside its methods, so importing this package never requires them.
__all__ = ["QISSStrategy", "SQRStrategy"]
