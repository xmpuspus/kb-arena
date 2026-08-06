"""Contracts for the strategy catalog shared by runtime surfaces."""

from __future__ import annotations

from kb_arena.benchmark.runner import STRATEGY_NAMES
from kb_arena.strategies import STRATEGY_REGISTRY
from kb_arena.strategies.catalog import STRATEGY_CATALOG, default_strategy_names


def test_catalog_matches_the_registered_strategy_order():
    assert tuple(spec.name for spec in STRATEGY_CATALOG) == tuple(STRATEGY_REGISTRY)


def test_catalog_separates_registered_and_default_strategies():
    assert len(STRATEGY_CATALOG) == 11
    assert default_strategy_names() == tuple(STRATEGY_NAMES)
    assert len(default_strategy_names()) == 10


def test_catalog_labels_quantum_strategies_as_experiments():
    by_name = {spec.name: spec for spec in STRATEGY_CATALOG}

    assert by_name["qiss"].experimental is True
    assert by_name["qiss"].optional_extra is None
    assert by_name["sqr"].experimental is True
    assert by_name["sqr"].default_benchmark is False
    assert by_name["sqr"].optional_extra == "quantum"
    assert by_name["sqr"].required_modules == ("qiskit", "qiskit_aer", "sklearn")
