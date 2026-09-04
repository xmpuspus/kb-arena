"""Repeats of one experiment give a spread, and only inside one key."""

from __future__ import annotations

import json

import pytest

from kb_arena.benchmark import variance
from kb_arena.benchmark.manifest import CORE_FIELDS, seed_identity
from kb_arena.settings import settings


def test_one_run_gives_a_point_and_says_it_carries_no_spread():
    spread = variance.summarize([0.5])

    assert spread.runs == 1
    assert spread.sd is None
    assert spread.half_width is None
    assert spread.thin is True


def test_three_runs_give_a_sample_standard_deviation_and_a_range():
    spread = variance.summarize([0.50, 0.60, 0.55])

    assert spread.runs == 3
    assert spread.mean == pytest.approx(0.55)
    assert spread.sd == pytest.approx(0.05)
    assert spread.half_width == pytest.approx(0.05)
    assert spread.thin is False


def test_the_standard_deviation_is_the_sample_one_not_the_population_one():
    """The runs are a sample of the runs that could have happened."""
    spread = variance.summarize([1.0, 3.0])

    # Population SD here is 1.0. Sample SD is sqrt(2).
    assert spread.sd == pytest.approx(2.0**0.5)


def test_a_non_finite_value_never_reaches_the_mean():
    spread = variance.summarize([0.5, float("nan"), float("inf"), 0.7])

    assert spread.runs == 2
    assert spread.mean == pytest.approx(0.6)


def test_nothing_measured_gives_nothing():
    assert variance.summarize([]) is None
    assert variance.summarize([None, None]) is None


def test_the_seed_stays_out_of_the_compatibility_key():
    """Two runs that differ only by seed measured the same experiment.

    A seed inside the key would split them into two groups of one run each,
    which is the exact opposite of what a spread needs.
    """
    assert "seed" not in CORE_FIELDS


def test_the_seed_record_says_what_it_does_not_control():
    """A captured seed must not read as a promise of an identical run."""
    identity = seed_identity()

    assert identity["value"] == settings.run_seed
    assert identity["does_not_control"], "the honest half of the record"
    assert any("model sampling" in claim for claim in identity["does_not_control"])


def _manifest(**overrides) -> dict:
    """A manifest complete enough for the reader to recompute its key."""
    base = {
        "schema_version": 1,
        "corpus": "c",
        "question_split": "all",
        "question_set_fingerprint": "abc123",
        "qrels_fingerprint": None,
        "generation": {},
        "scoring": {"reference_free": True, "ragas": False},
        "judge": None,
        "embedding": {},
        "chunk": {"tokens": 512, "overlap_tokens": 64},
        "top_k": 5,
        "question_count": 1,
    }
    return {**base, **overrides}


def test_two_keys_never_share_a_spread():
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": 0.5},
            "records": [{}],
            "manifest": _manifest(top_k=5),
        },
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": 0.9},
            "records": [{}],
            "manifest": _manifest(top_k=20),
        },
    ]

    rows = variance.spread_report(runs)

    assert len(rows) == 2, "a different top_k is a different experiment"
    assert all(row["metrics"]["accuracy_by_tier"]["runs"] == 1 for row in rows)


def test_repeats_of_one_experiment_group_into_one_row():
    manifest = _manifest()
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": {**manifest, "seed": {"value": seed}},
        }
        for seed, value in ((0, 0.50), (1, 0.60), (2, 0.55))
    ]

    [row] = variance.spread_report(runs)

    assert row["runs"] == 3
    assert row["seeds"] == ["0", "1", "2"]
    assert row["metrics"]["accuracy_by_tier"]["mean"] == pytest.approx(0.55)
    assert row["metrics"]["accuracy_by_tier"]["thin_evidence"] is False


def test_a_group_that_mixes_seeded_and_unseeded_runs_never_crashes():
    """Every result already on disk carries no seed, so the first upgrade hits this.

    Sorting a None beside an int raises, and a user who upgrades, runs
    `benchmark --runs 3` and then `variance` builds exactly that group.
    """
    manifest = _manifest()
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": 0.50},
            "records": [{}],
            "manifest": dict(manifest),
        },
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": 0.60},
            "records": [{}],
            "manifest": {**manifest, "seed": {"value": 0}},
        },
    ]

    [row] = variance.spread_report(runs)

    assert row["runs"] == 2
    # The unseeded run is named, not dropped. Dropping it would print "0" and
    # claim both runs used seed 0, a provenance the files do not carry.
    assert row["seeds"] == ["0", "unrecorded"]


def test_a_run_written_before_seeds_reports_an_unrecorded_seed():
    assert variance.seed_of({"manifest": {}}) is None
    assert variance.seed_of({"manifest": {"seed": {"value": True}}}) is None
    assert variance.seed_of({"manifest": {"seed": {"value": 7}}}) == 7


def test_the_loader_counts_a_run_once_even_when_it_sits_in_two_places(tmp_path, monkeypatch):
    """The newest run also lands at the top level, beside its own directory."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    record = {"corpus": "c", "strategy": "bm25", "run_id": "abc", "accuracy_by_tier": {"1": 0.5}}
    (tmp_path / "run_abc").mkdir()
    (tmp_path / "run_abc" / "c_bm25.json").write_text(json.dumps(record))
    (tmp_path / "c_bm25.json").write_text(json.dumps(record))

    runs = variance.load_runs("c")

    assert len(runs) == 1


def test_the_loader_skips_a_file_that_is_not_a_result(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps({"corpora": {}}))
    (tmp_path / "broken.json").write_text("{not json")

    assert variance.load_runs() == []
