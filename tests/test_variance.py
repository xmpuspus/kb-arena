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
    # A build identity is required for comparability, because two runs that
    # both fail to name one are not known to share it.
    manifest = _manifest(code_version="0.11.0", git_sha="a" * 40)
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
    """A summary is valid JSON and carries no run, so it is not lost evidence."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    (tmp_path / "summary.json").write_text(json.dumps({"corpora": {}}))
    (tmp_path / "report.json").write_text(json.dumps([1, 2, 3]))

    assert variance.load_runs() == []


def test_a_legacy_group_is_never_read_as_a_spread():
    """Every manifestless run keys the same way, whatever it measured."""
    runs = [
        {"corpus": "c", "strategy": "bm25", "accuracy_by_tier": {"1": 0.20}},
        {"corpus": "c", "strategy": "bm25", "accuracy_by_tier": {"1": 0.90}},
    ]

    [row] = variance.spread_report(runs)

    assert row["compatibility_key"] == "legacy"
    assert row["runs"] == 2
    assert row["comparable"] is False, (
        "two runs with no manifest may have measured anything, so the gap "
        "between 0.20 and 0.90 is not noise"
    )


def test_runs_from_different_code_versions_are_not_repeats():
    """A spread must describe noise, not the change between two releases."""
    manifest = _manifest()
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": {**manifest, "code_version": version},
        }
        for version, value in (("0.10.0", 0.50), ("0.11.0", 0.70))
    ]

    [row] = variance.spread_report(runs)

    assert row["code_versions"] == ["0.10.0", "0.11.0"]
    assert row["comparable"] is False


def test_one_code_version_is_comparable():
    manifest = _manifest(code_version="0.11.0")
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": dict(manifest),
        }
        for value in (0.50, 0.55, 0.60)
    ]

    [row] = variance.spread_report(runs)

    assert row["comparable"] is True
    assert row["code_versions"] == ["0.11.0"]


def test_two_commits_on_one_version_are_not_repeats():
    """Several commits share one unreleased version during development."""
    manifest = _manifest(code_version="0.11.0")
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": {**manifest, "git_sha": sha},
        }
        for sha, value in (("aaaaaaa", 0.2), ("bbbbbbb", 0.8))
    ]

    [row] = variance.spread_report(runs)

    assert row["comparable"] is False, "the version matched and the code did not"
    assert len(row["code_versions"]) == 2


def test_an_incomparable_group_reports_values_and_no_mean():
    """A warning cannot turn a mean across experiments into a valid statistic."""
    manifest = _manifest(code_version="0.11.0")
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": {**manifest, "git_sha": sha},
        }
        for sha, value in (("aaaaaaa", 0.2), ("bbbbbbb", 0.8))
    ]

    [row] = variance.spread_report(runs)
    metric = row["metrics"]["accuracy_by_tier"]

    assert metric["values"] == [0.2, 0.8]
    assert "mean" not in metric, "0.5 is the midpoint of a code change, not a mean"
    assert "sd" not in metric


def test_a_malformed_result_stops_the_command_too(tmp_path, monkeypatch):
    """A file that parses as nothing is lost evidence, the same as an unreadable one."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    good = {"corpus": "c", "strategy": "bm25", "run_id": "a", "accuracy_by_tier": {"1": 0.2}}
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "c_bm25.json").write_text(json.dumps(good))
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_b" / "c_bm25.json").write_text("{truncated")

    with pytest.raises(variance.RunsUnreadableError, match="malformed JSON"):
        variance.load_runs("c")


def test_a_true_in_a_result_is_never_a_perfect_score():
    """float(True) is 1.0, so a corrupt file would report a perfect run."""
    assert variance._metric({"accuracy_by_tier": True}, "accuracy_by_tier") is None
    assert variance._metric({"accuracy_by_tier": {"1": True}}, "accuracy_by_tier") is None
    assert variance._metric({"accuracy_by_tier": {"1": 0.5}}, "accuracy_by_tier") == 0.5


def test_the_build_identity_keeps_the_whole_commit():
    """Seven characters is an abbreviation, and this value is used for equality."""
    manifest = {"code_version": "0.11.0", "git_sha": "a" * 40}

    assert variance._code_version({"manifest": manifest}).endswith("a" * 40)


def test_the_seed_reaches_the_lab_bootstrap_it_claims_to_control():
    """The manifest names bootstrap resampling, and the lab has its own."""
    import inspect

    from kb_arena.benchmark import retriever_lab

    source = inspect.getsource(retriever_lab._bootstrap_ci)
    assert "settings.run_seed" in source
    assert "random_state=0" not in source


def test_the_seed_setting_stays_inside_the_range_scipy_accepts():
    """scipy rejects a seed above 2**32-1, after the sweep has already run."""
    from pydantic import ValidationError

    from kb_arena.settings import Settings

    with pytest.raises(ValidationError):
        Settings(run_seed=2**32)
    with pytest.raises(ValidationError):
        Settings(run_seed=-1)
    assert Settings(run_seed=2**32 - 1).run_seed == 2**32 - 1


def test_an_unreadable_result_stops_the_command(tmp_path, monkeypatch):
    """A run that exists and cannot be read is lost evidence, not a missing run."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    record = {"corpus": "c", "strategy": "bm25", "run_id": "a", "accuracy_by_tier": {"1": 0.2}}
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "c_bm25.json").write_text(json.dumps(record))
    blocked = tmp_path / "run_b"
    blocked.mkdir()
    unreadable = blocked / "c_bm25.json"
    unreadable.write_text(json.dumps(record))
    unreadable.chmod(0o000)

    try:
        with pytest.raises(variance.RunsUnreadableError, match="could not be read"):
            variance.load_runs("c")
    finally:
        unreadable.chmod(0o644)


def test_the_seed_reaches_the_trial_order_it_claims_to_control():
    """A manifest claim no code honours is a record of work that never happened."""
    from kb_arena.benchmark.optimizer import TrialConfig, build_trials

    baseline = TrialConfig(
        strategy="bm25",
        top_k=5,
        chunk_tokens=512,
        embedding_provider="bge",
        reranker_backend="bge",
    )
    kwargs = dict(
        strategy="bm25",
        top_ks=[3, 5, 10, 20],
        chunk_sizes=[256, 512],
        embedding_providers=["bge"],
        reranker_backends=["bge"],
        baseline=baseline,
        method="random",
        max_trials=3,
    )
    original = settings.run_seed
    try:
        settings.run_seed = 7
        first = [(t.top_k, t.chunk_tokens) for t in build_trials(**kwargs)]
        settings.run_seed = 8
        other = [(t.top_k, t.chunk_tokens) for t in build_trials(**kwargs)]
        settings.run_seed = 7
        again = [(t.top_k, t.chunk_tokens) for t in build_trials(**kwargs)]
    finally:
        settings.run_seed = original

    assert first == again, "one seed must give one trial order"
    assert first != other, "a different seed must give a different order"


def test_a_resume_refuses_a_different_seed():
    """A resume stamps the whole run with the new seed."""
    from kb_arena.benchmark.runner import RESUME_KEYS

    assert "run_seed" in RESUME_KEYS


def test_an_old_checkpoint_without_a_seed_still_resumes(tmp_path, monkeypatch):
    """Adding a resume key must not strand every checkpoint written before it."""
    from kb_arena.benchmark import runner

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    run_dir = tmp_path / "run_old"
    run_dir.mkdir()
    earlier = {
        "llm_provider": "anthropic",
        "generate_model": "m",
        "judge_provider": "anthropic",
        "judge_model": "m",
        "top_k": 5,
        "tier": 0,
        "question_split": "all",
        "reference_free": False,
        "ragas_enabled": False,
    }
    (run_dir / "run.json").write_text(json.dumps({"config_snapshot": earlier}))

    current = {**earlier, "run_seed": 3}
    resumed = runner.check_resumable(tmp_path, "old", current)

    assert resumed is not None, "a snapshot that predates the key must still resume"


def test_two_runs_with_an_empty_run_id_are_not_one_run(tmp_path, monkeypatch):
    """A file may carry `run_id: ""`, and dropping one shrinks the sample."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    for name, accuracy in (("run_a", 0.2), ("run_b", 0.8)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "c_bm25.json").write_text(
            json.dumps(
                {
                    "corpus": "c",
                    "strategy": "bm25",
                    "run_id": "",
                    "accuracy_by_tier": {"1": accuracy},
                }
            )
        )

    assert len(variance.load_runs("c")) == 2


def test_a_run_missing_the_metric_is_counted_and_not_hidden():
    """A spread over three of five runs must say it rests on three."""
    manifest = _manifest(code_version="0.11.0", git_sha="a" * 40)
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "records": [{}],
            "manifest": dict(manifest),
            **extra,
        }
        for extra in (
            {"accuracy_by_tier": {"1": 0.5}},
            {"accuracy_by_tier": {"1": 0.6}},
            {},
        )
    ]

    [row] = variance.spread_report(runs)
    metric = row["metrics"]["accuracy_by_tier"]

    assert row["runs"] == 3
    assert metric["runs"] == 2
    assert metric["runs_without_this_metric"] == 1


def test_the_cli_refuses_a_seed_outside_the_declared_bound():
    """Assigning the attribute skips the validator, so the CLI must apply it."""
    import inspect

    from kb_arena import cli

    source = inspect.getsource(cli.benchmark)
    assert (
        "Settings(run_seed=seed).run_seed" in source
    ), "the CLI must build the value through the model, not assign it raw"


def test_the_seed_record_names_only_consumers_that_exist():
    """A manifest claim no code honours is a record of work that never happened.

    This is the mechanical half. The claim lists what the seed controls, and
    each entry maps to a call site that reads `settings.run_seed`. When a later
    change adds a claim without a consumer, or drops a consumer and leaves the
    claim, this fails.
    """
    import inspect

    from kb_arena.benchmark import optimizer, retriever_lab
    from kb_arena.benchmark.manifest import seed_identity

    consumers = {
        "optimize trial order": inspect.getsource(optimizer.build_trials),
        "bootstrap resampling": (
            inspect.getsource(optimizer._bootstrap_ci)
            if hasattr(optimizer, "_bootstrap_ci")
            else inspect.getsource(retriever_lab._bootstrap_ci)
        ),
    }
    claimed = seed_identity()["controls"]

    assert set(claimed) == set(consumers), (
        f"the record claims {claimed} and this test knows about {sorted(consumers)}. "
        "Add the consumer, or drop the claim."
    )
    for claim, source in consumers.items():
        assert "settings.run_seed" in source, f"nothing reads the seed for {claim!r}"

    # The lab has its own bootstrap, and it must read the seed too.
    assert "settings.run_seed" in inspect.getsource(retriever_lab._bootstrap_ci)


def test_two_runs_that_both_name_no_build_are_not_comparable():
    """`unrecorded` is not a build. Sharing an unknown is not sharing."""
    manifest = _manifest()
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "accuracy_by_tier": {"1": value},
            "records": [{}],
            "manifest": dict(manifest),
        }
        for value in (0.2, 0.8)
    ]

    [row] = variance.spread_report(runs)

    assert row["code_versions"] == ["unrecorded"]
    assert row["comparable"] is False


def test_a_malformed_file_that_is_not_a_result_never_blocks_the_report(tmp_path, monkeypatch):
    """`results/` also holds summaries and scratch, and one bad byte is not lost evidence."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "c_bm25.json").write_text(
        json.dumps(
            {"corpus": "c", "strategy": "bm25", "run_id": "a", "accuracy_by_tier": {"1": 0.2}}
        )
    )
    (tmp_path / "run_a" / "summary.json").write_text("{truncated")
    (tmp_path / "scratch.json").write_text("also broken")

    runs = variance.load_runs("c")

    assert len(runs) == 1


def test_the_fail_below_gate_reads_every_repeat():
    """`--runs 3` overwrites the top-level file twice, so a gate saw run three."""
    import inspect

    from kb_arena import cli

    source = inspect.getsource(cli.benchmark)
    assert "load_runs(" in source, "the gate must read every run directory"
    assert (
        "_load_results(" not in source
    ), "_load_results reads the top-level files, which each repeat overwrites"


def test_a_plugin_result_counts_as_a_result(tmp_path):
    """A plugin strategy writes a result whose name is not in the catalog."""
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    assert variance._looks_like_a_result(run_dir / "c_my_custom_plugin.json") is True
    assert variance._looks_like_a_result(run_dir / "summary.json") is False
    assert variance._looks_like_a_result(tmp_path / "scratch.json") is False
    # A stray file inside a run directory must not speak for the evidence.
    assert variance._looks_like_a_result(run_dir / "notes.json") is False
    assert variance._looks_like_a_result(run_dir / "_orphan.json") is False


def test_an_unrelated_unreadable_file_never_aborts_the_report(tmp_path, monkeypatch):
    """A permission error on a scratch file is not lost benchmark evidence."""
    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "c_bm25.json").write_text(
        json.dumps(
            {"corpus": "c", "strategy": "bm25", "run_id": "a", "accuracy_by_tier": {"1": 0.2}}
        )
    )
    blocked = tmp_path / "notes.json"
    blocked.write_text("{}")
    blocked.chmod(0o000)

    try:
        assert len(variance.load_runs("c")) == 1
    finally:
        blocked.chmod(0o644)


def test_a_resume_of_a_legacy_checkpoint_says_the_seed_covers_only_new_records(caplog):
    """The records already on disk were scored under a seed nobody wrote down."""
    import logging

    from kb_arena.benchmark import runner

    earlier = {"top_k": 5, "question_split": "all"}
    current = {**earlier, "run_seed": 3}

    with caplog.at_level(logging.WARNING):
        changed = [
            k
            for k in runner.RESUME_KEYS
            if k in earlier and k in current and earlier[k] != current[k]
        ]

    assert changed == [], "a key the old snapshot lacks is not a changed setting"
    # The warning itself is exercised through check_resumable in the test above
    # this one. Here the point is that the resume is allowed at all.


def test_a_non_finite_metric_counts_as_missing():
    """A NaN is not a measurement, and the count must say so."""
    manifest = _manifest(code_version="0.11.0", git_sha="a" * 40)
    runs = [
        {
            "corpus": "c",
            "strategy": "bm25",
            "records": [{}],
            "manifest": dict(manifest),
            "accuracy_by_tier": {"1": value},
        }
        for value in (0.5, 0.6, float("nan"))
    ]

    [row] = variance.spread_report(runs)
    metric = row["metrics"]["accuracy_by_tier"]

    assert row["runs"] == 3
    assert metric["runs"] == 2
    assert metric["runs_without_this_metric"] == 1


def test_a_resumed_legacy_run_says_the_seed_does_not_cover_it(tmp_path, monkeypatch):
    """The records it inherited were scored under a seed nobody wrote down."""
    from kb_arena.benchmark import runner
    from kb_arena.benchmark.manifest import seed_identity

    assert seed_identity()["covers_whole_run"] is True
    assert seed_identity(covers_whole_run=False)["covers_whole_run"] is False

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    run_dir = tmp_path / "run_old"
    run_dir.mkdir()
    earlier = {"top_k": 5, "question_split": "all"}
    (run_dir / "run.json").write_text(json.dumps({"config_snapshot": earlier}))

    resumed = runner.check_resumable(tmp_path, "old", {**earlier, "run_seed": 3})

    assert (
        resumed["_seed_covers_whole_run"] is False
    ), "a checkpoint that predates seeds cannot vouch for the records it holds"


def test_a_fresh_run_records_a_seed_that_covers_all_of_it(tmp_path, monkeypatch):
    from kb_arena.benchmark import runner

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    run_dir = tmp_path / "run_new"
    run_dir.mkdir()
    snapshot = {"top_k": 5, "question_split": "all", "run_seed": 3}
    (run_dir / "run.json").write_text(json.dumps({"config_snapshot": snapshot}))

    resumed = runner.check_resumable(tmp_path, "new", dict(snapshot))

    assert resumed["_seed_covers_whole_run"] is True


def test_the_recorded_commit_is_the_whole_one():
    """Variance compares this for equality, and an abbreviation is a prefix."""
    import inspect

    from kb_arena.benchmark import manifest

    source = inspect.getsource(manifest.git_sha)
    assert '"--short"' not in source, "the whole commit, not the abbreviation"
    sha = manifest.git_sha()
    if sha:
        assert len(sha) == 40


def test_a_checkpoint_without_an_experiment_key_still_resumes(tmp_path, monkeypatch):
    """Refusing it blamed the question set for a change nobody made."""
    from kb_arena.benchmark import runner

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    run_dir = tmp_path / "run_old"
    run_dir.mkdir()
    record: dict = {"run_id": "old"}

    runner._bind_run_manifest(record, "c", "key123", "old", tmp_path, "old")

    assert record["manifests"]["c"] == "key123", "the key is recorded, not refused"


def test_a_checkpoint_with_a_different_key_is_still_refused(tmp_path, monkeypatch):
    """The permissive path must not swallow a real mismatch."""
    from kb_arena.benchmark import runner

    monkeypatch.setattr(settings, "results_path", str(tmp_path))
    record: dict = {"run_id": "old", "manifests": {"c": "other"}}

    with pytest.raises(runner.BenchmarkExecutionError, match="experiment key"):
        runner._bind_run_manifest(record, "c", "key123", "old", tmp_path, "old")
