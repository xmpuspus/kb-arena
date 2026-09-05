"""The retrieval-regression-gate action must pin every third-party action by
commit SHA and wire caller inputs through env, never straight into a script.

Follows the workflow-parsing pattern in test_release_metadata.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION_DIR = ROOT / ".github" / "actions" / "retrieval-regression-gate"
SHA_PIN = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _all_steps(workflow_or_action: dict) -> list[dict]:
    if "runs" in workflow_or_action:
        return workflow_or_action["runs"]["steps"]
    steps = []
    for job in workflow_or_action["jobs"].values():
        steps.extend(job["steps"])
    return steps


def test_action_file_exists_and_is_composite() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"


def test_every_third_party_action_is_pinned_by_full_commit_sha() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    example = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "retrieval-regression-example.yml").read_text()
    )

    used = [step["uses"] for step in _all_steps(action) if "uses" in step]
    used += [step["uses"] for step in _all_steps(example) if "uses" in step and step["uses"] != "."]
    # Local composite actions (./.github/actions/...) reference this repo's own
    # commit, not a third party, so they carry no version pin to check.
    third_party = [u for u in used if not u.startswith(".")]

    assert third_party, "expected at least one third-party action reference"
    for uses in third_party:
        assert SHA_PIN.match(uses), f"{uses} is not pinned by a full commit SHA"


def test_action_declares_the_inputs_the_gate_needs() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    inputs = action["inputs"]

    required = {"corpus", "corpus-path", "strategies", "metric", "baseline-path", "threshold"}
    for name in required:
        assert inputs[name]["required"] is True, f"{name} must be required"

    assert inputs["top-k"]["default"] == "5"
    assert inputs["kb-arena-version"]["default"] == "0.10.0"
    # pypi is the safe default for an external consumer; a caller gating its
    # own PR on its own retrieval code opts into checkout explicitly.
    assert inputs["install-from"]["default"] == "pypi"


def test_no_step_interpolates_a_caller_input_straight_into_a_script() -> None:
    """Action inputs are caller-controlled text.

    A `${{ inputs.* }}` paste happens before bash runs, so a crafted input
    would execute as code. Every input must ride in `env`, as the publish
    workflow already requires for `github.ref` (see
    test_a_dispatch_cannot_publish_an_arbitrary_branch in
    test_release_metadata.py).
    """
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    for step in _all_steps(action):
        run = step.get("run", "")
        assert "${{ inputs." not in run, f"step {step.get('name')} pastes an input into run:"


def test_the_benchmark_and_compare_steps_wire_inputs_through_env() -> None:
    action = yaml.safe_load((ACTION_DIR / "action.yml").read_text())
    steps = {step["name"]: step for step in _all_steps(action)}

    benchmark_env = steps["Run the retriever-lab benchmark"]["env"]
    assert benchmark_env["CORPUS"] == "${{ inputs.corpus }}"
    assert benchmark_env["STRATEGIES"] == "${{ inputs.strategies }}"
    assert benchmark_env["TOP_K"] == "${{ inputs.top-k }}"

    compare_env = steps["Compare against the stored baseline"]["env"]
    assert compare_env["METRIC"] == "${{ inputs.metric }}"
    assert compare_env["BASELINE_PATH"] == "${{ inputs.baseline-path }}"
    assert compare_env["THRESHOLD"] == "${{ inputs.threshold }}"


def test_compare_script_ships_next_to_the_action() -> None:
    script = ACTION_DIR / "compare_metric.py"
    assert script.exists()
    assert "sys.exit" in script.read_text(), "a regression or a data mismatch must fail the step"


def test_example_workflow_runs_a_real_corpus_against_a_stored_baseline() -> None:
    example = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "retrieval-regression-example.yml").read_text()
    )
    job = next(iter(example["jobs"].values()))
    assert job.get("timeout-minutes"), "the job needs a deadline like every other CI job"

    gate_step = next(s for s in job["steps"] if s.get("uses", "").startswith("./"))
    with_block = gate_step["with"]
    assert with_block["corpus"] == "aws-compute"
    assert with_block["strategies"] == "bm25"
    assert with_block["metric"] == "mean_recall_at_k"
    assert with_block["baseline-path"] == ".github/retrieval-baselines/aws-compute-bm25.json"
    threshold = float(with_block["threshold"])
    assert threshold > 0, "a zero threshold fails on the smallest floating-point noise"
    # This repo's own PRs must gate on the branch's retrieval code, not on the
    # last released PyPI build, or a broken bm25 ranking change would pass.
    assert with_block["install-from"] == "checkout"


def test_referenced_baseline_file_exists_and_matches_the_example_workflow() -> None:
    example = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "retrieval-regression-example.yml").read_text()
    )
    job = next(iter(example["jobs"].values()))
    gate_step = next(s for s in job["steps"] if s.get("uses", "").startswith("./"))
    with_block = gate_step["with"]

    baseline_path = ROOT / with_block["baseline-path"]
    baseline = json.loads(baseline_path.read_text())

    assert baseline["corpus"] == with_block["corpus"]
    assert baseline["metric"] == with_block["metric"]
    assert with_block["strategies"] in baseline["strategies"]
    assert 0.0 <= baseline["strategies"][with_block["strategies"]] <= 1.0


def test_corpus_raw_documents_are_tracked_so_the_example_can_actually_ingest() -> None:
    """The example workflow ingests datasets/aws-compute/raw on a fresh checkout.

    .gitignore excludes raw *.html/*.json/*.txt but not *.md, so the example
    depends on these markdown files being committed rather than generated.
    A file that only exists in this working tree would make the example fail
    on the first run against a clean checkout.
    """
    import subprocess

    output = subprocess.run(
        ["git", "ls-files", "datasets/aws-compute/raw"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tracked_md = [line for line in output.splitlines() if line.endswith(".md")]
    assert tracked_md, "no tracked markdown source for the example corpus"
