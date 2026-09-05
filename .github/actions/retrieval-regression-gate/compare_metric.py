"""Compare a fresh kb-arena retriever-lab run against a stored baseline.

Exits 1 when a named metric drops by more than the caller's threshold, so a
pull request that regresses retrieval quality fails its check instead of
merging silently. corpus, metric, and top_k are all checked against the
baseline file, not assumed, so a baseline never gets compared against a run
it was not recorded for.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_run(results_path: str) -> dict:
    matches = glob.glob(os.path.join(results_path, "run_*", "retriever_lab.json"))
    if len(matches) != 1:
        sys.exit(
            f"Expected exactly one retriever_lab.json under {results_path}, found {len(matches)}"
        )
    return _load_json(matches[0])


def _finite(raw: str, name: str) -> float:
    """One environment value as a real number, or a refusal.

    `float()` accepts `nan` and `inf`, and every comparison against NaN is
    false. A NaN threshold turned the gate green whatever the drop.
    """
    try:
        value = float(raw)
    except ValueError:
        sys.exit(f"{name} must be a number, got {raw!r}")
    if not math.isfinite(value):
        sys.exit(f"{name} must be a finite number, got {raw!r}")
    return value


def main() -> None:
    metric = os.environ["METRIC"]
    threshold = _finite(os.environ["THRESHOLD"], "THRESHOLD")
    corpus = os.environ["CORPUS"]
    top_k = int(os.environ["TOP_K"])
    baseline = _load_json(os.environ["BASELINE_PATH"])
    run = _load_run(os.environ["KB_ARENA_RESULTS_PATH"])

    for field, expected in (("corpus", corpus), ("metric", metric), ("top_k", top_k)):
        if baseline.get(field) != expected:
            sys.exit(
                f"Baseline records {field}={baseline.get(field)!r}, "
                f"the gate ran with {field}={expected!r}"
            )

    by_strategy = run.get("corpora", {}).get(corpus, {})
    failures = []
    # Every strategy the caller asked for has to be in the baseline. Walking
    # the baseline alone meant a newly added strategy could regress, or vanish
    # from the run, while the gate reported success over the ones it knew.
    requested = [
        name.strip() for name in os.environ.get("STRATEGIES", "").split(",") if name.strip()
    ]
    for name in requested:
        if name not in baseline["strategies"]:
            failures.append(f"{name}: the gate ran it and the baseline records no value for it")
    for strategy, baseline_value in baseline["strategies"].items():
        if not isinstance(baseline_value, int | float) or not math.isfinite(baseline_value):
            # Python's JSON parser accepts NaN, and every comparison against it
            # is false, so a NaN baseline made the gate pass whatever the run did.
            failures.append(f"{strategy}: the baseline records {baseline_value!r}, not a number")
            continue
        strategy_result = by_strategy.get(strategy)
        if strategy_result is None:
            failures.append(f"{strategy}: missing from the fresh run")
            continue
        new_value = strategy_result.get(metric)
        if new_value is None:
            failures.append(f"{strategy}: {metric} missing from the fresh run")
            continue
        if not math.isfinite(new_value):
            # Every comparison against NaN is false, so an unusable number made
            # the gate pass rather than fail. A gate that cannot read its own
            # input has to say so.
            failures.append(f"{strategy}: {metric} read {new_value!r}, which is not a number")
            continue
        drop = baseline_value - new_value
        status = "REGRESSION" if drop > threshold else "ok"
        print(
            f"{strategy}: baseline={baseline_value:.4f} new={new_value:.4f} "
            f"drop={drop:.4f} [{status}]"
        )
        if drop > threshold:
            failures.append(
                f"{strategy}: {metric} dropped {drop:.4f}, over the {threshold} threshold"
            )

    if failures:
        sys.exit("Retrieval regression gate failed:\n" + "\n".join(failures))
    print("Retrieval regression gate passed.")


if __name__ == "__main__":
    main()
