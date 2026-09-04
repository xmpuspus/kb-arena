"""Paired comparison of two strategies on the same questions.

Two means side by side hide whether one strategy beats the other on the
same questions or only on average. This pairs every question, reports the
per-question deltas, a percentile bootstrap CI on the mean delta, a
Wilcoxon p-value, a paired effect size, and the win, tie, and loss counts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean, pstdev

from kb_arena.benchmark.optimizer import _bootstrap_ci, _wilcoxon

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LOWER_IS_BETTER = {"latency_ms", "cost_usd", "retrieval_latency_ms", "generation_latency_ms"}
RECORD_METRICS = {"latency_ms", "cost_usd", "retrieval_latency_ms", "generation_latency_ms"}
LAB_METRICS = ("recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k")


def paired_compare(
    a: dict[str, float],
    b: dict[str, float],
    *,
    metric: str,
    label_a: str,
    label_b: str,
    n_resamples: int = 2000,
) -> dict:
    """Compare b against a on the questions both scored. Delta is b minus a."""
    shared = sorted(set(a) & set(b))
    lower = metric in LOWER_IS_BETTER
    deltas = [b[q] - a[q] for q in shared]
    better = [(d < 0) if lower else (d > 0) for d in deltas]
    worse = [(d > 0) if lower else (d < 0) for d in deltas]
    wins, losses = sum(better), sum(worse)
    mean_delta = fmean(deltas) if deltas else 0.0
    low, high = _bootstrap_ci(deltas, n_resamples=n_resamples) if deltas else (0.0, 0.0)
    spread = pstdev(deltas) if len(deltas) > 1 else 0.0
    p_value = _wilcoxon([a[q] for q in shared], [b[q] for q in shared])
    return {
        "metric": metric,
        "lower_is_better": lower,
        "a": label_a,
        "b": label_b,
        "n_paired": len(shared),
        "unpaired_a": len(set(a) - set(b)),
        "unpaired_b": len(set(b) - set(a)),
        "mean_a": fmean(a[q] for q in shared) if shared else 0.0,
        "mean_b": fmean(b[q] for q in shared) if shared else 0.0,
        "mean_delta": mean_delta,
        "delta_ci_95": [low, high],
        "ci_excludes_zero": bool(deltas) and (low > 0.0 or high < 0.0),
        # Paired Cohen's d: the mean delta over the spread of the deltas.
        "effect_size_d": mean_delta / spread if spread > 0 else 0.0,
        "wilcoxon_p": p_value,
        "significant": p_value is not None and p_value < 0.05 and mean_delta != 0.0,
        "wins": wins,
        "ties": len(deltas) - wins - losses,
        "losses": losses,
        "per_question": [
            {"question_id": q, "a": a[q], "b": b[q], "delta": b[q] - a[q]} for q in shared
        ],
        "note": (
            "Delta is b minus a on the paired questions only. Wins count questions where b "
            "does better. The CI is a percentile bootstrap on the paired deltas, seed 0."
        ),
    }


def resolve_result_path(results_dir: Path, corpus: str, strategy: str, run_id: str | None) -> Path:
    """The result file for one strategy. Ids are checked so no path leaves results_dir."""
    for name, value in (("corpus", corpus), ("strategy", strategy), ("run_id", run_id or "x")):
        if not SAFE_ID.match(value):
            raise ValueError(f"invalid {name}")
    if run_id:
        return Path(results_dir) / f"run_{run_id}" / f"{corpus}_{strategy}.json"
    return Path(results_dir) / f"{corpus}_{strategy}.json"


def benchmark_scores(path: Path, metric: str) -> tuple[dict[str, float], dict]:
    """Per-question values of one metric from a benchmark result file, plus its identity."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError(f"{path} is not a benchmark result file")
    scores: dict[str, float] = {}
    errors = 0
    for record in data["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("question_id"), str):
            continue
        # An error record holds a zero score and a zero latency that no
        # strategy earned. It leaves the pairing and gets counted instead,
        # so an outage on one side never reads as a win or a loss.
        if record.get("is_error") is True:
            errors += 1
            continue
        source = record if metric in RECORD_METRICS else record.get("score") or {}
        value = source.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        scores[record["question_id"]] = float(value)
    manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
    meta = {
        "path": str(path),
        "corpus": data.get("corpus"),
        "strategy": data.get("strategy"),
        "run_id": data.get("run_id"),
        "compatibility_key": manifest.get("compatibility_key"),
        "error_records": errors,
    }
    return scores, meta


def compare_result_files(path_a: Path, path_b: Path, metric: str = "accuracy") -> dict:
    """Two benchmark result files, paired by question id."""
    scores_a, meta_a = benchmark_scores(path_a, metric)
    scores_b, meta_b = benchmark_scores(path_b, metric)
    if not scores_a or not scores_b:
        raise ValueError(f"no per-question values for metric {metric!r}")
    result = paired_compare(
        scores_a,
        scores_b,
        metric=metric,
        label_a=str(meta_a["strategy"]),
        label_b=str(meta_b["strategy"]),
    )
    reasons: list[str] = []
    if meta_a["corpus"] != meta_b["corpus"]:
        reasons.append("different corpus")
    key_a, key_b = meta_a["compatibility_key"], meta_b["compatibility_key"]
    if key_a and key_b and key_a != key_b:
        reasons.append("different compatibility key: question set, judge, or top_k differ")
    elif not key_a or not key_b:
        missing = "neither file" if not key_a and not key_b else ("a" if not key_a else "b")
        reasons.append(
            f"{missing} carries a manifest, so the question set, judge, and top_k are unchecked"
        )
    for side, meta in (("a", meta_a), ("b", meta_b)):
        if meta["error_records"]:
            reasons.append(
                f"{meta['error_records']} error records in {side} left out of the pairing"
            )
    if result["unpaired_a"] or result["unpaired_b"]:
        reasons.append(
            f"{result['unpaired_a']} questions only in a, {result['unpaired_b']} only in b, "
            "compared on the shared ones"
        )
    result["meta"] = {"a": meta_a, "b": meta_b, "comparable": not reasons, "reasons": reasons}
    return result


def lab_scores(path: Path, strategy: str, metric: str) -> dict[str, float]:
    """Per-question values of one IR metric for one strategy in a retriever-lab file."""
    data = json.loads(Path(path).read_text())
    rows = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a retriever-lab file")
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("strategy") != strategy:
            continue
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            continue
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        scores[question_id] = float(value)
    return scores


def compare_lab(path: Path, a: str, b: str, metric: str = "ndcg_at_k") -> dict:
    """Two strategies inside one retriever-lab run, paired by question id."""
    scores_a, scores_b = lab_scores(path, a, metric), lab_scores(path, b, metric)
    if not scores_a or not scores_b:
        raise ValueError(f"no rows for {a!r} and {b!r} with metric {metric!r} in {path}")
    result = paired_compare(scores_a, scores_b, metric=metric, label_a=a, label_b=b)
    result["meta"] = {"path": str(path), "comparable": True, "reasons": []}
    return result
