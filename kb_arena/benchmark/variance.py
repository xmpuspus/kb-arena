"""Spread across repeats of one experiment.

`kb-arena benchmark --runs N` writes N result files that share a compatibility
key. One of them is a point. Together they say how much the number moves when
nothing about the experiment changed, which is the difference between a real
lift and noise.

A run is comparable to another only inside one compatibility key, so this
module never mixes keys. It reports the count, the mean, the sample standard
deviation, the range, and a half-width, and it says plainly when one run cannot
carry a spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from kb_arena.benchmark.manifest import LEGACY_KEY, UNRECORDED_BUILD, build_identity

# One run gives a point and no spread. Two give a range a reader can misread as
# a bound. The sample standard deviation needs two, so it is reported from two
# and the text says how thin it is below this.
MIN_RUNS_FOR_SPREAD = 2
THIN_EVIDENCE_RUNS = 3

# What a run written before this slice reports for its seed.
UNRECORDED_SEED = "unrecorded"
UNRECORDED_VERSION = UNRECORDED_BUILD

# Files a run writes beside its results. None of them is one.
_NON_RESULT_NAMES = frozenset(
    {
        "summary.json",
        "report.json",
        "optimize.json",
        "run.json",
        "arena_state.json",
    }
)


class RunsUnreadableError(RuntimeError):
    """A stored result exists and cannot be read, so the sample is incomplete."""


@dataclass(frozen=True)
class Spread:
    """What N repeats of one experiment say about a single metric."""

    runs: int
    mean: float
    sd: float | None
    minimum: float
    maximum: float

    @property
    def half_width(self) -> float | None:
        """Half the observed range, the plainest statement of the spread."""
        if self.runs < MIN_RUNS_FOR_SPREAD:
            return None
        return (self.maximum - self.minimum) / 2.0

    @property
    def thin(self) -> bool:
        """True when too few runs stand behind the number to read it as a bound."""
        return self.runs < THIN_EVIDENCE_RUNS

    def as_dict(self) -> dict:
        return {
            "runs": self.runs,
            "mean": round(self.mean, 6),
            "sd": None if self.sd is None else round(self.sd, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "half_width": None if self.half_width is None else round(self.half_width, 6),
            "thin_evidence": self.thin,
        }


def summarize(values: list[float]) -> Spread | None:
    """The spread of one metric over repeats, or None when nothing was measured."""
    # A non-finite value is not a measurement. The caller counts what it
    # passed against `Spread.runs`, so a dropped value shows up as missing.
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    sd = None
    if len(clean) >= MIN_RUNS_FOR_SPREAD:
        # Sample standard deviation. The runs are a sample of the runs that
        # could have happened, not the whole population of them.
        variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
        sd = math.sqrt(variance)
    return Spread(runs=len(clean), mean=mean, sd=sd, minimum=min(clean), maximum=max(clean))


def group_by_key(runs: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Runs grouped by (corpus, strategy, compatibility key).

    Two runs under different keys measured different things, so a spread across
    them would be a number about the difference and not about the noise.
    """
    from kb_arena.benchmark.manifest import compatibility_key

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for run in runs:
        base = compatibility_key(run)
        # A lab run that did not finish never joins one that did. A legacy run
        # already averages with nothing, so splitting it further would only
        # print a longer key that means no more than `legacy` does.
        key = base if base == LEGACY_KEY else base + _lab_sample_suffix(run)
        grouped.setdefault(
            (str(run.get("corpus", "")), str(run.get("strategy", "")), key), []
        ).append(run)
    return grouped


def _metric(run: dict, name: str) -> float | None:
    def _number(value) -> float | None:
        # A JSON true is a Python bool and float(True) is 1.0, so a corrupt
        # file would otherwise contribute a perfect score.
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        try:
            # A JSON integer has no size limit, and float() raises on one too
            # large to represent. An unreadable metric is not a crash.
            return float(value)
        except (OverflowError, ValueError):
            return None

    by_tier = run.get(name)
    if (direct := _number(by_tier)) is not None:
        return direct
    if isinstance(by_tier, dict) and by_tier:
        numeric = [n for v in by_tier.values() if (n := _number(v)) is not None]
        # Averaging the readable tiers of a run whose other tiers are corrupt
        # turns malformed evidence into an ordinary-looking result, and the
        # run then counts as carrying the metric. It does not.
        if len(numeric) == len(by_tier):
            return sum(numeric) / len(numeric)
    return None


def spread_report(runs: list[dict], metrics: tuple[str, ...] = ("accuracy_by_tier",)) -> list[dict]:
    """One row per (corpus, strategy, key), with the spread of each metric.

    A run with no manifest keys as `legacy`, and every such run keys the same
    way whatever it measured. A spread over that group would describe the
    differences between experiments, so those rows carry `comparable: False`
    and the caller must not read them as noise.
    """
    rows: list[dict] = []
    for (corpus, strategy, key), group in sorted(group_by_key(runs).items()):
        versions = sorted({_code_version(run) for run in group})
        row: dict = {
            "corpus": corpus,
            "strategy": strategy,
            "compatibility_key": key,
            # A legacy group holds whatever had no manifest. A group whose runs
            # came from different code is not a repeat of one experiment either.
            "comparable": (
                key != LEGACY_KEY
                and len(versions) == 1
                # "unrecorded" is not a build. Two runs that both fail to name
                # one are not known to share it, so they are not repeats.
                and versions != [UNRECORDED_VERSION]
            ),
            "code_versions": versions,
            "runs": len(group),
            # A run written before seeds existed reports None. Sorting that
            # beside an int raises, and every result already on disk is one of
            # them, so the first mixed group would crash the command.
            "seeds": _seed_labels(group),
            "metrics": {},
        }
        for name in metrics:
            values = [v for run in group if (v := _metric(run, name)) is not None]
            spread = summarize(values)
            if not spread:
                continue
            # A run in this group whose metric was missing or the wrong type is
            # not in the spread. Saying how many keeps the count honest.
            # Both a run that carries no metric and a run whose metric is not
            # finite are absent from the spread, so both are counted here.
            missing = len(group) - spread.runs
            if row["comparable"]:
                row["metrics"][name] = {**spread.as_dict(), "runs_without_this_metric": missing}
            else:
                # The runs measured different things, so a mean and a standard
                # deviation over them describe that difference. Report the
                # values instead, and let the reader see there is no spread.
                row["metrics"][name] = {
                    "runs": spread.runs,
                    "values": [round(v, 6) for v in sorted(values)],
                    "comparable": False,
                    "runs_without_this_metric": missing,
                }
        rows.append(row)
    return rows


def load_runs(corpus: str | None = None) -> list[dict]:
    """Every stored result, as the raw record, so the manifest survives.

    `--runs N` writes one directory per repeat, and the newest run also lands
    at the top level. The top-level copy duplicates a run id already read from
    its own directory, so it is counted once.
    """
    import json

    from kb_arena.settings import settings

    root = Path(settings.results_path)
    if not root.is_dir():
        return []
    runs: list[dict] = []
    unreadable: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    directories = sorted(root.glob("run_*")) + [root]
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except OSError as exc:
                # A RESULT that exists and cannot be read is lost evidence.
                # Silently shrinking the sample would report a spread over what
                # survived. An unrelated file is not evidence, so it is skipped
                # the same way a malformed one is.
                if _looks_like_a_result(path) and _is_for_corpus(path, corpus):
                    unreadable.append(f"{path}: {exc}")
                continue
            except json.JSONDecodeError as exc:
                # Only a file shaped like a result. `results/` also holds
                # summaries, reports and scratch, and one bad byte in any of
                # those must not block a report about the runs.
                if _looks_like_a_result(path) and _is_for_corpus(path, corpus):
                    unreadable.append(f"{path}: malformed JSON, {exc}")
                continue
            if not isinstance(data, dict):
                continue
            if "corpora" in data and "strategy" not in data:
                # A Retriever Lab run holds every strategy in one file, keyed by
                # corpus. Flattening it here is what lets `variance` read a lab
                # run at all: before this, the loader skipped the file and the
                # command answered "no run carries the metric".
                runs.extend(_flatten_lab_run(data, corpus))
                continue
            if "strategy" not in data or "corpus" not in data:
                continue
            if corpus and data.get("corpus") != corpus:
                continue
            # A file may carry `run_id: ""`. Falling back on the key's absence
            # alone would give two different runs one identity and drop one of
            # them, which silently shrinks the sample.
            run_id = data.get("run_id")
            run_id = str(run_id) if isinstance(run_id, str) and run_id.strip() else str(path)
            identity = (run_id, str(data.get("corpus")), str(data.get("strategy")))
            if identity in seen:
                continue
            seen.add(identity)
            runs.append(data)
    if unreadable:
        raise RunsUnreadableError(
            f"{len(unreadable)} result file(s) could not be read, so a spread over "
            f"the rest would hide lost evidence: " + "; ".join(unreadable[:5])
        )
    return runs


# A result file is `<corpus>_<strategy>.json`, and the strategies are known.
def _looks_like_a_result(path: Path) -> bool:
    from kb_arena.strategies.catalog import STRATEGY_CATALOG

    if any(path.name.endswith(f"_{spec.name}.json") for spec in STRATEGY_CATALOG):
        return True
    # A plugin strategy writes `<corpus>_<name>.json` under a run directory
    # too, and its name is not in the built-in catalog. Requiring the shape as
    # well as the location keeps a stray note.json in a run directory from
    # speaking for the evidence.
    if not path.parent.name.startswith("run_"):
        return False
    stem = path.stem
    if "_" not in stem:
        return False
    corpus, _, strategy = stem.partition("_")
    return bool(corpus) and bool(strategy) and path.name not in _NON_RESULT_NAMES


def _lab_sample_suffix(run: dict) -> str:
    """What makes one lab run a different sample from another lab run.

    Two things do. A run the lab halted scored fewer questions than one that
    finished. So did a run that finished while some queries errored: the lab
    drops a failed query from the metrics and still reports `complete`, so the
    status alone never separates them.

    The partial half copies the manifest's own shape, the count and then a
    digest of the question ids. The count alone would merge two runs of forty
    questions that scored forty different ones.
    """
    if run.get("source") != "retriever_lab":
        return ""
    parts = []
    status = run.get("lab_status")
    if status not in (None, "complete"):
        parts.append(str(status))
    scored = run.get("questions")
    if isinstance(scored, int) and not isinstance(scored, bool):
        expected = _expected_question_count(run)
        # A run that scored every question its manifest names is whole, and it
        # keys like one. Only a short run carries the extra suffix, which is
        # what `compatibility_key` does for a benchmark result.
        if expected is None or scored < expected:
            parts.append(f"partial-{scored}-{run.get('scored_fingerprint', 'none')}")
    return ("-" + "-".join(parts)) if parts else ""


def _expected_question_count(run: dict) -> int | None:
    """How many questions the run's manifest names, or None when it names none."""
    manifest = run.get("manifest")
    count = manifest.get("question_count") if isinstance(manifest, dict) else None
    return count if isinstance(count, int) and not isinstance(count, bool) else None


def _flatten_lab_run(data: dict, corpus: str | None) -> list[dict]:
    """One Retriever Lab file as one record per corpus and strategy.

    The lab reports every strategy in a single run, and the rest of this module
    compares one strategy at a time. The manifest travels with each record, so
    the compatibility key and the build identity still decide the grouping.
    """
    manifests = data.get("manifests")
    manifests = manifests if isinstance(manifests, dict) else {}
    run_id = str(data.get("run_id", ""))
    # A lab run records whether it finished. One that stopped early scored fewer
    # questions, so averaging it with a complete run reports a smaller sample as
    # the same measurement. The status rides into the grouping key, and so do
    # the questions each strategy actually scored. A run can report `complete`
    # after some queries errored, and then the status alone says nothing.
    status = str(data.get("status", "unknown"))
    scored_ids = _scored_question_ids(data)
    flat: list[dict] = []
    for corpus_name, strategies in (data.get("corpora") or {}).items():
        if corpus and corpus_name != corpus:
            continue
        if not isinstance(strategies, dict):
            continue
        for strategy, metrics in strategies.items():
            if not isinstance(metrics, dict):
                continue
            flat.append(
                {
                    **metrics,
                    "corpus": corpus_name,
                    "strategy": strategy,
                    "run_id": run_id,
                    "manifest": manifests.get(corpus_name, {}),
                    "source": "retriever_lab",
                    "lab_status": status,
                    "scored_fingerprint": scored_ids.get((corpus_name, strategy), "none"),
                }
            )
    return flat


def _scored_question_ids(data: dict) -> dict[tuple[str, str], str]:
    """A digest of the questions each corpus and strategy scored in one lab run.

    The lab writes one row per question it tried, so the rows say which
    questions a strategy scored. Two short runs of the same size that covered
    different questions are different samples, and only the ids show that.
    """
    from kb_arena.benchmark.manifest import _digest

    ids: dict[tuple[str, str], list[str]] = {}
    for row in data.get("questions") or []:
        if not isinstance(row, dict):
            continue
        # A question the strategy failed on still writes a row, and the lab
        # leaves it out of the metrics. Counting it here would give two runs
        # that failed on different questions the same digest.
        if row.get("execution_error") is not None:
            continue
        pair = (str(row.get("corpus", "")), str(row.get("strategy", "")))
        ids.setdefault(pair, []).append(str(row.get("question_id", "")))
    return {pair: _digest(sorted(v))[:8] for pair, v in ids.items()}


def _is_for_corpus(path: Path, corpus: str | None) -> bool:
    """Whether an unreadable result could belong to the corpus being reported.

    A file that cannot be read cannot name its corpus, so the name is all there
    is. A report about one corpus must not stop because another corpus has a
    broken file.
    """
    if not corpus:
        return True
    if path.name == "retriever_lab.json":
        # A lab file holds every corpus in one document, so a corpus-filtered
        # report cannot rule it out by name. An unreadable one may hold the
        # corpus being reported, and dropping it would shrink the sample in
        # silence.
        return True
    return path.stem.startswith(f"{corpus}_")


def _code_version(run: dict) -> str:
    """The build a run came from. One definition, shared with the leaderboard.

    Two copies of this rule would drift, and the two surfaces would then
    disagree about whether two runs are repeats of each other.
    """
    return build_identity(run)


def _seed_labels(group: list[dict]) -> list[str]:
    """The seeds a group used, with an unseeded run named and not hidden.

    Dropping the unseeded runs would print `0` for a group where only one run
    recorded a seed, which claims a provenance the files do not carry.
    """
    seeds = {seed_of(run) for run in group}
    labels = sorted(str(s) for s in seeds if s is not None)
    if None in seeds:
        labels.append(UNRECORDED_SEED)
    return labels


def seed_of(run: dict) -> int | None:
    """The seed a run recorded, or None for a run written before seeds existed."""
    manifest = run.get("manifest")
    seed = manifest.get("seed") if isinstance(manifest, dict) else None
    value = seed.get("value") if isinstance(seed, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
