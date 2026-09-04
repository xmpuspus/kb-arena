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

from kb_arena.benchmark.manifest import LEGACY_KEY

# One run gives a point and no spread. Two give a range a reader can misread as
# a bound. The sample standard deviation needs two, so it is reported from two
# and the text says how thin it is below this.
MIN_RUNS_FOR_SPREAD = 2
THIN_EVIDENCE_RUNS = 3

# What a run written before this slice reports for its seed.
UNRECORDED_SEED = "unrecorded"
UNRECORDED_VERSION = "unrecorded"


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
        key = (str(run.get("corpus", "")), str(run.get("strategy", "")), compatibility_key(run))
        grouped.setdefault(key, []).append(run)
    return grouped


def _metric(run: dict, name: str) -> float | None:
    by_tier = run.get(name)
    if isinstance(by_tier, int | float):
        return float(by_tier)
    if isinstance(by_tier, dict) and by_tier:
        numeric = [float(v) for v in by_tier.values() if isinstance(v, int | float)]
        if numeric:
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
            "comparable": key != LEGACY_KEY and len(versions) == 1,
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
            if row["comparable"]:
                row["metrics"][name] = spread.as_dict()
            else:
                # The runs measured different things, so a mean and a standard
                # deviation over them describe that difference. Report the
                # values instead, and let the reader see there is no spread.
                row["metrics"][name] = {
                    "runs": spread.runs,
                    "values": [round(v, 6) for v in sorted(values)],
                    "comparable": False,
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
                # A run that exists and cannot be read is lost evidence. Silently
                # shrinking the sample would report a spread over what survived.
                unreadable.append(f"{path}: {exc}")
                continue
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or "strategy" not in data or "corpus" not in data:
                continue
            if corpus and data.get("corpus") != corpus:
                continue
            identity = (
                str(data.get("run_id", path.stem)),
                str(data.get("corpus")),
                str(data.get("strategy")),
            )
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


def _code_version(run: dict) -> str:
    """The build a run came from: its version, and the commit inside it.

    Several commits share one unreleased version during development, so the
    version alone would call a code change run-to-run noise. The commit decides.
    """
    manifest = run.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    version = manifest.get("code_version")
    sha = manifest.get("git_sha")
    if not version and not sha:
        return UNRECORDED_VERSION
    label = str(version) if version else UNRECORDED_VERSION
    return f"{label}@{str(sha)[:7]}" if sha else label


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
