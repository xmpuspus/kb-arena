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

import json
import math
from dataclasses import dataclass
from pathlib import Path

from kb_arena.benchmark.manifest import LEGACY_KEY, UNRECORDED_BUILD, build_identity
from kb_arena.benchmark.result_schema import LabRun, Manifest, read_lab_run

# One run gives a point and no spread. Two give a range a reader can misread as
# a bound. The sample standard deviation needs two, so it is reported from two
# and the text says how thin it is below this.
MIN_RUNS_FOR_SPREAD = 2
THIN_EVIDENCE_RUNS = 3

# What a run written before this slice reports for its seed.
UNRECORDED_SEED = "unrecorded"
UNRECORDED_VERSION = UNRECORDED_BUILD

# Files a run writes beside its results. None of them is one. `kb-arena
# evidence` needed the same answer and a fixed name list could not give it, so
# it asks `evidence.is_bundle_result` what a result IS instead. This list stays
# here because the shape check below is already loose enough to need it.
_NON_RESULT_NAMES = frozenset(
    {
        "summary.json",
        "report.json",
        "optimize.json",
        "run.json",
        "arena_state.json",
        "evidence.json",
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
                # A record that cannot prove which questions it scored makes the
                # whole group unreadable as a spread, whatever the key says.
                and not any(run.get("sample_unproven") for run in group)
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


def load_runs(corpus: str | None = None, *, failures: list[str] | None = None) -> list[dict]:
    """Every stored result, as the raw record, so the manifest survives.

    `--runs N` writes one directory per repeat, and the newest run also lands
    at the top level. The top-level copy duplicates a run id already read from
    its own directory, so it is counted once.

    Pass `failures` to collect the lab runs that recorded a failure and hold no
    measurement. They belong to no corpus and no strategy, so they cannot take a
    row, and a caller that never mentions them reports a spread over the runs
    that happened to work.
    """
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
            # The lab writes one file, and its name is the only reliable mark.
            # `summary.json` from `generate_report` also carries a `corpora` key
            # and no `strategy`, so the shape alone counted it as a lab run and
            # then reported it as one that failed.
            if path.name == "retriever_lab.json":
                # A Retriever Lab run holds every strategy in one file, keyed by
                # corpus. Flattening it here is what lets `variance` read a lab
                # run at all: before this, the loader skipped the file and the
                # command answered "no run carries the metric".
                lab = read_lab_run(data)
                flat = _lab_records(lab, corpus)
                if failures is not None:
                    failures.extend(_lab_failures(path, lab, corpus))
                for record in flat:
                    # A lab record takes the same identity check as a benchmark
                    # result. Two copies of one lab file are one measurement, and
                    # counting them twice would report a spread of zero over a
                    # sample of one.
                    identity = (
                        record["run_id"] or str(path),
                        str(record.get("corpus")),
                        str(record.get("strategy")),
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    runs.append(record)
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
def _lab_failures(path: Path, lab: LabRun, corpus: str | None) -> list[str]:
    """Every loss a lab file records that a report about `corpus` must name.

    Three shapes lose evidence. A corpus whose summaries are not a record, a
    strategy whose summary is not a record, and a run that stopped before it
    finished. The third is the one four review rounds kept reopening: the
    earlier guard named an incomplete run only when the file yielded no record
    at all, so a run that died inside its second corpus reported nothing.

    An incomplete run cannot rule itself out by corpus either. The writer adds a
    corpus summary only after it finishes that corpus, so the map it wrote is
    not the list of corpora it meant to cover.
    """
    named = [
        f"{path}: corpus {name} has no readable summary"
        for name in lab.unreadable_corpora
        if not corpus or name == corpus
    ]
    named += [
        f"{path}: {corpus_name}/{strategy} has no readable summary"
        for corpus_name, strategy in lab.unreadable_strategies
        if not corpus or corpus_name == corpus
    ]
    # Only when nothing named the loss already. A file whose one strategy is
    # unreadable would otherwise be counted twice.
    if named:
        return named
    if not lab.complete:
        return [f"{path}: {lab.failure_reason}"]
    if lab.for_corpus(corpus):
        return []
    # A complete run that named other corpora and not this one belongs to
    # another report. A report about one corpus must not name it.
    if corpus and lab.corpora_named and corpus not in lab.corpora_named:
        return []
    return [f"{path}: {lab.failure_reason}"]


def _flatten_lab_run(data: dict, corpus: str | None) -> list[dict]:
    """One raw Retriever Lab document as one record per corpus and strategy.

    The reading and the flattening are two steps now. This keeps them behind one
    name, because the behaviour it guarantees is what the tests below it hold.
    """
    return _lab_records(read_lab_run(data), corpus)


def _lab_records(lab: LabRun, corpus: str | None) -> list[dict]:
    """One Retriever Lab file as one record per corpus and strategy.

    The lab reports every strategy in a single run, and the rest of this module
    compares one strategy at a time. The manifest travels with each record, so
    the compatibility key and the build identity still decide the grouping.
    """
    flat: list[dict] = []
    for result in lab.for_corpus(corpus):
        # Every query failed, and the lab writes each mean as 0.0 anyway.
        # Carrying those through would report an outage as a strategy that
        # retrieves nothing relevant. The record stays, so the group still
        # counts it under `runs_without_this_metric`, and it carries no number
        # for anybody to read.
        #
        # `reported` is the one definition of the grouping count, and the rows
        # outrank the summary inside it. A count of 2 over one scored row is a
        # file contradicting itself, and taking the 2 would call the run whole.
        measured = result.summary if result.measured else {}
        measured = {**measured, "questions": result.reported}
        manifest = lab.manifests.get(result.corpus)
        record = {
            **measured,
            "corpus": result.corpus,
            "strategy": result.strategy,
            "run_id": lab.run_id,
            "manifest": manifest.raw if manifest else {},
            "source": "retriever_lab",
            "lab_status": lab.status,
            "scored_fingerprint": result.row_digest,
        }
        # Nothing here proves which questions the run scored, so a mean over
        # this record and another would be a number about two unknowns. The key
        # still separates what it can, and the group says it cannot be read as
        # a spread.
        if not result.proven:
            record["sample_unproven"] = True
        flat.append(record)
    return flat


def _looks_like_a_result(path: Path) -> bool:
    from kb_arena.strategies.catalog import STRATEGY_CATALOG

    if any(path.name.endswith(f"_{spec.name}.json") for spec in STRATEGY_CATALOG):
        return True
    if path.name == "retriever_lab.json":
        # A lab run is evidence wherever it sits, and the check below would
        # refuse it outside a `run_` directory. A corrupt copy at the results
        # root then disappeared and the report never said the sample shrank.
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
    # The lab stubs the LLM client for the whole run, so a strategy that calls
    # the model (qiss decomposition, hybrid intent routing) retrieves
    # differently than it does under `kb-arena benchmark`. The lab also always
    # records `reference_free: True`, which a `--reference-free` benchmark
    # records too, so the manifest core alone cannot tell the two apart. Without
    # this the two averaged, and the gap between them read as noise.
    parts = ["lab"]
    status = run.get("lab_status")
    if status not in (None, "complete"):
        parts.append(str(status))
    digest = run.get("scored_fingerprint", "none")
    scored = run.get("questions")
    if not isinstance(scored, int) or isinstance(scored, bool):
        # A count that is missing, null, or the wrong type says nothing about
        # the sample. Reading it as whole would average an unknown number of
        # questions with a full run, which is the defect this suffix exists to
        # stop. An unknown sample groups only with itself.
        parts.append(f"partial-unknown-{digest}")
    else:
        expected = _expected_question_count(run)
        # A run that scored every question its manifest names is whole, and it
        # keys like one. Only a short run carries the extra suffix, which is
        # the shape `compatibility_key` gives a short benchmark result.
        #
        # The trigger differs from `manifest._scored_count` on one case, and
        # the difference is deliberate. That function reads a manifest with no
        # `question_count` as a whole run. This one refuses to, because a
        # missing count is not proof that the run scored everything. The refusal
        # costs no false split: two runs over the same questions carry the same
        # digest, so they still group. Reading it the other way would average an
        # unproven sample into a full run, which is the defect this suffix
        # exists to stop.
        # `!=` and not `<`. A run that scored MORE questions than its manifest
        # names is not the manifest's sample either, and reading it as whole
        # averaged it with a run that was.
        if expected is None or scored != expected:
            parts.append(f"partial-{scored}-{digest}")
    return ("-" + "-".join(parts)) if parts else ""


def _expected_question_count(run: dict) -> int | None:
    """How many questions the run's manifest names, or None when it names none.

    A count of zero is a count, and the test below is `!=`, so a run that scored
    anything against a manifest naming zero still splits out. An earlier version
    refused zero because the test was `<`, which made every run look whole.
    """
    return Manifest.read(run.get("manifest")).question_count


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
    return Manifest.read(run.get("manifest")).seed
