"""One validated reading of a result file, so the readers stop guessing its shape.

`variance.py` and `evidence.py` both read the files a run writes, and both used
to judge them through a growing pile of shape checks. Malformed shapes are
unbounded, so every review round found a gap in the round before. Three classes
came back over and over: a `str(None)` collision, a per-entry drop that let a
readable sibling speak for a broken one, and a summary count with no witness.

This module reads a file once and answers with a record. A field that a file
cannot support reads as absent or as the unreadable marker, never as a default
that looks like a fact. The callers then ask the record a question instead of
asking the JSON a shape.

Two things stay outside on purpose. Deciding WHICH files are results is a
name rule (`is_bundle_result`, `_looks_like_a_result`), because a rule that
reads the contents to decide hides a corrupt file. And parsing errors stay with
the caller, because a caller reports a file it could not read.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# A manifest that carries a fingerprint nobody can read. It is not a question
# set, and it is not absence either, so it needs a name of its own.
UNREADABLE_QUESTION_SET = "<unreadable>"


def is_count(value) -> bool:
    """A usable count is a whole number that is not negative, and `True` is not one."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class Manifest(BaseModel):
    """One manifest entry, as a result file carries it.

    An entry that is not a record still becomes a `Manifest`, with `readable`
    false. Dropping it was the defect three review rounds found at three depths:
    a bad fingerprint, then a bad entry, then a bad key. Each time a valid
    manifest beside it spoke for the whole file.
    """

    model_config = ConfigDict(frozen=True)

    readable: bool = True
    question_count: int | None = None
    question_set_fingerprint: str = ""
    question_split: str = ""
    seed: int | None = None
    raw: dict = Field(default_factory=dict)

    @property
    def names_a_question_set(self) -> str:
        """The question set this manifest names, or the unreadable marker.

        A fingerprint that is null, blank, or not a string names no set. An
        earlier reader asked only whether it was truthy, so each of those values
        skipped the comparison and passed as if it matched.
        """
        if not self.readable or not self.question_set_fingerprint:
            return UNREADABLE_QUESTION_SET
        return self.question_set_fingerprint

    @classmethod
    def read(cls, value) -> Manifest:
        if not isinstance(value, dict):
            return cls(readable=False)
        count = value.get("question_count")
        fingerprint = value.get("question_set_fingerprint")
        split = value.get("question_split")
        return cls(
            readable=True,
            question_count=count if is_count(count) else None,
            question_set_fingerprint=(
                fingerprint.strip() if isinstance(fingerprint, str) and fingerprint.strip() else ""
            ),
            question_split=split.strip() if isinstance(split, str) and split.strip() else "",
            seed=_seed_in(value),
            raw=value,
        )


def _seed_in(manifest: dict) -> int | None:
    """The seed a manifest recorded, or None for one written before seeds existed."""
    seed = manifest.get("seed")
    value = seed.get("value") if isinstance(seed, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class StrategyResult(BaseModel):
    """What one strategy measured over one corpus, in one run.

    Two numbers used to live here, and comparing them was left to each caller.
    The summary claims a count, and the per-question rows are the witness for
    it. This record holds both and answers the two questions callers actually
    ask: what number can be cited, and what number groups this run with another.
    """

    model_config = ConfigDict(frozen=True)

    corpus: str
    strategy: str
    claimed: int | None = None
    rows_present: bool = False
    row_count: int = 0
    row_digest: str = "none"
    rows_unidentified: bool = False
    summary: dict = Field(default_factory=dict)

    @property
    def scored(self) -> int | None:
        """How many questions this strategy scored, or None when nothing proves it.

        The rows are the witness. A summary count with no rows behind it is a
        claim about a measurement, and a citable result cannot rest on a claim
        the file does not support. A run whose every query failed writes rows
        that all carry an error, so it answers zero rather than nothing.
        """
        if not self.rows_present:
            return None
        return self.row_count

    @property
    def reported(self) -> int | None:
        """The count this file reports, for grouping runs that measured the same set.

        The rows outrank the summary when both exist and disagree. A count of 2
        over one scored row is a file contradicting itself, and taking the 2
        would call the run whole. When the file carries no rows at all, the
        claim is all there is, and `proven` says it has no witness.
        """
        if self.rows_present:
            return self.row_count
        return self.claimed

    @property
    def contradicted(self) -> bool:
        """Whether the summary count and the rows behind it disagree."""
        return self.rows_present and self.claimed is not None and self.claimed != self.row_count

    @property
    def proven(self) -> bool:
        """Whether the count has a witness, and the witness agrees with the claim.

        A summary that states no count, or states one that is not a whole
        number, makes no claim a witness can agree with. So it is unproven for
        the same reason a missing witness is.
        """
        return (
            self.rows_present
            and not self.rows_unidentified
            and self.claimed is not None
            and self.claimed == self.row_count
        )

    @property
    def measured(self) -> bool:
        """Whether the run measured anything at all under this strategy.

        Every query can fail, and the lab still writes each mean as `0.0`.
        Carrying those through reports an outage as a strategy that retrieves
        nothing relevant.
        """
        counted = self.claimed is not None and self.claimed > 0
        return counted or self.row_count > 0


class LabRun(BaseModel):
    """One Retriever Lab file, read once.

    The lab reports every strategy of every corpus in a single document. This
    record carries what the file says, what it cannot say, and what it lost.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = ""
    status: str = "unknown"
    manifests: dict[str, Manifest] = Field(default_factory=dict)
    entries: list[Manifest] = Field(default_factory=list)
    results: list[StrategyResult] = Field(default_factory=list)
    unreadable_strategies: list[tuple[str, str]] = Field(default_factory=list)
    unreadable_corpora: list[str] = Field(default_factory=list)
    corpora_named: list[str] = Field(default_factory=list)
    execution_error: dict = Field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Whether the run reached the end. A file written before `status` reads complete."""
        return self.status in ("unknown", "complete")

    @property
    def failure_reason(self) -> str:
        """What the file records about why it holds no measurement."""
        kind = str(self.execution_error.get("type", "")).strip()
        message = str(self.execution_error.get("message", "")).strip()
        if kind and message:
            return f"{kind}: {message}"
        if kind or message:
            return kind or message
        if self.corpora_named:
            return "no strategy summary could be read"
        return "incomplete" if self.status == "unknown" else self.status

    def for_corpus(self, corpus: str | None) -> list[StrategyResult]:
        return [r for r in self.results if not corpus or r.corpus == corpus]

    def scored_for(self, corpus: str) -> int | None:
        """The weakest strategy's proven count, because a bundle covers them all.

        An earlier reader took the maximum, and one strategy that scored every
        question then spoke for a sibling that scored none. A strategy whose
        count nothing proves refuses the whole corpus, for the same reason.
        """
        rows = self.for_corpus(corpus)
        if not rows:
            return None
        counts = [r.scored for r in rows]
        if any(c is None for c in counts):
            return None
        return min(counts)


class BenchmarkRun(BaseModel):
    """One `<corpus>_<strategy>.json` file, the shape `kb-arena benchmark` writes."""

    model_config = ConfigDict(frozen=True)

    corpus: str = ""
    strategy: str = ""
    run_id: str = ""
    manifest: Manifest | None = None
    entries: list[Manifest] = Field(default_factory=list)
    scored: int | None = None
    raw: dict = Field(default_factory=dict)


def manifest_entries(data: dict) -> list[Manifest]:
    """Every manifest entry a result file carries, discarding nothing.

    A present key always contributes one entry, whatever its value. That is the
    point: three rounds found the same defect at three depths, and each fix moved
    the drop one layer up instead of removing it.
    """
    candidates: list = []
    if "manifests" in data:
        manifests = data["manifests"]
        if isinstance(manifests, dict):
            candidates.extend(manifests.values())
        else:
            candidates.append(manifests)
    if "manifest" in data:
        candidates.append(data["manifest"])
    return [Manifest.read(value) for value in candidates]


def _run_id_in(data: dict) -> str:
    """The run id the file records, or an empty string.

    `str(None)` is "None", which is truthy, so two runs that both recorded no id
    shared one identity and a loader dropped one of them. Only a real string
    counts, and the caller falls back to the file path.
    """
    raw = data.get("run_id")
    return raw.strip() if isinstance(raw, str) and raw.strip() else ""


def _rows_by_pair(data: dict) -> dict[tuple[str, str], tuple[int, str, bool]]:
    """Per corpus and strategy: rows scored, a digest of their ids, and whether an id is unreadable.

    The lab writes one row per question it tried, so the rows say which questions
    a strategy scored. Two short runs of the same size that covered different
    questions are different samples, and only the ids show that.
    """
    from kb_arena.benchmark.manifest import question_digest

    ids: dict[tuple[str, str], list[str]] = {}
    unidentified: set[tuple[str, str]] = set()
    for row in data.get("questions") or []:
        if not isinstance(row, dict):
            continue
        pair = (str(row.get("corpus", "")), str(row.get("strategy", "")))
        # A question the strategy failed on still writes a row, and the lab
        # leaves it out of the metrics. The pair is still seen, so a strategy
        # whose every query failed reads as zero scored rather than as a
        # strategy with no witness at all.
        ids.setdefault(pair, [])
        if row.get("execution_error") is not None:
            continue
        qid = row.get("question_id")
        # `str()` maps null to "None", the same text a real id of "None" gives,
        # so two different question sets hashed alike. JSON keeps them apart.
        if not isinstance(qid, str) or not qid.strip():
            unidentified.add(pair)
        ids[pair].append(json.dumps(qid))
    return {
        pair: (len(rows), question_digest(rows), pair in unidentified) for pair, rows in ids.items()
    }


def read_lab_run(data: dict) -> LabRun:
    """One Retriever Lab document as a record. Never raises on a shape."""
    manifests_raw = data.get("manifests")
    by_corpus = {
        str(name): Manifest.read(value)
        for name, value in (manifests_raw.items() if isinstance(manifests_raw, dict) else [])
    }
    rows = _rows_by_pair(data)
    corpora = data.get("corpora")
    corpora = corpora if isinstance(corpora, dict) else {}
    results: list[StrategyResult] = []
    unreadable_strategies: list[tuple[str, str]] = []
    unreadable_corpora: list[str] = []
    for corpus_name, strategies in corpora.items():
        if not isinstance(strategies, dict):
            # A corpus nobody can read is lost the way an unreadable strategy
            # is. Skipping it in silence dropped a whole repeat from a spread.
            unreadable_corpora.append(str(corpus_name))
            continue
        for strategy, summary in strategies.items():
            if not isinstance(summary, dict):
                # A sibling strategy that read fine used to hide this one, since
                # the file still yielded records and the loader only reports a
                # file that yielded none. The strategy is lost on its own.
                unreadable_strategies.append((str(corpus_name), str(strategy)))
                continue
            pair = (str(corpus_name), str(strategy))
            row_count, digest, unidentified = rows.get(pair, (0, "none", False))
            claimed = summary.get("questions")
            results.append(
                StrategyResult(
                    corpus=str(corpus_name),
                    strategy=str(strategy),
                    claimed=claimed if is_count(claimed) else None,
                    rows_present=pair in rows,
                    row_count=row_count,
                    row_digest=digest,
                    rows_unidentified=unidentified,
                    summary=summary,
                )
            )
    # A lab file that carries a bare `manifest` key and no `manifests` map is
    # making one statement about every corpus it names. An earlier reader in
    # `evidence.py` read it that way, so dropping it here would lose a manifest
    # that the file does carry.
    if not isinstance(manifests_raw, dict) and isinstance(data.get("manifest"), dict):
        shared = Manifest.read(data["manifest"])
        by_corpus = {str(name): shared for name in corpora}
    error = data.get("execution_error")
    return LabRun(
        run_id=_run_id_in(data),
        status=str(data.get("status", "unknown")),
        manifests=by_corpus,
        entries=manifest_entries(data),
        results=results,
        unreadable_strategies=unreadable_strategies,
        unreadable_corpora=unreadable_corpora,
        corpora_named=[str(name) for name in corpora],
        execution_error=error if isinstance(error, dict) else {},
    )


def read_benchmark_run(data: dict) -> BenchmarkRun:
    """One single-strategy result document as a record. Never raises on a shape."""
    records = data.get("records")
    scored = None
    if isinstance(records, list):
        # A record marked `is_error` is a question the run failed to score. The
        # row exists, so counting the list counted an outage as a measurement.
        scored = sum(1 for r in records if isinstance(r, dict) and not r.get("is_error"))
    manifest = data.get("manifest")
    return BenchmarkRun(
        corpus=str(data.get("corpus", "")),
        strategy=str(data.get("strategy", "")),
        run_id=_run_id_in(data),
        manifest=Manifest.read(manifest) if "manifest" in data else None,
        entries=manifest_entries(data),
        scored=scored,
        raw=data,
    )


def read_result_document(data: dict) -> LabRun | BenchmarkRun | None:
    """One parsed result document as the record its shape supports.

    A document with a `corpora` map is a lab run. Anything else that is a record
    reads as a single-strategy result. A document that is not a record at all
    answers None, because nothing in it can be read.
    """
    if not isinstance(data, dict):
        return None
    if "corpora" in data:
        return read_lab_run(data)
    return read_benchmark_run(data)


def read_result_file(path: Path) -> LabRun | BenchmarkRun | None:
    """One result file as a record, or None when nothing in it can be read."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return read_result_document(data)
