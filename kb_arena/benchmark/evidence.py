"""A run bundle a reader can check without trusting the person who made it.

A benchmark result on its own says a number. A bundle says which corpus, which
questions, which code, which seed, which command, and what the result may not be
used for. The last one matters most: a run against labels no human checked is a
development signal, and a bundle that does not say so invites a citation it
cannot support.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from kb_arena.benchmark.atomic import atomic_write_text
from kb_arena.benchmark.result_schema import (
    UNREADABLE_QUESTION_SET,
    LabRun,
    read_result_file,
)

BUNDLE_VERSION = 1

# A commit is 40 hex characters, or 64 under sha256. Nothing else reaches git.
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _python_identity() -> dict:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }


def _package_version() -> str:
    from kb_arena import __version__

    return __version__


def _git_sha() -> str | None:
    from kb_arena.benchmark.manifest import git_sha

    return git_sha()


def build_bundle(
    *,
    command: list[str],
    result_paths: list[Path],
    review: dict,
    corpus: str,
    seed: int,
    question_set_fingerprint: str = "",
    review_question_set: str = "",
    review_split: str = "all",
    notes: str = "",
) -> dict:
    """The record that travels with a committed run.

    `command` is what a reader types to repeat this. `review` is the verdict
    from `review_summary`, and it decides whether the bundle calls itself
    citable evidence or a development signal.

    The bundle records two question sets, because they are two different
    facts. `question_set_fingerprint` is the set the RUN measured, read from the
    run's own manifest. `review_question_set` is the set the review verdict ran
    over, hashed from the corpus at bundle time, and `review_split` says which
    split that was. A bundle that records only one of them proves nothing: the
    earlier version copied the run's value and then compared it to the run.
    """
    return {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "command": command,
        "results": sorted(str(p) for p in result_paths),
        "environment": {
            "kb_arena": _package_version(),
            "git_sha": _git_sha(),
            "python": _python_identity(),
            "platform": platform.platform(),
        },
        "seed": seed,
        "question_set_fingerprint": question_set_fingerprint,
        "review_question_set": review_question_set,
        "review_split": review_split,
        "review": review,
        # The claim the bundle makes about itself, stated rather than implied.
        # A run that records no command cannot be repeated, so it cannot be
        # cited either, whatever its review verdict says about the questions.
        "citable": bool(review.get("publishable")) and bool(command),
        "why_not_citable": _why_not(review, command),
        "notes": notes,
    }


def _why_not(review: dict, command: list[str]) -> str:
    """Why this bundle is a development signal rather than citable evidence."""
    reasons = []
    if not review.get("publishable"):
        reasons.append(review.get("note", "the review verdict refuses"))
    if not command:
        reasons.append(
            "the run records no command, so nobody can repeat it. Re-run it with "
            "a build that records one."
        )
    return "; ".join(r for r in reasons if r)


def write_bundle(directory: Path, bundle: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "evidence.json"
    atomic_write_text(path, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return path


def repeat_command(bundle: dict) -> str:
    """The exact command line, so a reader repeats the run rather than guesses."""
    return " ".join(bundle.get("command") or [])


def check_bundle(bundle: dict, root: Path) -> list[str]:
    """What is missing or untrue in a bundle, as a list a caller can print."""
    problems: list[str] = []
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        problems.append(f"bundle_version is {bundle.get('bundle_version')!r}")
    if not bundle.get("command"):
        problems.append("no command, so nobody can repeat the run")
    for name in bundle.get("results") or []:
        if not (root / name).exists():
            problems.append(f"missing result file {name}")
    if not bundle.get("results"):
        problems.append("no result files, so the bundle describes nothing")
    review = bundle.get("review") or {}
    if bundle.get("citable") and not bundle.get("question_set_fingerprint"):
        problems.append(
            "calls itself citable and names no question set, so nobody can tell "
            "whether the review covers the questions the run scored"
        )
    for name in bundle.get("results") or []:
        problem = _question_set_problem(bundle, root / name, name)
        if problem:
            problems.append(problem)
    if bundle.get("citable"):
        live = _live_review(bundle.get("corpus"), bundle.get("review_split"))
        problems.extend(_measurement_problems(bundle, root, live))
        problems.extend(_review_scope_problems(bundle, live))
    if bundle.get("citable") and not review.get("publishable"):
        problems.append("calls itself citable while its own review verdict refuses")
    if not bundle.get("citable") and not bundle.get("why_not_citable"):
        problems.append("is not citable and does not say why")
    env = bundle.get("environment") or {}
    for field in ("kb_arena", "python", "platform"):
        if not env.get(field):
            problems.append(f"environment records no {field}")
    if bundle.get("citable"):
        problem = _commit_problem(env.get("git_sha"), root)
        if problem:
            problems.append(problem)
    return problems


def _commit_problem(sha, root: Path) -> str:
    """Why the commit a citable bundle names is not one a reader can check out.

    This repository squash-merges, so a commit made on a branch never becomes
    part of the default branch. A bundle built on a branch names a SHA a fresh
    clone does not hold, and the run is then unrepeatable from its own record.
    The bundle this repository shipped carried exactly that: `4120ce9`, which
    `git merge-base --is-ancestor` refuses against main.

    A dirty marker fails for the same reason under another name. `<sha>-dirty-<hash>`
    names a working tree nobody else has.

    The check stays silent when git cannot answer. A wheel install has no
    repository at all, and a shallow CI checkout holds too little history to
    decide. Silence there is honest, because the check has no evidence.
    """
    if not isinstance(sha, str) or not sha.strip():
        return "calls itself citable and records no commit, so nobody can get the code back"
    sha = sha.strip()
    # `manifest.git_sha` writes `<sha>-dirty-<hash>`, and `<sha>-dirty` when it
    # cannot hash the diff. Matching only the first form let the second through.
    if sha.endswith("-dirty") or "-dirty-" in sha:
        return (
            f"calls itself citable and was built from an uncommitted tree, {sha}. "
            f"Nobody can get that tree back, so the run cannot be repeated."
        )
    # The value comes out of a JSON file the reader did not write, so it is
    # checked before it reaches git. A value of `--help` was read as an option
    # rather than a commit, and git then answered something this function took
    # for silence.
    if not _COMMIT_SHA.fullmatch(sha):
        return (
            f"calls itself citable and records {sha[:24]!r} where a commit belongs, "
            f"so nobody can get the code back"
        )
    head = _default_branch_head(root)
    if head is None:
        # Not a repository at all, so there is nothing to check the commit
        # against. A wheel install lands here, and silence is honest.
        return ""
    # `rev-parse --verify --quiet` and not `cat-file -e`: cat-file exits 128
    # for an unknown object, the same code it uses for "not a repository", so
    # the two are indistinguishable. Quiet rev-parse exits 1 for an object this
    # repository does not hold and 128 only when it cannot answer at all.
    #
    # `False` is git saying no. `None` is git failing to answer, a timeout or
    # an OS error, and merging the two would reject a valid bundle over one
    # transient failure.
    # A shallow clone holds `origin/main` and almost none of its history, so an
    # object it lacks says nothing about whether the default branch holds it.
    # `actions/checkout` fetches one commit, so this is CI's ordinary state, and
    # the first version turned main red on all three Python jobs.
    #
    # The skip covers only the MISSING object. A commit the clone does hold
    # still gets its ancestry asked, because a shallow clone that holds a commit
    # and cannot reach it from `origin/main` is telling the truth about it.
    if _run_git(
        root, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"
    ) is False and not _is_shallow(root):
        # This IS a repository and it does not hold that object. Reading that
        # as "cannot answer" turned the exact failure this check exists to
        # catch into a pass: `git merge-base` exits 128 on an unknown object,
        # and the earlier version accepted every non-1 code as unknowable.
        return (
            f"calls itself citable and names commit {sha[:12]}, which this repository "
            f"does not hold. Nobody can check out the code this run measured."
        )
    answered = _run_git(root, "merge-base", "--is-ancestor", sha, head)
    if answered is None or answered:
        return ""
    return (
        f"calls itself citable and names commit {sha[:12]}, which is not on {head}. "
        f"The repository squash-merges, so a branch commit never reaches the default "
        f"branch, and a reader cannot check out the code this run measured."
    )


def _is_shallow(root: Path) -> bool:
    """Whether this checkout holds only a slice of the history.

    `actions/checkout` fetches one commit by default, so a shallow clone is the
    ordinary case in CI rather than a corner one.
    """
    import subprocess

    try:
        done = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=root,
        )
    except (OSError, subprocess.SubprocessError):
        # A failure answers "shallow", which suppresses the missing-object
        # refusal. That is deliberate, and it follows the rule the rest of this
        # function follows: no verdict without evidence. Not knowing whether the
        # history is truncated means not knowing whether a missing object means
        # anything, and the other reading refuses a valid bundle every time this
        # one command times out. That failure already turned main red once.
        return True
    return done.returncode != 0 or done.stdout.strip() == "true"


def _default_branch_head(root: Path) -> str | None:
    """The ref a reader would clone, or None when this checkout cannot say."""
    for ref in ("origin/main", "main"):
        if _run_git(root, "rev-parse", "--verify", "--quiet", ref):
            return ref
    return None


def _run_git(root: Path, *args: str) -> bool | None:
    """Whether the git command succeeded, or None when git could not answer.

    A return code of 1 is a plain no from both `merge-base --is-ancestor` and
    `rev-parse --verify`. Anything else means git refused the question: not a
    repository, an unknown object, a shallow clone.
    """
    import subprocess

    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, cwd=root)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode == 0:
        return True
    return False if done.returncode == 1 else None


def question_sets_in(path: Path) -> list[str] | None:
    """The question sets a result file says it measured, or None when it cannot say.

    None and an empty list are different answers. None means the file is
    unreadable or carries no manifest, so it proves nothing about its own
    provenance. An empty list means it carries a manifest that names no set.
    """
    record = read_result_file(path)
    if record is None or not record.entries:
        return None
    return [entry.names_a_question_set for entry in record.entries]


def question_splits_in(path: Path) -> list[str]:
    """The question splits a result file says it scored, one per manifest.

    An entry that names no split reads as an empty string, so a caller can see
    that the manifests disagree instead of reading a default as a fact.
    """
    record = read_result_file(path)
    return [entry.question_split for entry in (record.entries if record else [])]


def _question_set_problem(bundle: dict, path: Path, name: str) -> str:
    """Why a result file fails to back the question set the bundle names.

    A review verdict is a statement about a set of questions. Editing one
    question changes the set and leaves the verdict describing something else.
    """
    named = bundle.get("question_set_fingerprint")
    citable = bool(bundle.get("citable"))
    sets = question_sets_in(path) if path.exists() else None
    if sets is None:
        if citable:
            return f"{name} carries no manifest, so it cannot say which questions it scored"
        return ""
    if not sets:
        if citable:
            return f"{name} names no question set, so its provenance is unproven"
        return ""
    if UNREADABLE_QUESTION_SET in sets:
        if citable:
            return (
                f"{name} carries a manifest whose question set fingerprint nobody can "
                f"read, and a readable manifest beside it does not stand in for that one"
            )
        return ""
    if not named:
        return ""
    off = [s for s in sets if s != named]
    if off:
        return (
            f"{name} measured question set {off[0]}, and the bundle names {named}. "
            f"The review verdict describes a different set of questions."
        )
    return ""


def _measurement_problems(bundle: dict, root: Path, live) -> list[str]:
    """Why a citable bundle holds no measurement to cite.

    A run whose every query failed still writes a full summary, and every mean
    in it reads `0.0`. The review verdict says the questions are sound, which is
    a statement about the corpus and not about the run. So a bundle over an
    outage passed as evidence for a row of zeros.

    The pair of counts also tells a whole run from a partial one. A run that
    scored 60 of 75 read as a 75-question run, and the review verdict then
    covered 15 questions nobody measured.
    """
    corpus = bundle.get("corpus")
    if not isinstance(corpus, str) or not corpus.strip():
        return []
    # How many questions the review covers. Without this, a manifest that named
    # 1 question and a summary that scored 1 agreed with each other, and the
    # bundle still carried the fingerprint of all 75 reviewed ones.
    reviewed = len(live[2]) if live else None
    problems = []
    for name in bundle.get("results") or []:
        path = root / name
        if not path.exists():
            continue
        scored, expected, why = measurement_in(path, corpus)
        if expected is not None and reviewed is not None and expected != reviewed:
            problems.append(
                f"{name} names {expected} questions in its manifest, and the review "
                f"covers {reviewed}. The run is not over the set the bundle claims."
            )
            continue
        if scored is None:
            # The reader already decided this, and it says why. An earlier check
            # took the summary at its word, so emptying the per-question rows
            # left the bundle reading as citable.
            problems.append(f"{name} {why}")
        elif scored == 0:
            problems.append(f"{name} scored no questions, so it holds no measurement")
        elif expected is None:
            problems.append(f"{name} does not say how many questions it set out to score")
        elif scored != expected:
            problems.append(
                f"{name} scored {scored} of the {expected} questions its manifest names, "
                f"so the review verdict covers questions this run did not measure"
            )
    return problems


def _live_review(corpus, split) -> tuple[str, dict, list] | None:
    """The fingerprint, the review verdict, and the questions on disk now.

    Both come from one read, because they are two statements about the same set.
    None means the questions cannot be read, so nothing about them is provable.

    The catch is broad on purpose. A checker that crashes tells a reader less
    than one that reports a problem, and `load_questions` raises more than it
    documents: a question file holding a bare scalar reaches `for entry in raw`
    and raises TypeError, which no named exception here would have caught.
    """
    if not isinstance(corpus, str) or not corpus.strip():
        return None
    if not isinstance(split, str) or not split.strip():
        return None
    from kb_arena.benchmark.manifest import question_set_fingerprint
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.benchmark.review import review_summary

    try:
        questions = load_questions(corpus, split=split)
    except Exception:
        return None
    return question_set_fingerprint(questions), review_summary(questions), questions


def is_bundle_result(path: Path, corpus: str) -> bool:
    """Whether a file in a run directory is one of this corpus's measurements.

    A run directory holds more than measurements. It holds the bundle, the run
    record, a report, and a comparison whose name carries two strategies and a
    metric. Naming those one at a time cost three rounds, and the comparison
    ended it: its name varies, so no fixed list catches it.

    Deciding by the file's contents failed too, twice. A rule that asked for a
    manifest dropped a plugin result that was truncated, and then dropped one
    that simply carried no manifest. Both left the bundle in silence while the
    remaining results still read as citable.

    So the rule reads the name against the corpus the bundle names. The lab
    writes one file with one name. Every other result is `<corpus>_<strategy>`,
    which a plugin strategy satisfies as well as a built-in one. A comparison is
    `compare_...`, so it fails on the corpus, and it never needed a name of its
    own.
    """
    if path.suffix != ".json":
        return False
    if path.name == "retriever_lab.json":
        return True
    return bool(corpus) and path.name.startswith(f"{corpus}_")


def measurement_in(path: Path, corpus: str) -> tuple[int | None, int | None, str]:
    """What a result proved it scored, what its manifest names, and why it proved nothing.

    A run whose every query failed still writes a summary, and every mean in it
    reads `0.0`. So the scored count is the only field that tells a measurement
    from an outage, and the pair tells a whole run from a partial one.

    The proven count comes from the per-question rows, never from the summary
    alone. The third value carries the reader's own verdict, so a caller cannot
    read the count while ignoring the reason it is missing. That split is what
    let a summary the reader had already called unusable back a citable bundle.
    """
    record = read_result_file(path)
    if record is None:
        return None, None, "cannot be read"
    if isinstance(record, LabRun):
        manifest = record.manifests.get(corpus)
        scored, why = record.scored_for(corpus)
        return scored, manifest.question_count if manifest else None, why
    expected = record.manifest.question_count if record.manifest else None
    why = "" if record.scored is not None else "does not say how many questions it scored"
    return record.scored, expected, why


def _review_scope_problems(bundle: dict, live) -> list[str]:
    """Why a citable bundle fails to prove that its review covers the run.

    The earlier guard read the fingerprint out of the run, stored it in the
    bundle, and then compared the stored value to the run. That compares a value
    to a copy of itself, so it passed for every run, including a run over a
    question set nobody reviewed. Two separate facts must agree here: the set
    the run measured, and the set the review verdict ran over.

    One case stays unprovable, and the answer is to refuse it. A `--tier 1`
    benchmark run scores a subset and records `question_split: all`, so its
    manifest cannot tell that subset apart from a corpus somebody edited.
    """
    named = bundle.get("question_set_fingerprint")
    covered = bundle.get("review_question_set")
    split = bundle.get("review_split")
    if not isinstance(covered, str) or not covered.strip():
        return ["calls itself citable and does not say which question set its review covers"]
    if named and named != covered:
        return [
            f"the run measured question set {named} and the review covers {covered}. "
            f"The review verdict describes a different set of questions. A "
            f"tier-filtered run records no tier in its manifest, so a bundle over "
            f"one lands here too, and refusing it is the honest answer."
        ]
    if live is None:
        return [
            f"calls itself citable, and split {split!r} of corpus "
            f"{bundle.get('corpus')!r} cannot be read, so nobody can recheck the verdict"
        ]
    live_set, live_review, _ = live
    if live_set != covered:
        return [
            f"the questions on disk now measure {live_set} and the review covered "
            f"{covered}. Somebody changed a question after this bundle was written."
        ]
    # The set matched, which says WHICH questions. It says nothing about the
    # verdict over them. The stored review is data in the bundle, so a hand
    # edit that keeps `publishable` and flips every count used to pass here.
    stored = bundle.get("review") or {}
    if not live_review.get("publishable"):
        return [
            "calls itself citable, and the questions it names do not support that. "
            + "; ".join(_live_blockers(bundle.get("corpus"), split))
        ]
    if stored.get("counts") != live_review.get("counts"):
        return [
            f"records the review counts {stored.get('counts')!r} and the questions "
            f"it names count {live_review.get('counts')!r}"
        ]
    return []


def _live_blockers(corpus, split) -> list[str]:
    """Why the questions on disk do not support a citable claim."""
    from kb_arena.benchmark.questions import load_questions
    from kb_arena.benchmark.review import publication_blockers

    try:
        return publication_blockers(load_questions(corpus, split=split))
    except Exception:
        return ["the questions cannot be read"]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    """Check a bundle from the command line, for CI or a reviewer."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m kb_arena.benchmark.evidence <evidence.json>")
        return 2
    path = Path(args[0])
    bundle = json.loads(path.read_text())
    problems = check_bundle(bundle, path.parent.parent.parent)
    for problem in problems:
        print(f"{path}: {problem}")
    if not problems:
        print(f"{path}: complete. citable={bundle['citable']}")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
