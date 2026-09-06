# Every change in 0.11.0 traces to a finding, and every finding names its check

An eight-dimension audit on 2026-09-03 produced a ledger of 236 numbered
findings. This document maps what shipped to what it closed, and says what it
did not close. It is the record a reader checks instead of trusting the release
notes.

## The count, and what the count is not

The 236 rows carry five statuses, and they add up:

| Status | Rows | What it means |
|---|---|---|
| fixed | 166 | The change is on `origin/main`, and a script proves it by name |
| open | 62 | Not done. Most are enhancements, and each row states why it waits |
| in-progress | 3 | Started and not finished |
| checked | 4 | Investigated and found to need no change |
| deferred | 1 | Deliberately out of scope, with the reason recorded |

The script is `verify_fixed.py` in the run directory. It greps the default
branch for the name each fixed row quotes, because this repository squash-merges
and a branch SHA proves nothing about main. All 166 pass.

An earlier version of this document said 227 rows. The counter used a row-id
pattern that missed four-letter prefixes, so it skipped every `DEMO` and `DIST`
row. The count is now 236.

Three open rows are named in this document, because a reader deciding whether to
trust a number needs them.

## The one Critical and all thirteen High findings are closed

Each landed in its own pull request with its own review, and each is provable
against `origin/main` by name.

| Finding | What it was | What closed it |
|---|---|---|
| C-01 | A Critical from the audit | PR 17, `b585b3d` |
| H-01 to H-13 | Thirteen High findings | PRs 16, 17, 31 to 37, 38 |

## The evidence contract is what makes a number citable

A benchmark number on its own is a claim. These sixteen `EV` rows turned it into
a record a reader can check.

- The run manifest carries a compatibility key, so two runs from different
  commits never average together.
- The judge records its provider, so a score names the model that gave it.
- Paired bootstrap confidence intervals and a Wilcoxon test, with Holm
  correction across strategies.
- A per-record checkpoint, so a run resumes rather than restarts.
- A sealed holdout split, and a use counter that records every read.

## Three defects in this release were the same defect at different depths

Worth naming, because each was found only after the previous fix looked
complete.

**The recorded command did not reproduce the run.** `kb-arena evidence` built it
from the corpus name alone, so a bundle over a bm25-only run told a reader to
run eleven strategies. The lab and the benchmark runner record their own command
now, and running it reproduces the committed measurement to the last digit:
mean Recall@5 of `0.2745820105820106`.

**The recorded commit was one nobody could check out.** The committed bundle
named a dirty working tree, and the bundle before it named a branch commit the
squash merge erased. `kb-arena evidence --check` refuses both, and refuses a
value that is not a commit at all.

**Two readers judged a result file through a growing pile of shape checks.**
Eight review rounds over one slice found 26 defects, and three classes kept
returning. One validated read replaced the checks, which made those classes
unreachable rather than guarded one at a time.

## What this release still does not support

**It does not rank retrieval architectures.** The committed run scored one
strategy on one corpus. `docs/choosing-a-strategy.md` names what each of the 19
strategies is for and what it costs, and it names no winner.

**The aws-compute corpus cannot answer most of its own questions.** 58 of its 75
questions ask about services none of the three source documents holds, and 40
carry an empty expected-chunk list. `datasets/aws-compute/README.md` records the
count. So `0.275` is partly a corpus limit and not only a retriever limit.

**The NIST SP 800-171 r3 question set is machine-generated.** No qualified
reviewer has checked its questions, answers or constraints, and no strategy
winner may be published over it.

**One run carries no spread.** `kb-arena variance` needs repeats before it
reports a standard deviation.

## Three open findings a reader should know about

- **N-102**: a request that passes its deadline unblocks the caller and leaves
  the HTTP worker thread running. The run continues, which is the contract, but
  the work does not stop. Closing it needs an async HTTP client that shares the
  SSRF guard, and that is its own change.
- **N-103**: a strategy whose index is not built answers with a sentence a
  benchmark judge then scores, so an outage reads as a poor result. `bm25` has
  behaved this way since before this release, so both change together.
- **N-99**: the `pypi` environment carries no required reviewer. This release
  publishes with manual twine, so the workflow's publish path is unused.

## How to check any claim here

```bash
kb-arena evidence --check results/run_422209dd/evidence.json
python3 scripts/show_evidence.py
```

The first refuses a bundle that claims more than its run supports. The second
prints what the run measured, which questions it covered, and the commit it came
from.
