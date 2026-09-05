# Retrieval regression gate action

The retrieval-regression-gate action ingests a corpus, builds the selected
kb-arena strategy indexes, runs a retrieval-only benchmark, and fails the job
when a named metric drops by more than a threshold you set, against a
baseline file you store in your own repository.

Vector and graph strategies that need an embedding or an LLM provider are out
of scope for this action. Wire your own provider credentials before you
select one of them. The example in this repository runs `bm25`, which needs
no provider key.

## Inputs

| Input | Required | Default | Purpose |
|---|---|---|---|
| `corpus` | yes | | Corpus name to benchmark, for example `aws-compute`. |
| `corpus-path` | yes | | Path to the corpus's raw documents, for example `datasets/aws-compute/raw`. |
| `strategies` | yes | | Comma-separated kb-arena strategy filter, for example `bm25`. |
| `top-k` | no | `"5"` | Top-k chunks per query. |
| `metric` | yes | | Metric name from `retriever_lab.json` to compare, for example `mean_recall_at_k`. |
| `baseline-path` | yes | | Path to the stored baseline JSON, relative to your checkout. |
| `threshold` | yes | | Maximum allowed drop in the metric before the gate fails. |
| `kb-arena-version` | no | `"0.11.0"` | Pinned kb-arena version to install from PyPI. |
| `install-from` | no | `"pypi"` | Install kb-arena from `pypi` (the pinned version) or from your `checkout`. |

Use `install-from: checkout` to gate a pull request on its own retrieval
code. Leave it at the default `pypi` to benchmark with a released kb-arena
build, which is the right choice for a consumer repository that does not
carry kb-arena's own source.

## What the action does, step by step

1. Sets up Python 3.12.
2. Installs kb-arena, from PyPI at the pinned version, or from your checkout.
3. Runs `kb-arena ingest "$corpus-path" --corpus "$corpus"`.
4. Runs `kb-arena build-vectors --corpus "$corpus" --strategy "$name"` for
   each strategy named in `strategies`.
5. Runs `kb-arena retriever-lab --corpus "$corpus" --strategies "$strategies"
   --top-k "$top-k" --min-recall 0`, writing `retriever_lab.json` under a
   results path scoped to this job.
6. Runs `compare_metric.py`, which reads that file and the baseline, and
   exits with a failure when a check below does not pass.

## The baseline file's shape

The baseline is a JSON file you commit to your repository. This repository's
own baseline, `.github/retrieval-baselines/aws-compute-bm25.json`, looks like
this.

```json
{
  "corpus": "aws-compute",
  "metric": "mean_recall_at_k",
  "top_k": 5,
  "strategies": {
    "bm25": 0.274582
  },
  "recorded_from": "kb-arena 0.10.0, kb-arena retriever-lab --corpus aws-compute --strategies bm25 --top-k 5 --min-recall 0, 75 questions",
  "note": "CI regression reference only. Not a published benchmark claim."
}
```

The gate reads these fields.

- `corpus`, `metric`, and `top_k` must match the values the action ran with.
  A mismatch on any of the three stops the gate before it compares numbers.
- `strategies` maps a strategy name to its recorded metric value, one entry
  per strategy you want checked.
- `recorded_from` and `note` are for a human reader. The gate does not read
  them.

Write `recorded_from` and `note` yourself when you record a new baseline,
the same way this file does. They tell a later reader which kb-arena version
and command produced the number, and that the number is not a benchmark claim to publish, but a regression
reference.

## What makes the gate fail

The gate exits with a failure, and prints every reason, under any of these
conditions.

- `corpus`, `metric`, or `top_k` in the baseline does not match the values
  the action ran with.
- `THRESHOLD` does not parse as a finite number. `float()` accepts `nan` and
  `inf`, and a comparison against either is always false, so the gate treats
  a non-finite threshold not as an always-pass value, but as a refusal to run.
- A strategy named in the action's `strategies` input has no entry in the
  baseline's `strategies` map. Reading only the baseline's own keys would let
  a newly added strategy regress, or drop out of the run, while the gate
  reported success.
- A baseline value is not a finite number. `json.load` accepts `NaN`, and a
  comparison against it is always false, so a `NaN` baseline would otherwise
  pass whatever the run measured.
- A strategy in the baseline is missing from the fresh `retriever_lab.json`
  run, or the named metric is missing from that strategy's entry.
- The fresh metric value is not a finite number.
- `baseline_value - new_value` is greater than `threshold`, for any checked
  strategy. This is the regression check itself.
- The results directory holds no `retriever_lab.json`, or more than one. The
  gate compares one run, so it refuses rather than pick between two.

When every check passes, the gate prints `Retrieval regression gate passed.`
and exits zero.

## A complete example workflow

This is the workflow this repository runs on its own pull requests,
`.github/workflows/retrieval-regression-example.yml`.

```yaml
name: Retrieval regression example

on:
  pull_request:
    branches: [main]
    paths:
      - "datasets/aws-compute/**"
      - "kb_arena/strategies/**"
      - "kb_arena/benchmark/retriever_lab.py"
      - ".github/actions/retrieval-regression-gate/**"
      - ".github/retrieval-baselines/aws-compute-bm25.json"
  workflow_dispatch:

jobs:
  regression-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4.4.0
      - name: Gate on a BM25 recall regression
        uses: ./.github/actions/retrieval-regression-gate
        with:
          corpus: aws-compute
          corpus-path: datasets/aws-compute/raw
          strategies: bm25
          top-k: "5"
          metric: mean_recall_at_k
          baseline-path: .github/retrieval-baselines/aws-compute-bm25.json
          threshold: "0.03"
          install-from: checkout
```

This example is the workflow this repository runs on itself, so its paths point
at files here. Change `corpus`, `corpus-path`, `baseline-path` and the `paths`
filter to your own before you copy it. `install-from: checkout` gates the code
in the pull request, and a caller outside this repository wants
`install-from: pypi` instead, which gates against a released version.

The `paths` filter scopes the workflow to changes that can move the metric,
so it does not run on every pull request. `workflow_dispatch` lets you run it
by hand against the current baseline. This workflow ran green on three pull
requests in this repository, gating each one on its own retrieval code
through `install-from: checkout`.

To gate a consumer repository against a released kb-arena instead of a
checkout, reference the action from this repository and drop
`install-from`, so it defaults to `pypi` at the pinned `kb-arena-version`.

```yaml
      - name: Gate on a BM25 recall regression
        uses: xmpuspus/kb-arena/.github/actions/retrieval-regression-gate@main
        with:
          corpus: aws-compute
          corpus-path: datasets/aws-compute/raw
          strategies: bm25
          top-k: "5"
          metric: mean_recall_at_k
          baseline-path: .github/retrieval-baselines/aws-compute-bm25.json
          threshold: "0.03"
```
