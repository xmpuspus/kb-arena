# Walk your own documents through KB Arena, from ingest to an evidence bundle

This walkthrough runs every command below against a small real corpus named
`my-docs-test`, three headings pulled from a made-up onboarding page. Every
output block is what that command printed. Swap in your own `raw/` files and
your own corpus name and the same commands apply.

Two steps in a full run need a paid provider, writing questions with an LLM
and building a dense strategy such as `naive_vector`. This walkthrough has
neither key, so it writes questions by hand and builds the strategy that
needs no key at all, BM25. Read [getting-started.md](getting-started.md) for
the dense-strategy and graph setup once you have a provider.

## Scaffold the corpus and add your files

```bash
kb-arena init-corpus my-docs-test
```

```
Created corpus scaffold: datasets/my-docs-test/
  raw/         <- drop your documents here
  processed/   <- ingest output goes here
  questions/   <- benchmark questions (YAML, see sample)

Next: kb-arena run --corpus my-docs-test (orchestrates ingest -> build-graph ->
build-vectors -> benchmark)
```

Drop your files in `datasets/my-docs-test/raw/`. This run uses one short
Markdown file with three headings, requesting a laptop, VPN access, and a
first-week checklist.

## Ingest

```bash
kb-arena ingest ./datasets/my-docs-test/raw/ --corpus my-docs-test
```

```
Done. 1 documents, 4 sections -> datasets/my-docs-test/processed/documents.jsonl
```

Ingest splits the file into sections by heading. `documents.jsonl` carries a
stable section ID under each heading, for example `requesting-a-laptop`,
which the next steps cite as ground truth.

## Write questions by hand, since automatic generation needs a key

`kb-arena generate-questions` calls an LLM, so it needs
`KB_ARENA_ANTHROPIC_API_KEY` or another provider you set up. Without one,
copy the schema `init-corpus` drops at
`datasets/my-docs-test/questions/tier1_factoid.yaml.example` and write your
own questions. Rename the file to drop `.example` so KB Arena reads it. Each
question needs `ground_truth.source_refs` naming the document ID that
answers it, so retrieval scoring works even before you run
`kb-arena label-chunks`, which needs an LLM key too, for chunk-level
labels.

This run wrote two questions, one per document section.

```yaml
- id: my-docs-test-t1-001
  tier: 1
  type: factoid
  hops: 1
  split: development
  review_status: human-reviewed
  reviewed_by: "Xavier Puspus"
  rationale: "Checks that the laptop request policy is retrievable"
  source_anchors: ["onboarding#requesting-a-laptop"]
  question: "How long does laptop approval take for a new hire?"
  ground_truth:
    answer: "Two business days."
    source_refs: ["onboarding"]
    required_entities: []
  constraints:
    must_mention: ["two business days"]
    must_not_claim: []
```

## Build the retrieval index

```bash
kb-arena build-vectors --corpus my-docs-test --strategy bm25
```

```
BM25 index built for my-docs-test: 3 passages
Done. Built 1 vector index(es) from 1 documents
```

BM25 is the one strategy in the catalog that needs no embedding provider and
no LLM. Building any of the other 18, for example `naive_vector`, needs
`KB_ARENA_EMBEDDING_PROVIDER` set to `openai`, `voyage`, `cohere`, `bge`,
`gemini`, or `ollama`, plus that provider's key. See the strategy table in
[the README](../README.md#strategy-catalog) for which strategies need what.

## Run the retrieval-only evaluation

```bash
kb-arena retriever-lab --corpus my-docs-test --strategies bm25 --top-k 5 --min-recall 0
```

```
Run ID: bc3fd2f6 | top-k: 5 | ceiling-k: 20
  bm25: n=2, mean Recall@5=1.000
Run bc3fd2f6 written to results/run_bc3fd2f6/
```

`--min-recall 0` turns off the pass/fail gate `retriever-lab` applies by
default, so the run finishes cleanly on a toy corpus. Two hand-written
questions over one short document is not evidence about retrieval quality.
It shows the pipeline runs end to end. A number worth citing needs a corpus
and question count like `aws-compute`'s, which
[its README](../datasets/aws-compute/README.md) documents in full, including
what the corpus cannot answer.

## Compare two strategies

`kb-arena compare` pairs two strategies question by question from the same
retriever-lab run. This corpus only has BM25 built, so a second strategy
needs an embedding provider first. To show the command's real output without
that key, this run reads the repository's own committed multi-strategy
result instead.

```bash
kb-arena compare --lab results/run_855aac4e/retriever_lab.json --a bm25 --b naive_vector --metric recall_at_k
```

```
Comparable: 75 paired questions
n=75  mean bm25=0.2746  mean naive_vector=0.3524  delta=+0.0778
95% CI [+0.0407, +0.1204]  p=0.001  d=+0.43  W/T/L=18/56/1
```

Run the same command against your own corpus once you build a second
strategy there, pointing `--lab` at your run's `retriever_lab.json`. The
delta above describes one small, already-caveated historical run. The
[README](../README.md#evidence-included-in-this-repository) and
[the aws-compute corpus notes](../datasets/aws-compute/README.md) both say
this corpus is too small and too incompletely labeled to declare a winner.
Read those caveats before you read this number as anything more than a
worked example of what `compare` prints.

## Export the evidence bundle

```bash
kb-arena evidence --corpus my-docs-test --run-id bc3fd2f6
```

```
results/run_bc3fd2f6/evidence.json
Every scored question is human-reviewed, so this run is citable.
```

`evidence.json` records the command, the package version, the commit, the
platform, and whether every scored question carries `review_status:
human-reviewed`. Here is this run's bundle.

```json
"citable": true,
"review": {
  "counts": {"human-reviewed": 2, "machine-assisted-draft": 0, "unspecified": 0},
  "publishable": true,
  "questions": 2
}
```

`kb-arena evidence --check <path>` reads a bundle back later and refuses one
that claims `citable: true` while its own review counts disagree. See
[docs/reference-cli.md](reference-cli.md) for every `evidence` flag.

## Three commands need an LLM key, benchmark, report, and label-chunks

This walkthrough does not run `kb-arena benchmark`, `kb-arena report`, or
`kb-arena label-chunks`. All three call an LLM for generation or judging, on
top of the embedding key retrieval already needs. Set
`KB_ARENA_LLM_PROVIDER` and `KB_ARENA_ANTHROPIC_API_KEY` (or
`KB_ARENA_LLM_PROVIDER=ollama` for a local, keyless model) and follow
[getting-started.md](getting-started.md) for those steps.
