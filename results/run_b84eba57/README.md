# Committed run b84eba57: BM25 reaches Recall@5 of 0.275 on 75 reviewed questions

One bounded, repeatable retrieval run on the built-in `aws-compute` corpus, kept
in the repository so a reader can check the numbers rather than trust them.

## Repeat it

```bash
kb-arena ingest ./datasets/aws-compute/raw/ --corpus aws-compute
kb-arena build-vectors --corpus aws-compute --strategy bm25
kb-arena retriever-lab --corpus aws-compute --strategies bm25
kb-arena evidence --corpus aws-compute --run-id <the id it prints>
```

BM25 needs no API key and no embedding provider, so the run costs nothing.
`evidence.json` records the package version, the commit, the Python build, the
platform, the seed, the exact command, and the question set the review covers.

## What this run shows

75 questions, `mean Recall@5 = 0.275`. The full metric set is in
`retriever_lab.json`.

## What this run does NOT show

**It does not rank retrieval architectures.** One strategy ran. A single number
from a single retriever says nothing about which architecture wins.

**It does not measure this corpus fairly.** 58 of the 75 questions ask about AWS
services none of the three source documents holds, and 40 carry an empty list in
`expected_chunks.yaml`. So 0.275 is partly a corpus that cannot answer most of
its own questions. `datasets/aws-compute/README.md` gives the full count.

**It carries no spread.** One run gives a point. `kb-arena variance` needs
repeats before it reports a standard deviation.

## Why this run replaced run 59b5b60d

Run 59b5b60d measured question set `3aecce3d26b1`. On 2026-09-05 a review of all
75 answer keys found six wrong ones and corrected them, which changed the set to
`3083f59c5d22`. The old run then described questions the corpus no longer held.
`kb-arena evidence --check` refuses that now, by name:

> retriever_lab.json measured question set 3aecce3d26b1, and the bundle names
> 3083f59c5d22. The review verdict describes a different set of questions.

Both runs report the same retrieval numbers, because the lab scores against
`expected_chunks.yaml` and the corrections touched answer text and constraints.

## What makes this one citable

`evidence.json` says `citable: true`. Every scored question carries
`review_status: human-reviewed`. `datasets/aws-compute/README.md` records what
that review was and lists the six corrections it produced.
