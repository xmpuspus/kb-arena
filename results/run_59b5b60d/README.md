# Committed run 59b5b60d

One bounded, reproducible retrieval run on the built-in `aws-compute` corpus,
kept in the repository so a reader can check the numbers rather than trust them.

## Repeat it

```bash
kb-arena build-vectors --corpus aws-compute --strategy bm25
kb-arena retriever-lab --corpus aws-compute --strategies bm25
kb-arena evidence --corpus aws-compute --run-id <the id it prints>
```

BM25 needs no API key and no embedding provider, so the run costs nothing and
anyone can repeat it. `evidence.json` records the package version, the commit,
the Python build, the platform, the seed and the exact command.

## What this run shows

75 questions, `mean Recall@5 = 0.275`. The full metric set is in
`retriever_lab.json`.

## What this run does NOT show

`evidence.json` says `citable: false`, and the reason is in the file:

> publishable is true only when every scored question is human-reviewed.

All 75 questions carry no review status at all. Nobody recorded who wrote them
or whether anybody checked them. The chunk-level labels under them were made by
a model, not a person. So this run is a development signal and it is not
evidence anybody should cite.

That is the honest state, and it is why the bundle exists. A number with no
provenance beside it invites a citation it cannot support.

## What would make it citable

A person reviews the questions and the labels, and records that review in the
question files as `review_status: human-reviewed`. Only a human can do that, and
this repository will not mark it on their behalf.
