# aws-compute: 75 questions Xavier Puspus wrote and confirmed

The corpus KB Arena ships as its worked example. Seven AWS compute documents, 75
questions across five difficulty tiers, and chunk-level ground truth for 35 of
them in `questions/expected_chunks.yaml`.

## What the review status means

Every question carries `review_status: human-reviewed` and
`reviewed_by: "Xavier Puspus"`. That records two facts:

- Xavier wrote the questions and their answer keys, and committed them in
  `3b644b5` on 2026-03-02.
- Xavier confirmed the review on 2026-09-05, after the corpus had shipped in
  five releases as the canonical example.

The label is about the answer key, not about the retrieval numbers. It says a
person stands behind each question and its expected answer. It says nothing
about whether any strategy answers them well.

`kb-arena` reads the field through `kb_arena/benchmark/review.py`. A result is
`publishable` only when every question it scored is `human-reviewed`, so this
corpus can carry a citable result and a machine-drafted corpus cannot.

## How this corpus differs from nist-800-171-r3

The NIST corpus carries `review_status: machine-assisted-draft` and
`reviewed_by: "Codex draft pass"`. Nobody has checked those answer keys, so a
result over them is a development signal and never citable. That label stays
until a person reviews them.

Two corpora, two honest labels. The gate is what makes the difference readable.

## Layout

| Path | What it holds |
|---|---|
| `raw/` | The seven source documents |
| `processed/` | Parsed JSONL and the BM25 index a run builds |
| `questions/tier1_factoid.yaml` | 20 single-fact questions |
| `questions/tier2_procedural.yaml` | 20 step-sequence questions |
| `questions/tier3_comparative.yaml` | 15 service-versus-service questions |
| `questions/tier4_relational.yaml` | 12 questions over how services connect |
| `questions/tier5_multihop.yaml` | 8 questions needing several documents |
| `questions/expected_chunks.yaml` | Chunk-level ground truth for 35 questions |

## Repeat a run over it

```bash
kb-arena ingest ./datasets/aws-compute/raw/ --corpus aws-compute
kb-arena build-vectors --corpus aws-compute
kb-arena retriever-lab --corpus aws-compute
```

The lab needs no API key. `kb-arena benchmark --corpus aws-compute` calls a
judge and does need one.
