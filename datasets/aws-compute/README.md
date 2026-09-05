# aws-compute: 75 reviewed questions over 1,549 words, and 58 of them the corpus cannot answer

The corpus KB Arena ships as its worked example. Three AWS documents, 75
questions across five difficulty tiers, and chunk-level ground truth for 35 of
them in `questions/expected_chunks.yaml`.

Read the coverage section before you read any number this corpus produces.

## What the review status covers

Every question carries `review_status: human-reviewed` and
`reviewed_by: "Xavier Puspus"`. Three facts back that:

- Xavier wrote the questions and their answer keys. He committed them in
  `3b644b5` on 2026-03-02.
- On 2026-09-05 every answer key got a line-by-line check against AWS behaviour
  and against the three source documents. The check used three review models
  and each finding was read back against the file before anybody changed it.
- That check found six wrong keys out of 75. All six are corrected. Xavier
  confirmed the review.

The six corrections:

| Question | What was wrong |
|---|---|
| `compute-t3-007` | `must_not_claim` banned "containers cannot use Lambda layers", which is true |
| `compute-t3-010` | CloudFront Functions quotas read 10 ms of CPU and 2 MB of code, about 200x too high |
| `compute-t3-011` | The key denied automatic rollback, and the ECS deployment circuit breaker gives it |
| `compute-t3-013` | Two banned statements about placement groups were both true |
| `compute-t4-010` | ENI trunking read as a size minimum, and it is a Nitro family list |
| `compute-t5-003` | The ECS and Batch Step Functions integrations read `.sync:2` instead of `.sync` |

The label covers the answer key. It says a person stands behind each question
and its expected answer. It says nothing about how well any strategy answers
them, and nothing about whether this corpus contains the answer.

`kb-arena` reads the field through `kb_arena/benchmark/review.py`. A result is
`publishable` only when every question it scored is `human-reviewed`.

## 58 of the 75 questions ask about services this corpus does not hold

The same 2026-09-05 check counted how many questions the three documents can
answer. The answer is 17.

| Verdict | Count | What it means |
|---|---|---|
| Correct and covered | 11 | The key is right and a source document holds the facts |
| Correct, not covered | 58 | The key is right and no source document holds the facts |
| Wrong, now corrected | 6 | The key stated something false about AWS |

The keys cite 107 distinct AWS documentation URLs, covering EKS, Batch, Step
Functions, App Runner, Lightsail, CloudFront, Route 53, DynamoDB, RDS and more.
This corpus holds three of those subjects. So the questions were written against
a much larger body of documentation than the corpus contains.

That is why 40 of the 75 carry an empty list in `expected_chunks.yaml`. No chunk
in this corpus answers them.

**Read every retrieval number from this corpus with that in mind.** A recall of
0.2746 for BM25 is not only a statement about BM25. It is partly a corpus that
cannot answer most of its own questions. Use this corpus to see the pipeline run
end to end. Do not use it to rank retrieval architectures.

## How this corpus differs from nist-800-171-r3

The NIST corpus carries `review_status: machine-assisted-draft` and
`reviewed_by: "Codex draft pass"`. Nobody has checked those answer keys, so a
result over them is a development signal and never citable. That label stays
until a person reviews them.

Two corpora, two honest labels. The gate is what makes the difference readable.

## Layout

| Path | What it holds |
|---|---|
| `raw/` | Three source documents: Lambda, API Gateway, ECS and Fargate |
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
