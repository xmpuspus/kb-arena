# KB Arena 0.10.0 release post draft

> Draft only. Do not post automatically.

Choosing a retrieval architecture should not start with a generic leaderboard but with evidence
from your own documents.

KB Arena 0.10.0 reframes the project as a retrieval architecture decision lab. It runs lexical,
dense, graph, hybrid, hierarchical, and reranked retrieval against the same corpus and question
set, then records retrieval quality, answer quality, latency, cost, and reproducible run artifacts.

This release adds a retrieval-only lab, paired comparisons with confidence intervals, an explicit
strategy catalog, a no-key packaged demo, and a deterministic NIST SP 800-171 Revision 3 corpus
with 130 control documents and 80 draft questions for method development.

The historical AWS example shows why the framing matters. It has 75 questions across five
tiers, but only 35 have chunk-level labels. Contextual Vector recorded 0.355 Recall@5 and Naive
Vector recorded 0.352. That 0.003 gap documents the run, but it does not support a general winner
claim.

Run the precomputed demo without an API key or external service.

```bash
pip install kb-arena==0.10.0
kb-arena demo
```

[Project repository and visual walkthrough](https://github.com/xmpuspus/kb-arena)

[Published package](https://pypi.org/project/kb-arena/0.10.0/)

The included NIST questions are machine-generated drafts. A qualified reviewer should check the
questions, answers, constraints, and holdout isolation before anyone publishes a strategy winner
from that corpus.
