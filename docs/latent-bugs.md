# Latent bugs found during the 0.10.0 review

The 0.10.0 review found five pre-existing defects that the release itself did not introduce. They
stayed out of that release so the in-flight change kept its scope. All five are fixed now, and this
file records what they were.

Source: cross-model adversarial review of the 0.10.0 branch diff, 2026-08-06.

## Fixed after 0.10.0

### Arena matches accepted mock answers and moved persistent ELO ratings

`kb_arena/arena/engine.py` recorded whatever `query()` returned. With Neo4j down,
`KnowledgeGraphStrategy.query()` returns a mock-data message with `mock=True`, and `create_match()`
stored it without checking the flag. A later vote wrote that outcome into `arena_state.json`, so an
outage permanently changed the leaderboard.

`create_match()` now raises `ArenaError` when either result carries `mock=True`, and
`POST /api/arena/match` answers 503 `strategy_unavailable` rather than 500.

### The Arena page sent no corpus, so the backend used its aws-compute default

`web/app/arena/page.tsx` posted only the question, and `ArenaMatchRequest` defaults `corpus` to
`aws-compute`. A deployment carrying only `nist-800-171-r3` answered from a corpus the operator
never chose. The page now carries a corpus selector fed by `GET /api/corpora` and sends the choice.

### A slow first graph fetch could overwrite a completed live build

`web/app/graph/page.tsx` invalidated the first `GET /api/graph/data` response only on a corpus
change. A live build did not stop that request, and both paths wrote the same nodes, edges, and
connected state. The effect now captures `buildEpochRef` when it starts. It drops its response when
a build advanced the epoch.

### ArenaEngine startup failures looked like ordinary unavailability

`kb_arena/chatbot/api.py` turned any `ArenaEngine(...)` construction failure into `arena = None`,
which is also what a deployment without an arena looks like. The original exception was lost.
`_build_arena()` now logs the exception before it returns None, so the two cases stay apart in the
log.

### Q&A generation accepted the all-corpora sentinel and wrote a phantom corpus

`POST /api/tools/generate` accepted `corpus: "all"`, read sections from every corpus, then wrote one
joint file to `datasets/all/qa-pairs/qa_pairs.jsonl`. No per-corpus read uses that directory, so the
pairs lost the link to the corpora their sections came from. `GenerateRequest` now rejects the
sentinel. The read-only audit and fix endpoints still accept it.
