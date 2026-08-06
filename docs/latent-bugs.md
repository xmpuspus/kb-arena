# Latent bugs found during the 0.10.0 review and not fixed in it

Each item below is pre-existing behavior that the 0.10.0 branch did not introduce. They are
recorded here instead of being bundled into the release, so the in-flight change stays scoped.
Source: cross-model adversarial review of the branch diff, 2026-08-06.

## The Arena page sends no corpus and the backend defaults to aws-compute

`web/app/arena/page.tsx` posts only the question. `ArenaMatchRequest` defaults `corpus` to
`aws-compute`, so a deployment that carries only `nist-800-171-r3` answers from a corpus the
operator did not choose, or fails to match. The page also has no corpus selector, unlike the
benchmark and retriever-lab pages.

Fix direction: add a corpus selector fed by `GET /api/corpora`, and send that choice.

## A slow first graph fetch can overwrite a completed live build

`web/app/graph/page.tsx` invalidates the first `GET /api/graph/data` response only when the corpus
changes. A live build does not stop that request, and both paths write the same nodes, edges, and
connected state. A delayed first response that lands after the build completes replaces the new
graph with the pre-build one.

Fix direction: count request generations, or cancel the first fetch when a build starts.

## ArenaEngine startup failures look like ordinary unavailability

`kb_arena/chatbot/api.py` turns any `ArenaEngine(...)` construction failure into `arena = None`.
The arena endpoints then return 503 `arena_unavailable`, which is the same response a deployment
without an arena gives. A corrupt arena state or a programming error looks like a disabled
optional feature, and the original exception is lost.

Fix direction: log the exception. Then decide if an arena failure must stop startup, the way LLM
initialization now does.

## Arena matches accept mock answers and let them move persistent ELO ratings

`kb_arena/arena/engine.py` records whatever `query()` returns. When Neo4j is down,
`KnowledgeGraphStrategy.query()` returns a mock-data message with `mock=True`, and `create_match()`
stores it without checking the flag. A later vote then writes that outcome into `arena_state.json`,
so an infrastructure fallback permanently changes the leaderboard. The benchmark path already
rejects mock results.

Fix direction: skip a strategy whose result carries `mock=True`, or refuse the vote for that match.

## Q&A generation accepts the all-corpora sentinel and writes a phantom corpus

`POST /api/tools/generate` accepts `corpus: "all"` because the shared validator permits the
sentinel. Generation then reads sections from every corpus, but writes one joint output to
`datasets/all/qa-pairs/qa_pairs.jsonl`. No per-corpus read path uses that directory. The generated
pairs lose the link to the corpora their sections came from.

Fix direction: reject the sentinel on write endpoints, or write each pair back to its own corpus.
