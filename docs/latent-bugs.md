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

## Open, found by the 2026-09-03 review of the SSRF DNS-rebinding pin

A cross-model review of the DNS-rebinding fix in `kb_arena/ingest/parsers/web.py` found three
pre-existing defects that the fix did not introduce. They stayed out of that PR to keep its scope.
Each one has a ledger row in the enhancement train, N-12 to N-14. All three are fixed now: the URL
ingest defect in PR 32, the crawl cap and the resolver failure in the section below.

### `kb-arena ingest https://example.com` never sends a request, because `Path()` turns `https://` into `https:/`

`kb_arena/ingest/pipeline.py` wraps every source in `Path()` before it calls the parser.
`Path("https://example.com")` normalises to `https:/example.com`. `WebParser.parse` then sees a
string that does not start with `http://` or `https://`, tries to read it as a file that holds a
URL, fails, and returns an empty list. The CLI reports that no documents came out of the source.
A URL only works today when it sits inside a file.

## Fixed by the 2026-09-04 web-parser slice

The slice on branch `slice-b2-web-parser` closes the two items below. The crawl cap now counts
every fetch the crawler starts. A resolver failure, `EAI_AGAIN` or `EAI_FAIL`, raises
`DNSFailureError` and reaches the operator as a failed ingest. An unreachable entry page does
the same. A name that does not exist stays a refusal.

### The crawler's page cap counts extracted pages, so failed fetches let it send more requests than `max_pages`

`WebParser._crawl` loops while `len(pages) < max_pages`. A fetch that fails, or a page with no
text, adds nothing to `pages` but still adds its links to the queue. With `max_pages=1` and
`max_depth=3`, a site whose pages each link to two empty pages draws 15 requests and returns
nothing. The cap must count every fetch it starts.

### A temporary DNS failure reads as an SSRF refusal and an empty corpus

`_validate_url` turns every `socket.gaierror` into `SSRFBlocked`, including `EAI_AGAIN`, the
resolver's "try again later" code. `_scrape` catches that, logs a refusal, and returns an empty
list. An outage and a source with no documents look the same to the operator. A retryable
resolver error must surface as its own failure.

## Open, found while adding the D-01 to D-06 dataset adapters

### `kb-arena datasets` prints "may ship: yes" for an adapter its own code refuses to vendor

`cli.py`'s `datasets_command` sets the "may ship" column from `template.redistributable`, which
reads only the licence. `multihop-rag`, `frames`, `bright`, `beir-scifact`, `miracl`, and
`longbench-v2` carry open licences, so that column prints "yes", while each adapter still sets
`download_only = True` and its own `check_destination` refuses an in-repo destination. The column
should read `download_only` instead of `redistributable`, or check both.

## Open, found while adding the MCP registry entry

### `kb-arena quantum-diagnostics` drops the extra name from its install line

`cli.py` prints `"[red]The [quantum] extra is required.[/red] Install with: pip install
'kb-arena[quantum]'"` through Rich. Rich reads `[quantum]` as markup and removes it, so the operator
reads "The  extra is required. Install with: pip install 'kb-arena'", which names no extra and
installs the wrong thing. The brackets need the Rich escape, the way the new `kb-arena mcp` command
escapes them.
