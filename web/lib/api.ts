import { apiFetch } from "./auth";

// A refusal to read is not a benchmark result.
// A refusal to read is not an outage of the graph database.
// A rate limit or a server error is not an outage of the graph database.
export const GRAPH_UNAVAILABLE =
  "The graph could not be read just now. Try again in a moment.";

export const GRAPH_UNAUTHORIZED =
  "The graph needs an API token. Enter one to read it.";

// A rate limit or a server error is not a benchmark result either.
export const BENCHMARK_UNAVAILABLE =
  "The benchmark results could not be read just now. Try again in a moment.";

export const BENCHMARK_UNAUTHORIZED =
  "The benchmark results need an API token. Enter one to read them.";

export const NETWORK_UNREACHABLE = "The browser could not reach the API.";

// `fetch` rejects with a TypeError, and the browser writes its own message
// there: "Failed to fetch", "Load failed", "NetworkError when attempting to
// fetch resource". All three are developer strings with no action in them, so
// every page says the one sentence a reader can act on instead.
export function readFailureMessage(err: unknown, fallback: string): string {
  if (err instanceof TypeError) return NETWORK_UNREACHABLE;
  return err instanceof Error && err.message ? err.message : fallback;
}

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

// Known built-in names. Runtime availability comes from GET /strategies.
export const STRATEGIES = [
  "naive_vector",
  "contextual_vector",
  "qna_pairs",
  "knowledge_graph",
  "lightrag",
  "hybrid",
  "raptor",
  "pageindex",
  "bm25",
  "metadata_filtered",
  "temporal",
  "rerank_vector",
  "qiss",
  "sqr",
  "hyde",
  "multi_query",
  "late_interaction",
  "splade",
  "agentic",
] as const;

export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_LABELS: Record<Strategy, string> = {
  naive_vector: "Naive Vector",
  contextual_vector: "Contextual Vector",
  qna_pairs: "QnA Pairs",
  knowledge_graph: "Knowledge Graph",
  lightrag: "LightRAG",
  hybrid: "Hybrid",
  raptor: "RAPTOR",
  pageindex: "PageIndex",
  bm25: "BM25",
  metadata_filtered: "Metadata Filtered",
  temporal: "Temporal",
  rerank_vector: "Rerank Vector",
  qiss: "QISS (quantum)",
  sqr: "SQR (optional quantum)",
  hyde: "HyDE",
  multi_query: "Multi-Query",
  late_interaction: "Late Interaction",
  splade: "SPLADE",
  agentic: "Agentic (experimental)",
};

export const STRATEGY_COLORS: Record<Strategy, string> = {
  naive_vector: "#64748b",
  contextual_vector: "#3b82f6",
  qna_pairs: "#8b5cf6",
  knowledge_graph: "#22c55e",
  lightrag: "#16a34a",
  hybrid: "#f59e0b",
  raptor: "#ef4444",
  pageindex: "#ec4899",
  bm25: "#0ea5e9",
  metadata_filtered: "#84cc16",
  temporal: "#06b6d4",
  rerank_vector: "#14b8a6",
  qiss: "#6366f1",
  sqr: "#a855f7",
  hyde: "#eab308",
  multi_query: "#06b6d4",
  late_interaction: "#0d9488",
  splade: "#d946ef",
  agentic: "#d946ef",
};

export const TIER_INFO: Record<number, { label: string; description: string }> = {
  1: {
    label: "Lookup",
    description:
      "Single fact retrieval from one document. Example: 'What is the default timeout?'",
  },
  2: {
    label: "How-To",
    description:
      "Step-by-step procedure within one topic. Example: 'How do I enable server-side encryption?'",
  },
  3: {
    label: "Comparison",
    description:
      "Choosing between two options or configurations. Example: 'Compare hot storage vs cold archive for compliance.'",
  },
  4: {
    label: "Integration",
    description:
      "Cross-topic dependencies requiring 3-4 connected components. Example: 'What permissions does service A need for B and C?'",
  },
  5: {
    label: "Architecture",
    description:
      "Full system design spanning 3-5+ topics. Example: 'How does a request flow from ingress through processing to storage?'",
  },
};

export const STRATEGY_DESCRIPTIONS: Record<Strategy, string> = {
  naive_vector:
    "Chunks documents, embeds each chunk, and retrieves top-k by cosine similarity as a dense baseline.",
  contextual_vector:
    "Prepends parent topic context to each chunk before embedding so you can measure whether document context changes retrieval.",
  qna_pairs:
    "Generates question-answer pairs at index time and retrieves against those pairs instead of only source chunks.",
  knowledge_graph:
    "Extracts entities and relationships into Neo4j, then queries the graph through intent-matched Cypher templates.",
  lightrag:
    "Reads the same Neo4j graph two ways: a local entity neighborhood walk and a global community summary, and labels which one produced each chunk.",
  hybrid:
    "Routes by intent between vector and graph paths, then uses reciprocal rank fusion when both paths contribute.",
  raptor:
    "Builds a recursive tree of chunk clusters and summaries, then queries leaf and summary levels together.",
  pageindex:
    "Builds a hierarchical tree from document structure and uses model-guided traversal without an embedding index.",
  bm25:
    "Uses BM25 keyword matching as a keyless lexical baseline with no embeddings or graph service.",
  metadata_filtered:
    "Applies an access filter (tags, owner, classification, doc ID allow-list) inside retrieval, so a restricted chunk never reaches the ranked list.",
  temporal:
    "Prefers each document's newest version and supports an as-of date, so a superseded chunk never outranks its replacement.",
  rerank_vector:
    "Naive Vector retrieves a wide candidate pool, then a cross-encoder reranker (BGE, Cohere, or Voyage) rescores and keeps the top-k for a measured latency-quality tradeoff.",
  qiss:
    "Rescores dense candidates with a pure-NumPy state-fidelity calculation and offers an experimental multi-query mode.",
  sqr:
    "Experimental Qiskit Aer SWAP-test reranker. It is excluded from the default benchmark and needs the optional quantum dependency group.",
  hyde:
    "Asks the model for a hypothetical answer and embeds that instead of the question before retrieving over naive_vector.",
  multi_query:
    "Asks the model for several sub-queries, retrieves each over naive_vector, and fuses the ranked lists with Reciprocal Rank Fusion.",
  late_interaction:
    "ColBERT-style reranker that keeps one embedding per token and scores by MaxSim instead of a single pooled vector. Needs the optional late-interaction dependency group.",
  splade:
    "Expands a query into weighted vocabulary terms and scores against its own sparse term-weight index. Needs the optional splade dependency group.",
  agentic:
    "Retrieves, judges whether the context is enough, and retrieves again with a refined query, under a hard iteration and LLM-call budget. Excluded from the default benchmark because it costs several LLM calls per question.",
};

export interface StrategyCatalogRecord {
  name: Strategy;
  label: string;
  architecture: string;
  default_benchmark: boolean;
  api_supported: boolean;
  experimental: boolean;
  optional_extra: string | null;
  required_modules: string[];
  status: "loaded" | "unavailable" | "unknown";
  unavailable_reason: string | null;
}

// What `--strategies all` covers. A rule such as `name !== "sqr"` drifts the
// moment a strategy leaves the default set for another reason, and this list
// read rerank_vector as a default when the backend excludes it.
export const DEFAULT_BENCHMARK_STRATEGIES: readonly Strategy[] = [
  "naive_vector",
  "contextual_vector",
  "qna_pairs",
  "knowledge_graph",
  "hybrid",
  "raptor",
  "pageindex",
  "bm25",
  "qiss",
] as const;

// This fallback is what the browser shows when `/api/strategies` cannot be
// reached. It mirrors `kb_arena/strategies/catalog.py`, and the parity tests
// fail when the two disagree.
const EXPERIMENTAL_STRATEGIES: readonly Strategy[] = [
  "metadata_filtered",
  "temporal",
  "qiss",
  "sqr",
  "hyde",
  "multi_query",
  "agentic",
  "lightrag",
] as const;

const STRATEGY_EXTRAS: Partial<Record<Strategy, string>> = {
  rerank_vector: "rerank",
  sqr: "quantum",
  late_interaction: "late-interaction",
  splade: "splade",
};

const STRATEGY_REQUIRED_MODULES: Partial<Record<Strategy, string[]>> = {
  rerank_vector: ["sentence_transformers"],
  sqr: ["qiskit", "qiskit_aer", "sklearn"],
  late_interaction: ["transformers", "torch"],
  splade: ["transformers", "torch"],
};

export const DEFAULT_STRATEGY_CATALOG: StrategyCatalogRecord[] = STRATEGIES.map((name) => ({
  name,
  label: STRATEGY_LABELS[name],
  architecture: name === "bm25" ? "lexical" : "retrieval",
  default_benchmark: DEFAULT_BENCHMARK_STRATEGIES.includes(name),
  api_supported: true,
  // A chain of ternaries drifted from the backend every time a strategy
  // landed, and it already misreported rerank_vector's extra as null. These
  // two maps carry the same values `STRATEGY_CATALOG` does, so a reader sees
  // one list per fact instead of a condition per strategy.
  experimental: EXPERIMENTAL_STRATEGIES.includes(name),
  optional_extra: STRATEGY_EXTRAS[name] ?? null,
  required_modules: STRATEGY_REQUIRED_MODULES[name] ?? [],
  status: "unknown",
  unavailable_reason: "Runtime status unavailable.",
}));

export type StrategyGroup = "baseline" | "advanced" | "experimental";

export const STRATEGY_GROUP_ORDER: readonly StrategyGroup[] = [
  "baseline",
  "advanced",
  "experimental",
];

export const STRATEGY_GROUP_LABELS: Record<StrategyGroup, string> = {
  baseline: "Baseline",
  advanced: "Advanced",
  experimental: "Experimental",
};

// A strategy's group comes from the same two catalog facts the backend
// already reports (`experimental`, `optional_extra`), not from a name list.
// A picker keyed off a name list misses the next strategy the day it lands.
export function strategyGroup(
  record: Pick<StrategyCatalogRecord, "experimental" | "optional_extra">
): StrategyGroup {
  if (record.experimental) return "experimental";
  if (record.optional_extra) return "advanced";
  return "baseline";
}

export interface CorpusInfo {
  value: string;
  label: string;
  questionCount?: number;
  reviewedQuestionCount?: number;
  draftQuestionCount?: number;
  /** Question files the server could not parse, so their statuses are unknown. */
  unreadableQuestionFiles?: number;
  hasProcessed?: boolean;
  hasResults?: boolean;
  hasQaPairs?: boolean;
  qaPairCount?: number;
}

export const DEFAULT_CORPORA: CorpusInfo[] = [
  { value: "aws-compute", label: "AWS Compute" },
];

// Kept for backward compatibility with components that do not fetch dynamically.
export const CORPORA = DEFAULT_CORPORA;

export async function fetchStrategyCatalog(): Promise<StrategyCatalogRecord[]> {
  try {
    const res = await fetch(`${API_URL}/strategies`);
    if (!res.ok) return DEFAULT_STRATEGY_CATALOG;
    const data = await res.json();
    return data.catalog?.length ? data.catalog : DEFAULT_STRATEGY_CATALOG;
  } catch {
    return DEFAULT_STRATEGY_CATALOG;
  }
}

// The built-in list names what KB Arena ships, not what this deployment holds.
// A page that acts on a corpus, rather than only naming one, needs to know
// which of the two it received, so this reports the failure and `fetchCorpora`
// keeps the fallback for the pages that only list names.
export async function fetchCorporaResult(): Promise<{
  corpora: CorpusInfo[];
  failed: boolean;
}> {
  try {
    const res = await fetch(`${API_URL}/api/corpora`);
    if (!res.ok) return { corpora: [], failed: true };
    const data = await res.json();
    // A body of another shape reached the page as a corpus list, and the page
    // then mapped over whatever it held. A crash tells the reader less than
    // the failure the page already renders, so this is that failure.
    if (!data || typeof data !== "object") return { corpora: [], failed: true };
    const listed = (data as Record<string, unknown>).corpora;
    if (listed !== undefined && !Array.isArray(listed)) return { corpora: [], failed: true };
    const corpora = (listed ?? []).filter(
      (entry: unknown): entry is CorpusInfo =>
        Boolean(entry) &&
        typeof entry === "object" &&
        typeof (entry as CorpusInfo).value === "string" &&
        typeof (entry as CorpusInfo).label === "string",
    );
    // An empty answer is a deployment that holds no corpus. Handing back the
    // built-in list here reported that deployment as holding the built-in set,
    // and reported it as a success.
    return { corpora, failed: false };
  } catch {
    return { corpora: [], failed: true };
  }
}

export async function fetchCorpora(): Promise<CorpusInfo[]> {
  // A page that only names corpora keeps the built-in list when the read
  // fails, and its banner says the API did not answer. A page that acts on a
  // corpus reads `fetchCorporaResult` and stops instead.
  const result = await fetchCorporaResult();
  return result.failed ? DEFAULT_CORPORA : result.corpora;
}

export interface ServerStatus {
  demoMode: boolean;
  // The app sets demo mode for itself when no model key is configured. That
  // machine is not a hosted demo, and only this flag tells the two apart.
  // null when the answer did not carry it, which is neither of the two.
  demoModeAuto: boolean | null;
  // Whether this browser reached the API over the loopback address. Neither
  // demo flag says where the server runs, and the browser cannot tell.
  // null when the answer did not carry it, which is not "somewhere else".
  callerIsLocal: boolean | null;
}

// A missing flag is an absent answer, not a false one. `Boolean(undefined)`
// turned every gap into a positive claim: an older build that reports no
// locality read as a server on another machine.
function reportedFlag(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

// null means the server did not answer, which is not the same as live mode.
// The demo page shows its read-only banner only on a definite demoMode: true.
export async function fetchServerStatus(): Promise<ServerStatus | null> {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) return null;
    const data = await res.json();
    // A body with no demo flag is not this API's health answer, and reading
    // the gap as false would call it a live deployment.
    if (typeof data?.demo_mode !== "boolean") return null;
    return {
      demoMode: data.demo_mode,
      demoModeAuto: reportedFlag(data.demo_mode_auto),
      callerIsLocal: reportedFlag(data.caller_is_local),
    };
  } catch {
    return null;
  }
}

export const LEADERBOARD_MALFORMED =
  "The leaderboard answer did not carry the rows this page reads.";

// The three status strings `kb_arena/benchmark/review.py` writes into every
// record. A fourth name invented here would count nothing, so the parity test
// fails when the two lists disagree.
export const REVIEW_REVIEWED = "human-reviewed";
export const REVIEW_DRAFT = "machine-assisted-draft";
export const REVIEW_UNSPECIFIED = "unspecified";

// How much of a row rests on questions a reviewer checked. `publishable` is
// true only when every scored question is human-reviewed, so a row with one
// machine draft under it is a development signal and not citable evidence.
export interface RowReview {
  counts: Record<string, number>;
  questions: number;
  reviewed_share: number | null;
  publishable: boolean;
}

export interface LeaderboardRow {
  corpus: string;
  strategy: string;
  // Runs that differ in question set, qrels, judge or top_k never share a row.
  compatibility_key: string;
  build?: string;
  // Optional for the same reason `build` is: a server older than the field
  // still reports real rows, and dropping them would hide measured results.
  review?: RowReview;
  manifest: {
    question_split?: string | null;
    judge_model?: string | null;
    top_k?: number | null;
  };
  mixed_with: string[];
  runs: number;
  mean_accuracy: number | null;
  mean_recall_at_5: number | null;
  mean_ndcg_at_5: number | null;
  mean_cost_usd: number | null;
  mean_latency_ms: number | null;
}

export interface LeaderboardPage {
  corpora: string[];
  leaderboard: LeaderboardRow[];
  filter: string;
}

const strings = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

// A row from a server older than the review summary carries no `review`, and
// that row still holds measured numbers. An absent or malformed summary reads
// as undefined, so the page says the review status is unknown instead of
// claiming a count nobody reported.
function reviewOf(value: unknown): RowReview | undefined {
  if (!value || typeof value !== "object") return undefined;
  const review = value as Record<string, unknown>;
  if (typeof review.questions !== "number" || !Number.isFinite(review.questions)) return undefined;
  if (typeof review.publishable !== "boolean") return undefined;
  const rawCounts = review.counts;
  if (!rawCounts || typeof rawCounts !== "object") return undefined;
  const counts: Record<string, number> = {};
  for (const [status, count] of Object.entries(rawCounts as Record<string, unknown>)) {
    // A count this page drops still leaves `publishable` behind it, so the row
    // printed "a reviewer checked all 10" from a reply whose counts it threw
    // away. A count it cannot read makes the whole review block unusable.
    if (typeof count !== "number" || !Number.isFinite(count)) return undefined;
    counts[status] = count;
  }
  const share = review.reviewed_share;
  return {
    counts,
    questions: review.questions,
    reviewed_share: typeof share === "number" && Number.isFinite(share) ? share : null,
    publishable: review.publishable,
  };
}

// A metric the answer reports as null is a measurement nobody has, and the
// table prints "n/a" for it. A metric the answer never carried is a different
// thing: the row is not the shape this page reads, so undefined drops it.
const metric = (value: unknown): number | null | undefined => {
  if (value === null) return null;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return undefined;
};

/**
 * The one place the leaderboard answer becomes rows.
 *
 * A page that reads fields off an unchecked body renders whatever a wrong
 * answer carries, and a check added at the call site guards one field and
 * misses the next. `kb_arena/benchmark/result_schema.py` learned this on the
 * Python side: one validated read made a class of defects unreachable instead
 * of guarded.
 *
 * A body that is not an object, or that carries no `leaderboard` array, is a
 * failed read.
 *
 * A row must carry every field the table reads: `corpus`, `strategy`,
 * `compatibility_key`, `mixed_with`, `manifest`, `runs`, and the five metrics,
 * where a metric may be null for a measurement nobody has. A row that lacks
 * one is dropped, never completed. Filling a gap with "legacy", zero runs and
 * n/a metrics builds a row the server never sent, and a reader cannot tell it
 * from a measured one. `build` and `review` stay optional, because a server
 * older than either field still reports real rows.
 */
export function parseLeaderboard(body: unknown): LeaderboardPage {
  if (!body || typeof body !== "object") throw new Error(LEADERBOARD_MALFORMED);
  const answer = body as Record<string, unknown>;
  if (!Array.isArray(answer.leaderboard)) throw new Error(LEADERBOARD_MALFORMED);

  const rows: LeaderboardRow[] = [];
  for (const entry of answer.leaderboard) {
    if (!entry || typeof entry !== "object") continue;
    const row = entry as Record<string, unknown>;
    if (typeof row.corpus !== "string" || typeof row.strategy !== "string") continue;
    if (typeof row.compatibility_key !== "string") continue;
    if (!Array.isArray(row.mixed_with)) continue;
    if (typeof row.runs !== "number" || !Number.isFinite(row.runs)) continue;
    if (!row.manifest || typeof row.manifest !== "object") continue;
    const manifest = row.manifest as Record<string, unknown>;
    const metrics = {
      mean_accuracy: metric(row.mean_accuracy),
      mean_recall_at_5: metric(row.mean_recall_at_5),
      mean_ndcg_at_5: metric(row.mean_ndcg_at_5),
      mean_cost_usd: metric(row.mean_cost_usd),
      mean_latency_ms: metric(row.mean_latency_ms),
    };
    if (Object.values(metrics).some((value) => value === undefined)) continue;
    rows.push({
      corpus: row.corpus,
      strategy: row.strategy,
      compatibility_key: row.compatibility_key,
      build: typeof row.build === "string" ? row.build : undefined,
      review: reviewOf(row.review),
      manifest: {
        question_split:
          typeof manifest.question_split === "string" ? manifest.question_split : null,
        judge_model: typeof manifest.judge_model === "string" ? manifest.judge_model : null,
        top_k: metric(manifest.top_k) ?? null,
      },
      mixed_with: strings(row.mixed_with),
      runs: row.runs,
      ...(metrics as {
        mean_accuracy: number | null;
        mean_recall_at_5: number | null;
        mean_ndcg_at_5: number | null;
        mean_cost_usd: number | null;
        mean_latency_ms: number | null;
      }),
    });
  }

  return {
    corpora: strings(answer.corpora),
    leaderboard: rows,
    filter: typeof answer.filter === "string" ? answer.filter : "all",
  };
}

export interface GraphData {
  nodes: { id: string; name: string; type: string; description?: string }[];
  edges: { source: string; target: string; type: string }[];
  connected: boolean;
}

export async function fetchGraphData(corpus: string = "all"): Promise<GraphData> {
  // The route returns entities extracted from the documents, so it carries
  // the API token when one is set.
  const res = await apiFetch(`${API_URL}/api/graph/data?corpus=${corpus}`).catch(() => {
    // A network error reached the same `connected: false` return as a real
    // answer did, so an unreachable API read as an unreachable database.
    throw new Error(GRAPH_UNAVAILABLE);
  });
  if (res.status === 401) throw new Error(GRAPH_UNAUTHORIZED);
  // `connected: false` reads as "the graph database is down", which is a claim
  // about the deployment. A rate limit, a server error and every other failed
  // read are failures to read, and none of them is that claim.
  if (res.status === 429 || res.status >= 500 || !res.ok) {
    throw new Error(GRAPH_UNAVAILABLE);
  }
  return await res.json().catch(() => {
    throw new Error(GRAPH_UNAVAILABLE);
  });
}

export type GraphBuildEvent =
  | { type: "started"; corpus: string; total_sections: number }
  | { type: "entity"; id: string; name: string; nodeType: string }
  | { type: "relationship"; source: string; target: string; relType: string }
  | { type: "section_done"; doc_id: string; entities_count: number; rels_count: number }
  | { type: "complete"; total_entities: number; total_relationships: number }
  | { type: "error"; message: string }
  | { type: "heartbeat" };

export async function triggerGraphBuild(
  corpus: string
): Promise<{ status: string; build_id: string; corpus: string }> {
  const res = await apiFetch(`${API_URL}/api/graph/build`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corpus }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function* streamGraphBuild(
  buildId: string,
  signal?: AbortSignal
): AsyncGenerator<GraphBuildEvent> {
  const maxReconnects = 3;

  for (let attempt = 0; attempt <= maxReconnects; attempt += 1) {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");

    try {
      const response = await apiFetch(`${API_URL}/api/graph/build/stream/${buildId}`, { signal });
      if (!response.ok) {
        yield { type: "error", message: `HTTP ${response.status}` };
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        yield { type: "error", message: "No response body" };
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let eventType = "";
      let dataLine = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, "");
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLine = line.slice(5).trim();
          } else if (line === "" && eventType && dataLine) {
            let event: GraphBuildEvent | undefined;
            try {
              const parsed = JSON.parse(dataLine);
              if (eventType === "entity")
                event = { type: "entity", id: parsed.id, name: parsed.name, nodeType: parsed.type };
              else if (eventType === "relationship")
                event = { type: "relationship", source: parsed.source, target: parsed.target, relType: parsed.type };
              else if (eventType === "started")
                event = { type: "started", corpus: parsed.corpus, total_sections: parsed.total_sections };
              else if (eventType === "section_done")
                event = { type: "section_done", doc_id: parsed.doc_id, entities_count: parsed.entities_count, rels_count: parsed.rels_count };
              else if (eventType === "complete")
                event = { type: "complete", total_entities: parsed.total_entities, total_relationships: parsed.total_relationships };
              else if (eventType === "error")
                event = { type: "error", message: parsed.message };
              else if (eventType === "heartbeat")
                event = { type: "heartbeat" };
            } catch { /* skip malformed SSE */ }
            eventType = "";
            dataLine = "";
            if (event) {
              yield event;
              if (event.type === "complete" || event.type === "error") return;
            }
          }
        }
      }
    } catch (error) {
      if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
        throw error;
      }
      if (attempt === maxReconnects) {
        yield { type: "error", message: "Graph build stream disconnected after 4 attempts." };
        return;
      }
    }

    if (attempt === maxReconnects) {
      yield { type: "error", message: "Graph build stream disconnected after 4 attempts." };
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 250 * (attempt + 1)));
  }
}

// An empty list is a corpus with no recorded run. Every failure throws, because
// sample numbers in place of a failed read put invented results on screen under
// a real corpus name.
export async function fetchBenchmarkResults(
  corpus: string = "all"
): Promise<{ strategy: Strategy; tiers: number[]; latencyMs: number; costUsd: number }[]> {
  // This route returns per-question records, so it carries the API token
  // when one is set. A bare fetch would get 401 on a deployment with a token.
  const res = await apiFetch(`${API_URL}/api/benchmark/results?corpus=${corpus}`).catch(() => {
    throw new Error(BENCHMARK_UNAVAILABLE);
  });
  if (res.status === 401) throw new Error(BENCHMARK_UNAUTHORIZED);
  // A rate limit and a server error are failures to read, not results.
  if (res.status === 429 || res.status >= 500 || !res.ok) {
    throw new Error(BENCHMARK_UNAVAILABLE);
  }
  const data = await res.json().catch(() => {
    throw new Error(BENCHMARK_UNAVAILABLE);
  });
  return data.results ?? [];
}

export interface Source {
  title: string;
  url?: string;
}

export interface ChatStreamResult {
  messageId: string;
  answer: string;
  sources: string[];
  strategyUsed: string;
  latencyMs: number;
  tokensUsed: number;
  costUsd: number;
  mock?: boolean;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export async function* streamChat(
  query: string,
  strategy: Strategy,
  corpus: string,
  history: Message[],
  signal?: AbortSignal
): AsyncGenerator<
  | { type: "token"; text: string }
  | { type: "done"; sources: string[]; strategyUsed: string }
  | { type: "meta"; latencyMs: number; tokensUsed: number; costUsd: number }
  | { type: "error"; message: string }
> {
  const response = await apiFetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, strategy, corpus, history }),
    signal,
  });

  if (!response.ok) {
    yield { type: "error", message: `HTTP ${response.status}` };
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    yield { type: "error", message: "No response body" };
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let eventType = "";
  let dataLine = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const line = rawLine.replace(/\r$/, "");
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLine = line.slice(5).trim();
      } else if (line === "" && eventType && dataLine) {
        try {
          const parsed = JSON.parse(dataLine);
          if (eventType === "token") {
            yield { type: "token", text: parsed.text };
          } else if (eventType === "done") {
            yield {
              type: "done",
              sources: parsed.sources ?? [],
              strategyUsed: parsed.strategy_used ?? "",
            };
          } else if (eventType === "meta") {
            yield {
              type: "meta",
              latencyMs: parsed.latency_ms ?? 0,
              tokensUsed: parsed.tokens_used ?? 0,
              costUsd: parsed.cost_usd ?? 0,
            };
          } else if (eventType === "error") {
            yield { type: "error", message: parsed.message ?? "Unknown error" };
          }
        } catch {
          // Ignore malformed SSE lines.
        }
        eventType = "";
        dataLine = "";
      }
    }
  }
}

