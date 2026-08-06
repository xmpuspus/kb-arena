import { apiFetch } from "./auth";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

// Known built-in names. Runtime availability comes from GET /strategies.
export const STRATEGIES = [
  "naive_vector",
  "contextual_vector",
  "qna_pairs",
  "knowledge_graph",
  "hybrid",
  "raptor",
  "pageindex",
  "bm25",
  "rerank_vector",
  "qiss",
  "sqr",
] as const;

export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_LABELS: Record<Strategy, string> = {
  naive_vector: "Naive Vector",
  contextual_vector: "Contextual Vector",
  qna_pairs: "QnA Pairs",
  knowledge_graph: "Knowledge Graph",
  hybrid: "Hybrid",
  raptor: "RAPTOR",
  pageindex: "PageIndex",
  bm25: "BM25",
  rerank_vector: "Rerank Vector",
  qiss: "QISS (quantum)",
  sqr: "SQR (optional quantum)",
};

export const STRATEGY_COLORS: Record<Strategy, string> = {
  naive_vector: "#64748b",
  contextual_vector: "#3b82f6",
  qna_pairs: "#8b5cf6",
  knowledge_graph: "#22c55e",
  hybrid: "#f59e0b",
  raptor: "#ef4444",
  pageindex: "#ec4899",
  bm25: "#0ea5e9",
  rerank_vector: "#14b8a6",
  qiss: "#6366f1",
  sqr: "#a855f7",
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
  hybrid:
    "Routes by intent between vector and graph paths, then uses reciprocal rank fusion when both paths contribute.",
  raptor:
    "Builds a recursive tree of chunk clusters and summaries, then queries leaf and summary levels together.",
  pageindex:
    "Builds a hierarchical tree from document structure and uses model-guided traversal without an embedding index.",
  bm25:
    "Uses BM25 keyword matching as a keyless lexical baseline with no embeddings or graph service.",
  rerank_vector:
    "Naive Vector retrieves a wide candidate pool, then a cross-encoder reranker (BGE, Cohere, or Voyage) rescores and keeps the top-k for a measured latency-quality tradeoff.",
  qiss:
    "Rescores dense candidates with a pure-NumPy state-fidelity calculation and offers an experimental multi-query mode.",
  sqr:
    "Experimental Qiskit Aer SWAP-test reranker. It is excluded from the default benchmark and needs the optional quantum dependency group.",
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

export const DEFAULT_STRATEGY_CATALOG: StrategyCatalogRecord[] = STRATEGIES.map((name) => ({
  name,
  label: STRATEGY_LABELS[name],
  architecture: name === "bm25" ? "lexical" : "retrieval",
  default_benchmark: name !== "sqr",
  api_supported: true,
  experimental: name === "qiss" || name === "sqr",
  optional_extra: name === "sqr" ? "quantum" : null,
  required_modules: name === "sqr" ? ["qiskit", "qiskit_aer", "sklearn"] : [],
  status: "unknown",
  unavailable_reason: "Runtime status unavailable.",
}));

export interface CorpusInfo {
  value: string;
  label: string;
  questionCount?: number;
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

export async function fetchCorpora(): Promise<CorpusInfo[]> {
  try {
    const res = await fetch(`${API_URL}/api/corpora`);
    if (!res.ok) return DEFAULT_CORPORA;
    const data = await res.json();
    return data.corpora?.length ? data.corpora : DEFAULT_CORPORA;
  } catch {
    return DEFAULT_CORPORA;
  }
}

export interface GraphData {
  nodes: { id: string; name: string; type: string; description?: string }[];
  edges: { source: string; target: string; type: string }[];
  connected: boolean;
}

export async function fetchGraphData(corpus: string = "all"): Promise<GraphData> {
  try {
    const res = await fetch(`${API_URL}/api/graph/data?corpus=${corpus}`);
    if (!res.ok) return { nodes: [], edges: [], connected: false };
    return await res.json();
  } catch {
    return { nodes: [], edges: [], connected: false };
  }
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

export async function fetchBenchmarkResults(
  corpus: string = "all"
): Promise<{ strategy: Strategy; tiers: number[]; latencyMs: number; costUsd: number }[]> {
  try {
    const res = await fetch(`${API_URL}/api/benchmark/results?corpus=${corpus}`);
    if (!res.ok) return MOCK_BENCHMARK_DATA;
    const data = await res.json();
    return data.results?.length ? data.results : MOCK_BENCHMARK_DATA;
  } catch {
    return MOCK_BENCHMARK_DATA;
  }
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

// Mock data used when API is unavailable
export const MOCK_BENCHMARK_DATA = [
  {
    strategy: "qna_pairs" as Strategy,
    tiers: [79, 85, 83, 84, 66],
    latencyMs: 9043,
    costUsd: 0.48,
  },
  {
    strategy: "knowledge_graph" as Strategy,
    tiers: [72, 69, 61, 77, 79],
    latencyMs: 20322,
    costUsd: 1.37,
  },
  {
    strategy: "hybrid" as Strategy,
    tiers: [39, 81, 61, 80, 62],
    latencyMs: 41549,
    costUsd: 3.02,
  },
  {
    strategy: "raptor" as Strategy,
    tiers: [30, 16, 15, 36, 30],
    latencyMs: 7240,
    costUsd: 0.69,
  },
  {
    strategy: "naive_vector" as Strategy,
    tiers: [27, 15, 14, 26, 22],
    latencyMs: 6421,
    costUsd: 0.33,
  },
  {
    strategy: "contextual_vector" as Strategy,
    tiers: [25, 11, 9, 26, 11],
    latencyMs: 5114,
    costUsd: 0.29,
  },
  {
    strategy: "pageindex" as Strategy,
    tiers: [19, 12, 7, 21, 12],
    latencyMs: 10933,
    costUsd: 0.29,
  },
  {
    strategy: "bm25" as Strategy,
    tiers: [24, 11, 9, 16, 10],
    latencyMs: 4514,
    costUsd: 0.26,
  },
];
