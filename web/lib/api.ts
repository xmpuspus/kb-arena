export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const STRATEGIES = [
  "naive_vector",
  "contextual_vector",
  "qna_pairs",
  "knowledge_graph",
  "hybrid",
] as const;

export type Strategy = (typeof STRATEGIES)[number];

export const STRATEGY_LABELS: Record<Strategy, string> = {
  naive_vector: "Naive Vector",
  contextual_vector: "Contextual Vector",
  qna_pairs: "QnA Pairs",
  knowledge_graph: "Knowledge Graph",
  hybrid: "Hybrid",
};

export const STRATEGY_COLORS: Record<Strategy, string> = {
  naive_vector: "#64748b",
  contextual_vector: "#3b82f6",
  qna_pairs: "#8b5cf6",
  knowledge_graph: "#22c55e",
  hybrid: "#f59e0b",
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
      "Cross-topic dependencies requiring 3\u20134 connected components. Example: 'What permissions does service A need for B and C?'",
  },
  5: {
    label: "Architecture",
    description:
      "Full system design spanning 3\u20135+ topics. Example: 'How does a request flow from ingress through processing to storage?'",
  },
};

export const STRATEGY_DESCRIPTIONS: Record<Strategy, string> = {
  naive_vector:
    "Chunks documents, embeds with text-embedding-3-large, retrieves top-k by cosine similarity. Fast and simple, but no cross-topic understanding.",
  contextual_vector:
    "Same as Naive Vector, but prepends parent topic context to each chunk before embedding. Better at disambiguating domain-specific terms.",
  qna_pairs:
    "LLM pre-generates question-answer pairs from each doc page at index time. Direct question-to-answer matching, but misses novel cross-topic questions.",
  knowledge_graph:
    "Extracts entities and relationships into Neo4j. Queries via Cypher templates. Excels at multi-hop dependency chains.",
  hybrid:
    "Routes by intent \u2014 vector path for lookups, graph path for integration queries, both paths fused via RRF for how-to questions.",
};

export const DEFAULT_CORPORA = [
  { value: "aws-compute", label: "AWS Compute" },
  { value: "aws-storage", label: "AWS Storage" },
  { value: "aws-networking", label: "AWS Networking" },
];

// Kept for backward compatibility — components that don't fetch dynamically
export const CORPORA = DEFAULT_CORPORA;

export async function fetchCorpora(): Promise<{ value: string; label: string }[]> {
  try {
    const res = await fetch(`${API_URL}/api/corpora`);
    if (!res.ok) return DEFAULT_CORPORA;
    const data = await res.json();
    return data.corpora?.length ? data.corpora : DEFAULT_CORPORA;
  } catch {
    return DEFAULT_CORPORA;
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
  const response = await fetch(`${API_URL}/chat/stream`, {
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
          // malformed SSE line — skip
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
    strategy: "naive_vector" as Strategy,
    tiers: [72, 61, 48, 35, 22],
    latencyMs: 420,
    costUsd: 0.0012,
  },
  {
    strategy: "contextual_vector" as Strategy,
    tiers: [78, 69, 55, 41, 28],
    latencyMs: 510,
    costUsd: 0.0018,
  },
  {
    strategy: "qna_pairs" as Strategy,
    tiers: [82, 74, 61, 44, 30],
    latencyMs: 680,
    costUsd: 0.0024,
  },
  {
    strategy: "knowledge_graph" as Strategy,
    tiers: [85, 82, 78, 71, 62],
    latencyMs: 890,
    costUsd: 0.0031,
  },
  {
    strategy: "hybrid" as Strategy,
    tiers: [88, 85, 81, 74, 65],
    latencyMs: 1050,
    costUsd: 0.0038,
  },
];
