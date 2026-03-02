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

export const CORPORA = [
  { value: "aws-compute", label: "AWS Compute" },
  { value: "aws-storage", label: "AWS Storage" },
  { value: "aws-networking", label: "AWS Networking" },
];

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

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    let eventType = "";
    let dataLine = "";

    for (const line of lines) {
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
