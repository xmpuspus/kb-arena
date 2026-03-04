import { API_URL } from "./api";

// ── Types ──

export interface QaPair {
  question: string;
  answer: string;
  source_id: string;
  section_id: string;
  section_ref: string;
}

export interface QuestionResult {
  question_text: string;
  accuracy: number;
  completeness: number;
  answer_snippet: string;
}

export interface SectionAuditResult {
  section_index: number;
  total: number;
  section_id: string;
  section_title: string;
  doc_id: string;
  heading_path: string[];
  questions_tested: number;
  avg_accuracy: number;
  worst_question: string;
  worst_accuracy: number;
  classification: "strong" | "weak" | "gap";
  question_results: QuestionResult[];
}

export interface FixRecommendation {
  fix_index: number;
  total_fixes: number;
  priority: number;
  section_title: string;
  doc_id: string;
  diagnosis: string;
  suggested_content: string;
  placement: string;
  estimated_impact: string;
  failing_questions: string[];
  current_accuracy: number;
}

// ── Event types ──

export type GenerateEvent =
  | { type: "started"; total_sections: number }
  | { type: "progress"; section_index: number; total: number; doc_id: string; section_title: string }
  | { type: "pair"; pair: QaPair }
  | { type: "complete"; total_pairs: number; output_path: string }
  | { type: "error"; message: string };

export type AuditEvent =
  | { type: "started"; total_sections: number }
  | { type: "section_result"; result: SectionAuditResult }
  | { type: "complete"; strong: number; weak: number; gaps: number; total_questions: number }
  | { type: "error"; message: string };

export type FixEvent =
  | { type: "phase"; phase: "audit" | "fix" }
  | { type: "audit_progress"; section_index: number; total: number; section_title: string }
  | { type: "audit_complete"; strong: number; weak: number; gaps: number; total_questions: number }
  | { type: "fix_result"; recommendation: FixRecommendation }
  | { type: "complete" }
  | { type: "error"; message: string };

// ── SSE helper (same pattern as streamChat in api.ts) ──

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function* parseSSE<T>(response: Response, eventMap: Record<string, (data: any) => T | null>): AsyncGenerator<T> {
  const reader = response.body?.getReader();
  if (!reader) return;

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
          const mapper = eventMap[eventType];
          if (mapper) {
            const event = mapper(parsed);
            if (event) yield event;
          }
        } catch {
          // malformed SSE line
        }
        eventType = "";
        dataLine = "";
      }
    }
  }
}

// ── Generate Q&A ──

export async function* streamGenerate(corpus: string, signal?: AbortSignal): AsyncGenerator<GenerateEvent> {
  const response = await fetch(`${API_URL}/api/tools/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corpus }),
    signal,
  });

  if (!response.ok) {
    yield { type: "error", message: `HTTP ${response.status}` };
    return;
  }

  yield* parseSSE<GenerateEvent>(response, {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    started: (d: any) => ({ type: "started", total_sections: d.total_sections }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    progress: (d: any) => ({ type: "progress", section_index: d.section_index, total: d.total, doc_id: d.doc_id, section_title: d.section_title }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    pair: (d: any) => ({ type: "pair", pair: d }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    complete: (d: any) => ({ type: "complete", total_pairs: d.total_pairs, output_path: d.output_path }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    error: (d: any) => ({ type: "error", message: d.message ?? "Unknown error" }),
  });
}

// ── Audit ──

export async function* streamAudit(corpus: string, maxSections: number, signal?: AbortSignal): AsyncGenerator<AuditEvent> {
  const response = await fetch(`${API_URL}/api/tools/audit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corpus, max_sections: maxSections }),
    signal,
  });

  if (!response.ok) {
    yield { type: "error", message: `HTTP ${response.status}` };
    return;
  }

  yield* parseSSE<AuditEvent>(response, {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    started: (d: any) => ({ type: "started", total_sections: d.total_sections }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    section_result: (d: any) => ({ type: "section_result", result: d }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    complete: (d: any) => ({ type: "complete", strong: d.strong, weak: d.weak, gaps: d.gaps, total_questions: d.total_questions }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    error: (d: any) => ({ type: "error", message: d.message ?? "Unknown error" }),
  });
}

// ── Fix ──

export async function* streamFix(corpus: string, maxSections: number, maxFixes: number, signal?: AbortSignal): AsyncGenerator<FixEvent> {
  const response = await fetch(`${API_URL}/api/tools/fix`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corpus, max_sections: maxSections, max_fixes: maxFixes }),
    signal,
  });

  if (!response.ok) {
    yield { type: "error", message: `HTTP ${response.status}` };
    return;
  }

  yield* parseSSE<FixEvent>(response, {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    phase: (d: any) => ({ type: "phase", phase: d.phase }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    audit_progress: (d: any) => ({ type: "audit_progress", section_index: d.section_index, total: d.total, section_title: d.section_title }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    audit_complete: (d: any) => ({ type: "audit_complete", strong: d.strong, weak: d.weak, gaps: d.gaps, total_questions: d.total_questions }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    fix_result: (d: any) => ({ type: "fix_result", recommendation: d }),
    complete: () => ({ type: "complete" as const }),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    error: (d: any) => ({ type: "error", message: d.message ?? "Unknown error" }),
  });
}

// ── REST ──

export async function fetchQaPairs(corpus: string): Promise<{ pairs: QaPair[]; total: number }> {
  try {
    const res = await fetch(`${API_URL}/api/tools/qa-pairs?corpus=${corpus}`);
    if (!res.ok) return { pairs: [], total: 0 };
    return await res.json();
  } catch {
    return { pairs: [], total: 0 };
  }
}
