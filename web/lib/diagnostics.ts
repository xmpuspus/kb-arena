import { API_URL } from "./api";

/**
 * The reads behind the diagnostics page, each one able to report its failure.
 *
 * `fetchStrategyCatalog` and `fetchCorpora` hand back the built-in list when a
 * read fails, which is the right answer for a page that only names strategies.
 * A diagnostics page that did the same would say this deployment loaded the
 * built-in set, which is a claim about a server that never answered. Every
 * read here reports the failure instead.
 */
export type Read<T> = { ok: true; value: T } | { ok: false; reason: string };

export const READ_FAILED = "The API did not answer this read.";
export const WRONG_SHAPE = "The answer did not carry the fields this page reads.";

// A missing flag is an absent answer, not a false one. The diagnostics page
// prints "not reported" for null and never turns a gap into a capability.
function flag(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

export interface HealthReport {
  version: string | null;
  neo4jConnected: boolean | null;
  neo4jUri: string | null;
  neo4jLastError: string | null;
  llmProvider: string | null;
  llmConfigured: boolean | null;
  llmAvailable: boolean | null;
  arenaAvailable: boolean | null;
  arenaLastError: string | null;
  loadedStrategies: string[];
  demoMode: boolean;
  demoModeAuto: boolean | null;
  callerIsLocal: boolean | null;
}

export function parseHealth(body: unknown): Read<HealthReport> {
  if (!body || typeof body !== "object") return { ok: false, reason: WRONG_SHAPE };
  const data = body as Record<string, unknown>;
  // The same gate the state banner uses. A body with no demo flag is not this
  // API's health answer, and reading the gap as false would call it live.
  if (typeof data.demo_mode !== "boolean") return { ok: false, reason: WRONG_SHAPE };
  const neo4j = (data.neo4j ?? {}) as Record<string, unknown>;
  const llm = (data.llm ?? {}) as Record<string, unknown>;
  const arena = (data.arena ?? {}) as Record<string, unknown>;
  const strategies = Array.isArray(data.strategies)
    ? data.strategies.filter((name): name is string => typeof name === "string")
    : [];
  return {
    ok: true,
    value: {
      version: text(data.version),
      neo4jConnected: flag(neo4j.connected),
      neo4jUri: text(neo4j.uri),
      neo4jLastError: text(neo4j.last_error),
      llmProvider: text(llm.provider),
      llmConfigured: flag(llm.configured),
      llmAvailable: flag(llm.available),
      arenaAvailable: flag(arena.available),
      arenaLastError: text(arena.last_error),
      loadedStrategies: strategies,
      demoMode: data.demo_mode,
      demoModeAuto: flag(data.demo_mode_auto),
      callerIsLocal: flag(data.caller_is_local),
    },
  };
}

export async function readHealth(signal?: AbortSignal): Promise<Read<HealthReport>> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal });
    if (!res.ok) return { ok: false, reason: `The server answered ${res.status}.` };
    return parseHealth(await res.json());
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    return { ok: false, reason: READ_FAILED };
  }
}

export interface StrategyStatus {
  name: string;
  label: string;
  status: "loaded" | "unavailable" | "unknown";
  unavailableReason: string | null;
  optionalExtra: string | null;
  experimental: boolean;
}

export function parseStrategies(body: unknown): Read<StrategyStatus[]> {
  if (!body || typeof body !== "object") return { ok: false, reason: WRONG_SHAPE };
  const data = body as Record<string, unknown>;
  if (!Array.isArray(data.catalog)) return { ok: false, reason: WRONG_SHAPE };
  const records: StrategyStatus[] = [];
  for (const entry of data.catalog) {
    if (!entry || typeof entry !== "object") continue;
    const record = entry as Record<string, unknown>;
    if (typeof record.name !== "string") continue;
    // A record with no status says nothing about this deployment. Filling it
    // with "unknown" here would invent the one fact the page is asked for.
    if (record.status !== "loaded" && record.status !== "unavailable") continue;
    records.push({
      name: record.name,
      label: typeof record.label === "string" ? record.label : record.name,
      status: record.status,
      unavailableReason: text(record.unavailable_reason),
      optionalExtra: text(record.optional_extra),
      experimental: record.experimental === true,
    });
  }
  return { ok: true, value: records };
}

export async function readStrategies(signal?: AbortSignal): Promise<Read<StrategyStatus[]>> {
  try {
    const res = await fetch(`${API_URL}/strategies`, { signal });
    if (!res.ok) return { ok: false, reason: `The server answered ${res.status}.` };
    return parseStrategies(await res.json());
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    return { ok: false, reason: READ_FAILED };
  }
}
