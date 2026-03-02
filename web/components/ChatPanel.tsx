"use client";

import { useEffect, useRef, useState } from "react";
import { streamChat, STRATEGY_LABELS, type Message, type Strategy } from "@/lib/api";

type PanelState = "idle" | "loading" | "streaming" | "complete" | "error";

interface Props {
  strategy: Strategy;
  query: string;
  corpus: string;
  history: Message[];
  trigger: number; // increment to fire a new query
}

interface Result {
  answer: string;
  sources: string[];
  latencyMs: number;
  tokensUsed: number;
  costUsd: number;
}

export default function ChatPanel({ strategy, query, corpus, history, trigger }: Props) {
  const [state, setState] = useState<PanelState>("idle");
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (trigger === 0 || !query) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState("loading");
    setAnswer("");
    setResult(null);
    setError("");

    let accumulated = "";
    let sources: string[] = [];
    let latencyMs = 0;
    let tokensUsed = 0;
    let costUsd = 0;

    async function run() {
      try {
        for await (const event of streamChat(query, strategy, corpus, history, ctrl.signal)) {
          if (event.type === "token") {
            accumulated += event.text;
            setAnswer(accumulated);
            setState("streaming");
          } else if (event.type === "done") {
            sources = event.sources;
          } else if (event.type === "meta") {
            latencyMs = event.latencyMs;
            tokensUsed = event.tokensUsed;
            costUsd = event.costUsd;
          } else if (event.type === "error") {
            setError(event.message);
            setState("error");
            return;
          }
        }
        setResult({ answer: accumulated, sources, latencyMs, tokensUsed, costUsd });
        setState("complete");
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message ?? "Unknown error");
        setState("error");
      }
    }

    run();
    return () => ctrl.abort();
  }, [trigger]); // eslint-disable-line react-hooks/exhaustive-deps

  const label = STRATEGY_LABELS[strategy];

  const accent =
    strategy === "knowledge_graph"
      ? "#22c55e"
      : strategy === "hybrid"
      ? "#f59e0b"
      : strategy === "qna_pairs"
      ? "#8b5cf6"
      : strategy === "contextual_vector"
      ? "#3b82f6"
      : "#64748b";

  return (
    <div
      className="flex flex-col rounded-lg border overflow-hidden h-full min-h-[400px]"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center gap-2 border-b text-xs font-semibold uppercase tracking-wider"
        style={{ borderColor: "var(--border)", color: accent }}
      >
        <span
          className="w-2 h-2 rounded-full"
          style={{ background: accent }}
        />
        {label}
      </div>

      {/* Body */}
      <div className="flex-1 p-3 overflow-y-auto">
        {state === "idle" && (
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            Enter a query above to compare strategies.
          </p>
        )}

        {state === "loading" && (
          <div className="flex items-center gap-2" style={{ color: "var(--muted)" }}>
            <Spinner />
            <span className="text-xs">Querying...</span>
          </div>
        )}

        {(state === "streaming" || state === "complete") && (
          <div>
            <p className="text-sm whitespace-pre-wrap leading-relaxed" style={{ color: "var(--foreground)" }}>
              {answer}
              {state === "streaming" && (
                <span
                  className="inline-block w-0.5 h-4 ml-0.5 animate-pulse"
                  style={{ background: accent, verticalAlign: "text-bottom" }}
                />
              )}
            </p>
          </div>
        )}

        {state === "error" && (
          <div>
            <p className="text-xs mb-2" style={{ color: "var(--danger)" }}>
              {error || "An error occurred"}
            </p>
            <button
              onClick={() => setState("idle")}
              className="text-xs px-2 py-1 rounded border transition-colors hover:opacity-70"
              style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
            >
              Dismiss
            </button>
          </div>
        )}
      </div>

      {/* Footer — metrics + sources */}
      {state === "complete" && result && (
        <div className="px-3 py-2 border-t space-y-2" style={{ borderColor: "var(--border)" }}>
          <div className="flex flex-wrap gap-1.5">
            <Badge label={`${result.latencyMs.toFixed(0)} ms`} />
            <Badge label={`${result.tokensUsed} tok`} />
            <Badge label={`$${result.costUsd.toFixed(5)}`} />
          </div>
          {result.sources.length > 0 && (
            <div>
              <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>
                Sources
              </p>
              <ul className="space-y-0.5">
                {result.sources.slice(0, 5).map((src, i) => (
                  <li key={i} className="text-xs truncate" style={{ color: "var(--accent)" }}>
                    {src.startsWith("http") ? (
                      <a href={src} target="_blank" rel="noopener noreferrer" className="hover:underline">
                        {src}
                      </a>
                    ) : (
                      src
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Badge({ label }: { label: string }) {
  return (
    <span
      className="mono text-xs px-2 py-0.5 rounded"
      style={{ background: "var(--border)", color: "var(--muted)" }}
    >
      {label}
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-4 w-4"
      style={{ color: "var(--muted)" }}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}
