"use client";

import { useState, useRef } from "react";
import ChatPanel from "@/components/ChatPanel";
import { STRATEGIES, STRATEGY_LABELS, CORPORA, type Strategy, type Message } from "@/lib/api";

export default function DemoPage() {
  const [query, setQuery] = useState("");
  const [corpus, setCorpus] = useState("python-stdlib");
  const [selectedStrategies, setSelectedStrategies] = useState<Strategy[]>([...STRATEGIES]);
  const [trigger, setTrigger] = useState(0);
  const [history, setHistory] = useState<Message[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setHistory((prev) => [...prev, { role: "user", content: q }]);
    setTrigger((t) => t + 1);
  }

  function toggleStrategy(s: Strategy) {
    setSelectedStrategies((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  }

  function handleClear() {
    setQuery("");
    setHistory([]);
    setTrigger(0);
    inputRef.current?.focus();
  }

  const gridCols =
    selectedStrategies.length <= 2
      ? "grid-cols-1 sm:grid-cols-2"
      : selectedStrategies.length <= 3
      ? "grid-cols-1 sm:grid-cols-3"
      : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Strategy comparison
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Ask a question and see how each retrieval strategy responds side-by-side.
        </p>
      </div>

      {/* Controls */}
      <div className="space-y-3">
        {/* Strategy toggles */}
        <div className="flex flex-wrap gap-2">
          {STRATEGIES.map((s) => {
            const active = selectedStrategies.includes(s);
            return (
              <button
                key={s}
                onClick={() => toggleStrategy(s)}
                className="text-xs px-3 py-1.5 rounded-lg border transition-all"
                style={{
                  borderColor: active ? "var(--accent)" : "var(--border)",
                  background: active ? "var(--accent)" : "transparent",
                  color: active ? "#fff" : "var(--muted)",
                  opacity: active ? 1 : 0.6,
                }}
              >
                {STRATEGY_LABELS[s]}
              </button>
            );
          })}
        </div>

        {/* Query input */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <select
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            className="px-3 py-2 rounded-lg border text-sm shrink-0"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            {CORPORA.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the documentation..."
            className="flex-1 px-4 py-2 rounded-lg border text-sm outline-none"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          />
          <button
            type="submit"
            disabled={!query.trim() || selectedStrategies.length === 0}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Ask
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="px-3 py-2 rounded-lg border text-sm transition-opacity hover:opacity-70"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            Clear
          </button>
        </form>

        {/* Example queries */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs" style={{ color: "var(--muted)" }}>Try:</span>
          {[
            "How do I expose a StatefulSet through an Ingress with TLS?",
            "Compare ConfigMap vs Secret vs ServiceAccount in Kubernetes",
            "Which executives serve on boards of companies with material litigation?",
            "What's the chain of modules when urllib makes an HTTPS connection?",
          ].map((q) => (
            <button
              key={q}
              onClick={() => { setQuery(q); inputRef.current?.focus(); }}
              className="text-xs px-2.5 py-1 rounded-lg border transition-opacity hover:opacity-70 text-left"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              {q.length > 50 ? q.slice(0, 50) + "..." : q}
            </button>
          ))}
        </div>
      </div>

      {/* Chat panels grid */}
      {selectedStrategies.length === 0 ? (
        <div className="text-center py-16">
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            Select at least one strategy above to begin.
          </p>
        </div>
      ) : (
        <div className={`grid ${gridCols} gap-4`}>
          {selectedStrategies.map((s) => (
            <ChatPanel
              key={s}
              strategy={s}
              query={query}
              corpus={corpus}
              history={history}
              trigger={trigger}
            />
          ))}
        </div>
      )}
    </div>
  );
}
