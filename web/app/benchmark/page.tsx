"use client";

import { useState } from "react";
import BenchmarkTable from "@/components/BenchmarkTable";
import TierChart from "@/components/TierChart";
import { MOCK_BENCHMARK_DATA, CORPORA } from "@/lib/api";

type ViewMode = "table" | "chart" | "both";

export default function BenchmarkPage() {
  const [corpus, setCorpus] = useState("all");
  const [view, setView] = useState<ViewMode>("both");

  // In a real deployment, this would fetch from /api/benchmark/results
  // For now, use mock data with a clear label
  const rows = MOCK_BENCHMARK_DATA;

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Benchmark results
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Accuracy by tier, latency, and cost across all 5 retrieval strategies.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>Corpus</label>
          <select
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            className="px-3 py-1.5 rounded-lg border text-sm"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            <option value="all">All corpora</option>
            {CORPORA.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        <div className="flex rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
          {(["table", "chart", "both"] as ViewMode[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className="px-3 py-1.5 text-xs font-medium capitalize transition-colors"
              style={{
                background: view === v ? "var(--accent)" : "transparent",
                color: view === v ? "#fff" : "var(--muted)",
              }}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="space-y-8">
        {(view === "table" || view === "both") && (
          <BenchmarkTable rows={rows} />
        )}

        {(view === "chart" || view === "both") && (
          <div
            className="rounded-lg border p-4"
            style={{ borderColor: "var(--border)", background: "var(--card)" }}
          >
            <h2 className="text-sm font-semibold mb-4" style={{ color: "var(--foreground)" }}>
              Accuracy by tier
            </h2>
            <TierChart rows={rows} />
          </div>
        )}
      </div>

      {/* Methodology note */}
      <div
        className="rounded-lg border p-4 space-y-2"
        style={{ borderColor: "var(--border)", background: "var(--card)" }}
      >
        <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>Methodology</h3>
        <div className="text-xs leading-relaxed space-y-1" style={{ color: "var(--muted)" }}>
          <p>
            Each question is sent to all 5 strategies. Answers are evaluated through a 4-pass pipeline:
            structural checks (must_mention / must_not_claim), entity coverage, source attribution,
            and LLM-as-judge scoring for accuracy, completeness, and faithfulness.
          </p>
          <p>
            Composite ranking: 0.5 * accuracy + 0.3 * reliability + 0.2 * latency_score.
            Latency score inverts p95 so lower is better.
          </p>
          <p>
            Tiers: 1 = factoid (single lookup), 2 = procedural (how-to), 3 = comparative (A vs B),
            4 = relational (requires joins), 5 = multi-hop (3+ concepts, cross-module).
          </p>
        </div>
      </div>
    </div>
  );
}
