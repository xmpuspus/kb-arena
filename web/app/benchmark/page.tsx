"use client";

import { useState, useEffect } from "react";
import BenchmarkTable from "@/components/BenchmarkTable";
import TierChart from "@/components/TierChart";
import StrategyCompare from "@/components/StrategyCompare";
import StrategyRunPanel from "@/components/StrategyRunPanel";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";
import {
  BENCHMARK_UNAVAILABLE,
  CORPORA,
  fetchBenchmarkResults,
  fetchCorpora,
} from "@/lib/api";
import { useTokenEpoch } from "@/lib/useTokenEpoch";

type ViewMode = "table" | "chart" | "both" | "compare";
type Row = Awaited<ReturnType<typeof fetchBenchmarkResults>>[number];

export default function BenchmarkPage() {
  // A saved token must retry the read it was entered for.
  const tokenEpoch = useTokenEpoch();
  const [corpus, setCorpus] = useState("all");
  const [view, setView] = useState<ViewMode>("both");
  const [rows, setRows] = useState<Row[]>([]);
  const [status, setStatus] = useState<"loading" | "ok" | "refused">("loading");
  const [failure, setFailure] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [corpora, setCorpora] = useState(CORPORA);

  useEffect(() => {
    fetchCorpora().then(setCorpora);
  }, []);

  useEffect(() => {
    let active = true;
    setStatus("loading");
    fetchBenchmarkResults(corpus)
      .then((data) => {
        if (!active) return;
        setRows(data);
        setStatus("ok");
      })
      .catch((err: unknown) => {
        // The read failed. Sample rows under a real corpus name would read as
        // that corpus's results, so the table goes and the reason takes its
        // place.
        if (!active) return;
        setRows([]);
        setFailure(err instanceof Error ? err.message : BENCHMARK_UNAVAILABLE);
        setStatus("refused");
      });
    return () => {
      active = false;
    };
  }, [corpus, attempt, tokenEpoch]);

  const corpusLabel = corpora.find((c) => c.value === corpus)?.label ?? "all corpora";
  const runCommand = `kb-arena benchmark --corpus ${corpus === "all" ? "aws-compute" : corpus}`;

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <StateBanner />

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Benchmark results
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Accuracy by tier, latency, and cost for the strategies recorded in each run.
        </p>
      </div>

      <StrategyRunPanel corpora={corpora} />

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="benchmark-corpus" className="text-xs font-medium" style={{ color: "var(--muted)" }}>Corpus</label>
          <select
            id="benchmark-corpus"
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            className="px-3 py-1.5 rounded-lg border text-sm"
            style={{ background: "var(--card)", borderColor: "var(--border-strong)", color: "var(--foreground)" }}
          >
            <option value="all">All corpora</option>
            {corpora.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        <div
          className="flex rounded-lg border overflow-hidden"
          style={{ borderColor: "var(--border-strong)" }}
          role="group"
          aria-label="Result view"
        >
          {(["table", "chart", "both", "compare"] as ViewMode[]).map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => setView(v)}
              // The pressed state carried colour alone, which a reader who
              // cannot tell the two apart never sees.
              aria-pressed={view === v}
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

        {status === "loading" && (
          <span className="text-xs px-2 py-1" style={{ color: "var(--muted)" }} role="status">
            Reading the recorded runs...
          </span>
        )}
      </div>

      {/* Content */}
      {status === "refused" && (
        <FetchError
          title="The benchmark results did not load"
          message={failure}
          hint="Sample rows under a real corpus name would read as that corpus's results, so the table stays empty. Start the API, or enter an API token with the key button, then try again."
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

      {status === "ok" && rows.length === 0 && (
        <div
          className="rounded-lg border border-dashed p-6 text-sm"
          style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        >
          No recorded run for {corpusLabel}. Run <code className="mono">{runCommand}</code> to
          record one. This page shows measured runs only.
        </div>
      )}

      {status === "ok" && rows.length > 0 && (
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

          {view === "compare" && (
            <StrategyCompare rows={rows} />
          )}
        </div>
      )}

      {/* Methodology note */}
      <div
        className="rounded-lg border p-4 space-y-2"
        style={{ borderColor: "var(--border)", background: "var(--card)" }}
      >
        <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>Methodology</h3>
        <div className="text-xs leading-relaxed space-y-1" style={{ color: "var(--muted)" }}>
          <p>
            Each question is sent to the strategies recorded in the run. Answers are evaluated through a 4-pass pipeline:
            structural checks (must_mention / must_not_claim), entity coverage against source documentation,
            source attribution, and LLM-as-judge scoring for accuracy, completeness, and faithfulness.
          </p>
          <p>
            Composite ranking: 0.5 * accuracy + 0.3 * reliability + 0.2 * latency_score.
            Latency score inverts p95 so lower is better.
          </p>
          <p>
            Tiers: 1 = lookup (single fact retrieval), 2 = how-to (procedure within one topic),
            3 = comparison (option A vs B), 4 = integration (cross-topic dependencies),
            5 = architecture (3+ topics, system design).
          </p>
        </div>
      </div>
    </div>
  );
}
