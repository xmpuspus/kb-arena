"use client";

import { useState, useEffect, useMemo } from "react";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";
import MetricNote from "@/components/MetricNote";
import RowProvenance from "@/components/RowProvenance";
import {
  parseLeaderboard,
  readFailureMessage,
  type LeaderboardPage,
} from "@/lib/api";
import { useScopeReset } from "@/lib/useScopeReset";

const apiBase = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export default function LeaderboardPage() {
  const [data, setData] = useState<LeaderboardPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [attempt, setAttempt] = useState(0);
  const [corpusNames, setCorpusNames] = useState<string[]>([]);

  // The filter names the corpus over the table, so the rows for the last one
  // go before the new read starts.
  useScopeReset(filter, () => {
    setData(null);
    setError(null);
  });

  useEffect(() => {
    const controller = new AbortController();
    const url = `${apiBase}/api/leaderboard?corpus=${encodeURIComponent(filter)}`;
    fetch(url, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`The server answered ${r.status}.`);
        return r.json();
      })
      // Every field the table reads comes back checked, so a 200 with the
      // wrong body is a failed read rather than a rendered one.
      .then((body: unknown) => parseLeaderboard(body))
      .then((d) => {
        setData(d);
        setCorpusNames(d.corpora ?? []);
        setError(null);
      })
      .catch((e) => {
        if (e instanceof Error && e.name === "AbortError") return;
        // Rows from the last filter under the new corpus name read as that
        // corpus's runs, so the table goes with the failed read.
        setData(null);
        setError(readFailureMessage(e, "The leaderboard did not load."));
      });
    return () => controller.abort();
  }, [filter, attempt]);

  // The picker lists every corpus this deployment holds, whatever the filter
  // is, so it survives a filter change while the rows do not.
  const corpora = useMemo(() => ["all", ...corpusNames], [corpusNames]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <StateBanner />

      <header>
        <h1 className="text-2xl font-bold tracking-tight">
          KB Arena Leaderboard
        </h1>
        <p className="text-sm mt-2 max-w-2xl" style={{ color: "var(--muted)" }}>
          Runs stored in this deployment, grouped by corpus, strategy, experiment key
          and build. Higher accuracy, Recall@5 and NDCG@5 are better. Lower cost and
          latency are better. Two rows with a different key or a different build
          measured different things, so read them side by side, not as one ranking.
          A row is citable only when a reviewer checked every question under it.
        </p>
      </header>

      <div className="flex items-center gap-3">
        <label className="text-sm font-medium" htmlFor="corpus-filter">
          Corpus:
        </label>
        <select
          id="corpus-filter"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="rounded border px-3 py-1 text-sm"
          style={{ background: "var(--card)", borderColor: "var(--border-strong)", color: "var(--foreground)" }}
        >
          {corpora.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <FetchError
          title="The leaderboard did not load"
          message={error}
          hint="A table from an earlier read would name the wrong corpus, so the rows stay off screen. Start the API, then try again."
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

      {!data && !error && <p>Loading...</p>}

      {data && data.leaderboard.length === 0 && (
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          No benchmark runs yet. Run <code>kb-arena benchmark --corpus aws-compute</code>.
        </p>
      )}

      {data && data.leaderboard.length > 0 && (
        <div className="overflow-x-auto rounded border" style={{ borderColor: "var(--border)" }}>
          <table className="min-w-full text-sm">
            <thead className="text-left" style={{ background: "var(--subtle)" }}>
              <tr>
                <th className="px-3 py-2 font-medium">Corpus</th>
                <th className="px-3 py-2 font-medium">Strategy</th>
                <th className="px-3 py-2 font-medium">Where this row came from</th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  Accuracy
                  <MetricNote metric="accuracy" align="left" />
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  Recall@5
                  <MetricNote metric="recall_at_5" align="left" />
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  NDCG@5
                  <MetricNote metric="ndcg_at_5" align="left" />
                </th>
                {/* The API averages each run's whole cost, not its cost for one
                    question, so the header names a run. */}
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  Cost per run (USD)
                  <MetricNote metric="cost_per_run" align="left" />
                </th>
                <th className="px-3 py-2 font-medium text-right whitespace-nowrap">
                  Latency (ms)
                  <MetricNote metric="latency" align="left" />
                </th>
                <th className="px-3 py-2 font-medium text-right">Runs</th>
              </tr>
            </thead>
            <tbody>
              {data.leaderboard.map((row, i) => (
                <tr key={`${row.corpus}-${row.strategy}-${row.compatibility_key}-${i}`} className="border-t">
                  <td className="px-3 py-2 font-mono">{row.corpus}</td>
                  <td className="px-3 py-2 font-mono">{row.strategy}</td>
                  <td className="px-3 py-2 align-top" style={{ minWidth: 220 }}>
                    <RowProvenance
                      compatibilityKey={row.compatibility_key}
                      buildLabel={
                        row.build && row.build !== "unrecorded"
                          ? `build ${
                              row.build.length > 14 ? `${row.build.slice(0, 14)}...` : row.build
                            }`
                          : "build unrecorded"
                      }
                      review={row.review}
                      manifest={row.manifest}
                      mixedWith={row.mixed_with ?? []}
                    />
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.mean_accuracy != null ? (row.mean_accuracy * 100).toFixed(1) + "%" : "n/a"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.mean_recall_at_5 != null
                      ? (row.mean_recall_at_5 * 100).toFixed(1) + "%"
                      : "n/a"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.mean_ndcg_at_5 != null ? row.mean_ndcg_at_5.toFixed(3) : "n/a"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.mean_cost_usd != null ? "$" + row.mean_cost_usd.toFixed(2) : "n/a"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {row.mean_latency_ms != null ? row.mean_latency_ms.toFixed(0) : "n/a"}
                  </td>
                  <td className="px-3 py-2 text-right">{row.runs}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
