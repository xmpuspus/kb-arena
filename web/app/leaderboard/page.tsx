"use client";

import { useState, useEffect, useMemo } from "react";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";
import { readFailureMessage } from "@/lib/api";

type LeaderRow = {
  corpus: string;
  strategy: string;
  // Runs that differ in question set, qrels, judge, or top_k never share a
  // row. The key names the experiment, and mixed_with lists the other keys
  // seen for the same corpus and strategy.
  compatibility_key: string;
  // Which build produced these runs. The API groups by it, so the page shows it.
  build?: string;
  manifest: {
    question_split?: string | null;
    judge_model?: string | null;
    top_k?: number | null;
  };
  mixed_with: string[];
  runs: number;
  mean_accuracy: number | null;
  mean_recall_at_5: number | null;
  mean_ndcg_at_5: number | null;
  mean_cost_usd: number | null;
  mean_latency_ms: number | null;
};

type LeaderboardResponse = {
  corpora: string[];
  leaderboard: LeaderRow[];
  filter: string;
};

const apiBase = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export default function LeaderboardPage() {
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const url = `${apiBase}/api/leaderboard?corpus=${encodeURIComponent(filter)}`;
    fetch(url, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`The server answered ${r.status}.`);
        return r.json();
      })
      .then((d: LeaderboardResponse) => {
        setData(d);
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

  const corpora = useMemo(() => ["all", ...(data?.corpora ?? [])], [data?.corpora]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <StateBanner />

      <header>
        <h1 className="text-2xl font-bold tracking-tight">
          KB Arena Leaderboard
        </h1>
        <p className="text-sm text-gray-600 mt-2 max-w-2xl">
          Runs stored in this deployment, grouped by corpus, strategy, experiment key
          and build. Higher accuracy, Recall@5 and NDCG@5 are better. Lower cost and
          latency are better. Two rows with a different key or a different build
          measured different things, so read them side by side, not as one ranking.
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
          className="border rounded px-3 py-1 text-sm"
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
        <p className="text-sm text-gray-600">
          No benchmark runs yet. Run <code>kb-arena benchmark --corpus aws-compute</code>.
        </p>
      )}

      {data && data.leaderboard.length > 0 && (
        <div className="overflow-x-auto border rounded">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 text-left">
              <tr>
                <th className="px-3 py-2 font-medium">Corpus</th>
                <th className="px-3 py-2 font-medium">Strategy</th>
                <th className="px-3 py-2 font-medium">Experiment</th>
                <th className="px-3 py-2 font-medium text-right">Accuracy</th>
                <th className="px-3 py-2 font-medium text-right">Recall@5</th>
                <th className="px-3 py-2 font-medium text-right">NDCG@5</th>
                <th className="px-3 py-2 font-medium text-right">Cost (USD)</th>
                <th className="px-3 py-2 font-medium text-right">Latency (ms)</th>
                <th className="px-3 py-2 font-medium text-right">Runs</th>
              </tr>
            </thead>
            <tbody>
              {data.leaderboard.map((row, i) => (
                <tr key={`${row.corpus}-${row.strategy}-${row.compatibility_key}-${i}`} className="border-t">
                  <td className="px-3 py-2 font-mono">{row.corpus}</td>
                  <td className="px-3 py-2 font-mono">{row.strategy}</td>
                  <td className="px-3 py-2 text-xs">
                    <span
                      className="font-mono"
                      title={
                        row.compatibility_key === "legacy"
                          ? "Result file without a manifest"
                          : `judge ${row.manifest?.judge_model ?? "?"}, split ${
                              row.manifest?.question_split ?? "?"
                            }, top_k ${row.manifest?.top_k ?? "?"}`
                      }
                    >
                      {row.compatibility_key === "legacy" ? "legacy" : row.compatibility_key.slice(0, 6)}
                    </span>
                    {row.build && row.build !== "unrecorded" && (
                      <div style={{ color: "var(--muted)" }} title="The build that produced these runs">
                        {row.build.length > 14 ? `${row.build.slice(0, 14)}...` : row.build}
                      </div>
                    )}
                    {row.build === "unrecorded" && (
                      <div style={{ color: "var(--muted)" }} title="These runs recorded no version or commit">
                        build unrecorded
                      </div>
                    )}
                    {row.mixed_with?.length > 0 && (
                      <div style={{ color: "var(--muted)" }}>
                        {row.mixed_with.length} other experiment{row.mixed_with.length === 1 ? "" : "s"} for this pair, not comparable
                      </div>
                    )}
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
