"use client";

import { useState, useEffect, useMemo } from "react";

type LeaderRow = {
  corpus: string;
  strategy: string;
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

  useEffect(() => {
    const controller = new AbortController();
    const url = `${apiBase}/api/leaderboard?corpus=${encodeURIComponent(filter)}`;
    fetch(url, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: LeaderboardResponse) => {
        setData(d);
        setError(null);
      })
      .catch((e) => {
        if (e instanceof Error && e.name !== "AbortError") setError(String(e));
      });
    return () => controller.abort();
  }, [filter]);

  const corpora = useMemo(() => ["all", ...(data?.corpora ?? [])], [data?.corpora]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight">
          KB Arena Leaderboard
        </h1>
        <p className="text-sm text-gray-600 mt-2 max-w-2xl">
          Aggregated benchmark scores across every run in this deployment. Higher
          accuracy + Recall@5 + NDCG@5 are better; lower cost + latency are better.
          To submit a run, open a PR with your <code>results/run_*</code> JSON.
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
        <div className="border border-red-300 bg-red-50 text-red-900 p-4 rounded">
          Failed to load leaderboard: {error}
        </div>
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
                <tr key={`${row.corpus}-${row.strategy}-${i}`} className="border-t">
                  <td className="px-3 py-2 font-mono">{row.corpus}</td>
                  <td className="px-3 py-2 font-mono">{row.strategy}</td>
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
