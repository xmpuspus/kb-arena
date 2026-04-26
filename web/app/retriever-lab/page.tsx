"use client";

import { useState, useEffect, useMemo } from "react";
import { API_URL, STRATEGY_LABELS, type Strategy } from "@/lib/api";

type StrategySummary = {
  mean_recall_at_k: number;
  mean_precision_at_k: number;
  mean_hit_at_k: number;
  mean_mrr: number;
  mean_ndcg_at_k: number;
  questions: number;
};

type RetrievedItem = {
  chunk_id: string;
  doc_id: string;
  rank: number;
  score: number;
  source_strategy: string;
  is_hit: boolean;
};

type QuestionRow = {
  corpus: string;
  strategy: string;
  question_id: string;
  question: string;
  recall_at_k: number;
  precision_at_k: number;
  hit_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  fallback_doc_level: boolean;
  retrieved: RetrievedItem[];
};

type RunData = {
  run_id: string;
  timestamp: string;
  top_k: number;
  corpora: Record<string, Record<string, StrategySummary>>;
  questions: QuestionRow[];
};

type RunListEntry = { run_id: string; timestamp: string; top_k: number; corpora: string[] };

const fmtPct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmt3 = (v: number) => v.toFixed(3);

function MetricsCard({ strategy, m, topK }: { strategy: string; m: StrategySummary; topK: number }) {
  return (
    <div
      className="border rounded-xl p-5"
      style={{ background: "var(--card)", borderColor: "var(--border)" }}
    >
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="font-semibold text-base" style={{ color: "var(--foreground)" }}>
          {STRATEGY_LABELS[strategy as Strategy] ?? strategy}
        </h3>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          n={m.questions}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Recall@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {fmtPct(m.mean_recall_at_k)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Precision@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {fmtPct(m.mean_precision_at_k)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Hit@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {fmtPct(m.mean_hit_at_k)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            MRR
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {fmt3(m.mean_mrr)}
          </div>
        </div>
        <div className="col-span-2">
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            NDCG@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {fmt3(m.mean_ndcg_at_k)}
          </div>
        </div>
      </div>
    </div>
  );
}

function ChunkRow({ item }: { item: RetrievedItem }) {
  return (
    <div
      className="flex items-center gap-3 border rounded-lg px-3 py-2 text-sm"
      style={{
        background: item.is_hit ? "rgba(34, 197, 94, 0.05)" : "var(--card)",
        borderColor: item.is_hit ? "rgba(34, 197, 94, 0.4)" : "var(--border)",
      }}
    >
      <span
        className="font-mono text-xs px-2 py-0.5 rounded"
        style={{
          background: item.is_hit ? "rgba(34, 197, 94, 0.15)" : "var(--background)",
          color: item.is_hit ? "rgb(22, 163, 74)" : "var(--muted)",
        }}
      >
        #{item.rank}
      </span>
      <span className="font-mono text-xs flex-1 truncate" style={{ color: "var(--foreground)" }}>
        {item.chunk_id}
      </span>
      <span className="font-mono text-xs" style={{ color: "var(--muted)" }}>
        {item.score.toFixed(3)}
      </span>
      <span className="text-xs uppercase tracking-wider" style={{ color: "var(--muted)" }}>
        {item.is_hit ? "hit" : "miss"}
      </span>
    </div>
  );
}

export default function RetrieverLabPage() {
  const [runs, setRuns] = useState<RunListEntry[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [data, setData] = useState<RunData | null>(null);
  const [selectedCorpus, setSelectedCorpus] = useState<string>("");
  const [selectedQid, setSelectedQid] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    fetch(`${API_URL}/api/retriever-lab/runs`)
      .then((r) => r.json())
      .then((j) => {
        const entries: RunListEntry[] = j.runs ?? [];
        setRuns(entries);
        if (entries.length > 0) setSelectedRun(entries[0].run_id);
      })
      .catch((e) => setError(`Failed to load runs: ${e}`));
  }, []);

  useEffect(() => {
    if (!selectedRun) return;
    setLoading(true);
    setError("");
    fetch(`${API_URL}/api/retriever-lab/${selectedRun}`)
      .then((r) => {
        if (!r.ok) throw new Error(`status ${r.status}`);
        return r.json();
      })
      .then((j: RunData) => {
        setData(j);
        const firstCorpus = Object.keys(j.corpora)[0] ?? "";
        setSelectedCorpus(firstCorpus);
        setSelectedQid("");
      })
      .catch((e) => setError(`Failed to load run: ${e}`))
      .finally(() => setLoading(false));
  }, [selectedRun]);

  const corpusSummary = useMemo(() => {
    if (!data || !selectedCorpus) return null;
    return data.corpora[selectedCorpus] ?? null;
  }, [data, selectedCorpus]);

  const questionsForCorpus = useMemo(() => {
    if (!data || !selectedCorpus) return [];
    return data.questions.filter((q) => q.corpus === selectedCorpus);
  }, [data, selectedCorpus]);

  const uniqueQuestions = useMemo(() => {
    const seen = new Map<string, QuestionRow>();
    for (const q of questionsForCorpus) {
      if (!seen.has(q.question_id)) seen.set(q.question_id, q);
    }
    return Array.from(seen.values());
  }, [questionsForCorpus]);

  const drillDownRows = useMemo(() => {
    if (!selectedQid) return [];
    return questionsForCorpus.filter((q) => q.question_id === selectedQid);
  }, [questionsForCorpus, selectedQid]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Retriever Lab
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Classical IR metrics — Recall@k, Precision@k, Hit@k, MRR, NDCG@k. See exactly which chunks
          surfaced and which the strategy missed.
        </p>
      </div>

      {error && (
        <div
          className="border rounded-lg px-3 py-2 text-sm"
          style={{ borderColor: "rgba(220, 38, 38, 0.4)", color: "rgb(185, 28, 28)" }}
        >
          {error}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>
            Run
          </label>
          <select
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            className="px-3 py-1.5 rounded-lg border text-sm"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--foreground)",
            }}
          >
            {runs.length === 0 && <option value="">No runs yet</option>}
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} · top-{r.top_k} · {r.timestamp.slice(0, 19)}
              </option>
            ))}
          </select>
        </div>
        {data && (
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>
              Corpus
            </label>
            <select
              value={selectedCorpus}
              onChange={(e) => {
                setSelectedCorpus(e.target.value);
                setSelectedQid("");
              }}
              className="px-3 py-1.5 rounded-lg border text-sm"
              style={{
                background: "var(--card)",
                borderColor: "var(--border)",
                color: "var(--foreground)",
              }}
            >
              {Object.keys(data.corpora).map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        )}
        {data && (
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            top-k = {data.top_k}
          </span>
        )}
      </div>

      {loading && <div style={{ color: "var(--muted)" }}>Loading…</div>}

      {!loading && corpusSummary && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
            Aggregate metrics — {selectedCorpus}
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(corpusSummary).map(([strategy, m]) => (
              <MetricsCard key={strategy} strategy={strategy} m={m} topK={data!.top_k} />
            ))}
          </div>
        </section>
      )}

      {!loading && uniqueQuestions.length > 0 && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
            Per-question drill-down
          </h2>
          <div className="flex items-center gap-2">
            <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>
              Question
            </label>
            <select
              value={selectedQid}
              onChange={(e) => setSelectedQid(e.target.value)}
              className="px-3 py-1.5 rounded-lg border text-sm flex-1 max-w-2xl"
              style={{
                background: "var(--card)",
                borderColor: "var(--border)",
                color: "var(--foreground)",
              }}
            >
              <option value="">— pick a question to drill in —</option>
              {uniqueQuestions.map((q) => (
                <option key={q.question_id} value={q.question_id}>
                  {q.question_id}: {q.question.slice(0, 90)}
                </option>
              ))}
            </select>
          </div>
          {selectedQid && (
            <div className="space-y-3">
              {drillDownRows.map((row) => (
                <div
                  key={row.strategy}
                  className="border rounded-lg p-4"
                  style={{ background: "var(--card)", borderColor: "var(--border)" }}
                >
                  <div className="flex items-baseline justify-between mb-3">
                    <h3
                      className="font-semibold text-sm"
                      style={{ color: "var(--foreground)" }}
                    >
                      {STRATEGY_LABELS[row.strategy as Strategy] ?? row.strategy}
                    </h3>
                    <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                      R@{data!.top_k}={fmtPct(row.recall_at_k)} · MRR={fmt3(row.mrr)} · NDCG=
                      {fmt3(row.ndcg_at_k)}
                      {row.fallback_doc_level && " · doc-level"}
                    </span>
                  </div>
                  <div className="space-y-1.5">
                    {row.retrieved.length === 0 && (
                      <span className="text-xs" style={{ color: "var(--muted)" }}>
                        No chunks retrieved.
                      </span>
                    )}
                    {row.retrieved.map((it) => (
                      <ChunkRow key={`${row.strategy}-${it.rank}-${it.chunk_id}`} item={it} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {!loading && !corpusSummary && runs.length === 0 && (
        <div
          className="border border-dashed rounded-lg p-6 text-sm"
          style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        >
          No retriever-lab runs yet. Run <code>kb-arena retriever-lab --corpus aws-compute</code>{" "}
          to populate this page.
        </div>
      )}
    </div>
  );
}
