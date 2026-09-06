"use client";

import { useState, useEffect, useMemo } from "react";
import { API_URL, STRATEGY_LABELS, readFailureMessage, type Strategy } from "@/lib/api";
import { apiFetch } from "@/lib/auth";
import { useTokenEpoch } from "@/lib/useTokenEpoch";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";

type StrategySummary = {
  mean_recall_at_k: number;
  mean_precision_at_k: number;
  mean_hit_at_k: number;
  mean_mrr: number;
  mean_ndcg_at_k: number;
  questions: number;
  execution_errors?: number;
};

type RetrievedItem = {
  chunk_id: string;
  doc_id: string;
  rank: number;
  score: number;
  source_strategy: string;
  is_hit: boolean;
};

type QuestionBase = {
  corpus: string;
  strategy: string;
  question_id: string;
  question: string;
};

type QuestionResultRow = QuestionBase & {
  recall_at_k: number;
  precision_at_k: number;
  hit_at_k: number;
  mrr: number;
  ndcg_at_k: number;
  fallback_doc_level: boolean;
  retrieved: RetrievedItem[];
};

type QuestionErrorRow = QuestionBase & {
  execution_error: { type: string; message: string };
};

type QuestionRow = QuestionResultRow | QuestionErrorRow;

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
  const metric = (value: number, format: (v: number) => string) =>
    m.questions > 0 ? format(value) : "n/a";

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
          {(m.execution_errors ?? 0) > 0 && ` | errors=${m.execution_errors}`}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Recall@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {metric(m.mean_recall_at_k, fmtPct)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Precision@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {metric(m.mean_precision_at_k, fmtPct)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            Hit@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {metric(m.mean_hit_at_k, fmtPct)}
          </div>
        </div>
        <div>
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            MRR
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {metric(m.mean_mrr, fmt3)}
          </div>
        </div>
        <div className="col-span-2">
          <div className="text-xs" style={{ color: "var(--muted)" }}>
            NDCG@{topK}
          </div>
          <div className="font-mono text-lg" style={{ color: "var(--foreground)" }}>
            {metric(m.mean_ndcg_at_k, fmt3)}
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
  // A saved token must retry the read it was entered for.
  const tokenEpoch = useTokenEpoch();
  const [runs, setRuns] = useState<RunListEntry[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const [data, setData] = useState<RunData | null>(null);
  const [selectedCorpus, setSelectedCorpus] = useState<string>("");
  const [selectedQid, setSelectedQid] = useState<string>("");
  const [loading, setLoading] = useState(false);
  // `loading` covers the detail read. Without its own flag, the run list read
  // is pending while the page already says "No retriever-lab runs yet", which
  // is an answer nobody has yet.
  const [listPending, setListPending] = useState(true);
  const [error, setError] = useState<string>("");
  // The two reads fail for different reasons and need different remedies. One
  // error state said the run list failed even when the list had loaded.
  const [detailError, setDetailError] = useState<string>("");
  const [attempt, setAttempt] = useState(0);
  const [detailAttempt, setDetailAttempt] = useState(0);

  useEffect(() => {
    setError("");
    setListPending(true);
    fetch(`${API_URL}/api/retriever-lab/runs`)
      .then((r) => {
        // An empty run list reads as a deployment that never ran the lab, so a
        // failed read must not arrive as one.
        if (!r.ok) throw new Error(`The server answered ${r.status}.`);
        return r.json();
      })
      .then((j) => {
        const entries: RunListEntry[] = j.runs ?? [];
        setRuns(entries);
        if (entries.length > 0) setSelectedRun(entries[0].run_id);
      })
      .catch((e: unknown) => {
        setRuns([]);
        setSelectedRun("");
        setData(null);
        setError(readFailureMessage(e, "The run list did not load."));
      })
      .finally(() => setListPending(false));
  }, [attempt]);

  useEffect(() => {
    if (!selectedRun) return;
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setDetailError("");
    // Question-level records, so this read carries the API token when set.
    apiFetch(`${API_URL}/api/retriever-lab/${selectedRun}`, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`The server answered ${r.status}.`);
        return r.json();
      })
      .then((j: RunData) => {
        if (!active) return;
        setData(j);
        const firstCorpus = Object.keys(j.corpora)[0] ?? "";
        setSelectedCorpus(firstCorpus);
        setSelectedQid("");
      })
      .catch((e) => {
        if (e instanceof Error && e.name !== "AbortError") {
          // Leaving the old run on screen puts one run's numbers under
          // another run's name. Clear it and say what happened.
          if (active) setData(null);
          setDetailError(readFailureMessage(e, "The run did not load."));
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [selectedRun, attempt, detailAttempt, tokenEpoch]);

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
      <StateBanner />

      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Retriever Lab
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Classical IR metrics: Recall@k, Precision@k, Hit@k, MRR, NDCG@k. See exactly which chunks
          surfaced and which the strategy missed.
        </p>
      </div>

      {error && (
        <FetchError
          title="The retriever-lab run list did not load"
          message={error}
          hint="An empty run list would read as a deployment that never ran the lab, so this page shows none. Start the API, or enter an API token with the key button, then try again."
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

      {detailError && (
        <FetchError
          title={`Run ${selectedRun} did not load`}
          message={detailError}
          hint="The run list loaded, so this run alone is unreadable. Pick another run, or try this one again."
          onRetry={() => setDetailAttempt((n) => n + 1)}
        />
      )}

      <div className="flex flex-wrap items-center gap-4">
        {/* A run label reads `9beeb4aa | top-5 | 2026-04-26T13:35:43`, and a
            select sizes to its widest option. At 375 that pushed one pixel past
            the viewport, so both the row and the control may shrink. */}
        <div className="flex min-w-0 items-center gap-2">
          <label htmlFor="lab-run" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
            Run
          </label>
          <select
            id="lab-run"
            value={selectedRun}
            onChange={(e) => setSelectedRun(e.target.value)}
            disabled={Boolean(error) || listPending}
            className="min-w-0 px-3 py-1.5 rounded-lg border text-sm disabled:opacity-50"
            style={{
              background: "var(--card)",
              borderColor: "var(--border-strong)",
              color: "var(--foreground)",
            }}
          >
            {/* "No runs yet" is a claim about the deployment, and a failed read
                supports no such claim. */}
            {runs.length === 0 && (
              <option value="">
                {listPending
                  ? "Reading the run list..."
                  : error
                    ? "Run list unavailable"
                    : "No runs yet"}
              </option>
            )}
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} | top-{r.top_k} | {r.timestamp.slice(0, 19)}
              </option>
            ))}
          </select>
        </div>
        {data && (
          <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-center">
            <label htmlFor="lab-corpus" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
              Corpus
            </label>
            <select
              id="lab-corpus"
              value={selectedCorpus}
              onChange={(e) => {
                setSelectedCorpus(e.target.value);
                setSelectedQid("");
              }}
              className="px-3 py-1.5 rounded-lg border text-sm"
              style={{
                background: "var(--card)",
                borderColor: "var(--border-strong)",
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

      {(loading || listPending) && (
        <div style={{ color: "var(--muted)" }}>
          {listPending ? "Reading the run list..." : "Loading..."}
        </div>
      )}

      {!loading && corpusSummary && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold" style={{ color: "var(--foreground)" }}>
            Aggregate metrics: {selectedCorpus}
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
            <label htmlFor="lab-question" className="text-xs font-medium" style={{ color: "var(--muted)" }}>
              Question
            </label>
            <select
              id="lab-question"
              value={selectedQid}
              onChange={(e) => setSelectedQid(e.target.value)}
              className="w-full min-w-0 px-3 py-1.5 rounded-lg border text-sm sm:flex-1 sm:max-w-2xl"
              style={{
                background: "var(--card)",
                borderColor: "var(--border-strong)",
                color: "var(--foreground)",
              }}
            >
              <option value="">Select a question to inspect</option>
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
                    {"execution_error" in row ? (
                      <span className="text-xs font-mono" style={{ color: "rgb(185, 28, 28)" }}>
                        Retrieval failed
                      </span>
                    ) : (
                      <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                        R@{data!.top_k}={fmtPct(row.recall_at_k)} | MRR={fmt3(row.mrr)} | NDCG=
                        {fmt3(row.ndcg_at_k)}
                        {row.fallback_doc_level && " | doc-level"}
                      </span>
                    )}
                  </div>
                  {"execution_error" in row ? (
                    <div
                      className="break-words border px-3 py-2 text-xs"
                      style={{ borderColor: "rgba(220, 38, 38, 0.4)", color: "rgb(185, 28, 28)" }}
                    >
                      {row.execution_error.type}: {row.execution_error.message}
                    </div>
                  ) : (
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
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* A failed read empties the list too, and a pending read has answered
          nothing yet. "No runs yet" is a claim, and neither state supports it. */}
      {!loading && !listPending && !error && !corpusSummary && runs.length === 0 && (
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
