"use client";

import { useEffect, useRef, useState } from "react";
import { streamGenerate, fetchQaPairs, type QaPair } from "@/lib/tools-api";
import ProgressBar from "@/components/tools/ProgressBar";
import EmptyState from "@/components/tools/EmptyState";

interface Props {
  corpus: string;
}

type State = "idle" | "running" | "complete" | "error";

const PAGE_SIZE = 20;

export default function GenerateTab({ corpus }: Props) {
  const [state, setState] = useState<State>("idle");
  const [pairs, setPairs] = useState<QaPair[]>([]);
  const [progress, setProgress] = useState({ current: 0, total: 0, label: "" });
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  // Load existing pairs on mount / corpus change
  useEffect(() => {
    setState("idle");
    setPairs([]);
    setProgress({ current: 0, total: 0, label: "" });
    fetchQaPairs(corpus).then(({ pairs: existing }) => {
      if (existing.length > 0) {
        setPairs(existing);
        setState("complete");
      }
    });
  }, [corpus]);

  const handleGenerate = async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState("running");
    setPairs([]);
    setError("");
    setProgress({ current: 0, total: 0, label: "" });

    try {
      for await (const event of streamGenerate(corpus, ctrl.signal)) {
        if (event.type === "started") {
          setProgress((p) => ({ ...p, total: event.total_sections }));
        } else if (event.type === "progress") {
          setProgress({ current: event.section_index + 1, total: event.total, label: event.section_title });
        } else if (event.type === "pair") {
          setPairs((prev) => [...prev, event.pair]);
        } else if (event.type === "complete") {
          setState("complete");
        } else if (event.type === "error") {
          setError(event.message);
          setState("error");
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError((err as Error).message);
      setState("error");
    }
  };

  const handleCancel = () => {
    abortRef.current?.abort();
    setState(pairs.length > 0 ? "complete" : "idle");
  };

  const filtered = search
    ? pairs.filter(
        (p) =>
          p.question.toLowerCase().includes(search.toLowerCase()) ||
          p.answer.toLowerCase().includes(search.toLowerCase())
      )
    : pairs;
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const displayed = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <div className="space-y-4">
      {/* Action bar */}
      <div className="flex items-center gap-3">
        <button
          onClick={state === "running" ? handleCancel : handleGenerate}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: state === "running" ? "var(--danger)" : "var(--accent)",
            color: "#fff",
          }}
        >
          {state === "running" ? "Cancel" : pairs.length > 0 ? "Regenerate" : "Generate Q&A Pairs"}
        </button>
        {pairs.length > 0 && (
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {pairs.length} pairs generated
          </span>
        )}
      </div>

      {/* Progress */}
      {state === "running" && (
        <ProgressBar
          current={progress.current}
          total={progress.total}
          label="Generating..."
          sublabel={progress.label}
        />
      )}

      {/* Error */}
      {state === "error" && (
        <div
          className="rounded-lg border p-3"
          style={{ borderColor: "var(--danger)", background: "rgba(239, 68, 68, 0.05)" }}
        >
          <p className="text-sm" style={{ color: "var(--danger)" }}>
            {error || "An error occurred"}
          </p>
        </div>
      )}

      {/* Results table */}
      {pairs.length > 0 && (
        <div className="space-y-3">
          <input
            type="text"
            placeholder="Search questions and answers..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
            className="w-full px-3 py-2 rounded-lg border text-sm"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--foreground)",
            }}
          />

          <div className="rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr style={{ background: "var(--background)" }}>
                  <th
                    className="text-left px-3 py-2 text-xs font-medium"
                    style={{ color: "var(--muted)" }}
                  >
                    Question
                  </th>
                  <th
                    className="text-left px-3 py-2 text-xs font-medium"
                    style={{ color: "var(--muted)" }}
                  >
                    Answer
                  </th>
                  <th
                    className="text-left px-3 py-2 text-xs font-medium"
                    style={{ color: "var(--muted)" }}
                  >
                    Source
                  </th>
                </tr>
              </thead>
              <tbody>
                {displayed.map((pair, i) => (
                  <tr
                    key={`${pair.section_id}-${i}`}
                    className="border-t"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td
                      className="px-3 py-2 align-top"
                      style={{ color: "var(--foreground)", maxWidth: "300px" }}
                    >
                      <p className="text-sm line-clamp-3">{pair.question}</p>
                    </td>
                    <td
                      className="px-3 py-2 align-top"
                      style={{ color: "var(--muted)", maxWidth: "400px" }}
                    >
                      <p className="text-xs line-clamp-3">{pair.answer}</p>
                    </td>
                    <td className="px-3 py-2 align-top text-xs" style={{ color: "var(--muted)" }}>
                      {pair.section_ref || pair.source_id}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between">
              <span className="text-xs" style={{ color: "var(--muted)" }}>
                Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of{" "}
                {filtered.length}
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-2 py-1 text-xs rounded border transition-colors disabled:opacity-30"
                  style={{ borderColor: "var(--border)", color: "var(--muted)" }}
                >
                  Prev
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-2 py-1 text-xs rounded border transition-colors disabled:opacity-30"
                  style={{ borderColor: "var(--border)", color: "var(--muted)" }}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {state === "idle" && pairs.length === 0 && (
        <EmptyState
          title="No Q&A pairs yet"
          description="Generate question-answer pairs from your documentation corpus. These pairs are used for evaluation and can be exported."
          command={`kb-arena generate-qa --corpus ${corpus}`}
        />
      )}
    </div>
  );
}
