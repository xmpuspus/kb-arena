"use client";

import { useRef, useState } from "react";
import { streamAudit, type SectionAuditResult } from "@/lib/tools-api";
import ProgressBar from "@/components/tools/ProgressBar";
import SectionTable from "@/components/tools/SectionTable";
import EmptyState from "@/components/tools/EmptyState";

interface Props {
  corpus: string;
}

type State = "idle" | "running" | "complete" | "error";

interface Summary {
  strong: number;
  weak: number;
  gaps: number;
  totalQuestions: number;
}

export default function AuditTab({ corpus }: Props) {
  const [state, setState] = useState<State>("idle");
  const [maxSections, setMaxSections] = useState(50);
  const [progress, setProgress] = useState({ current: 0, total: 0, label: "" });
  const [sections, setSections] = useState<SectionAuditResult[]>([]);
  const [summary, setSummary] = useState<Summary>({ strong: 0, weak: 0, gaps: 0, totalQuestions: 0 });
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const handleAudit = async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState("running");
    setSections([]);
    setError("");
    setProgress({ current: 0, total: 0, label: "" });
    setSummary({ strong: 0, weak: 0, gaps: 0, totalQuestions: 0 });

    try {
      for await (const event of streamAudit(corpus, maxSections, ctrl.signal)) {
        if (event.type === "started") {
          setProgress((p) => ({ ...p, total: event.total_sections }));
        } else if (event.type === "section_result") {
          const r = event.result;
          setSections((prev) => [...prev, r]);
          setProgress({ current: r.section_index + 1, total: r.total, label: r.section_title });
          setSummary((prev) => ({
            strong: prev.strong + (r.classification === "strong" ? 1 : 0),
            weak: prev.weak + (r.classification === "weak" ? 1 : 0),
            gaps: prev.gaps + (r.classification === "gap" ? 1 : 0),
            totalQuestions: prev.totalQuestions + r.questions_tested,
          }));
        } else if (event.type === "complete") {
          setSummary({
            strong: event.strong,
            weak: event.weak,
            gaps: event.gaps,
            totalQuestions: event.total_questions,
          });
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
    setState(sections.length > 0 ? "complete" : "idle");
  };

  const overallAcc =
    sections.length > 0
      ? sections.reduce((sum, s) => sum + s.avg_accuracy * s.questions_tested, 0) /
        Math.max(summary.totalQuestions, 1)
      : 0;

  return (
    <div className="space-y-4">
      {/* Action bar */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-xs" style={{ color: "var(--muted)" }}>
            Max sections
          </label>
          <input
            type="number"
            value={maxSections}
            onChange={(e) => setMaxSections(Math.max(1, Math.min(500, Number(e.target.value))))}
            className="w-20 px-2 py-1.5 rounded-lg border text-sm"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--foreground)",
            }}
            min={1}
            max={500}
          />
        </div>
        <button
          onClick={state === "running" ? handleCancel : handleAudit}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: state === "running" ? "var(--danger)" : "var(--accent)",
            color: "#fff",
          }}
        >
          {state === "running" ? "Cancel" : "Run Audit"}
        </button>
      </div>

      {/* Progress */}
      {state === "running" && (
        <ProgressBar
          current={progress.current}
          total={progress.total}
          label="Auditing..."
          sublabel={progress.label}
        />
      )}

      {/* Summary cards */}
      {(state === "running" || state === "complete") && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <SummaryCard label="Overall" value={`${Math.round(overallAcc * 100)}%`} color="var(--accent)" />
          <SummaryCard label="Strong" value={String(summary.strong)} color="#22c55e" />
          <SummaryCard label="Weak" value={String(summary.weak)} color="#f59e0b" />
          <SummaryCard label="Gaps" value={String(summary.gaps)} color="#ef4444" />
        </div>
      )}

      {/* Section table */}
      {sections.length > 0 && <SectionTable sections={sections} />}

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

      {/* Empty state */}
      {state === "idle" && sections.length === 0 && (
        <EmptyState
          title="Audit documentation quality"
          description="Evaluate how well your documentation answers its own questions. Sections are classified as strong, weak, or gap."
          command={`kb-arena audit --corpus ${corpus}`}
        />
      )}
    </div>
  );
}

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div
      className="rounded-lg border p-3 text-center"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <p className="text-2xl font-bold mono" style={{ color }}>
        {value}
      </p>
      <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
        {label}
      </p>
    </div>
  );
}
