"use client";

import { useRef, useState } from "react";
import { streamFix, type FixRecommendation } from "@/lib/tools-api";
import ProgressBar from "@/components/tools/ProgressBar";
import FixCard from "@/components/tools/FixCard";
import EmptyState from "@/components/tools/EmptyState";

interface Props {
  corpus: string;
}

type State = "idle" | "auditing" | "fixing" | "complete" | "error";

interface AuditSummary {
  strong: number;
  weak: number;
  gaps: number;
  totalQuestions: number;
}

export default function FixTab({ corpus }: Props) {
  const [state, setState] = useState<State>("idle");
  const [maxSections, setMaxSections] = useState(50);
  const [maxFixes, setMaxFixes] = useState(10);
  const [auditProgress, setAuditProgress] = useState({ current: 0, total: 0, label: "" });
  const [auditSummary, setAuditSummary] = useState<AuditSummary | null>(null);
  const [fixes, setFixes] = useState<FixRecommendation[]>([]);
  const [fixProgress, setFixProgress] = useState({ current: 0, total: 0 });
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const handleRun = async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setState("auditing");
    setFixes([]);
    setError("");
    setAuditProgress({ current: 0, total: 0, label: "" });
    setAuditSummary(null);
    setFixProgress({ current: 0, total: 0 });

    try {
      for await (const event of streamFix(corpus, maxSections, maxFixes, ctrl.signal)) {
        if (event.type === "phase") {
          setState(event.phase === "audit" ? "auditing" : "fixing");
        } else if (event.type === "audit_progress") {
          setAuditProgress({
            current: event.section_index + 1,
            total: event.total,
            label: event.section_title,
          });
        } else if (event.type === "audit_complete") {
          setAuditSummary({
            strong: event.strong,
            weak: event.weak,
            gaps: event.gaps,
            totalQuestions: event.total_questions,
          });
        } else if (event.type === "fix_result") {
          const rec = event.recommendation;
          setFixes((prev) => [...prev, rec]);
          setFixProgress({ current: rec.fix_index + 1, total: rec.total_fixes });
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
    setState(fixes.length > 0 ? "complete" : "idle");
  };

  const isRunning = state === "auditing" || state === "fixing";

  return (
    <div className="space-y-4">
      {/* Action bar */}
      <div className="flex flex-wrap items-center gap-3">
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
        <div className="flex items-center gap-2">
          <label className="text-xs" style={{ color: "var(--muted)" }}>
            Max fixes
          </label>
          <input
            type="number"
            value={maxFixes}
            onChange={(e) => setMaxFixes(Math.max(1, Math.min(50, Number(e.target.value))))}
            className="w-20 px-2 py-1.5 rounded-lg border text-sm"
            style={{
              background: "var(--card)",
              borderColor: "var(--border)",
              color: "var(--foreground)",
            }}
            min={1}
            max={50}
          />
        </div>
        <button
          onClick={isRunning ? handleCancel : handleRun}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: isRunning ? "var(--danger)" : "var(--accent)",
            color: "#fff",
          }}
        >
          {isRunning ? "Cancel" : "Run Audit + Fix"}
        </button>
      </div>

      {/* Pipeline stepper */}
      {(isRunning || state === "complete") && (
        <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
          <span
            className="px-2 py-0.5 rounded"
            style={{
              background: state === "auditing" ? "var(--accent)" : "#22c55e",
              color: "#fff",
            }}
          >
            1. Audit
          </span>
          <span>→</span>
          <span
            className="px-2 py-0.5 rounded"
            style={{
              background:
                state === "fixing"
                  ? "var(--accent)"
                  : state === "complete"
                  ? "#22c55e"
                  : "var(--border)",
              color: state === "fixing" || state === "complete" ? "#fff" : "var(--muted)",
            }}
          >
            2. Fix
          </span>
        </div>
      )}

      {/* Audit progress */}
      {state === "auditing" && (
        <ProgressBar
          current={auditProgress.current}
          total={auditProgress.total}
          label="Auditing..."
          sublabel={auditProgress.label}
        />
      )}

      {/* Audit summary */}
      {auditSummary && (
        <div
          className="rounded-lg border p-3 flex flex-wrap gap-4 text-xs"
          style={{ borderColor: "var(--border)", background: "var(--card)" }}
        >
          <span>Audit complete:</span>
          <span style={{ color: "#22c55e" }}>{auditSummary.strong} strong</span>
          <span style={{ color: "#f59e0b" }}>{auditSummary.weak} weak</span>
          <span style={{ color: "#ef4444" }}>{auditSummary.gaps} gaps</span>
          <span style={{ color: "var(--muted)" }}>{auditSummary.totalQuestions} questions tested</span>
        </div>
      )}

      {/* Fix progress */}
      {state === "fixing" && fixProgress.total > 0 && (
        <ProgressBar
          current={fixProgress.current}
          total={fixProgress.total}
          label="Generating fixes..."
        />
      )}

      {/* Fix cards */}
      {fixes.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>
            Fix Recommendations ({fixes.length})
          </h3>
          {fixes.map((rec, i) => (
            <FixCard key={i} recommendation={rec} />
          ))}
        </div>
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

      {/* Empty state */}
      {state === "idle" && fixes.length === 0 && (
        <EmptyState
          title="Fix documentation gaps"
          description="Run a two-phase pipeline: first audit your docs to find weak spots, then generate actionable fix recommendations with draft content you can copy into your docs."
          command={`kb-arena fix --corpus ${corpus}`}
        />
      )}
    </div>
  );
}
