"use client";

import { useState } from "react";
import type { FixRecommendation } from "@/lib/tools-api";

interface Props {
  recommendation: FixRecommendation;
}

const PRIORITY_COLORS: Record<string, { bg: string; text: string }> = {
  high: { bg: "rgba(239, 68, 68, 0.1)", text: "#ef4444" },
  medium: { bg: "rgba(245, 158, 11, 0.1)", text: "#f59e0b" },
  low: { bg: "rgba(34, 197, 94, 0.1)", text: "#22c55e" },
};

export default function FixCard({ recommendation: rec }: Props) {
  const [copied, setCopied] = useState(false);
  const [showQuestions, setShowQuestions] = useState(false);

  const priorityLevel = rec.priority <= 3 ? "high" : rec.priority <= 6 ? "medium" : "low";
  const colors = PRIORITY_COLORS[priorityLevel];

  const handleCopy = async () => {
    await navigator.clipboard.writeText(rec.suggested_content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="rounded-lg border p-4 space-y-3" style={{ borderColor: "var(--border)", background: "var(--card)" }}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold px-2 py-0.5 rounded" style={{ background: colors.bg, color: colors.text }}>
            #{rec.priority}
          </span>
          <span className="text-sm font-medium" style={{ color: "var(--foreground)" }}>{rec.section_title}</span>
          <span className="text-xs" style={{ color: "var(--muted)" }}>{rec.doc_id}</span>
        </div>
        <span className="mono text-xs px-2 py-0.5 rounded" style={{ background: "var(--border)", color: rec.current_accuracy < 0.3 ? "#ef4444" : "#f59e0b" }}>
          {Math.round(rec.current_accuracy * 100)}%
        </span>
      </div>

      {/* Diagnosis */}
      <p className="text-sm" style={{ color: "var(--foreground)" }}>{rec.diagnosis}</p>

      {/* Suggested content */}
      <div className="relative">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-medium" style={{ color: "var(--muted)" }}>Suggested content</span>
          <button onClick={handleCopy} className="text-xs px-2 py-0.5 rounded border transition-colors hover:opacity-70" style={{ borderColor: "var(--border)", color: "var(--accent)" }}>
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <pre className="mono text-xs p-3 rounded-lg border overflow-x-auto whitespace-pre-wrap" style={{ background: "var(--background)", borderColor: "var(--border)", color: "var(--foreground)" }}>
          {rec.suggested_content}
        </pre>
      </div>

      {/* Metadata */}
      <div className="flex flex-wrap gap-3 text-xs" style={{ color: "var(--muted)" }}>
        <span><strong>Placement:</strong> {rec.placement}</span>
        <span><strong>Impact:</strong> {rec.estimated_impact}</span>
      </div>

      {/* Failing questions */}
      {rec.failing_questions.length > 0 && (
        <div>
          <button onClick={() => setShowQuestions(!showQuestions)} className="text-xs transition-colors hover:opacity-70" style={{ color: "var(--accent)" }}>
            {showQuestions ? "Hide" : "Show"} failing questions ({rec.failing_questions.length})
          </button>
          {showQuestions && (
            <ul className="mt-2 space-y-1">
              {rec.failing_questions.map((q, i) => (
                <li key={i} className="text-xs pl-3 border-l-2" style={{ borderColor: "#ef4444", color: "var(--muted)" }}>
                  {q}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
