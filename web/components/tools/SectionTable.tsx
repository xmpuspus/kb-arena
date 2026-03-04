"use client";

import { Fragment, useState } from "react";
import type { SectionAuditResult } from "@/lib/tools-api";

type Filter = "all" | "strong" | "weak" | "gap";

interface Props {
  sections: SectionAuditResult[];
}

const CLASS_COLORS: Record<string, { bg: string; text: string; label: string }> = {
  strong: { bg: "rgba(34, 197, 94, 0.08)", text: "#22c55e", label: "Strong" },
  weak: { bg: "rgba(245, 158, 11, 0.08)", text: "#f59e0b", label: "Weak" },
  gap: { bg: "rgba(239, 68, 68, 0.08)", text: "#ef4444", label: "Gap" },
};

export default function SectionTable({ sections }: Props) {
  const [filter, setFilter] = useState<Filter>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<"title" | "accuracy" | "questions">("accuracy");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const filtered = filter === "all" ? sections : sections.filter((s) => s.classification === filter);

  const sorted = [...filtered].sort((a, b) => {
    let cmp = 0;
    if (sortKey === "title") cmp = a.section_title.localeCompare(b.section_title);
    else if (sortKey === "accuracy") cmp = a.avg_accuracy - b.avg_accuracy;
    else cmp = a.questions_tested - b.questions_tested;
    return sortDir === "asc" ? cmp : -cmp;
  });

  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  };

  const counts = {
    all: sections.length,
    strong: sections.filter((s) => s.classification === "strong").length,
    weak: sections.filter((s) => s.classification === "weak").length,
    gap: sections.filter((s) => s.classification === "gap").length,
  };

  return (
    <div className="space-y-3">
      {/* Filter tabs */}
      <div className="flex gap-1">
        {(["all", "strong", "weak", "gap"] as Filter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className="px-3 py-1 text-xs font-medium rounded-lg capitalize transition-colors"
            style={{
              background: filter === f ? "var(--accent)" : "transparent",
              color: filter === f ? "#fff" : "var(--muted)",
            }}
          >
            {f} ({counts[f]})
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ background: "var(--background)" }}>
              <th className="text-left px-3 py-2 text-xs font-medium cursor-pointer" style={{ color: "var(--muted)" }} onClick={() => handleSort("title")}>
                Section {sortKey === "title" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th className="text-left px-3 py-2 text-xs font-medium" style={{ color: "var(--muted)" }}>Doc</th>
              <th className="text-right px-3 py-2 text-xs font-medium cursor-pointer" style={{ color: "var(--muted)" }} onClick={() => handleSort("accuracy")}>
                Accuracy {sortKey === "accuracy" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th className="text-right px-3 py-2 text-xs font-medium cursor-pointer" style={{ color: "var(--muted)" }} onClick={() => handleSort("questions")}>
                Questions {sortKey === "questions" ? (sortDir === "asc" ? "↑" : "↓") : ""}
              </th>
              <th className="text-center px-3 py-2 text-xs font-medium" style={{ color: "var(--muted)" }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => {
              const cls = CLASS_COLORS[s.classification] ?? CLASS_COLORS.gap;
              const isExpanded = expandedId === s.section_id;
              return (
                <Fragment key={s.section_id}>
                  <tr
                    className="border-t cursor-pointer hover:opacity-80 transition-opacity"
                    style={{ borderColor: "var(--border)", background: cls.bg }}
                    onClick={() => setExpandedId(isExpanded ? null : s.section_id)}
                  >
                    <td className="px-3 py-2" style={{ color: "var(--foreground)" }}>{s.section_title}</td>
                    <td className="px-3 py-2 text-xs" style={{ color: "var(--muted)" }}>{s.doc_id}</td>
                    <td className="px-3 py-2 text-right mono" style={{ color: cls.text }}>
                      {Math.round(s.avg_accuracy * 100)}%
                    </td>
                    <td className="px-3 py-2 text-right mono" style={{ color: "var(--muted)" }}>{s.questions_tested}</td>
                    <td className="px-3 py-2 text-center">
                      <span className="text-xs px-2 py-0.5 rounded" style={{ background: cls.text, color: "#fff" }}>
                        {cls.label}
                      </span>
                    </td>
                  </tr>
                  {isExpanded && s.question_results.length > 0 && (
                    <tr style={{ background: "var(--background)" }}>
                      <td colSpan={5} className="px-6 py-3 border-t" style={{ borderColor: "var(--border)" }}>
                        <div className="space-y-2">
                          <p className="text-xs font-medium" style={{ color: "var(--muted)" }}>Question Details</p>
                          {s.question_results.map((q, qi) => (
                            <div key={qi} className="flex items-start gap-3 text-xs">
                              <span className="mono shrink-0" style={{ color: q.accuracy >= 0.7 ? "#22c55e" : q.accuracy >= 0.3 ? "#f59e0b" : "#ef4444" }}>
                                {Math.round(q.accuracy * 100)}%
                              </span>
                              <span style={{ color: "var(--foreground)" }}>{q.question_text}</span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
        {sorted.length === 0 && (
          <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--muted)" }}>
            No sections match the selected filter.
          </div>
        )}
      </div>
    </div>
  );
}
