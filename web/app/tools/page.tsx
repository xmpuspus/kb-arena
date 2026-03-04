"use client";

import { useState, useEffect } from "react";
import { fetchCorpora, type CorpusInfo } from "@/lib/api";
import GenerateTab from "@/components/tools/GenerateTab";
import AuditTab from "@/components/tools/AuditTab";
import FixTab from "@/components/tools/FixTab";

type Tab = "generate" | "audit" | "fix";

const TAB_LABELS: Record<Tab, string> = {
  generate: "Generate Q&A",
  audit: "Audit",
  fix: "Fix Docs",
};

export default function ToolsPage() {
  const [tab, setTab] = useState<Tab>("generate");
  const [corpus, setCorpus] = useState("");
  const [corpora, setCorpora] = useState<CorpusInfo[]>([]);

  useEffect(() => {
    fetchCorpora().then((data) => {
      setCorpora(data);
      if (data.length > 0) setCorpus((prev) => prev || data[0].value);
    });
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Documentation Tools
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Generate Q&A pairs, audit documentation quality, and get fix recommendations.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium" style={{ color: "var(--muted)" }}>Corpus</label>
          <select
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            className="px-3 py-1.5 rounded-lg border text-sm"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            {corpora.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>

        <div className="flex rounded-lg border overflow-hidden" style={{ borderColor: "var(--border)" }}>
          {(["generate", "audit", "fix"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-3 py-1.5 text-xs font-medium transition-colors"
              style={{
                background: tab === t ? "var(--accent)" : "transparent",
                color: tab === t ? "#fff" : "var(--muted)",
              }}
            >
              {TAB_LABELS[t]}
            </button>
          ))}
        </div>
      </div>

      {/* Active tab */}
      {corpus && (
        <>
          {tab === "generate" && <GenerateTab corpus={corpus} />}
          {tab === "audit" && <AuditTab corpus={corpus} />}
          {tab === "fix" && <FixTab corpus={corpus} />}
        </>
      )}
    </div>
  );
}
