"use client";

import { useState, useEffect } from "react";
import { fetchCorporaResult, type CorpusInfo } from "@/lib/api";
import GenerateTab from "@/components/tools/GenerateTab";
import AuditTab from "@/components/tools/AuditTab";
import FixTab from "@/components/tools/FixTab";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";

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
  const [failed, setFailed] = useState(false);
  // A read still running is not a server that holds no corpus, and the empty
  // list looks the same either way.
  const [pending, setPending] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setPending(true);
    fetchCorporaResult().then((result) => {
      // Every tool here acts on the corpus this list names. The built-in
      // default under a failed read would send a fix or an audit at a corpus
      // this deployment may not hold.
      setFailed(result.failed);
      setCorpora(result.failed ? [] : result.corpora);
      if (!result.failed && result.corpora.length > 0) {
        setCorpus((prev) => prev || result.corpora[0].value);
      }
      setPending(false);
    });
  }, [attempt]);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 space-y-6">
      <StateBanner />

      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Documentation Tools
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Generate Q&A pairs, audit documentation quality, and get fix recommendations.
        </p>
      </div>

      {pending && (
        <p className="text-sm" style={{ color: "var(--muted)" }} role="status">
          Reading the corpus list...
        </p>
      )}

      {!pending && !failed && corpora.length === 0 && (
        <div
          className="rounded-lg border border-dashed p-6 text-sm"
          style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        >
          This server holds no corpus. Run <code className="mono">kb-arena ingest</code> to add
          one, then reload this page.
        </div>
      )}

      {failed && (
        <FetchError
          title="The corpus list did not load"
          message="The API did not answer with the corpora this deployment holds."
          hint="The built-in list would send an audit or a fix at a corpus this server may not hold, so the tools stay off. Start the API, then try again."
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="tools-corpus" className="text-xs font-medium" style={{ color: "var(--muted)" }}>Corpus</label>
          <select
            id="tools-corpus"
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            disabled={failed || pending}
            className="px-3 py-1.5 rounded-lg border text-sm disabled:opacity-50"
            style={{ background: "var(--card)", borderColor: "var(--border-strong)", color: "var(--foreground)" }}
          >
            {/* Every tool acts on this corpus, so a failed read leaves nothing
                to pick and nothing to start. */}
            {failed && <option value="">Corpus list unavailable</option>}
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
              disabled={failed || pending}
              title={failed ? "The corpus list did not load, so the tools stay off" : undefined}
              className="px-3 py-1.5 text-xs font-medium transition-colors disabled:opacity-40"
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
      {corpus && !failed && (
        <>
          {tab === "generate" && <GenerateTab corpus={corpus} />}
          {tab === "audit" && <AuditTab corpus={corpus} />}
          {tab === "fix" && <FixTab corpus={corpus} />}
        </>
      )}
    </div>
  );
}
