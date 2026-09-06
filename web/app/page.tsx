"use client";

import { useEffect, useState } from "react";
import {
  CORPORA,
  DEFAULT_STRATEGY_CATALOG,
  STRATEGY_COLORS,
  STRATEGY_DESCRIPTIONS,
  fetchCorpora,
  fetchStrategyCatalog,
  type CorpusInfo,
  type Strategy,
  type StrategyCatalogRecord,
} from "@/lib/api";
import StateBanner from "@/components/StateBanner";
import { useServerState } from "@/components/ServerStateProvider";

const TIER_LABELS = [
  "Tier 1: Factoid",
  "Tier 2: Procedural",
  "Tier 3: Comparative",
  "Tier 4: Relational",
  "Tier 5: Multi-hop",
];

function StrategyCard({ record }: { record: StrategyCatalogRecord }) {
  const name = record.name as Strategy;
  return (
    <div
      className="rounded-lg border p-4 flex flex-col gap-2"
      style={{ borderColor: "var(--border)", background: "var(--card)", boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}
    >
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: STRATEGY_COLORS[name] }} />
        <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>{record.label}</h3>
      </div>
      <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>{STRATEGY_DESCRIPTIONS[name]}</p>
      <p className="text-xs" style={{ color: record.status === "loaded" ? "var(--success)" : "var(--muted)" }}>
        {record.status === "loaded"
          ? "Loaded"
          : record.status === "unknown"
            ? "Runtime status unavailable"
            : record.optional_extra
              ? `Optional: ${record.optional_extra}`
              : "Unavailable"}
        {record.experimental ? " | Experimental" : ""}
      </p>
    </div>
  );
}

export default function Home() {
  const [corpora, setCorpora] = useState<CorpusInfo[]>(CORPORA);
  const [catalog, setCatalog] = useState<StrategyCatalogRecord[]>(DEFAULT_STRATEGY_CATALOG);
  const { state } = useServerState();

  useEffect(() => {
    fetchCorpora().then(setCorpora);
    fetchStrategyCatalog().then(setCatalog);
  }, []);

  // With no server answer, both lists below are the built-in defaults. They
  // name what KB Arena ships, not what this deployment loaded.
  const sample =
    state === "unreachable"
      ? "The strategy catalog and the corpus list below are the built-in defaults."
      : undefined;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
      <StateBanner sample={sample} />

      {/* Hero */}
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          KB Arena
        </h1>
        <p className="text-lg leading-relaxed max-w-3xl" style={{ color: "var(--muted)" }}>
          Compare retrieval architectures on the same documentation and questions, then choose from recorded quality, latency, cost, and limits.
        </p>
        <div className="flex gap-3 pt-2">
          <a
            href="/demo/"
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Try the demo
          </a>
          <a
            href="/benchmark/"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            View benchmarks
          </a>
          <a
            href="https://github.com/xmpuspus/kb-arena"
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            GitHub
          </a>
        </div>
      </section>

      {/* How it works */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>How it works</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { step: "1", title: "Same question", desc: "Send each question, from direct lookups to multi-topic chains, to the selected strategies." },
            { step: "2", title: "4-pass evaluation", desc: "Structural checks, entity coverage, source attribution against your docs, then LLM-as-judge scoring." },
            { step: "3", title: "Ranked report", desc: "Accuracy by tier, latency percentiles, reliability rates, and cross-strategy composite ranking across your documentation." },
          ].map((item) => (
            <div
              key={item.step}
              className="rounded-lg border p-4 space-y-2"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <span
                className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                {item.step}
              </span>
              <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>{item.title}</h3>
              <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Strategies */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>Strategy catalog</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {catalog.map((record) => (
            <StrategyCard key={record.name} record={record} />
          ))}
        </div>
      </section>

      {/* Question tiers */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>5 difficulty tiers, auto-generated or hand-crafted</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {corpora.map((c) => (
            <div
              key={c.value}
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--foreground)" }}>{c.label}</h3>
              <p className="text-xs" style={{ color: "var(--muted)" }}>
                {c.questionCount != null ? `${c.questionCount} questions` : "Not labeled"}
              </p>
            </div>
          ))}
        </div>
        <div className="flex flex-wrap gap-2 pt-1">
          {TIER_LABELS.map((t, i) => (
            <span
              key={i}
              className="text-xs px-2.5 py-1 rounded-full border"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              {t}
            </span>
          ))}
        </div>
      </section>

      {/* Tech stack */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>Built with</h2>
        <div className="flex flex-wrap gap-2">
          {[
            "Python 3.11+", "Pydantic v2", "FastAPI", "Neo4j 5", "ChromaDB",
            "Anthropic Claude", "OpenAI Embeddings", "Next.js 16", "Tailwind CSS", "Recharts",
          ].map((tech) => (
            <span
              key={tech}
              className="text-xs px-3 py-1.5 rounded-lg border"
              style={{ borderColor: "var(--border)", color: "var(--muted)", background: "var(--card)" }}
            >
              {tech}
            </span>
          ))}
        </div>
      </section>
    </div>
  );
}
