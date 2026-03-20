"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { STRATEGIES, STRATEGY_LABELS, STRATEGY_COLORS, CORPORA, fetchCorpora, type CorpusInfo, type Strategy } from "@/lib/api";

const TIER_LABELS = [
  "Tier 1 — Factoid",
  "Tier 2 — Procedural",
  "Tier 3 — Comparative",
  "Tier 4 — Relational",
  "Tier 5 — Multi-hop",
];

function StrategyCard({ label, desc, color }: { label: string; desc: string; color: string }) {
  return (
    <div
      className="rounded-lg border p-4 flex flex-col gap-2"
      style={{ borderColor: "var(--border)", background: "var(--card)", boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}
    >
      <div className="flex items-center gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
        <h3 className="text-sm font-semibold" style={{ color: "var(--foreground)" }}>{label}</h3>
      </div>
      <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>{desc}</p>
    </div>
  );
}

const STRATEGY_DESCS: Record<Strategy, string> = {
  naive_vector: "Embed doc pages as chunks, retrieve by cosine similarity. Baseline approach — fast, simple, no cross-topic awareness.",
  contextual_vector: "Embed chunks with parent topic context prepended. Better at disambiguating domain-specific terms across large documentation sets.",
  qna_pairs: "Pre-generate Q&A pairs from docs using an LLM, then embed and retrieve the pairs. High precision on common domain questions.",
  knowledge_graph: "Extract entities, components, and dependencies into Neo4j. Query with Cypher templates matched to question intent. Best on multi-topic architectures.",
  hybrid: "Route by intent: factoid → vector, cross-topic → graph, complex → both with Reciprocal Rank Fusion. Adapts per question.",
  raptor: "Build a recursive tree of LLM cluster summaries over the corpus. Query all levels simultaneously — leaf chunks + broad topic synthesis for Tier 4/5 questions.",
  pageindex: "Vectorless, reasoning-based retrieval. Builds a hierarchical tree index from document structure, then uses LLM reasoning to traverse the tree — no embeddings, no chunking.",
  bm25: "Classic keyword matching with BM25Okapi scoring. The lexical baseline — no embeddings, no LLM retrieval. Shows whether neural retrieval adds value for your docs.",
};

export default function Home() {
  const [corpora, setCorpora] = useState<CorpusInfo[]>(CORPORA);

  useEffect(() => { fetchCorpora().then(setCorpora); }, []);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
      {/* Hero */}
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          KB Arena
        </h1>
        <p className="text-lg leading-relaxed max-w-3xl" style={{ color: "var(--muted)" }}>
          Which retrieval architecture works best for your documentation? 7 strategies, tiered difficulty questions — empirical evidence so you don&apos;t have to guess.
        </p>
        <div className="flex gap-3 pt-2">
          <Link
            href="/demo"
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity hover:opacity-80"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Try the demo
          </Link>
          <Link
            href="/benchmark"
            className="px-4 py-2 rounded-lg text-sm font-medium border transition-opacity hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            View benchmarks
          </Link>
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
            { step: "1", title: "Same question", desc: "Each question — from simple lookups to multi-topic dependency chains — is sent to all 7 strategies simultaneously." },
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
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>The 7 strategies</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {STRATEGIES.map((s) => (
            <StrategyCard
              key={s}
              label={STRATEGY_LABELS[s]}
              desc={STRATEGY_DESCS[s]}
              color={STRATEGY_COLORS[s]}
            />
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
                {c.questionCount != null ? `${c.questionCount} questions` : "—"}
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
            "Anthropic Claude", "OpenAI Embeddings", "Next.js 14", "Tailwind CSS", "Recharts",
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
