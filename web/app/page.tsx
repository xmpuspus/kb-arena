"use client";

import Link from "next/link";
import { STRATEGIES, STRATEGY_LABELS, STRATEGY_COLORS, CORPORA, type Strategy } from "@/lib/api";

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
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
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
  naive_vector: "Embed documents as chunks, retrieve by cosine similarity. Baseline approach — fast, simple, no context awareness.",
  contextual_vector: "Embed chunks with parent document context prepended. Better semantic grounding than naive, same retrieval speed.",
  qna_pairs: "Pre-generate Q&A pairs from documents using an LLM, then embed and retrieve the pairs. High precision on anticipated questions.",
  knowledge_graph: "Extract entities and relationships into Neo4j. Query with Cypher templates matched to question intent. Best on relational and multi-hop.",
  hybrid: "Route by intent: factoid → vector, relational → graph, complex → both with Reciprocal Rank Fusion. Adapts per question.",
};

export default function Home() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-16">
      {/* Hero */}
      <section className="space-y-4">
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          KB Arena
        </h1>
        <p className="text-lg leading-relaxed max-w-3xl" style={{ color: "var(--muted)" }}>
          Benchmark knowledge graphs vs vector RAG on real documentation. 200 questions, 5 strategies, 3 corpora — empirical evidence for which retrieval architecture works.
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
            { step: "1", title: "Same question", desc: "Each question from a tiered difficulty set is sent to all 5 strategies simultaneously." },
            { step: "2", title: "4-pass evaluation", desc: "Structural checks, entity coverage, source attribution, then LLM-as-judge scoring." },
            { step: "3", title: "Ranked report", desc: "Accuracy by tier, latency percentiles, reliability rates, and cross-strategy composite ranking." },
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
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>The 5 strategies</h2>
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
        <h2 className="text-xl font-semibold" style={{ color: "var(--foreground)" }}>200 questions across 5 difficulty tiers</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {CORPORA.map((c) => (
            <div
              key={c.value}
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--card)" }}
            >
              <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--foreground)" }}>{c.label}</h3>
              <p className="text-xs" style={{ color: "var(--muted)" }}>
                {c.value === "python-stdlib" ? "75 questions" : c.value === "kubernetes" ? "65 questions" : "60 questions"}
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
