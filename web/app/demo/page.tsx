"use client";

import { useState, useRef } from "react";
import ChatPanel, { type DemoResult } from "@/components/ChatPanel";
import { STRATEGIES, STRATEGY_LABELS, CORPORA, type Strategy, type Message } from "@/lib/api";

const DEMO_QUESTION = "How do I expose a StatefulSet through an Ingress with TLS?";

const DEMO_RESULTS: Partial<Record<Strategy, DemoResult>> = {
  naive_vector: {
    answer: "To expose a StatefulSet, you can create a Service and then configure an Ingress resource. The Ingress should have TLS configured with a certificate stored in a Kubernetes Secret.\n\nHere's the general approach:\n1. Create a Service targeting your StatefulSet pods\n2. Create a TLS Secret with your certificate\n3. Create an Ingress with the TLS section referencing the Secret\n\nNote: StatefulSets use stable network identities, so each pod gets a predictable hostname.",
    sources: ["concepts/services-networking/ingress.md", "concepts/services-networking/service.md"],
    latencyMs: 420,
    tokensUsed: 134,
    costUsd: 0.0012,
  },
  contextual_vector: {
    answer: "Exposing a StatefulSet through an Ingress with TLS involves several Kubernetes resources working together.\n\nFirst, create a headless Service (clusterIP: None) for stable DNS entries per pod. Then create a regular ClusterIP Service for the Ingress to route to. The Ingress resource needs a TLS section with:\n- A Secret containing tls.crt and tls.key\n- The host matching your certificate's CN/SAN\n\nStatefulSets with volumeClaimTemplates get stable persistent storage per pod. The Ingress controller handles TLS termination at the edge.",
    sources: ["concepts/services-networking/ingress.md", "concepts/workloads/controllers/statefulset.md", "concepts/services-networking/service.md"],
    latencyMs: 510,
    tokensUsed: 167,
    costUsd: 0.0018,
  },
  knowledge_graph: {
    answer: "Exposing a StatefulSet through an Ingress with TLS requires connecting 5 Kubernetes resources in a specific dependency chain:\n\nStatefulSet → headless Service (clusterIP: None) → provides stable DNS (pod-0.svc.cluster.local)\nStatefulSet → volumeClaimTemplates → PersistentVolumeClaim (one per pod)\nService (ClusterIP) → targets StatefulSet pods via selector\nIngress → references Service as backend\nIngress → references TLS Secret (tls.crt + tls.key)\n\nThe headless Service is required for StatefulSet pod identity. A separate ClusterIP Service is needed because Ingress controllers cannot route to headless Services. The TLS Secret must contain a valid certificate matching the Ingress host.\n\nKey constraint: volumeClaimTemplates in the StatefulSet spec are immutable after creation — plan storage before deploying.",
    sources: ["concepts/workloads/controllers/statefulset.md", "concepts/services-networking/ingress.md", "concepts/services-networking/service.md", "concepts/storage/persistent-volumes.md"],
    latencyMs: 890,
    tokensUsed: 203,
    costUsd: 0.0031,
  },
  hybrid: {
    answer: "This is a relational question requiring graph traversal (5 interconnected resources). Routing to knowledge graph path.\n\nThe dependency chain for StatefulSet → Ingress with TLS:\n\n1. StatefulSet defines pods with volumeClaimTemplates for persistent storage (creates one PVC per replica)\n2. Headless Service (clusterIP: None) provides stable DNS: {pod-name}.{service}.{namespace}.svc.cluster.local\n3. ClusterIP Service routes traffic to StatefulSet pods (Ingress cannot target headless Services)\n4. TLS Secret stores the certificate (tls.crt) and private key (tls.key)\n5. Ingress references the ClusterIP Service as backend and the TLS Secret for termination\n\nCommon pitfall: using the headless Service directly in the Ingress backend — this fails silently. You need both a headless Service (for pod identity) and a regular Service (for Ingress routing).\n\nDownstream effects of StatefulSet + PVC: pod scheduling is constrained to nodes where the PV is accessible. With ReadWriteOnce access mode, pods cannot be rescheduled to a different availability zone.",
    sources: ["concepts/workloads/controllers/statefulset.md", "concepts/services-networking/ingress.md", "concepts/services-networking/service.md", "concepts/storage/persistent-volumes.md", "concepts/scheduling-eviction/assign-pod-node.md"],
    latencyMs: 1050,
    tokensUsed: 248,
    costUsd: 0.0038,
  },
  qna_pairs: {
    answer: "To expose a StatefulSet via Ingress with TLS:\n\n1. Create a headless Service for pod DNS identity\n2. Create a ClusterIP Service for Ingress routing\n3. Store your TLS certificate in a Secret\n4. Configure Ingress with TLS termination pointing to the ClusterIP Service\n\nMake sure your volumeClaimTemplates are configured before deployment since they can't be changed later.",
    sources: ["concepts/services-networking/ingress.md", "concepts/workloads/controllers/statefulset.md"],
    latencyMs: 680,
    tokensUsed: 112,
    costUsd: 0.0024,
  },
};

export default function DemoPage() {
  const [query, setQuery] = useState(DEMO_QUESTION);
  const [corpus, setCorpus] = useState("kubernetes");
  const [selectedStrategies, setSelectedStrategies] = useState<Strategy[]>([...STRATEGIES]);
  const [trigger, setTrigger] = useState(0);
  const [history, setHistory] = useState<Message[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setHistory((prev) => [...prev, { role: "user", content: q }]);
    setTrigger((t) => t + 1);
  }

  function toggleStrategy(s: Strategy) {
    setSelectedStrategies((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  }

  function handleClear() {
    setQuery("");
    setHistory([]);
    setTrigger(0);
    inputRef.current?.focus();
  }

  const gridCols =
    selectedStrategies.length <= 2
      ? "grid-cols-1 sm:grid-cols-2"
      : selectedStrategies.length <= 3
      ? "grid-cols-1 sm:grid-cols-3"
      : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
          Strategy comparison
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
          Ask a question and see how each retrieval strategy responds side-by-side.
        </p>
      </div>

      {/* Controls */}
      <div className="space-y-3">
        {/* Strategy toggles */}
        <div className="flex flex-wrap gap-2">
          {STRATEGIES.map((s) => {
            const active = selectedStrategies.includes(s);
            return (
              <button
                key={s}
                onClick={() => toggleStrategy(s)}
                className="text-xs px-3 py-1.5 rounded-lg border transition-all"
                style={{
                  borderColor: active ? "var(--accent)" : "var(--border)",
                  background: active ? "var(--accent)" : "transparent",
                  color: active ? "#fff" : "var(--muted)",
                  opacity: active ? 1 : 0.6,
                }}
              >
                {STRATEGY_LABELS[s]}
              </button>
            );
          })}
        </div>

        {/* Query input */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <select
            value={corpus}
            onChange={(e) => setCorpus(e.target.value)}
            className="px-3 py-2 rounded-lg border text-sm shrink-0"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            {CORPORA.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about the documentation..."
            className="flex-1 px-4 py-2 rounded-lg border text-sm outline-none"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          />
          <button
            type="submit"
            disabled={!query.trim() || selectedStrategies.length === 0}
            className="px-4 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "#fff" }}
          >
            Ask
          </button>
          <button
            type="button"
            onClick={handleClear}
            className="px-3 py-2 rounded-lg border text-sm transition-opacity hover:opacity-70"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            Clear
          </button>
        </form>

        {/* Example queries */}
        <div className="flex flex-wrap gap-2">
          <span className="text-xs" style={{ color: "var(--muted)" }}>Try:</span>
          {[
            "How do I expose a StatefulSet through an Ingress with TLS?",
            "Compare ConfigMap vs Secret vs ServiceAccount in Kubernetes",
            "Which executives serve on boards of companies with material litigation?",
            "What's the chain of modules when urllib makes an HTTPS connection?",
          ].map((q) => (
            <button
              key={q}
              onClick={() => { setQuery(q); inputRef.current?.focus(); }}
              className="text-xs px-2.5 py-1 rounded-lg border transition-opacity hover:opacity-70 text-left"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              {q.length > 50 ? q.slice(0, 50) + "..." : q}
            </button>
          ))}
        </div>
      </div>

      {/* Chat panels grid */}
      {selectedStrategies.length === 0 ? (
        <div className="text-center py-16">
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            Select at least one strategy above to begin.
          </p>
        </div>
      ) : (
        <div className={`grid ${gridCols} gap-4`}>
          {selectedStrategies.map((s) => (
            <ChatPanel
              key={s}
              strategy={s}
              query={query}
              corpus={corpus}
              history={history}
              trigger={trigger}
              demoResult={trigger === 0 ? DEMO_RESULTS[s] : undefined}
            />
          ))}
        </div>
      )}
    </div>
  );
}
