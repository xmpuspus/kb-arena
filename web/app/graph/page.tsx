"use client";

import { useState, useEffect, useRef } from "react";
import GraphViewer, { type GraphNode, type GraphEdge } from "@/components/GraphViewer";
import {
  CORPORA,
  fetchCorpora,
  fetchGraphData,
  triggerGraphBuild,
  streamGraphBuild,
  type CorpusInfo,
} from "@/lib/api";
import { useTokenEpoch } from "@/lib/useTokenEpoch";
import StateBanner from "@/components/StateBanner";
import FetchError from "@/components/FetchError";
import { useScopeReset } from "@/lib/useScopeReset";

// The example map, shown only when a read succeeds and reports no graph
// database behind it. A failed read never reaches it: sample entities under a
// corpus name read as that corpus's graph.
const SAMPLE_NODES: GraphNode[] = [
  { id: "lambda", label: "Lambda", type: "Topic", properties: { category: "Compute" } },
  { id: "api-gw", label: "API Gateway", type: "Topic", properties: { category: "Networking" } },
  { id: "rds", label: "RDS", type: "Topic", properties: { category: "Database" } },
  { id: "s3", label: "S3", type: "Topic", properties: { category: "Storage" } },
  { id: "ec2", label: "EC2", type: "Topic", properties: { category: "Compute" } },
  { id: "vpc", label: "VPC", type: "Topic", properties: { category: "Networking" } },
  { id: "iam", label: "IAM", type: "Topic", properties: { category: "Security" } },
  { id: "cloudfront", label: "CloudFront", type: "Topic", properties: { category: "Networking" } },
  { id: "dynamodb", label: "DynamoDB", type: "Topic", properties: { category: "Database" } },
  { id: "sqs", label: "SQS", type: "Topic", properties: { category: "Integration" } },
  { id: "sns", label: "SNS", type: "Topic", properties: { category: "Integration" } },
  { id: "ecs", label: "ECS", type: "Topic", properties: { category: "Compute" } },
  { id: "cloudwatch", label: "CloudWatch", type: "Topic", properties: { category: "Monitoring" } },
  { id: "route53", label: "Route 53", type: "Topic", properties: { category: "Networking" } },
  { id: "alb", label: "ALB", type: "Component", properties: { parent: "ELB" } },
  { id: "sg", label: "Security Group", type: "Component", properties: { parent: "VPC" } },
  { id: "subnet", label: "Subnet", type: "Component", properties: { parent: "VPC" } },
  { id: "nat-gw", label: "NAT Gateway", type: "Component", properties: { parent: "VPC" } },
  { id: "exec-role", label: "Execution Role", type: "Constraint", properties: { parent: "IAM" } },
  { id: "vpc-endpoint", label: "VPC Endpoint", type: "Component", properties: { parent: "VPC" } },
  { id: "acm", label: "ACM Certificate", type: "Config", properties: { parent: "ACM" } },
  { id: "waf", label: "WAF", type: "Topic", properties: { category: "Security" } },
  { id: "secrets-mgr", label: "Secrets Manager", type: "Topic", properties: { category: "Security" } },
  { id: "ecr", label: "ECR", type: "Topic", properties: { category: "Compute" } },
  { id: "eni", label: "ENI", type: "Component", properties: { parent: "VPC" } },
];

const SAMPLE_EDGES: GraphEdge[] = [
  { id: "e1", source: "api-gw", target: "lambda", label: "TRIGGERS" },
  { id: "e2", source: "lambda", target: "rds", label: "CONNECTS_TO" },
  { id: "e3", source: "vpc", target: "lambda", label: "CONTAINS" },
  { id: "e4", source: "lambda", target: "exec-role", label: "DEPENDS_ON" },
  { id: "e5", source: "lambda", target: "s3", label: "CONNECTS_TO" },
  { id: "e6", source: "lambda", target: "dynamodb", label: "CONNECTS_TO" },
  { id: "e7", source: "sqs", target: "lambda", label: "TRIGGERS" },
  { id: "e8", source: "lambda", target: "eni", label: "DEPENDS_ON" },
  { id: "e9", source: "lambda", target: "cloudwatch", label: "CONNECTS_TO" },
  { id: "e10", source: "vpc", target: "subnet", label: "CONTAINS" },
  { id: "e11", source: "vpc", target: "sg", label: "CONTAINS" },
  { id: "e12", source: "vpc", target: "nat-gw", label: "CONTAINS" },
  { id: "e13", source: "vpc", target: "vpc-endpoint", label: "CONTAINS" },
  { id: "e14", source: "sg", target: "rds", label: "CONFIGURES" },
  { id: "e15", source: "sg", target: "lambda", label: "CONFIGURES" },
  { id: "e16", source: "subnet", target: "rds", label: "CONTAINS" },
  { id: "e17", source: "cloudfront", target: "s3", label: "CONNECTS_TO" },
  { id: "e18", source: "cloudfront", target: "alb", label: "CONNECTS_TO" },
  { id: "e19", source: "cloudfront", target: "acm", label: "DEPENDS_ON" },
  { id: "e20", source: "cloudfront", target: "waf", label: "DEPENDS_ON" },
  { id: "e21", source: "route53", target: "cloudfront", label: "CONNECTS_TO" },
  { id: "e22", source: "route53", target: "alb", label: "CONNECTS_TO" },
  { id: "e23", source: "alb", target: "ecs", label: "CONNECTS_TO" },
  { id: "e24", source: "ecs", target: "ecr", label: "DEPENDS_ON" },
  { id: "e25", source: "ecs", target: "cloudwatch", label: "CONNECTS_TO" },
  { id: "e26", source: "sns", target: "sqs", label: "CONNECTS_TO" },
  { id: "e27", source: "sns", target: "lambda", label: "TRIGGERS" },
  { id: "e28", source: "exec-role", target: "iam", label: "DEPENDS_ON" },
  { id: "e29", source: "lambda", target: "secrets-mgr", label: "CONNECTS_TO" },
  { id: "e30", source: "vpc", target: "nat-gw", label: "CONTAINS" },
];

function apiToGraphNodes(data: { id: string; name: string; type: string; description?: string }[]): GraphNode[] {
  return data.map((n) => {
    const props: Record<string, string> = {};
    if (n.description) props.description = n.description;
    return { id: n.id, label: n.name, type: n.type, properties: props };
  });
}

function apiToGraphEdges(data: { source: string; target: string; type: string }[]): GraphEdge[] {
  return data.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    label: e.type,
  }));
}

export default function GraphPage() {
  const [corpus, setCorpus] = useState("aws-compute");
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [corpora, setCorpora] = useState<CorpusInfo[]>(CORPORA);
  const [connected, setConnected] = useState(false);
  const [sampleMap, setSampleMap] = useState(false);
  const [loading, setLoading] = useState(true);
  // A refused read, distinct from the graph database being unreachable.
  const [readError, setReadError] = useState("");
  const [attempt, setAttempt] = useState(0);
  // Entering a token is the moment a refused read should be tried again.
  const tokenEpoch = useTokenEpoch();
  const [buildStatus, setBuildStatus] = useState<"idle" | "building" | "done" | "error">("idle");
  const [buildProgress, setBuildProgress] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const buildEpochRef = useRef(0);
  // Only the newest read writes the graph or clears the pending state. The
  // reply for the corpus the reader left can land last, and it turned the
  // pending state off for the corpus now on screen.
  const readTicket = useRef(0);

  useEffect(() => { fetchCorpora().then(setCorpora); }, []);

  // The corpus name over the map changed, so the entities under the old one go
  // before the new read starts.
  useScopeReset(corpus, () => {
    setNodes([]);
    setEdges([]);
    setSampleMap(false);
    setConnected(false);
    setReadError("");
    setLoading(true);
  });

  useEffect(() => () => {
    buildEpochRef.current += 1;
    abortRef.current?.abort();
  }, []);

  async function handleLiveBuild() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const buildEpoch = buildEpochRef.current + 1;
    buildEpochRef.current = buildEpoch;
    const buildCorpus = corpus;
    const isCurrentBuild = () =>
      buildEpochRef.current === buildEpoch && !controller.signal.aborted;
    setBuildStatus("building");
    setBuildProgress("Starting...");
    // A read still in flight when the build starts describes the graph the
    // build is replacing. Retiring the ticket here stops its failure from
    // writing an error over a build that is already streaming entities.
    //
    // The read owned the pending state, and only its own handler cleared it.
    // Retiring the ticket alone left the page loading for ever, so the build
    // takes that state over here. `buildStatus` is the indicator from now on.
    readTicket.current += 1;
    setLoading(false);
    // A build streams its own entities, so the earlier read failure no longer
    // describes what is on screen.
    setReadError("");
    // Clear existing graph so nodes animate in fresh
    setNodes([]);
    setEdges([]);
    setSampleMap(false);
    try {
      const build = await triggerGraphBuild(buildCorpus);
      if (!isCurrentBuild()) return;
      for await (const event of streamGraphBuild(build.build_id, controller.signal)) {
        if (!isCurrentBuild()) break;
        if (event.type === "started") {
          setBuildProgress(`Extracting ${event.total_sections} sections...`);
        } else if (event.type === "entity") {
          setNodes((prev) => {
            if (prev.find((n) => n.id === event.id)) return prev;
            return [...prev, { id: event.id, label: event.name, type: event.nodeType, properties: {} }];
          });
        } else if (event.type === "relationship") {
          setEdges((prev) => {
            const id = `live-${prev.length}`;
            return [...prev, { id, source: event.source, target: event.target, label: event.relType }];
          });
        } else if (event.type === "section_done") {
          setBuildProgress(`Processing documents...`);
        } else if (event.type === "complete") {
          const data = await fetchGraphData(buildCorpus);
          if (!isCurrentBuild()) break;
          setConnected(data.connected);
          if (!data.connected) {
            // The build just ran on this corpus, so the example map here would
            // read as the graph it produced. Show nothing and say so.
            setNodes([]);
            setEdges([]);
            setSampleMap(false);
            setBuildStatus("error");
            setBuildProgress("The graph build finished, and the saved graph did not load.");
            continue;
          }
          setSampleMap(false);
          setNodes(apiToGraphNodes(data.nodes));
          setEdges(apiToGraphEdges(data.edges));
          setBuildStatus("done");
          setBuildProgress(`Complete: ${event.total_entities} entities, ${event.total_relationships} relationships`);
        } else if (event.type === "error") {
          setBuildStatus("error");
          setBuildProgress(event.message);
        }
      }
    } catch (err: unknown) {
      if (isCurrentBuild() && err instanceof Error && err.name !== "AbortError") {
        setBuildStatus("error");
        setBuildProgress(err.message);
      }
    } finally {
      if (isCurrentBuild()) {
        setBuildStatus((status) => (status === "building" ? "idle" : status));
      }
    }
  }

  function handleCorpusChange(nextCorpus: string) {
    buildEpochRef.current += 1;
    abortRef.current?.abort();
    setBuildStatus("idle");
    setBuildProgress("");
    setCorpus(nextCorpus);
  }

  useEffect(() => {
    let active = true;
    // A live build advances the epoch. Without this check, a slow first response
    // lands after the build finishes and overwrites the graph it just produced.
    const fetchEpoch = buildEpochRef.current;
    const ticket = ++readTicket.current;
    const isCurrentRead = () => active && ticket === readTicket.current;
    setLoading(true);
    setReadError("");
    fetchGraphData(corpus)
      .catch((err: unknown) => {
        // A refusal is not an outage of the graph database, so it must not
        // set `connected: false` and show an empty graph as a real answer.
        // The example map goes too, because it would read as this corpus.
        if (isCurrentRead()) {
          setNodes([]);
          setEdges([]);
          setSampleMap(false);
          setConnected(false);
          setReadError(err instanceof Error ? err.message : "The graph did not load.");
        }
        return null;
      })
      .then((data) => {
      // A reply for the corpus the reader left says nothing about this one,
      // and clearing the pending state on it names the new corpus over the
      // last corpus's map.
      if (!isCurrentRead()) return;
      if (data === null) {
        setLoading(false);
        return;
      }
      if (buildEpochRef.current !== fetchEpoch) {
        setLoading(false);
        return;
      }
      setConnected(data.connected);
      if (data.connected) {
        setSampleMap(false);
        setNodes(apiToGraphNodes(data.nodes));
        setEdges(apiToGraphEdges(data.edges));
      } else {
        // The server answered, and it reports no graph database behind it.
        // That is the sample state, and the banner names it.
        setSampleMap(true);
        setNodes(SAMPLE_NODES);
        setEdges(SAMPLE_EDGES);
      }
      setLoading(false);
    });
    return () => {
      active = false;
    };
  }, [corpus, attempt, tokenEpoch]);




  const maxDegree = Math.max(
    0,
    ...nodes.map((n) => edges.filter((e) => e.source === n.id || e.target === n.id).length),
  );

  const stats = {
    nodes: nodes.length,
    edges: edges.length,
    types: Array.from(new Set(nodes.map((n) => n.type))).length,
    maxDegree,
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      <StateBanner
        sample={
          sampleMap
            ? "This server reports no Neo4j connection, so the map below is a built-in example."
            : undefined
        }
      />

      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
            Knowledge graph
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
            Explore the entity dependency graph extracted from your documentation. Hover to highlight neighbors, drag to pan, scroll to zoom, double-click to focus.
          </p>
        </div>
        <div className="flex items-center gap-2 sm:shrink-0">
          <select
            aria-label="Corpus"
            value={corpus}
            onChange={(e) => handleCorpusChange(e.target.value)}
            disabled={buildStatus === "building"}
            className="px-3 py-1.5 rounded-lg border text-sm"
            style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            {corpora.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
          <button
            onClick={handleLiveBuild}
            disabled={buildStatus === "building"}
            className="px-3 py-1.5 rounded-lg text-sm font-medium transition-colors"
            style={{
              background: buildStatus === "building" ? "var(--border)" : "var(--accent)",
              color: buildStatus === "building" ? "var(--muted)" : "#fff",
              cursor: buildStatus === "building" ? "not-allowed" : "pointer",
            }}
          >
            {buildStatus === "building" ? "Building..." : "Build live"}
          </button>
        </div>
      </div>

      {/* Status banner */}
      {readError && !loading && (
        <FetchError
          title="The graph did not load"
          message={readError}
          hint="A failed read is not an empty graph, so the map stays off screen. Start the API, or enter an API token with the key button, then try again."
          onRetry={() => setAttempt((n) => n + 1)}
        />
      )}

      {!readError && !connected && !loading && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ background: "var(--border)", color: "var(--muted)" }}
        >
          Showing sample data because Neo4j is not connected. Run <code className="mono">docker compose up -d neo4j</code> and <code className="mono">kb-arena build-graph --corpus {corpus}</code> to see your graph.
        </div>
      )}

      {connected && !loading && nodes.length === 0 && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ background: "var(--border)", color: "var(--muted)" }}
        >
          Connected to Neo4j, but this corpus has no graph data. Run <code className="mono">kb-arena build-graph --corpus {corpus}</code> to build it.
        </div>
      )}

      {loading && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ background: "var(--border)", color: "var(--muted)" }}
        >
          Loading graph data...
        </div>
      )}

      {buildProgress && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{
            background: buildStatus === "error" ? "#fef2f2" : buildStatus === "done" ? "#f0fdf4" : "var(--border)",
            color: buildStatus === "error" ? "#dc2626" : buildStatus === "done" ? "#16a34a" : "var(--muted)",
          }}
        >
          {buildProgress}
        </div>
      )}

      {/* Stats bar and graph. A failed read shows neither, and neither does a
          read still running: zero nodes under a corpus name reads as a corpus
          with no entities, and the last corpus's nodes read as this one's. */}
      {!readError && !loading && (
        <>
          <div className="flex gap-6">
            {[
              { label: "Nodes", value: stats.nodes },
              { label: "Edges", value: stats.edges },
              { label: "Types", value: stats.types },
              { label: "Max degree", value: stats.maxDegree },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <p className="text-xl font-bold mono" style={{ color: "var(--foreground)" }}>{s.value}</p>
                <p className="text-xs" style={{ color: "var(--muted)" }}>{s.label}</p>
              </div>
            ))}
          </div>

          <div
            className="rounded-lg border overflow-hidden"
            style={{ borderColor: "var(--border)", height: "calc(100vh - 300px)", minHeight: 500 }}
          >
            <GraphViewer
              nodes={nodes}
              edges={edges}
            />
          </div>
        </>
      )}

      {/* Info panel */}
      <div
        className="rounded-lg border p-4"
        style={{ borderColor: "var(--border)", background: "var(--card)" }}
      >
        <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--foreground)" }}>About the graph</h3>
        <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
          The knowledge graph is built by the LLM entity extractor during <code className="mono">kb-arena build-graph</code>.
          Entities and relationships use a shared schema: Topics, Components, Processes, Configs,
          and Constraints, connected by DEPENDS_ON, CONTAINS, CONNECTS_TO, TRIGGERS, CONFIGURES, ALTERNATIVE_TO,
          and EXTENDS relationships. The graph is stored in Neo4j and queried with Cypher templates matched to question intent.
        </p>
      </div>
    </div>
  );
}
