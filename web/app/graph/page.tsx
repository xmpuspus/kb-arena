"use client";

import { useState } from "react";
import GraphViewer, { type GraphNode, type GraphEdge } from "@/components/GraphViewer";
import { CORPORA } from "@/lib/api";

// Sample graph data for demonstration when Neo4j is not connected
const SAMPLE_NODES: GraphNode[] = [
  { id: "json", label: "json", type: "Module" },
  { id: "json.JSONDecodeError", label: "JSONDecodeError", type: "Exception", properties: { module: "json" } },
  { id: "json.loads", label: "json.loads", type: "Function", properties: { module: "json", returns: "Any" } },
  { id: "json.dumps", label: "json.dumps", type: "Function", properties: { module: "json", returns: "str" } },
  { id: "json.JSONEncoder", label: "JSONEncoder", type: "Class", properties: { module: "json" } },
  { id: "json.JSONDecoder", label: "JSONDecoder", type: "Class", properties: { module: "json" } },
  { id: "json.load", label: "json.load", type: "Function", properties: { module: "json" } },
  { id: "json.dump", label: "json.dump", type: "Function", properties: { module: "json" } },
  { id: "io", label: "io", type: "Module" },
  { id: "io.StringIO", label: "StringIO", type: "Class", properties: { module: "io" } },
  { id: "io.BytesIO", label: "BytesIO", type: "Class", properties: { module: "io" } },
  { id: "os.path", label: "os.path", type: "Module" },
  { id: "pathlib", label: "pathlib", type: "Module" },
  { id: "pathlib.Path", label: "Path", type: "Class", properties: { module: "pathlib" } },
  { id: "pathlib.PurePath", label: "PurePath", type: "Class", properties: { module: "pathlib" } },
  { id: "collections", label: "collections", type: "Module" },
  { id: "collections.OrderedDict", label: "OrderedDict", type: "Class", properties: { module: "collections" } },
  { id: "collections.defaultdict", label: "defaultdict", type: "Class", properties: { module: "collections" } },
  { id: "collections.namedtuple", label: "namedtuple", type: "Function", properties: { module: "collections" } },
  { id: "typing", label: "typing", type: "Module" },
  { id: "typing.Any", label: "Any", type: "Class", properties: { module: "typing" } },
  { id: "str_param", label: "s: str", type: "Parameter" },
  { id: "obj_param", label: "obj: Any", type: "Parameter" },
];

const SAMPLE_EDGES: GraphEdge[] = [
  { id: "e1", source: "json", target: "json.loads", label: "CONTAINS" },
  { id: "e2", source: "json", target: "json.dumps", label: "CONTAINS" },
  { id: "e3", source: "json", target: "json.load", label: "CONTAINS" },
  { id: "e4", source: "json", target: "json.dump", label: "CONTAINS" },
  { id: "e5", source: "json", target: "json.JSONEncoder", label: "CONTAINS" },
  { id: "e6", source: "json", target: "json.JSONDecoder", label: "CONTAINS" },
  { id: "e7", source: "json.loads", target: "json.JSONDecodeError", label: "RAISES" },
  { id: "e8", source: "json.load", target: "json.JSONDecodeError", label: "RAISES" },
  { id: "e9", source: "json.loads", target: "str_param", label: "REQUIRES" },
  { id: "e10", source: "json.dumps", target: "obj_param", label: "REQUIRES" },
  { id: "e11", source: "json.loads", target: "typing.Any", label: "RETURNS" },
  { id: "e12", source: "json.dumps", target: "typing.Any", label: "RETURNS" },
  { id: "e13", source: "io", target: "io.StringIO", label: "CONTAINS" },
  { id: "e14", source: "io", target: "io.BytesIO", label: "CONTAINS" },
  { id: "e15", source: "json.load", target: "io.StringIO", label: "REFERENCES" },
  { id: "e16", source: "pathlib", target: "pathlib.Path", label: "CONTAINS" },
  { id: "e17", source: "pathlib", target: "pathlib.PurePath", label: "CONTAINS" },
  { id: "e18", source: "pathlib.Path", target: "pathlib.PurePath", label: "INHERITS" },
  { id: "e19", source: "pathlib.Path", target: "os.path", label: "REFERENCES" },
  { id: "e20", source: "collections", target: "collections.OrderedDict", label: "CONTAINS" },
  { id: "e21", source: "collections", target: "collections.defaultdict", label: "CONTAINS" },
  { id: "e22", source: "collections", target: "collections.namedtuple", label: "CONTAINS" },
  { id: "e23", source: "typing", target: "typing.Any", label: "CONTAINS" },
];

export default function GraphPage() {
  const [corpus, setCorpus] = useState("python-stdlib");
  const [nodes] = useState<GraphNode[]>(SAMPLE_NODES);
  const [edges] = useState<GraphEdge[]>(SAMPLE_EDGES);

  const stats = {
    nodes: nodes.length,
    edges: edges.length,
    types: Array.from(new Set(nodes.map((n) => n.type))).length,
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
            Knowledge graph
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
            Explore the extracted entity-relationship graph. Click nodes to inspect, scroll to zoom, drag to move.
          </p>
        </div>
        <select
          value={corpus}
          onChange={(e) => setCorpus(e.target.value)}
          className="px-3 py-1.5 rounded-lg border text-sm"
          style={{ background: "var(--card)", borderColor: "var(--border)", color: "var(--foreground)" }}
        >
          {CORPORA.map((c) => (
            <option key={c.value} value={c.value}>{c.label}</option>
          ))}
        </select>
      </div>

      {/* Stats bar */}
      <div className="flex gap-6">
        {[
          { label: "Nodes", value: stats.nodes },
          { label: "Edges", value: stats.edges },
          { label: "Types", value: stats.types },
        ].map((s) => (
          <div key={s.label} className="text-center">
            <p className="text-xl font-bold mono" style={{ color: "var(--foreground)" }}>{s.value}</p>
            <p className="text-xs" style={{ color: "var(--muted)" }}>{s.label}</p>
          </div>
        ))}
      </div>

      {/* Graph */}
      <div
        className="rounded-lg border overflow-hidden"
        style={{ borderColor: "var(--border)", height: "calc(100vh - 300px)", minHeight: 500 }}
      >
        <GraphViewer
          nodes={nodes}
          edges={edges}
        />
      </div>

      {/* Info panel */}
      <div
        className="rounded-lg border p-4"
        style={{ borderColor: "var(--border)", background: "var(--card)" }}
      >
        <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--foreground)" }}>About the graph</h3>
        <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
          The knowledge graph is built by the LLM entity extractor during <code className="mono">kb-arena build-graph</code>.
          Entities and relationships are extracted using corpus-specific schemas — Python stdlib has 10 node types
          (Module, Class, Function, Parameter, ReturnType, Exception, etc.) and 10 relationship types
          (CONTAINS, REQUIRES, RETURNS, RAISES, INHERITS, etc.). The graph is stored in Neo4j and queried
          with Cypher templates matched to question intent.
        </p>
        <p className="text-xs mt-2" style={{ color: "var(--muted)" }}>
          Showing sample data. Connect to Neo4j to explore the full graph.
        </p>
      </div>
    </div>
  );
}
