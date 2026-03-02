"use client";

import { useState } from "react";
import GraphViewer, { type GraphNode, type GraphEdge } from "@/components/GraphViewer";
import { CORPORA } from "@/lib/api";

// AWS services knowledge graph — extracted from AWS documentation
const SAMPLE_NODES: GraphNode[] = [
  { id: "lambda", label: "Lambda", type: "Service", properties: { category: "Compute" } },
  { id: "api-gw", label: "API Gateway", type: "Service", properties: { category: "Networking" } },
  { id: "rds", label: "RDS", type: "Service", properties: { category: "Database" } },
  { id: "s3", label: "S3", type: "Service", properties: { category: "Storage" } },
  { id: "ec2", label: "EC2", type: "Service", properties: { category: "Compute" } },
  { id: "vpc", label: "VPC", type: "Service", properties: { category: "Networking" } },
  { id: "iam", label: "IAM", type: "Service", properties: { category: "Security" } },
  { id: "cloudfront", label: "CloudFront", type: "Service", properties: { category: "Networking" } },
  { id: "dynamodb", label: "DynamoDB", type: "Service", properties: { category: "Database" } },
  { id: "sqs", label: "SQS", type: "Service", properties: { category: "Integration" } },
  { id: "sns", label: "SNS", type: "Service", properties: { category: "Integration" } },
  { id: "ecs", label: "ECS", type: "Service", properties: { category: "Compute" } },
  { id: "cloudwatch", label: "CloudWatch", type: "Service", properties: { category: "Monitoring" } },
  { id: "route53", label: "Route 53", type: "Service", properties: { category: "Networking" } },
  { id: "alb", label: "ALB", type: "Resource", properties: { service: "ELB" } },
  { id: "sg", label: "Security Group", type: "Resource", properties: { service: "VPC" } },
  { id: "subnet", label: "Subnet", type: "Resource", properties: { service: "VPC" } },
  { id: "nat-gw", label: "NAT Gateway", type: "Resource", properties: { service: "VPC" } },
  { id: "exec-role", label: "Execution Role", type: "Policy", properties: { service: "IAM" } },
  { id: "vpc-endpoint", label: "VPC Endpoint", type: "Resource", properties: { service: "VPC" } },
  { id: "acm", label: "ACM Certificate", type: "Resource", properties: { service: "ACM" } },
  { id: "waf", label: "WAF", type: "Service", properties: { category: "Security" } },
  { id: "secrets-mgr", label: "Secrets Manager", type: "Service", properties: { category: "Security" } },
  { id: "ecr", label: "ECR", type: "Service", properties: { category: "Compute" } },
  { id: "eni", label: "ENI", type: "Resource", properties: { service: "VPC" } },
];

const SAMPLE_EDGES: GraphEdge[] = [
  { id: "e1", source: "api-gw", target: "lambda", label: "INVOKES" },
  { id: "e2", source: "lambda", target: "rds", label: "CONNECTS_TO" },
  { id: "e3", source: "lambda", target: "vpc", label: "DEPLOYED_IN" },
  { id: "e4", source: "lambda", target: "exec-role", label: "ASSUMES" },
  { id: "e5", source: "lambda", target: "s3", label: "READS_FROM" },
  { id: "e6", source: "lambda", target: "dynamodb", label: "READS_WRITES" },
  { id: "e7", source: "lambda", target: "sqs", label: "TRIGGERED_BY" },
  { id: "e8", source: "lambda", target: "eni", label: "CREATES" },
  { id: "e9", source: "lambda", target: "cloudwatch", label: "LOGS_TO" },
  { id: "e10", source: "vpc", target: "subnet", label: "CONTAINS" },
  { id: "e11", source: "vpc", target: "sg", label: "CONTAINS" },
  { id: "e12", source: "vpc", target: "nat-gw", label: "CONTAINS" },
  { id: "e13", source: "vpc", target: "vpc-endpoint", label: "CONTAINS" },
  { id: "e14", source: "sg", target: "rds", label: "PROTECTS" },
  { id: "e15", source: "sg", target: "lambda", label: "ATTACHED_TO" },
  { id: "e16", source: "subnet", target: "rds", label: "HOSTS" },
  { id: "e17", source: "cloudfront", target: "s3", label: "ORIGIN" },
  { id: "e18", source: "cloudfront", target: "alb", label: "ORIGIN" },
  { id: "e19", source: "cloudfront", target: "acm", label: "USES" },
  { id: "e20", source: "cloudfront", target: "waf", label: "PROTECTED_BY" },
  { id: "e21", source: "route53", target: "cloudfront", label: "ROUTES_TO" },
  { id: "e22", source: "route53", target: "alb", label: "ROUTES_TO" },
  { id: "e23", source: "alb", target: "ecs", label: "FORWARDS_TO" },
  { id: "e24", source: "ecs", target: "ecr", label: "PULLS_FROM" },
  { id: "e25", source: "ecs", target: "cloudwatch", label: "LOGS_TO" },
  { id: "e26", source: "sns", target: "sqs", label: "PUBLISHES_TO" },
  { id: "e27", source: "sns", target: "lambda", label: "TRIGGERS" },
  { id: "e28", source: "exec-role", target: "iam", label: "MANAGED_BY" },
  { id: "e29", source: "lambda", target: "secrets-mgr", label: "READS_FROM" },
  { id: "e30", source: "nat-gw", target: "subnet", label: "DEPLOYED_IN" },
];

export default function GraphPage() {
  const [corpus, setCorpus] = useState("aws-compute");
  const [nodes] = useState<GraphNode[]>(SAMPLE_NODES);
  const [edges] = useState<GraphEdge[]>(SAMPLE_EDGES);

  const maxDegree = Math.max(
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
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
            Knowledge graph
          </h1>
          <p className="text-sm mt-1" style={{ color: "var(--muted)" }}>
            Explore the AWS service dependency graph extracted from documentation. Hover to highlight neighbors, drag to pan, scroll to zoom, double-click to focus.
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
          { label: "Max degree", value: stats.maxDegree },
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
          Entities and relationships are extracted from AWS documentation using service-specific schemas — AWS Compute has
          node types for Services, Resources, Policies, and Features, with relationship types like INVOKES, DEPLOYED_IN,
          ASSUMES, CONNECTS_TO, and TRIGGERED_BY. The graph is stored in Neo4j and queried with Cypher templates
          matched to question intent.
        </p>
        <p className="text-xs mt-2" style={{ color: "var(--muted)" }}>
          Showing sample data extracted from AWS documentation. Connect to Neo4j to explore the full graph.
        </p>
      </div>
    </div>
  );
}
