"use client";

import { useState, useRef, useEffect } from "react";
import ChatPanel, { type DemoResult } from "@/components/ChatPanel";
import { STRATEGIES, STRATEGY_LABELS, CORPORA, fetchCorpora, type Strategy, type Message } from "@/lib/api";

const DEMO_QUESTION = "How do I set up a Lambda function behind API Gateway with VPC access to an RDS database?";

const DEMO_RESULTS: Partial<Record<Strategy, DemoResult>> = {
  naive_vector: {
    answer: "To connect Lambda to API Gateway and RDS, you need to:\n1. Create a Lambda function with your application code\n2. Create an API Gateway REST API and connect it to Lambda\n3. Configure VPC settings on the Lambda function to access RDS\n4. Set up the RDS database in a private subnet\n\nMake sure the Lambda function has the right IAM permissions and that the security groups allow traffic between Lambda and RDS.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "apigateway/latest/developerguide/getting-started.html"],
    latencyMs: 420,
    tokensUsed: 134,
    costUsd: 0.0012,
  },
  contextual_vector: {
    answer: "Setting up Lambda behind API Gateway with VPC access to RDS involves configuring several AWS services together.\n\nFirst, deploy your RDS instance in private subnets within a VPC. Then configure your Lambda function with VPC settings — attach it to the same VPC's private subnets and assign a security group. The Lambda security group must allow outbound traffic to the RDS security group on port 3306 (MySQL) or 5432 (PostgreSQL).\n\nFor API Gateway, create a REST API with a Lambda proxy integration. The API Gateway invokes Lambda via the AWS service network, so it doesn't need VPC access itself.\n\nKey consideration: Lambda functions in a VPC need a NAT Gateway to access the internet (e.g., for calling other AWS APIs). Use VPC endpoints for services like S3 and DynamoDB to avoid NAT costs.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html", "apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html"],
    latencyMs: 510,
    tokensUsed: 167,
    costUsd: 0.0018,
  },
  knowledge_graph: {
    answer: "This requires connecting 6 AWS resources in a specific dependency chain:\n\nAPI Gateway (REST API) → Lambda Function (proxy integration)\nLambda Function → IAM Execution Role (AssumeRole: lambda.amazonaws.com)\nLambda Function → VPC Configuration (SubnetIds + SecurityGroupIds)\nVPC → Private Subnets (where both Lambda ENIs and RDS reside)\nSecurity Group (Lambda) → Security Group (RDS) on port 5432\nRDS Instance → Subnet Group (private subnets across 2+ AZs)\n\nThe IAM execution role needs: AWSLambdaVPCAccessExecutionRole (for ENI creation in VPC), plus any RDS-specific permissions if using IAM database authentication.\n\nCritical constraint: Lambda in a VPC loses direct internet access. If your function needs to call external APIs or other AWS services, you need either a NAT Gateway ($0.045/hr + data) or VPC endpoints for specific services.\n\nCold start impact: VPC-attached Lambda adds 1-10s cold start for ENI attachment. Use Provisioned Concurrency to mitigate this for latency-sensitive APIs.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html", "apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html", "lambda/latest/dg/lambda-intro-execution-role.html"],
    latencyMs: 890,
    tokensUsed: 203,
    costUsd: 0.0031,
  },
  hybrid: {
    answer: "This is a relational question requiring graph traversal (6 interconnected AWS resources). Routing to knowledge graph path.\n\nThe dependency chain for API Gateway → Lambda → RDS:\n\n1. API Gateway REST API with Lambda proxy integration handles HTTP routing and invokes Lambda\n2. Lambda Execution Role (IAM) grants lambda.amazonaws.com AssumeRole + AWSLambdaVPCAccessExecutionRole managed policy\n3. Lambda VPC Configuration attaches the function to private subnets via Elastic Network Interfaces (ENIs)\n4. Security Group chain: Lambda SG allows outbound to RDS SG on port 5432/3306; RDS SG allows inbound from Lambda SG\n5. RDS Subnet Group spans 2+ AZs in the same VPC private subnets\n6. Optional: NAT Gateway or VPC Endpoints if Lambda needs internet or AWS API access\n\nCommon pitfall: forgetting AWSLambdaVPCAccessExecutionRole on the execution role — Lambda silently fails to create ENIs and times out. Another: placing Lambda in public subnets thinking it will have internet access — it won't without a NAT Gateway.\n\nCost implications: NAT Gateway adds ~$32/month base + $0.045/GB processed. For high-throughput APIs, consider VPC endpoints ($0.01/hr each) for frequently accessed AWS services like S3, DynamoDB, and Secrets Manager.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html", "apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html", "lambda/latest/dg/lambda-intro-execution-role.html", "vpc/latest/userguide/vpc-nat-gateway.html"],
    latencyMs: 1050,
    tokensUsed: 248,
    costUsd: 0.0038,
  },
  qna_pairs: {
    answer: "To set up Lambda behind API Gateway with VPC access to RDS:\n\n1. Deploy RDS in private subnets with a DB subnet group\n2. Create a Lambda function and attach it to the same VPC private subnets\n3. Configure security groups: Lambda SG → RDS SG on the database port\n4. Attach AWSLambdaVPCAccessExecutionRole to the Lambda execution role\n5. Create an API Gateway REST API with Lambda proxy integration\n\nRemember that VPC-attached Lambda needs a NAT Gateway for internet access and adds cold start latency.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html"],
    latencyMs: 680,
    tokensUsed: 112,
    costUsd: 0.0024,
  },
  raptor: {
    answer: "Setting up Lambda behind API Gateway with VPC access to RDS requires coordinating several AWS services across networking, compute, and database layers.\n\nAt the networking level, your VPC needs private subnets (for Lambda ENIs and RDS instances) across at least 2 AZs, security groups allowing Lambda-to-RDS traffic on the database port, and optionally a NAT Gateway if Lambda needs internet access for external API calls.\n\nFor the compute layer, the Lambda function must be configured with VPC settings (SubnetIds and SecurityGroupIds) and an execution role that includes AWSLambdaVPCAccessExecutionRole for ENI management. Be aware that VPC-attached Lambda functions experience 1-10s cold starts due to ENI attachment.\n\nThe API Gateway REST API connects to Lambda via proxy integration, routing HTTP requests to your function. API Gateway itself doesn't need VPC access — it invokes Lambda through the AWS service network.\n\nKey trade-off: VPC attachment adds cold start latency but is required for RDS access. Use Provisioned Concurrency for latency-sensitive APIs.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html", "apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html"],
    latencyMs: 720,
    tokensUsed: 189,
    costUsd: 0.0028,
  },
  pageindex: {
    answer: "Based on the documentation structure, this question spans three main topic areas: Lambda VPC configuration, API Gateway integration, and RDS networking.\n\nLambda VPC Configuration: Attach your Lambda function to private subnets in the same VPC as RDS. The function needs an execution role with AWSLambdaVPCAccessExecutionRole to create Elastic Network Interfaces (ENIs) in your subnets. Lambda functions in a VPC lose default internet access — add a NAT Gateway or VPC endpoints if needed.\n\nAPI Gateway Setup: Create a REST API with Lambda proxy integration. API Gateway invokes Lambda via the AWS internal network, so no VPC configuration is needed on the API Gateway side. The proxy integration passes the full HTTP request to your function.\n\nRDS Connectivity: Place RDS in a DB subnet group spanning private subnets across 2+ AZs. Configure security groups so the Lambda security group can reach the RDS security group on port 3306 (MySQL) or 5432 (PostgreSQL).\n\nCritical path: API Gateway → Lambda (proxy integration) → VPC ENI → Private Subnet → RDS. Cold starts add 1-10s for ENI attachment; use Provisioned Concurrency for production APIs.",
    sources: ["lambda/latest/dg/configuration-vpc.html", "AmazonRDS/latest/UserGuide/USER_VPC.html", "apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html"],
    latencyMs: 950,
    tokensUsed: 215,
    costUsd: 0.0032,
  },
};

export default function DemoPage() {
  const [query, setQuery] = useState(DEMO_QUESTION);
  const [corpus, setCorpus] = useState("aws-compute");
  const [corpora, setCorpora] = useState(CORPORA);
  const [selectedStrategies, setSelectedStrategies] = useState<Strategy[]>([...STRATEGIES]);
  const [trigger, setTrigger] = useState(0);
  const [history, setHistory] = useState<Message[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { fetchCorpora().then(setCorpora); }, []);

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
          Ask a question about your documentation and see how each retrieval strategy responds side-by-side.
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
            {corpora.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your documentation..."
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
            "How do I set up a Lambda function behind API Gateway with VPC access to RDS?",
            "Compare S3 Standard vs Glacier vs Glacier Deep Archive storage classes",
            "What IAM policies does an ECS task need to pull from ECR and write to CloudWatch?",
            "How does CloudFront route to an ALB origin with WAF and ACM certificate?",
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

      {/* Pre-computed example note */}
      {trigger === 0 && selectedStrategies.length > 0 && (
        <div
          className="px-3 py-2 rounded-lg text-xs"
          style={{ background: "var(--border)", color: "var(--muted)" }}
        >
          Showing a pre-computed example. Submit a query to get live results from your data.
        </div>
      )}

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
              key={`${s}-${trigger}`}
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
