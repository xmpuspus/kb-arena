# KB Arena — Build Prompt (Enhance & Migrate)

Copy this entire prompt into a new Claude Code session from `~/Desktop/kb-arena/`.

---

## PROMPT START

Ultrawork this. Read PLAN.md first, then CLAUDE.md for conventions. The codebase already exists — your job is to **migrate, enhance, and verify** it. Execute in parallel agent teams. Don't stop until everything passes tests, looks polished, and is visually verified with screenshots.

### Context

KB Arena benchmarks 5 retrieval strategies (vector RAG vs knowledge graphs) on **AWS documentation** — the most sprawling, cross-referenced, and inconsistently structured professional documentation in existence. 200+ services, each with overlapping concepts (VPCs appear in EC2, Lambda, RDS, ECS docs), implicit dependencies (Lambda needs an Execution Role, which needs IAM policies, which reference S3 ARNs), and documentation spread across guides, API references, and FAQs with no unified structure. This makes it the perfect stress test for retrieval architectures.

The project proves empirically: knowledge graphs beat vector RAG on multi-hop, relational, and comparative queries. Pure vector wins on simple lookups but collapses when queries touch 3+ interconnected AWS services.

**3 AWS corpora:**
- **aws-compute** — Lambda, EC2, ECS, Fargate, Batch, Step Functions (75 questions)
- **aws-storage** — S3, EBS, EFS, FSx, Glacier, Storage Gateway (65 questions)
- **aws-networking** — VPC, Route 53, CloudFront, ALB/NLB, API Gateway, Direct Connect (60 questions)

### Current State

The codebase is **partially migrated** from its original Python-stdlib/Kubernetes/SEC-EDGAR content to AWS. The code structure, models, strategies, frontend, and graph viewer all exist and work. What remains:

**Backend (partially migrated):**
- `kb_arena/graph/schema.py` — Still has Python/K8s/SEC schemas in `_CORPUS_SCHEMA`. Needs AWS node/rel types: Service, Resource, Policy, Feature + DEPENDS_ON, INVOKES, CONNECTS_TO, ASSUMES, etc.
- `kb_arena/strategies/knowledge_graph.py` — Mock data still references json/json.loads Python examples. CYPHER_GEN_PROMPT still has Python node/rel types. Needs AWS examples and schema.
- `kb_arena/chatbot/router.py` — AWS corpus references partially done. Verify keywords cover AWS service names.

**Question files (old content, need full rewrite):**
- `datasets/python-stdlib/questions/*.yaml` — DELETE these entirely
- `datasets/kubernetes/questions/*.yaml` — DELETE these entirely
- `datasets/sec-edgar/questions/*.yaml` — DELETE these entirely
- CREATE new `datasets/aws-compute/questions/*.yaml`, `datasets/aws-storage/questions/*.yaml`, `datasets/aws-networking/questions/*.yaml` with 200 total questions across 5 tiers

**Tests (old fixtures):**
- `tests/conftest.py` — Fixtures reference Python stdlib (json.loads, os.path.join). Migrate to AWS examples (Lambda functions, S3 bucket policies).
- `tests/integration/test_ingest_to_vector.py` — Sample data references Python modules. Migrate to AWS service documents.
- All other test files — scan for Python/K8s/SEC references, replace with AWS equivalents.

**Frontend (theme done, content partially migrated):**
- Cloudwright light theme is applied and builds clean
- `web/app/page.tsx` — AWS content mostly in place. Verify all references.
- `web/app/demo/page.tsx` — Pre-filled with AWS Lambda question. Verify answers.
- `web/app/graph/page.tsx` — 25 AWS service nodes, 30 edges already defined. Verify completeness.
- `web/app/benchmark/page.tsx` — Methodology text updated. Verify.
- `web/components/GraphViewer.tsx` — Already rewritten (~600 lines, Gephi-quality canvas). DO NOT MODIFY unless broken.
- `web/components/BenchmarkTable.tsx` — Footer references AWS. Verify.

**Screenshots (ALL stale, need retake):**
- `docs/screenshot-home.png` — Styled but shows pre-AWS content
- `docs/screenshot-demo.png` — Styled but shows Kubernetes question
- `docs/screenshot-benchmark.png` — Styled but old methodology text
- `docs/screenshot-graph.png` — BROKEN: captured without CSS, shows unstyled HTML

**README.md** — Exists with AWS focus but may reference stale content.

---

### Wave 1: MIGRATE (3 parallel agents in worktrees)

**Agent 1: "migrate-backend" (implementer, worktree, sonnet)**

Branch: `migrate/backend-aws`

Migrate all backend Python files from Python-stdlib/K8s/SEC references to AWS. Read each file before modifying.

Files to modify:
- `kb_arena/graph/schema.py` — Replace `_CORPUS_SCHEMA` dispatch. NodeTypes: Service, Resource, Policy, Feature, Configuration, Limit, API_Action, ARN_Pattern. RelTypes: DEPENDS_ON, INVOKES, CONNECTS_TO, ASSUMES, CONTAINS, PROTECTS, ROUTES_TO, LOGS_TO, TRIGGERS, DEPLOYED_IN, MANAGES, READS_FROM, WRITES_TO. Three corpus schemas: aws-compute, aws-storage, aws-networking.
- `kb_arena/strategies/knowledge_graph.py` — Replace mock data with AWS examples (Lambda -> IAM Role -> S3 access). Update CYPHER_GEN_PROMPT with AWS node/rel types. Update example Cypher queries.
- `kb_arena/chatbot/router.py` — Verify AWS service name keywords in intent classifier. Add any missing: Lambda, EC2, S3, RDS, VPC, IAM, CloudFront, DynamoDB, SQS, SNS, ECS, EKS, Route 53, ALB, NLB, API Gateway, CloudWatch, Secrets Manager, etc.
- `kb_arena/graph/cypher_templates.py` — If exists, update templates for AWS service relationships.
- `kb_arena/graph/extractor.py` — If exists, update system prompt to use AWS schema.
- Any other `.py` file referencing "python-stdlib", "kubernetes", "sec-edgar", "json.loads", "os.path", "pathlib", or K8s/SEC concepts.

Scan with: `grep -r "python-stdlib\|kubernetes\|sec-edgar\|json\.loads\|os\.path\|pathlib\|kubectl\|EDGAR\|filing" kb_arena/ --include="*.py" -l`

Rules:
- Read before modifying — understand existing structure.
- Keep the same code style. Don't add docstrings, comments, or type annotations beyond what exists.
- Don't restructure or refactor — just migrate content references.
- Run `python3 -m pytest tests/ -x` after all changes to verify nothing broke.

---

**Agent 2: "migrate-questions" (implementer, worktree, sonnet)**

Branch: `migrate/aws-questions`

Delete old question files and create 200 AWS documentation questions across 3 corpora and 5 tiers.

Step 1 — Delete old:
```bash
rm -rf datasets/python-stdlib/questions/
rm -rf datasets/kubernetes/questions/
rm -rf datasets/sec-edgar/questions/
```

Step 2 — Create directory structure:
```
datasets/aws-compute/questions/
datasets/aws-storage/questions/
datasets/aws-networking/questions/
```

Step 3 — Write question files:

`datasets/aws-compute/questions/` (75 questions):
- `tier1_factoid.yaml` — 20 questions. Single AWS service lookups. "What is the maximum timeout for a Lambda function?" "What instance types support EBS optimization?"
- `tier2_procedural.yaml` — 20 questions. How-to with one service. "How do you configure a Lambda function to run inside a VPC?"
- `tier3_comparative.yaml` — 15 questions. Service A vs B. "When should you use Lambda vs Fargate for container workloads?"
- `tier4_relational.yaml` — 12 questions. Cross-service dependencies. "What IAM permissions does a Lambda function need to read from DynamoDB and write to S3?"
- `tier5_multihop.yaml` — 8 questions. 3+ services, architecture patterns. "Design the IAM policy chain for: API Gateway -> Lambda -> RDS in private subnet with Secrets Manager credential rotation."

`datasets/aws-storage/questions/` (65 questions):
- `tier1_factoid.yaml` — 18 questions. "What are the S3 storage classes?" "What is the maximum size of an EBS volume?"
- `tier2_procedural.yaml` — 17 questions. "How do you configure S3 lifecycle rules to transition objects to Glacier?"
- `tier3_comparative.yaml` — 13 questions. "When should you use EFS vs FSx for Lustre?"
- `tier4_relational.yaml` — 10 questions. "How do S3 bucket policies interact with IAM policies for cross-account access?"
- `tier5_multihop.yaml` — 7 questions. "Trace the data flow: application writes to EFS mounted on ECS Fargate, backed up by AWS Backup to S3, replicated cross-region."

`datasets/aws-networking/questions/` (60 questions):
- `tier1_factoid.yaml` — 17 questions. "What is the maximum number of VPCs per region?"
- `tier2_procedural.yaml` — 15 questions. "How do you configure a VPC endpoint for S3?"
- `tier3_comparative.yaml` — 12 questions. "ALB vs NLB: when to use which?"
- `tier4_relational.yaml` — 10 questions. "How do security groups, NACLs, and WAF rules layer together?"
- `tier5_multihop.yaml` — 6 questions. "Design the complete network path: user -> CloudFront -> WAF -> ALB -> ECS in private subnet -> RDS via VPC endpoint."

Each YAML entry must have: id, tier, type, hops, question, ground_truth (answer, source_refs, required_entities), constraints (must_mention, must_not_claim).

Ground truth answers must be factually correct against real AWS documentation. Source refs should be real AWS docs URLs (https://docs.aws.amazon.com/...).

---

**Agent 3: "migrate-tests" (implementer, worktree, sonnet)**

Branch: `migrate/aws-tests`

Migrate all test fixtures and assertions from Python/K8s/SEC to AWS content.

Files to modify:
- `tests/conftest.py` — Replace `sample_section` (json.loads) with AWS example (Lambda function configuration). Replace `sample_document` (python-stdlib-json) with AWS document (aws-compute-lambda). Replace `sample_documents` (json + os modules) with AWS documents (Lambda + S3). Keep exact same fixture structure and field types.
- `tests/integration/test_ingest_to_vector.py` — Replace Python module sample data with AWS service documents. Keep same test logic.
- All files in `tests/` — scan for old references: `grep -r "python-stdlib\|kubernetes\|sec-edgar\|json\.loads\|os\.path\|pathlib" tests/ -l`

Rules:
- Read each test file fully before modifying.
- Keep the same test structure and assertions — only change content/data.
- Don't add new tests. Don't remove tests. Don't change assertion logic.
- Run `python3 -m pytest tests/ -x -v` after changes.

---

### Wave 1 Merge (orchestrator)

After all 3 agents complete:
1. Merge `migrate/backend-aws` -> main
2. Merge `migrate/aws-questions` -> main
3. Merge `migrate/aws-tests` -> main
4. Run `python3 -m pytest tests/ -x` on merged main — fix any integration issues
5. Commit merged main

---

### Wave 2: FRONTEND VERIFY + GRAPH (2 parallel agents)

**Agent 4: "frontend-verify" (implementer, worktree, sonnet)**

Branch: `enhance/frontend-content`

Verify and fix all frontend files for AWS consistency. The Cloudwright light theme and component structure are already done — this is content verification only.

Checklist:
1. `web/lib/api.ts` — Verify CORPORA array is `["aws-compute", "aws-storage", "aws-networking"]` with correct labels and question counts (75, 65, 60).
2. `web/app/page.tsx` — Verify hero text, strategy descriptions, corpus cards, tier labels all reference AWS. No Python/K8s/SEC mentions.
3. `web/app/demo/page.tsx` — Verify pre-filled question is AWS-themed. Verify 5 strategy answers are realistic for AWS query.
4. `web/app/benchmark/page.tsx` — Verify methodology text references AWS corpora. Verify tier descriptions match question files.
5. `web/components/BenchmarkTable.tsx` — Verify footer text.
6. `web/app/graph/page.tsx` — Verify 25 AWS service nodes and 30 edges are present and correctly typed.
7. **DO NOT modify** `web/components/GraphViewer.tsx` — it's already complete.
8. Run `cd web && npm run build` — must compile with zero errors.
9. Scan all `web/` files: `grep -r "python\|kubernetes\|kubectl\|sec-edgar\|EDGAR\|filing\|stdlib" web/ --include="*.ts" --include="*.tsx" -l`

Fix any stale references found.

---

**Agent 5: "frontend-graph" (qa-verifier, worktree, sonnet)**

Branch: `verify/graph-page`

Verify the graph page renders correctly with AWS service nodes.

1. `cd web && npm install && npx next dev -p 3001 &`
2. Wait for "Ready" message
3. Use `agent-browser` to open http://localhost:3001/graph
4. Wait for networkidle + 3 seconds for canvas animation
5. Take a test screenshot — verify:
   - Nodes are visible with glow effects (colored circles, not plain dots)
   - Edges are curved bezier (not straight lines)
   - At least 20 nodes visible
   - Labels appear near nodes
   - Background is light (#f8fafc), not dark
6. If graph looks broken, check browser console for errors
7. Report findings but DO NOT modify GraphViewer.tsx

---

### Wave 2 Merge (orchestrator)

1. Merge `enhance/frontend-content` -> main
2. Merge `verify/graph-page` findings — fix if needed
3. Run `cd web && npm run build` on merged main
4. Commit

---

### Wave 3: QA + DOCS + SCREENSHOTS (3 parallel agents)

**Agent 6: "qa-enforce" (qa-verifier, worktree, sonnet)**

Branch: `qa/final`

Enforce all quality gates:

1. **Python lint**: `ruff check . && ruff format --check .` — zero warnings. Fix any issues.
2. **Python tests**: `python3 -m pytest tests/ -v` — all pass. Fix failures.
3. **Frontend build**: `cd web && npm run build` — zero errors.
4. **No old references**: These greps must return ZERO results (excluding BUILD_PROMPT.md, PLAN.md, git history, and .md documentation):
   ```bash
   grep -r "python-stdlib\|kubernetes\|sec-edgar" --include="*.py" --include="*.ts" --include="*.tsx" --include="*.yaml" -l
   ```
5. **No hardcoded secrets**: `grep -r "sk-\|AKIA\|password=" --include="*.py" --include="*.ts" -l` — zero results (excluding .env.example).
6. **Question count validation**: Count questions in each corpus YAML — must total 200 (75 + 65 + 60).
7. **YAML validity**: Load every question YAML file and validate schema.
8. Write QA report to `QA_REPORT.md`.

---

**Agent 7: "readme-docs" (implementer, worktree, sonnet)**

Branch: `enhance/readme`

Update README.md to match current state:

1. Read current README.md first.
2. Verify all content references AWS documentation (not Python/K8s/SEC).
3. Verify screenshot references point to `docs/screenshot-*.png`.
4. Verify "Quick Start" commands work: `pip install -e .`, `kb-arena --help`.
5. Verify architecture description matches actual code structure.
6. Fix any stale content found.
7. Verify `.env.example` has all required vars.

Do NOT rewrite the README from scratch — just verify and fix inconsistencies.

---

**Agent 8: "screenshots" (implementer, worktree, sonnet)**

Branch: `enhance/screenshots`

Retake all 4 publication-quality screenshots. The old ones are stale or broken.

Prerequisites:
```bash
cd web && npm install && npx next dev -p 3001 &
# Wait for "Ready" message
```

Use `agent-browser` (NOT Playwright CLI — it misses CSS hydration):

**Screenshot 1: Home page**
```bash
agent-browser open http://localhost:3001
agent-browser wait --load networkidle
agent-browser wait 2000
agent-browser screenshot --full docs/screenshot-home.png
```
Verify: styled page with AWS content, Cloudwright light theme, strategy cards visible.

**Screenshot 2: Demo page**
```bash
agent-browser open http://localhost:3001/demo
agent-browser wait --load networkidle
agent-browser wait 2000
agent-browser screenshot --full docs/screenshot-demo.png
```
Verify: AWS Lambda question pre-filled, strategy tabs visible, answer panels with metrics.

**Screenshot 3: Benchmark page**
```bash
agent-browser open http://localhost:3001/benchmark
agent-browser wait --load networkidle
agent-browser wait 2000
agent-browser screenshot --full docs/screenshot-benchmark.png
```
Verify: table with strategy comparison, AWS methodology text, tier descriptions.

**Screenshot 4: Graph page**
```bash
agent-browser open http://localhost:3001/graph
agent-browser wait --load networkidle
agent-browser wait 5000
agent-browser screenshot --full docs/screenshot-graph.png
```
Verify: AWS service nodes with glow, curved edges, labels, light background. Wait 5s for force simulation to settle.

For each screenshot: if it shows unstyled HTML (no colors, no borders, plain text), the CSS didn't load. Close browser, restart dev server, try again.

---

### Wave 3 Merge + Final Commit

1. Merge all branches -> main
2. Run final checks: `python3 -m pytest tests/ && ruff check . && cd web && npm run build`
3. Verify all 4 screenshots show properly styled pages with AWS content
4. Commit: "Migrate to AWS documentation, enforce tests, retake screenshots"

---

### Code Quality Rules

All agents must follow these:

1. **Read before modifying** — understand existing code structure before making changes.
2. **Minimal changes** — modify only what's needed for AWS migration. Don't refactor, restructure, or "improve" surrounding code.
3. **No AI fingerprints** — no excessive docstrings, no numbered step comments (`# Step 1:`), no isinstance() on every parameter, no defensive returns for impossible cases, no 3+ helpers for a 20-line operation.
4. **Match existing style** — if the file uses single quotes, use single quotes. If it has no docstrings, don't add them. If functions are 40 lines, don't break them into 10-line helpers.
5. **Test enforcement** — `pytest` must pass after every agent's changes. Not optional.
6. **Lint enforcement** — `ruff check .` must pass. Not optional.
7. **No stale references** — zero mentions of python-stdlib, kubernetes, sec-edgar, json.loads (as example data), os.path (as example data), pathlib (as example data), kubectl, EDGAR, filing in any non-documentation file.

### Routing and Constraints

- All agents: restrict searches to project directory only. Never search ~/Desktop or ~/.claude recursively.
- All agents: read PLAN.md and CLAUDE.md before starting.
- All agents: use the shared models from kb_arena/models/ — do NOT create parallel model definitions.
- Build/migrate agents: implementer type, worktree isolation, sonnet model.
- QA agents: qa-verifier type, worktree isolation, sonnet model.
- Orchestrator budget: stay under 25% context. Write summaries, not full file contents.
- Max 3 concurrent agents per wave. Wait for wave completion before starting next wave.
- If an agent hits 3+ failures on same issue: stop, report to orchestrator, move on.

## PROMPT END
