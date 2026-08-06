# KB Arena Relevance Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make KB Arena's public promise, onboarding, benchmark evidence, and current project surfaces truthful, reproducible, and useful to practitioners choosing a retrieval architecture.

**Architecture:** Runtime strategy distinctions come from one strategy catalog, while the CLI orchestrator treats ingestion and graph availability as explicit stage outcomes. Public documentation is rebuilt around an evidence contract, and a second NIST corpus uses stable control identifiers for source-grounded qrels and holdout evaluation.

**Tech Stack:** Python 3.11+, Typer, FastAPI, pytest, ChromaDB, Next.js, Markdown, VHS-compatible terminal recordings.

## Global Constraints

- Preserve all pre-existing untracked datasets, results, recordings, plans, and `uv.lock`.
- Do not publish, deploy, post, or open third-party pull requests in this repository pass.
- Do not present the AWS Compute corpus as general performance evidence.
- Keep QISS and SQR labeled experimental; SQR remains an optional dependency.
- Every public metric names corpus, question count, ground-truth method, run ID, date, and limitation.
- Write behavior changes test-first and see the intended failing assertion before implementation.

---

### Task 1: Repair the one-shot local workflow

**Files:**
- Change: `kb_arena/cli.py`
- Create: `tests/test_run_cli.py`

**Interfaces:**
- Consumes: `datasets/<corpus>/{raw,processed,questions}` and existing stage functions.
- Produces: `run()` behavior that automatically ingests non-empty `raw/` input and continues after a recoverable graph failure without checkpointing that failed stage.

- [ ] Add a CLI test that places a document in `raw/`, omits `--docs`, and asserts ingestion and all later stages run.
- [ ] Run the focused test and confirm the current command exits at the missing-processed-documents check.
- [ ] Make `run()` select the non-empty raw directory as its ingestion source.
- [ ] Add a CLI test whose graph extractor raises and assert vector/question/benchmark stages still run while `build_graph` remains absent from pipeline state.
- [ ] Run the focused test and confirm the current `typer.Exit(0)` stops later stages.
- [ ] Let recoverable stages return an incomplete outcome so `_stage()` continues without writing a successful checkpoint.
- [ ] Run `pytest tests/test_run_cli.py -q`.

### Task 2: Make no-key and demo modes honest

**Files:**
- Change: `kb_arena/cli.py`
- Change: `kb_arena/chatbot/api.py`
- Change: `tests/test_settings.py`
- Change: `tests/integration/test_chatbot_api.py`

**Interfaces:**
- Consumes: `Settings.llm_provider`, `Settings.embedding_provider`, and `Settings.demo_mode`.
- Produces: provider-specific preflight validation and a demo lifespan that skips intentionally unavailable Neo4j access and disables Chroma telemetry.

- [ ] Add parameterized preflight tests for Ollama generation with OpenAI embeddings and for fully local Ollama generation and embeddings.
- [ ] Run the focused tests and confirm that the code infers embedding credentials from the wrong setting.
- [ ] Validate embedding credentials independently from the generation provider.
- [ ] Add an API lifespan test. Make it fail if demo mode constructs a Neo4j driver, and check that it disables telemetry.
- [ ] Run the focused test and confirm that the current lifespan tries the connection.
- [ ] Skip Neo4j construction in demo mode and configure Chroma telemetry before client construction.
- [ ] Run `pytest tests/test_settings.py tests/integration/test_chatbot_api.py -q`.

### Task 3: Set up one strategy catalog

**Files:**
- Create: `kb_arena/strategies/catalog.py`
- Change: `kb_arena/strategies/__init__.py`
- Change: `kb_arena/benchmark/runner.py`
- Change: `kb_arena/chatbot/api.py`
- Change: `web/lib/api.ts`
- Change: strategy and API tests selected after reading their fixtures.

**Interfaces:**
- Produces: immutable strategy metadata with `name`, `label`, `architecture`, `default_benchmark`, `api_supported`, `experimental`, and `optional_extra`; helpers for default names and runtime availability.
- Consumers: benchmark defaults, `/strategies`, public copy, and frontend choice controls.

- [ ] Add catalog tests for 11 registered strategies, 10 defaults, experimental QISS/SQR labels, and the SQR optional-extra reason.
- [ ] Run them and confirm the catalog does not exist.
- [ ] Implement the catalog and derive benchmark defaults from it.
- [ ] Add API contract tests for loaded and unavailable strategy records.
- [ ] Run them and confirm `/strategies` only returns the hard-coded loaded map.
- [ ] Instantiate QISS for API use, conditionally instantiate SQR, and return status metadata for every catalog entry.
- [ ] Update the frontend contract and rebuild the bundled static application.
- [ ] Run the catalog, benchmark, API, and frontend checks.

### Task 4: Rebuild current project documentation

**Files:**
- Replace: `README.md`
- Change: `CHANGELOG.md`
- Change: `SECURITY.md`
- Change: `CONTRIBUTING.md`
- Change: `.zenodo.json`
- Create: `docs/getting-started.md`
- Create: `docs/methodology.md`
- Create: `docs/strategy-catalog.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`

**Interfaces:**
- Consumes: checked CLI help, strategy catalog, tracked benchmark artifacts, tags, and citation metadata.
- Produces: a short decision-oriented README and focused long-form references.

- [ ] Backfill changelog entries for every tag from v0.7.0 through v0.9.3 using tagged commits as authority.
- [ ] Correct the security support table and metadata descriptions.
- [ ] Move detailed setup, method, strategy, and contributor material into focused pages.
- [ ] Replace the README with the approved five-part path.
- [ ] Remove the competitor grid, release archive, fabricated screenshots, universal winner language, and quantum-first framing.
- [ ] Add issue routing and a contributor code of conduct.
- [ ] Check headings, relative links, version strings, strategy language, image references, and protected URLs mechanically.

### Task 5: Add the NIST SP 800-171 Rev. 3 evidence corpus

**Files:**
- Create: `datasets/nist-800-171-r3/README.md`
- Create: `datasets/nist-800-171-r3/source-manifest.json`
- Create: `datasets/nist-800-171-r3/raw/`
- Create: `datasets/nist-800-171-r3/processed/`
- Create: `datasets/nist-800-171-r3/questions/`
- Create or change: corpus-validation tests under `tests/`.

**Interfaces:**
- Consumes: the official NIST publication found by DOI `10.6028/NIST.SP.800-171r3`.
- Produces: source-hashed control documents, source-defined hierarchy, and cross-references. It also produces 80 draft questions with control-ID qrels and 48/12/20 development/validation/holdout splits.

- [ ] Add schema tests for source URL, retrieval date, SHA-256, license, control IDs, and source anchors.
- [ ] Check question type, split, rationale, and non-empty qrels in the same schema tests.
- [ ] Run them against an empty corpus and confirm the expected missing-artifact failure.
- [ ] Snapshot the official source and record exact provenance and license text.
- [ ] Transform controls into deterministic documents while preserving official identifiers and links.
- [ ] Author and review 80 questions across the approved 20/20/20/10/10 type distribution.
- [ ] Validate 48/12/20 splits, qrel resolvability, family leakage constraints, duplicate questions, source hashes, and license attribution.
- [ ] Keep holdout answers out of tuning and public intermediate reports.

### Task 6: Replace public evidence media

**Files:**
- Change: `docs/tapes/hero-demo.tape`
- Create: a tape for the own-documents workflow.
- Replace: only the canonical media referenced by the new README.
- Create: media validation notes with captured command, version, run ID, frame timestamps, and numerical source.

**Interfaces:**
- Consumes: current CLI output and checked-in benchmark evidence.
- Produces: no more than three README media assets with reproducible capture instructions.

- [ ] Remove all stale or synthetic terminal images from README references.
- [ ] Record a 12-15 second hero around a real decision result.
- [ ] Record a 30-45 second own-documents path after checks prove the runtime fixes.
- [ ] Generate one static benchmark visual only from checked-in result data.
- [ ] Extract keyframes and compare every visible number, version, strategy, and command with its authoritative source.

### Task 7: Check the complete refresh

**Files:**
- Inspect all files changed by Tasks 1-6.

**Interfaces:**
- Consumes: the completed implementation and its evidence artifacts.
- Produces: a need-by-need completion record.

- [ ] Run `ruff check .` and `ruff format --check .`.
- [ ] Run `pytest tests/ -v --ignore=tests/live`.
- [ ] Run `npm ci` and `npx next build` from `web/`.
- [ ] Build the wheel and inspect its bundled results/static assets.
- [ ] Install the wheel in a clean environment and smoke-test `kb-arena demo` and local onboarding.
- [ ] Render and inspect the README and canonical media at desktop and mobile widths.
- [ ] Inspect `git diff --check`, changed-file scope, untracked preservation, and every completion gate in `docs/relevance-design.md`.
