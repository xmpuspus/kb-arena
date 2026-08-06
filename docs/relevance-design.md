# KB Arena relevance refresh

## Decision

KB Arena will present itself as a retrieval architecture decision lab for a team's own
documentation. Its immediate constraint is public trust and distribution, so this refresh
prioritizes reproducible evidence, clean onboarding, and a compact public surface before new
retrieval features.

## Product promise

> Compare materially different retrieval architectures on the same corpus and question set,
> then choose with evidence about retrieval quality, answer quality, latency, cost, and limits.

This promise is narrower than a general RAG evaluation framework. KB Arena is useful when a team
has not yet decided between lexical, dense, graph, hybrid, hierarchical, or reranked retrieval.
It is not a production observability service or evidence that one architecture wins universally.

## Evidence contract

Every public benchmark result must state:

- the corpus and source version;
- the question count and question-type distribution;
- the ground-truth method and holdout policy;
- the strategies and dependency profile actually available in that surface;
- the metric definition, run identifier, and run date;
- material cost, latency, and interpretation limitations.

The current AWS Compute corpus remains a pedagogical example. It has three short documents
and incomplete chunk-level judgments, so it cannot support general product-performance claims.
The public comparison will be rerun on NIST SP 800-171 Revision 3, using control identifiers
as auditable relevance targets and source-defined hierarchy and cross-references.

## Strategy language

One number cannot describe every current surface. The core registry, default benchmark, API, and
optional quantum installation expose different strategy sets. Public copy will describe the
architecture families or distinguish registered, default, loaded, and optional strategies. Runtime
surfaces will derive those distinctions from one strategy catalog.

## Public surface

The README will be a short decision path:

1. literal product offer and checked demo;
2. one reproducible result with limitations;
3. run on your own documents;
4. architecture families and method;
5. focused links for evaluation, extension, contribution, security, and citation.

Release history belongs in `CHANGELOG.md`. Time-sensitive competitor feature grids and unverified
terminal mockups will be removed. The refreshed surface will use at most three canonical assets:
a short hero result, an own-documents workflow, and a reproducible result visual.

Quantum methods stay available as clearly labeled experiments. Their null result can support a
technical article after the larger-corpus run, but quantum is not the main product category.

## Onboarding behavior

The documented local path must work as written:

- `run` automatically ingests files already placed in a corpus `raw/` directory;
- a failed graph stage leaves that checkpoint incomplete and continues with vector-capable stages;
- the no-key Ollama path configures both generation and embeddings;
- demo mode avoids connection tries and warning output for services it intentionally does not use;
- strategy endpoints report available and unavailable strategies explicitly.

## Distribution boundary

This repository pass may prepare the hosted demo, social preview, integration guide, launch copy,
and contribution paths. Deployments, community posts, and pull requests to other projects need
separate approval at once before the external write.

## Completion gates

- Focused regression tests prove every onboarding behavior change.
- Backend lint, formatting, non-live tests, and the frontend production build pass.
- A clean installation launches the no-key demo without service-error noise.
- Every README metric and media asset traces to checked-in evidence.
- README, package metadata, citation metadata, changelog, and security policy agree.
- The NIST corpus includes provenance, license, source hashes, reviewed qrels, and a sealed holdout.
- We inspect the rendered README and canonical media, including more than link existence.
