# Contributing to KB Arena

Thanks for your interest in contributing. This guide covers setup, conventions, and how to add new corpora, strategies, and questions.

## Development Setup

```bash
git clone https://github.com/xmpuspus/kb-arena
cd kb-arena
pip install -e '.[dev]'

# Start Neo4j for integration tests
docker compose up neo4j -d
```

## Running Tests

```bash
# Unit tests (fast, no dependencies)
pytest tests/test_benchmark.py tests/test_strategies.py tests/test_graph/ tests/test_ingest.py tests/test_router.py -q

# Full suite
pytest tests/ -q

# Integration tests (requires Neo4j + ChromaDB)
pytest tests/integration/ -v

# Live tests (requires API keys)
ANTHROPIC_API_KEY=sk-ant-... OPENAI_API_KEY=sk-... pytest tests/live/ -v
```

## Code Style

Ruff handles both linting and formatting:

```bash
ruff check .          # lint
ruff format .         # format
ruff check --fix .    # auto-fix
```

Run before every commit. CI will reject PRs that don't pass `ruff check . && ruff format --check .`

## Project Conventions

- **Python 3.11+** — f-strings, `match` statements, `X | Y` union types
- **Pydantic v2** — All data models use `BaseModel`. No dataclasses, no TypedDicts for interchange
- **Async first** — Strategies, LLM calls, Neo4j queries are all async
- **Type hints** — All public functions and methods are typed
- **Tests** — `pytest` + `pytest-asyncio` with `asyncio_mode = "auto"`
- **Imports** — `from __future__ import annotations` in every file

## Adding a New Corpus

A corpus is a documentation domain (e.g., Python stdlib, Kubernetes, SEC EDGAR). To add one:

### 1. Parser

Create `kb_arena/ingest/parsers/your_corpus.py`:

```python
from kb_arena.models.document import Document

def parse(path: str) -> list[Document]:
    """Parse raw files into Document models."""
    documents = []
    # ... your parsing logic
    return documents
```

Register in `kb_arena/ingest/pipeline.py`.

### 2. Graph Schema

Add node and relationship enums to `kb_arena/graph/schema.py`:

```python
class YourNodeType(str, Enum):
    ENTITY_A = "EntityA"
    ENTITY_B = "EntityB"

class YourRelType(str, Enum):
    RELATES_TO = "RELATES_TO"
```

Register in `_CORPUS_SCHEMA`:

```python
_CORPUS_SCHEMA["your-corpus"] = (YourNodeType, YourRelType)
```

### 3. Questions

Create 5 YAML files in `datasets/your-corpus/questions/`:

```
tier1_factoid.yaml        # 10-20 single-fact lookups
tier2_multi_entity.yaml   # 10-15 questions involving 2+ entities
tier3_comparative.yaml    # 10-15 compare/contrast questions
tier4_relational.yaml     # 8-12 relationship traversal questions
tier5_temporal.yaml       # 8-12 version/change tracking questions
```

### 4. Tests

- Parser test in `tests/test_ingest.py`
- Schema validation test in `tests/test_graph/`
- At least one integration test showing the full pipeline

## Adding a New Strategy

### 1. Implementation

Create `kb_arena/strategies/your_strategy.py`:

```python
from kb_arena.strategies.base import AnswerResult, Strategy
from kb_arena.models.document import Document

class YourStrategy(Strategy):
    name = "your_strategy"

    async def build_index(self, documents: list[Document]) -> None:
        """Build the retrieval index."""
        ...

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        """Answer a question."""
        start = self._start_timer()
        # ... your retrieval + generation logic
        latency = self._record_metrics(start, sources=sources)
        return AnswerResult(
            answer=answer,
            sources=sources,
            strategy=self.name,
            latency_ms=latency,
        )
```

### 2. Registration

In `kb_arena/strategies/__init__.py`:

```python
STRATEGY_REGISTRY["your_strategy"] = YourStrategy
```

Update `get_strategy()` if it needs special initialization (e.g., database clients).

### 3. Chatbot Integration

Add to the lifespan in `kb_arena/chatbot/api.py`:

```python
app.state.strategies["your_strategy"] = YourStrategy(...)
```

### 4. Tests

Add tests in `tests/test_strategies.py` covering:
- Index building (or mocked)
- Query with expected answer structure
- Error handling (timeout, empty result)

## Writing Benchmark Questions

### Format

```yaml
- id: "corpus-tN-NNN"
  tier: 1-5
  type: factoid|comparison|relational|temporal|causal
  hops: 1-5
  question: "The actual question"
  ground_truth:
    answer: "Complete reference answer"
    source_refs:
      - "doc.html#section"
    required_entities:
      - "entity.name"
  constraints:
    must_mention:
      - "term"
    must_not_claim:
      - "false claim"
```

### Guidelines

- **ID format:** `{corpus_prefix}-t{tier}-{3-digit-number}` (e.g., `py-t1-001`, `k8s-t3-012`)
- **Tier 1-2:** Answerable by any strategy. Tests basic retrieval.
- **Tier 3:** Requires comparing two entities. Tests the knowledge graph's ability to find shared neighbors.
- **Tier 4:** Requires traversing 3+ relationships. Only graph-backed strategies should score well.
- **Tier 5:** Requires temporal reasoning (version changes, deprecations). The hardest tier.
- **`must_mention`:** Terms that a correct answer definitely includes. Be specific but not overly restrictive.
- **`must_not_claim`:** Common misconceptions. Don't include obscure edge cases — test for real confusions.
- **Ground truth:** Write the ideal answer, not just a keyword. The LLM judge uses it as reference.
- **Source refs:** Point to specific document sections, not entire documents.

## Pull Requests

- One feature per PR
- Include tests
- Run `ruff check . && ruff format --check .` before submitting
- Simple commit messages, no prefix format

## Reporting Issues

Use GitHub Issues. Include:
- What you expected
- What happened
- Steps to reproduce
- Python version, OS
