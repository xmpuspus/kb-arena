# Contributing to KB Arena

KB Arena accepts bug reports, documentation fixes, new corpus adapters, retrieval strategies, and
evaluation improvements. Read the [code of conduct](CODE_OF_CONDUCT.md) before participating.

## Choose a contribution path

- Report a reproducible product defect with the bug template.
- Propose a user problem with the feature template before a large implementation.
- Improve an existing corpus, qrel, or evidence record.
- Add a strategy when it shows a distinct retrieval method and includes a fair baseline.
- Improve onboarding or documentation with commands checked against the current CLI.

For a first contribution, start with a documentation correction, a missing regression test, or a
small qrel review. Avoid changing benchmark numbers without the run artifacts that produced them.

## Development setup

```bash
git clone https://github.com/xmpuspus/kb-arena
cd kb-arena
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Start Neo4j only for graph integration work:

```bash
export KB_ARENA_NEO4J_PASSWORD=choose-a-password
docker compose up neo4j -d
```

## Checks

Backend:

```bash
ruff check .
ruff format --check .
pytest tests/ -q --ignore=tests/live
```

Frontend:

```bash
cd web
npm ci
npm run lint
npm run build
```

Frontend development needs Node.js 20.9 or later.

Live tests need provider keys and can spend money. Run them only when your change needs that proof:

```bash
pytest tests/live/ -v
```

## Project rules

- Support Python 3.11 and 3.12.
- Use Pydantic models for stored and exchanged data.
- Keep strategy and service calls asynchronous where the existing interface is asynchronous.
- Add a regression test before a behavior fix.
- Keep benchmark inputs, model versions, settings, and run IDs with public metrics.
- Do not commit API keys, corpus secrets, local indexes, or unrelated run output.
- Do not change a test only to hide a product failure.

## Add a corpus

1. Create `datasets/<name>/raw`, `processed`, and `questions`.
2. Record the source URL, version, retrieval date, license, and content hash.
3. Parse source files into the common `Document` model.
4. Use stable document and section identifiers.
5. Add reviewed questions and qrels with source anchors and reasons.
6. Split development, validation, and holdout questions before tuning.
7. Add validation tests for source hashes, duplicate questions, qrel targets, and split counts.
8. Document limits and prohibited interpretations.

See [the method guide](docs/methodology.md) for the public evidence contract.

## Add a strategy

Create a `Strategy` subclass under `kb_arena/strategies/`:

```python
from kb_arena.models.document import Document
from kb_arena.strategies.base import AnswerResult, Strategy


class MyStrategy(Strategy):
    name = "my_strategy"

    async def build_index(self, documents: list[Document]) -> None:
        ...

    async def query(self, question: str, top_k: int = 5) -> AnswerResult:
        ...
```

Then:

1. Add it to `STRATEGY_REGISTRY`.
2. Add one `StrategySpec` to `kb_arena/strategies/catalog.py`.
3. Return stable `RetrievedChunk` identifiers and source information.
4. Add happy-path, empty-index, failure, and retrieval-trace tests.
5. Explain the architecture, dependencies, expected advantage, and fair baseline.
6. Include a run that can disprove the advantage.

External plugins can use `--strategy-module package.module`. A plugin module must export exactly one
`Strategy` subclass.

## Pull requests

A pull request should state:

- the user problem and chosen behavior;
- files and interfaces changed;
- tests and checks run;
- benchmark evidence, when the change affects results;
- costs, migrations, or compatibility limits;
- screenshots or recordings for visible UI changes.

Keep the change focused. Maintainers may ask to split unrelated behavior, data, or documentation.
