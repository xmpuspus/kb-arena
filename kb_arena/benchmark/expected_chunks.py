"""Generate expected_chunks.yaml using BM25 + Haiku judge.

For each question, BM25 retrieves the top-N candidate chunks, then a Haiku call
classifies which of those candidates are actually relevant. Output is written to
datasets/{corpus}/questions/expected_chunks.yaml as a `{question_id: [chunk_id]}`
mapping.

Idempotent: skips questions that already have labels unless force=True.
Cost-capped: stops once cumulative cost reaches KB_ARENA_COST_CAP_USD.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import yaml

from kb_arena.benchmark.atomic import atomic_write_text
from kb_arena.benchmark.questions import load_questions, validate_expected_chunks
from kb_arena.llm.client import LLMClient
from kb_arena.settings import settings
from kb_arena.strategies.bm25 import BM25Strategy

log = logging.getLogger(__name__)


def _write_expected_chunks(path: Path, labels: dict[str, list[str]]) -> None:
    """Atomically checkpoint labels so a later provider failure cannot erase progress."""
    atomic_write_text(path, yaml.safe_dump(labels, sort_keys=True, default_flow_style=False))


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


def _extract_json_array(text: str) -> str | None:
    """Extract the first balanced JSON array '[...]' from possibly noisy text."""
    text = _strip_fences(text)
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


async def label_one_question(
    question_text: str,
    bm25: BM25Strategy,
    llm: LLMClient,
    corpus: str,
    n_candidates: int = 20,
    extra_retrievers: list | None = None,
) -> tuple[list[str], float]:
    """Returns (relevant_chunk_ids, cost_usd) for a single question.

    Builds the candidate pool from the UNION of every retriever in
    `extra_retrievers` plus BM25 — not BM25 alone. Otherwise the gold set is
    structurally limited to chunks BM25 would surface, which biases IR metrics
    in favour of keyword-overlap strategies.
    """
    result = await bm25.query(question_text, top_k=n_candidates, corpus=corpus)
    candidates: list = (
        list(result.retrieval.retrieved) if result.retrieval and result.retrieval.retrieved else []
    )
    cost = 0.0

    if extra_retrievers:
        for retriever in extra_retrievers:
            try:
                extra_result = await retriever.query(
                    question_text, top_k=n_candidates, corpus=corpus
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Extra retriever {retriever.name} failed while building the label pool"
                ) from exc
            if extra_result.retrieval and extra_result.retrieval.retrieved:
                candidates.extend(extra_result.retrieval.retrieved)
            cost += float(getattr(extra_result, "cost_usd", 0.0) or 0.0)

    if not candidates:
        return [], cost

    # Deduplicate by chunk_id, keep the highest-ranked instance.
    by_id: dict[str, object] = {}
    for c in candidates:
        cid = c.chunk_id
        existing = by_id.get(cid)
        if existing is None or c.rank < existing.rank:  # type: ignore[attr-defined]
            by_id[cid] = c
    deduped = list(by_id.values())

    candidates_text = "\n\n".join(f"[{c.chunk_id}] {c.content[:400]}" for c in deduped)
    prompt = JUDGE_PROMPT.format(question=question_text, candidates=candidates_text)

    resp = await llm.extract(
        text=prompt,
        system_prompt="You output only a JSON array literal. No prose. No markdown.",
    )
    cost += float(resp.cost_usd or 0.0)
    valid = {c.chunk_id for c in deduped}
    array_text = _extract_json_array(resp.text)
    if array_text is None:
        log.warning("No JSON array in judge output: %.200s", resp.text)
        return [], cost
    try:
        ids = json.loads(array_text)
    except json.JSONDecodeError:
        log.warning("Failed to parse judge output: %.200s", array_text)
        return [], cost
    if not isinstance(ids, list):
        return [], cost
    return [str(x) for x in ids if str(x) in valid], cost


async def label_corpus(corpus: str, force: bool = False, n_candidates: int = 20) -> dict:
    """Label every question in a corpus. Idempotent unless force=True. Cost-capped.

    Candidate pool is the union of BM25 + naive_vector + contextual_vector top-N
    when those indexes exist. If only BM25 is built, we still proceed (with a
    documented bias warning) so users running label-chunks before build-vectors
    aren't blocked.
    """
    questions = load_questions(corpus)
    bm25 = BM25Strategy()
    if not bm25._ensure_index(corpus):
        raise RuntimeError(
            f"BM25 index missing for {corpus}. Run: kb-arena build-vectors --corpus "
            f"{corpus} --strategy bm25"
        )
    llm = LLMClient()

    # Best-effort extra retrievers — silently skip if their index isn't built yet.
    extra_retrievers: list = []
    try:
        import chromadb

        from kb_arena.strategies.contextual_vector import ContextualVectorStrategy
        from kb_arena.strategies.naive_vector import NaiveVectorStrategy

        chroma = chromadb.PersistentClient(path=settings.chroma_path)
        for cls in (NaiveVectorStrategy, ContextualVectorStrategy):
            try:
                inst = cls(chroma_client=chroma)
                # Probe — query a trivial string; failure means no index built.
                await inst.query("kb_arena_index_probe", top_k=1, corpus=corpus)
                extra_retrievers.append(inst)
            except Exception as exc:  # noqa: BLE001 — best-effort
                log.info(
                    "Skipping %s for ground-truth pool (index not ready: %s)",
                    cls.__name__,
                    exc,
                )
    except ImportError:
        pass

    if not extra_retrievers:
        log.warning(
            "Ground-truth pool is BM25-only — vector indexes not built yet. "
            "For unbiased labels run `kb-arena build-vectors` first, then re-label."
        )

    out_path = (
        settings.datasets_path
        if not isinstance(settings.datasets_path, str)
        else settings.datasets_path
    )
    out_dir = Path(settings.datasets_path) / corpus / "questions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expected_chunks.yaml"

    existing: dict[str, list[str]] = {}
    if out_path.exists():
        try:
            loaded = yaml.safe_load(out_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid expected chunks YAML: {out_path}") from exc
        existing = validate_expected_chunks(loaded, out_path)

    cost_cap = settings.benchmark_cost_cap_usd
    total_cost = 0.0
    out_dict: dict[str, list[str]] = dict(existing)
    skipped = 0
    labeled = 0
    halted = False

    for q in questions:
        if q.id in existing and not force:
            skipped += 1
            continue
        if cost_cap > 0 and total_cost >= cost_cap:
            log.warning("Cost cap reached at $%.2f", total_cost)
            halted = True
            break
        ids, cost = await label_one_question(
            q.question,
            bm25,
            llm,
            corpus,
            n_candidates,
            extra_retrievers=extra_retrievers,
        )
        total_cost += cost
        out_dict[q.id] = ids
        labeled += 1
        _write_expected_chunks(out_path, out_dict)

    if not out_path.exists():
        _write_expected_chunks(out_path, out_dict)
    return {
        "labeled": labeled,
        "skipped": skipped,
        "cost_usd": total_cost,
        "path": str(out_path),
        "halted_by_cost_cap": halted,
        "total_questions": len(questions),
    }
