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

import yaml

from kb_arena.benchmark.questions import load_questions
from kb_arena.llm.client import LLMClient
from kb_arena.settings import settings
from kb_arena.strategies.bm25 import BM25Strategy

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are labeling retrieval ground truth. Given a QUESTION and CANDIDATE chunks,
return a JSON array of chunk_ids that contain information directly relevant to answering it.
Be strict: a chunk is "relevant" only if it contains content needed to answer.

QUESTION: {question}

CANDIDATES:
{candidates}

Return ONLY a JSON array of chunk_id strings. Example:
["lambda-overview::pricing", "ec2-overview::instance-types"]"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


async def label_one_question(
    question_text: str,
    bm25: BM25Strategy,
    llm: LLMClient,
    n_candidates: int = 20,
) -> tuple[list[str], float]:
    """Returns (relevant_chunk_ids, cost_usd) for a single question."""
    result = await bm25.query(question_text, top_k=n_candidates)
    if not result.retrieval or not result.retrieval.retrieved:
        return [], 0.0

    candidates_text = "\n\n".join(
        f"[{c.chunk_id}] {c.content[:400]}" for c in result.retrieval.retrieved
    )
    prompt = JUDGE_PROMPT.format(question=question_text, candidates=candidates_text)

    resp = await llm.extract(text=prompt, system_prompt="Return only valid JSON.")
    text = _strip_fences(resp.text)
    try:
        ids = json.loads(text)
        if isinstance(ids, list):
            valid = {c.chunk_id for c in result.retrieval.retrieved}
            return [str(x) for x in ids if str(x) in valid], resp.cost_usd
    except json.JSONDecodeError:
        log.warning("Failed to parse judge output: %.200s", text)
    return [], resp.cost_usd


async def label_corpus(
    corpus: str, force: bool = False, n_candidates: int = 20
) -> dict:
    """Label every question in a corpus. Idempotent unless force=True. Cost-capped."""
    questions = load_questions(corpus)
    bm25 = BM25Strategy()
    if not bm25._ensure_index(corpus):
        raise RuntimeError(
            f"BM25 index missing for {corpus}. Run: kb-arena build-vectors --corpus "
            f"{corpus} --strategy bm25"
        )
    llm = LLMClient()

    out_path = (
        settings.datasets_path
        if not isinstance(settings.datasets_path, str)
        else settings.datasets_path
    )
    from pathlib import Path

    out_dir = Path(settings.datasets_path) / corpus / "questions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "expected_chunks.yaml"

    existing: dict[str, list[str]] = {}
    if out_path.exists():
        loaded = yaml.safe_load(out_path.read_text()) or {}
        if isinstance(loaded, dict):
            existing = {str(k): list(v) if isinstance(v, list) else [] for k, v in loaded.items()}

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
        ids, cost = await label_one_question(q.question, bm25, llm, n_candidates)
        total_cost += cost
        out_dict[q.id] = ids
        labeled += 1

    out_path.write_text(yaml.dump(out_dict, sort_keys=True, default_flow_style=False))
    return {
        "labeled": labeled,
        "skipped": skipped,
        "cost_usd": total_cost,
        "path": str(out_path),
        "halted_by_cost_cap": halted,
        "total_questions": len(questions),
    }
