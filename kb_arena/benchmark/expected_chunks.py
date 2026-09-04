"""Generate expected_chunks.yaml: graded ground truth from a pooled judge.

For each question the candidate pool is the union of BM25 and every built
retrieval-only index, plus a seeded random sample of the rest of the corpus.
A judge grades every candidate: 2 answers the question, 1 supports the answer,
0 means the judge read the chunk and rejected it. A grade of 0 is what lets
bpref count a real negative instead of guessing from what a run retrieved.

Output goes to datasets/{corpus}/questions/expected_chunks.yaml in the version 2
shape `{version, pool, labels}`, where labels maps question id to a grade per
chunk. Every stored chunk id is canonical, with no strategy prefix, so a label
is reachable for every strategy.

Idempotent: skips questions that already have labels unless force=True.
Cost-capped: stops once cumulative cost reaches KB_ARENA_COST_CAP_USD.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import yaml

from kb_arena.benchmark.atomic import atomic_write_text
from kb_arena.benchmark.ir_metrics import canonical_chunk_id
from kb_arena.benchmark.questions import load_qrels, load_questions
from kb_arena.llm.client import LLMClient
from kb_arena.models.retrieval import RetrievedChunk
from kb_arena.settings import settings
from kb_arena.strategies.bm25 import BM25Strategy

log = logging.getLogger(__name__)


QRELS_VERSION = 2


def _write_expected_chunks(
    path: Path, labels: dict[str, dict[str, int]], pool: dict | None = None
) -> None:
    """Atomically checkpoint graded labels so a later provider failure cannot erase progress.

    The file carries a version, the labels as chunk-to-grade per question,
    and a record of the pool the judge saw, so a reader knows which
    retrievers and how many random chunks fed the labels.
    """
    payload = {"version": QRELS_VERSION, "pool": pool or {}, "labels": labels}
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=True, default_flow_style=False))


JUDGE_PROMPT = """You are labeling retrieval ground truth for a documentation QA benchmark.

Given a QUESTION and CANDIDATE chunks, grade EVERY candidate you were shown.
A chunk counts as relevant when it holds information that helps answer the
question, including partial information, supporting context, and related
details. Grade a plausibly useful chunk as relevant. Grade an off-topic chunk
0. Never drop a candidate from your answer.

QUESTION: {question}

CANDIDATES:
{candidates}

OUTPUT FORMAT — strict:
Return ONLY a single JSON object literal that maps chunk_id to a grade. No prose,
no reasoning, no code fences. Grade 2: the chunk answers the question on its own.
Grade 1: the chunk supports the answer but does not give it. Grade 0: you read
the chunk and it is irrelevant. Grade every candidate you were shown, so a
later reader knows which chunks a judge rejected. If nothing is relevant,
grade every candidate 0.

Example, for three candidates:
{{"lambda-overview::pricing": 2, "ec2-overview::instance-types": 1, "s3-overview::lifecycle": 0}}"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
    return text.strip()


class JudgeParseError(RuntimeError):
    """The judge answered, and nothing in the answer parsed as grades.

    The call still cost money, so the error carries the cost. Without it a
    corpus whose every judgment fails reports a spend of zero and runs past
    the cost cap.
    """

    def __init__(self, message: str, cost_usd: float = 0.0) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd


def _parse_grades(text: str, valid: set[str], report: bool = False):
    """The judge's grades, and whether anything parsed at all when report is true.

    An object maps chunk to grade. A bare array is the old shape and every
    chunk becomes grade 1, which loses the grade-2 signal, so it is logged.
    """
    text = _strip_fences(text)
    decoder = json.JSONDecoder()
    for opener in ("{", "["):
        start = text.find(opener)
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find(opener, start + 1)
                continue
            if isinstance(parsed, dict):
                out: dict[str, int] = {}
                for cid, grade in parsed.items():
                    if str(cid) in valid and not isinstance(grade, bool) and grade in (0, 1, 2):
                        out[str(cid)] = int(grade)
                if not out:
                    # A wrapper key, a string grade, or a wrong chunk id all
                    # decode as JSON and grade nothing. Storing that empty
                    # label would skip the question on every later run, so the
                    # search moves on and the caller learns the judge failed.
                    log.warning("Judge returned a JSON object with no usable grade: %.120s", text)
                    start = text.find(opener, start + 1)
                    continue
                return (out, True) if report else out
            if isinstance(parsed, list):
                # The prompt asks for grades. A bare list is a model that
                # answered the old way, so the labels carry no grade-2
                # signal and the caller must know.
                log.warning(
                    "Judge returned a list, not grades. Every chunk gets grade 1: %.120s", text
                )
                listed = {str(c): 1 for c in parsed if str(c) in valid}
                return (listed, True) if report else listed
            start = text.find(opener, start + 1)
    return ({}, False) if report else {}


async def label_one_question(
    question_text: str,
    bm25: BM25Strategy,
    llm: LLMClient,
    corpus: str,
    n_candidates: int = 20,
    n_random: int = 10,
    extra_retrievers: list | None = None,
) -> tuple[dict[str, int], float]:
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
                # A retriever that fails would silently narrow the pool and
                # bias every label after it, so the run stops instead.
                raise RuntimeError(
                    f"Extra retriever {retriever.name} failed while building the label pool"
                ) from exc
            if extra_result.retrieval and extra_result.retrieval.retrieved:
                candidates.extend(extra_result.retrieval.retrieved)
            cost += float(getattr(extra_result, "cost_usd", 0.0) or 0.0)

    # A pool made only of what the retrievers rank high never shows the judge
    # a chunk they all missed. A random sample of the rest gives the labels
    # negatives and the odd hit outside every retriever's top-N.
    if n_random > 0:
        seen = {c.chunk_id for c in candidates}
        rest = [
            (cid, text)
            for cid, text in zip(bm25._chunk_ids, bm25._corpus_texts, strict=False)
            if cid not in seen
        ]
        rng = random.Random(f"{corpus}:{question_text}")
        for cid, text in rng.sample(rest, min(n_random, len(rest))):
            candidates.append(
                RetrievedChunk(
                    chunk_id=cid,
                    content=text,
                    doc_id=cid.split("::", 1)[0],
                    rank=len(candidates) + 1,
                    source_strategy="random",
                )
            )
    if not candidates:
        return {}, cost

    # RAPTOR emits `L1:doc::sec` and QnA emits `qna:...`. `ir_metrics` strips
    # those prefixes from a RETRIEVED id only, because a stored label carries
    # none. A prefixed label would be unreachable for bm25 and the vector
    # strategies, so every candidate id is stripped before the judge sees it.
    # Deduplicate on that canonical id, keeping the highest-ranked instance.
    by_id: dict[str, object] = {}
    for c in candidates:
        cid = canonical_chunk_id(c.chunk_id)
        existing = by_id.get(cid)
        if existing is None or c.rank < existing.rank:  # type: ignore[attr-defined]
            by_id[cid] = c
    deduped = list(by_id.values())

    candidates_text = "\n\n".join(
        f"[{canonical_chunk_id(c.chunk_id)}] {c.content[:400]}" for c in deduped
    )
    prompt = JUDGE_PROMPT.format(question=question_text, candidates=candidates_text)

    resp = await llm.extract(
        text=prompt,
        system_prompt=(
            "You output only a JSON object literal mapping chunk_id to a grade. "
            "No prose. No markdown."
        ),
    )
    cost += float(resp.cost_usd or 0.0)
    valid = {canonical_chunk_id(c.chunk_id) for c in deduped}
    grades, parsed_any = _parse_grades(resp.text, valid, report=True)
    if not parsed_any:
        # A truncated reply parses as nothing. Storing an empty label would
        # make force=False skip that question forever, so the caller learns
        # the judge failed and the question stays unlabeled.
        raise JudgeParseError(
            f"judge output did not parse as grades: {resp.text[:200]}", cost_usd=cost
        )
    return grades, cost


async def label_corpus(
    corpus: str, force: bool = False, n_candidates: int = 20, n_random: int = 10
) -> dict:
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
        from kb_arena.strategies.qna_pairs import QnAPairsStrategy
        from kb_arena.strategies.raptor import RaptorStrategy

        # Retrieval-only strategies join the pool, so labeling stays a
        # retrieval cost. A strategy that calls the LLM per query, such as
        # hybrid, pageindex, or qiss, would add a model call per question
        # per strategy and its own bias, so it stays out.
        pool_makers = [
            NaiveVectorStrategy,
            ContextualVectorStrategy,
            QnAPairsStrategy,
            RaptorStrategy,
        ]
        for cls in pool_makers:
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

    existing: dict[str, dict[str, int]] = {}
    if out_path.exists():
        try:
            loaded = yaml.safe_load(out_path.read_text())
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid expected chunks YAML: {out_path}") from exc
        existing, _ = load_qrels(loaded, out_path)

    cost_cap = settings.benchmark_cost_cap_usd
    total_cost = 0.0
    out_dict: dict[str, dict[str, int]] = dict(existing)
    pool_record = {
        "retrievers": ["bm25"] + [r.name for r in extra_retrievers],
        "n_candidates": n_candidates,
        "n_random": n_random,
    }
    skipped = 0
    labeled = 0
    unparsed = 0
    halted = False

    for q in questions:
        if q.id in existing and not force:
            skipped += 1
            continue
        if cost_cap > 0 and total_cost >= cost_cap:
            log.warning("Cost cap reached at $%.2f", total_cost)
            halted = True
            break
        try:
            grades, cost = await label_one_question(
                q.question,
                bm25,
                llm,
                corpus,
                n_candidates,
                n_random,
                extra_retrievers=extra_retrievers,
            )
        except JudgeParseError as exc:
            # The question stays unlabeled, so the next run tries it again
            # without --force. A stored empty label would be permanent. The
            # call still cost money, so the cap sees it.
            log.warning("Skipping %s: %s", q.id, exc)
            total_cost += getattr(exc, "cost_usd", 0.0)
            unparsed += 1
            continue
        total_cost += cost
        out_dict[q.id] = grades
        labeled += 1
        _write_expected_chunks(out_path, out_dict, pool_record)

    if not out_path.exists():
        _write_expected_chunks(out_path, out_dict, pool_record)
    return {
        "labeled": labeled,
        "skipped": skipped,
        "unparsed": unparsed,
        "cost_usd": total_cost,
        "path": str(out_path),
        "halted_by_cost_cap": halted,
        "total_questions": len(questions),
    }
