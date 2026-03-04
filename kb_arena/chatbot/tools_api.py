"""API endpoints for documentation tools — Q&A generation, audit, fix."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from kb_arena.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tools", tags=["tools"])


class GenerateRequest(BaseModel):
    corpus: str


class AuditRequest(BaseModel):
    corpus: str
    max_sections: int = Field(default=50, ge=1, le=500)


class FixRequest(BaseModel):
    corpus: str
    max_sections: int = Field(default=50, ge=1, le=500)
    max_fixes: int = Field(default=10, ge=1, le=50)


@router.post("/generate")
async def generate_qa(body: GenerateRequest, request: Request) -> EventSourceResponse:
    """SSE stream Q&A pair generation for a corpus."""
    from kb_arena.generate.qna import generate_pairs_for_section
    from kb_arena.llm.client import LLMClient
    from kb_arena.strategies import load_documents

    async def event_generator() -> AsyncIterator[dict]:
        documents = load_documents(body.corpus)
        if not documents:
            msg = f"No documents found for corpus '{body.corpus}'"
            yield {"event": "error", "data": json.dumps({"message": msg})}
            return

        llm = LLMClient()

        # Count total sections
        all_sections = []
        for doc in documents:
            for section in doc.sections:
                if section.content.strip():
                    all_sections.append((section, doc.id))

        total = len(all_sections)
        yield {"event": "started", "data": json.dumps({"total_sections": total})}

        all_pairs: list[dict] = []
        for idx, (section, doc_id) in enumerate(all_sections):
            if await request.is_disconnected():
                return

            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "section_index": idx,
                        "total": total,
                        "doc_id": doc_id,
                        "section_title": section.title,
                    }
                ),
            }

            try:
                pairs = await generate_pairs_for_section(section, doc_id, llm)
                for pair in pairs:
                    yield {"event": "pair", "data": json.dumps(pair)}
                    all_pairs.append(pair)
            except Exception as exc:
                logger.warning("Failed to generate pairs for %s/%s: %s", doc_id, section.id, exc)

        # Write to JSONL
        output_dir = Path(settings.datasets_path) / body.corpus / "qa-pairs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "qa_pairs.jsonl"
        with open(output_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")

        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "total_pairs": len(all_pairs),
                    "output_path": str(output_path),
                }
            ),
        }

    return EventSourceResponse(event_generator())


@router.post("/audit")
async def run_audit_stream(body: AuditRequest, request: Request) -> EventSourceResponse:
    """SSE stream documentation audit."""
    from kb_arena.audit.analyzer import audit_sections_iter

    async def event_generator() -> AsyncIterator[dict]:
        started = False
        strong, weak, gaps = 0, 0, 0
        total_questions = 0

        async for idx, total, audit in audit_sections_iter(body.corpus, body.max_sections):
            if await request.is_disconnected():
                return

            if not started:
                yield {"event": "started", "data": json.dumps({"total_sections": total})}
                started = True

            # Classify
            if audit.questions_tested == 0:
                classification = "gap"
                gaps += 1
            elif audit.avg_accuracy >= 0.70:
                classification = "strong"
                strong += 1
            elif audit.avg_accuracy >= 0.30:
                classification = "weak"
                weak += 1
            else:
                classification = "gap"
                gaps += 1

            total_questions += audit.questions_tested

            yield {
                "event": "section_result",
                "data": json.dumps(
                    {
                        "section_index": idx,
                        "total": total,
                        "section_id": audit.section_id,
                        "section_title": audit.section_title,
                        "doc_id": audit.doc_id,
                        "heading_path": audit.heading_path,
                        "questions_tested": audit.questions_tested,
                        "avg_accuracy": round(audit.avg_accuracy, 4),
                        "worst_question": audit.worst_question,
                        "worst_accuracy": round(audit.worst_accuracy, 4),
                        "classification": classification,
                        "question_results": [
                            {
                                "question_text": r.question_text,
                                "accuracy": round(r.accuracy, 4),
                                "completeness": round(r.completeness, 4),
                                "answer_snippet": r.answer_snippet,
                            }
                            for r in audit.question_results
                        ],
                    }
                ),
            }

        if not started:
            msg = f"No sections found for corpus '{body.corpus}'"
            yield {"event": "error", "data": json.dumps({"message": msg})}
            return

        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "strong": strong,
                    "weak": weak,
                    "gaps": gaps,
                    "total_questions": total_questions,
                }
            ),
        }

    return EventSourceResponse(event_generator())


@router.post("/fix")
async def run_fix_stream(body: FixRequest, request: Request) -> EventSourceResponse:
    """SSE stream audit + fix pipeline (two-phase)."""
    from kb_arena.audit.analyzer import AuditReport, SectionAudit, audit_sections_iter
    from kb_arena.audit.fixer import generate_fixes_iter
    from kb_arena.llm.client import LLMClient
    from kb_arena.strategies import load_documents

    async def event_generator() -> AsyncIterator[dict]:
        yield {"event": "phase", "data": json.dumps({"phase": "audit"})}

        # Phase 1: Audit
        audits: list[SectionAudit] = []
        strong_list, weak_list, gap_list = [], [], []
        total_questions = 0

        async for idx, total, audit in audit_sections_iter(body.corpus, body.max_sections):
            if await request.is_disconnected():
                return

            audits.append(audit)

            if audit.questions_tested == 0:
                gap_list.append(audit)
            elif audit.avg_accuracy >= 0.70:
                strong_list.append(audit)
            elif audit.avg_accuracy >= 0.30:
                weak_list.append(audit)
            else:
                gap_list.append(audit)

            total_questions += audit.questions_tested

            yield {
                "event": "audit_progress",
                "data": json.dumps(
                    {
                        "section_index": idx,
                        "total": total,
                        "section_title": audit.section_title,
                    }
                ),
            }

        # Build audit report for fixer
        audit_report = AuditReport(
            corpus=body.corpus,
            total_sections=len(audits),
            total_questions=total_questions,
            overall_accuracy=(
                sum(a.avg_accuracy * a.questions_tested for a in audits if a.questions_tested > 0)
                / max(total_questions, 1)
            ),
            strong=strong_list,
            weak=weak_list,
            gaps=gap_list,
        )

        yield {
            "event": "audit_complete",
            "data": json.dumps(
                {
                    "strong": len(strong_list),
                    "weak": len(weak_list),
                    "gaps": len(gap_list),
                    "total_questions": total_questions,
                }
            ),
        }

        # Phase 2: Fix
        yield {"event": "phase", "data": json.dumps({"phase": "fix"})}

        documents = load_documents(body.corpus)
        llm = LLMClient()

        async for idx, total, rec in generate_fixes_iter(
            audit_report, documents, llm, body.max_fixes
        ):
            if await request.is_disconnected():
                return

            yield {
                "event": "fix_result",
                "data": json.dumps(
                    {
                        "fix_index": idx,
                        "total_fixes": total,
                        "priority": rec.priority,
                        "section_title": rec.section_title,
                        "doc_id": rec.doc_id,
                        "diagnosis": rec.diagnosis,
                        "suggested_content": rec.suggested_content,
                        "placement": rec.placement,
                        "estimated_impact": rec.estimated_impact,
                        "failing_questions": rec.failing_questions,
                        "current_accuracy": round(rec.current_accuracy, 4),
                    }
                ),
            }

        yield {"event": "complete", "data": json.dumps({"status": "done"})}

    return EventSourceResponse(event_generator())


@router.get("/qa-pairs")
async def get_qa_pairs(corpus: str) -> dict:
    """Read stored Q&A pairs for a corpus."""
    qa_path = Path(settings.datasets_path) / corpus / "qa-pairs" / "qa_pairs.jsonl"
    if not qa_path.exists():
        return {"pairs": [], "total": 0}

    pairs = []
    with open(qa_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return {"pairs": pairs, "total": len(pairs)}
