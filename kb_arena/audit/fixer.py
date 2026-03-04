"""Fix-my-docs — generate actionable recommendations with draft content."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from kb_arena.audit.analyzer import AuditReport, SectionAudit
from kb_arena.llm.client import LLMClient
from kb_arena.models.document import Document

log = logging.getLogger(__name__)

FIX_PROMPT = """You are a technical writing consultant. A documentation section scored poorly \
in automated quality evaluation.

Section title: {section_title}
Document: {doc_id}
Current accuracy: {accuracy}%

Section content (preview):
{content_preview}

Worst-performing question: {worst_question}

Analyze what's missing and provide a fix recommendation.

Return valid JSON:
{{
  "diagnosis": "What's missing or unclear (1-2 sentences)",
  "suggested_content": "Draft paragraph to add (100-200 words, technical, actionable)",
  "placement": "Where to add it (e.g., 'After the Configuration heading')",
  "estimated_impact": "Which question types would improve (e.g., 'How-To and Integration')"
}}

No explanations outside the JSON."""


@dataclass
class FixRecommendation:
    priority: int
    section_title: str
    doc_id: str
    diagnosis: str
    suggested_content: str
    placement: str
    estimated_impact: str
    failing_questions: list[str] = field(default_factory=list)
    current_accuracy: float = 0.0


@dataclass
class FixReport:
    corpus: str
    total_fixes: int
    recommendations: list[FixRecommendation] = field(default_factory=list)


def _find_section_content(audit: SectionAudit, documents: list[Document]) -> str:
    """Find the actual section content from documents."""
    for doc in documents:
        if doc.id != audit.doc_id:
            continue
        for section in doc.sections:
            if section.id == audit.section_id:
                return section.content[:1500]
    return "(section content not found)"


async def generate_fixes_iter(
    audit_report: AuditReport,
    documents: list[Document],
    llm: LLMClient,
    max_fixes: int = 10,
) -> AsyncGenerator[tuple[int, int, FixRecommendation], None]:
    """Yield (index, total, recommendation) per fix.

    Shared by CLI (collects all) and API (streams per event).
    """
    fixable: list[SectionAudit] = []
    fixable.extend(audit_report.gaps)
    fixable.extend(audit_report.weak)
    fixable.sort(key=lambda s: s.questions_tested, reverse=True)
    fixable = fixable[:max_fixes]
    total = len(fixable)

    for idx, audit in enumerate(fixable):
        content_preview = _find_section_content(audit, documents)

        prompt = FIX_PROMPT.format(
            section_title=audit.section_title,
            doc_id=audit.doc_id,
            accuracy=int(audit.avg_accuracy * 100),
            content_preview=content_preview,
            worst_question=audit.worst_question,
        )

        try:
            resp = await llm.extract(
                text=prompt,
                system_prompt="Return only valid JSON. No prose.",
            )

            json_match = re.search(r"\{[^}]+\}", resp.text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                failing_qs = [r.question_text for r in audit.question_results if r.accuracy < 0.5]
                rec = FixRecommendation(
                    priority=idx + 1,
                    section_title=audit.section_title,
                    doc_id=audit.doc_id,
                    diagnosis=parsed.get("diagnosis", "Unable to diagnose"),
                    suggested_content=parsed.get("suggested_content", ""),
                    placement=parsed.get("placement", "End of section"),
                    estimated_impact=parsed.get("estimated_impact", "General improvement"),
                    failing_questions=failing_qs,
                    current_accuracy=audit.avg_accuracy,
                )
                yield (idx, total, rec)
            else:
                log.warning("No JSON in fix response for %s/%s", audit.doc_id, audit.section_id)
        except Exception as exc:
            log.warning("Failed to generate fix for %s/%s: %s", audit.doc_id, audit.section_id, exc)
            continue


async def generate_fixes(
    audit_report: AuditReport,
    documents: list[Document],
    llm: LLMClient,
    max_fixes: int = 10,
) -> FixReport:
    """Generate fix recommendations. Thin wrapper around generate_fixes_iter."""
    recommendations: list[FixRecommendation] = []
    async for _idx, _total, rec in generate_fixes_iter(audit_report, documents, llm, max_fixes):
        recommendations.append(rec)

    return FixReport(
        corpus=audit_report.corpus,
        total_fixes=len(recommendations),
        recommendations=recommendations,
    )
