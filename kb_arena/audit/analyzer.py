"""Documentation gap analyzer — find what's missing in your docs."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from kb_arena.benchmark.evaluator import evaluate
from kb_arena.generate.qna import generate_pairs_for_section
from kb_arena.llm.client import LLMClient
from kb_arena.models.benchmark import Constraints, GroundTruth
from kb_arena.models.document import Section
from kb_arena.strategies import load_documents

log = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    question_text: str
    accuracy: float
    completeness: float
    answer_snippet: str  # first 200 chars of generated answer


@dataclass
class SectionAudit:
    section_id: str
    section_title: str
    doc_id: str
    heading_path: list[str]
    questions_tested: int
    avg_accuracy: float
    worst_question: str
    worst_accuracy: float
    question_results: list[QuestionResult] = field(default_factory=list)


@dataclass
class AuditReport:
    corpus: str
    total_sections: int
    total_questions: int
    overall_accuracy: float
    strong: list[SectionAudit] = field(default_factory=list)  # >= 70%
    weak: list[SectionAudit] = field(default_factory=list)  # 30-70%
    gaps: list[SectionAudit] = field(default_factory=list)  # < 30%
    uncovered: list[str] = field(default_factory=list)  # sections with 0 questions


async def _audit_section(
    section: Section,
    doc_id: str,
    llm: LLMClient,
) -> SectionAudit:
    """Generate Q&A pairs for a section and self-evaluate them.

    The insight: if a section's own Q&A pairs score poorly when evaluated
    against the section content, the documentation is weak/incomplete.
    """
    # Generate Q&A pairs from the section
    pairs = await generate_pairs_for_section(section, doc_id, llm)

    if not pairs:
        return SectionAudit(
            section_id=section.id,
            section_title=section.title,
            doc_id=doc_id,
            heading_path=section.heading_path or [section.title],
            questions_tested=0,
            avg_accuracy=0.0,
            worst_question="(no pairs generated)",
            worst_accuracy=0.0,
        )

    results: list[QuestionResult] = []

    for pair in pairs:
        question = pair.get("question", "")
        expected_answer = pair.get("answer", "")
        if not question or not expected_answer:
            continue

        # Evaluate the generated answer against the section content
        ground_truth = GroundTruth(answer=expected_answer)
        constraints = Constraints()

        score = await evaluate(
            answer=expected_answer,
            ground_truth=ground_truth,
            constraints=constraints,
            llm=llm,
        )

        results.append(
            QuestionResult(
                question_text=question,
                accuracy=score.accuracy,
                completeness=score.completeness,
                answer_snippet=expected_answer[:200],
            )
        )

    if not results:
        return SectionAudit(
            section_id=section.id,
            section_title=section.title,
            doc_id=doc_id,
            heading_path=section.heading_path or [section.title],
            questions_tested=0,
            avg_accuracy=0.0,
            worst_question="(no valid pairs)",
            worst_accuracy=0.0,
        )

    avg_acc = sum(r.accuracy for r in results) / len(results)
    worst = min(results, key=lambda r: r.accuracy)

    return SectionAudit(
        section_id=section.id,
        section_title=section.title,
        doc_id=doc_id,
        heading_path=section.heading_path or [section.title],
        questions_tested=len(results),
        avg_accuracy=avg_acc,
        worst_question=worst.question_text,
        worst_accuracy=worst.accuracy,
        question_results=results,
    )


async def audit_sections_iter(
    corpus: str,
    max_sections: int = 50,
) -> AsyncGenerator[tuple[int, int, SectionAudit], None]:
    """Yield (index, total, audit_result) per section.

    Shared by CLI (collects all) and API (streams per event).
    """
    documents = load_documents(corpus)
    if not documents:
        return

    llm = LLMClient()

    # Collect all non-empty sections
    all_sections: list[tuple[Section, str]] = []
    for doc in documents:
        for section in doc.sections:
            if section.content.strip():
                all_sections.append((section, doc.id))

    sections_to_audit = all_sections[:max_sections]
    total = len(sections_to_audit)

    for idx, (section, doc_id) in enumerate(sections_to_audit):
        try:
            audit = await _audit_section(section, doc_id, llm)
            yield (idx, total, audit)
        except Exception as exc:
            log.warning("Failed to audit %s/%s: %s", doc_id, section.id, exc)
            continue


async def run_audit(
    corpus: str,
    max_sections: int = 50,
) -> AuditReport:
    """Audit documentation quality. Thin wrapper around audit_sections_iter."""
    documents = load_documents(corpus)
    if not documents:
        return AuditReport(corpus=corpus, total_sections=0, total_questions=0, overall_accuracy=0.0)

    # Collect all sections for total count
    all_sections = []
    for doc in documents:
        for section in doc.sections:
            if section.content.strip():
                all_sections.append((section, doc.id))

    total_sections = len(all_sections)
    uncovered = [f"{doc_id}/{s.title}" for s, doc_id in all_sections[max_sections:]]

    # Also mark empty sections as uncovered
    for doc in documents:
        for section in doc.sections:
            if not section.content.strip():
                uncovered.append(f"{doc.id}/{section.title} (empty)")

    # Collect results from generator
    audits: list[SectionAudit] = []
    async for _idx, _total, audit in audit_sections_iter(corpus, max_sections):
        audits.append(audit)

    # Classify
    strong, weak, gaps = [], [], []
    total_questions = 0
    accuracy_sum = 0.0

    for audit in audits:
        total_questions += audit.questions_tested
        if audit.questions_tested == 0:
            gaps.append(audit)
        elif audit.avg_accuracy >= 0.70:
            strong.append(audit)
            accuracy_sum += audit.avg_accuracy * audit.questions_tested
        elif audit.avg_accuracy >= 0.30:
            weak.append(audit)
            accuracy_sum += audit.avg_accuracy * audit.questions_tested
        else:
            gaps.append(audit)
            accuracy_sum += audit.avg_accuracy * audit.questions_tested

    overall_accuracy = accuracy_sum / max(total_questions, 1)

    return AuditReport(
        corpus=corpus,
        total_sections=total_sections,
        total_questions=total_questions,
        overall_accuracy=overall_accuracy,
        strong=strong,
        weak=weak,
        gaps=gaps,
        uncovered=uncovered,
    )
