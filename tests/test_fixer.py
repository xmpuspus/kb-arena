"""Tests for the fix-my-docs recommendation engine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kb_arena.audit.analyzer import AuditReport, QuestionResult, SectionAudit
from kb_arena.audit.fixer import FixRecommendation, FixReport, generate_fixes
from kb_arena.models.document import Document, Section


@pytest.fixture
def weak_section():
    return SectionAudit(
        section_id="s1",
        section_title="VPC Peering",
        doc_id="networking-guide",
        heading_path=["Networking", "VPC Peering"],
        questions_tested=3,
        avg_accuracy=0.40,
        worst_question="How to configure VPC peering between Lambda and RDS?",
        worst_accuracy=0.15,
        question_results=[
            QuestionResult(
                question_text="How to configure VPC peering between Lambda and RDS?",
                accuracy=0.15,
                completeness=0.20,
                answer_snippet="VPC peering allows...",
            ),
            QuestionResult(
                question_text="What is VPC peering?",
                accuracy=0.65,
                completeness=0.70,
                answer_snippet="VPC peering is a networking connection...",
            ),
        ],
    )


@pytest.fixture
def gap_section():
    return SectionAudit(
        section_id="s2",
        section_title="Security Groups",
        doc_id="networking-guide",
        heading_path=["Networking", "Security Groups"],
        questions_tested=2,
        avg_accuracy=0.10,
        worst_question="How to configure inbound rules for RDS access?",
        worst_accuracy=0.05,
        question_results=[
            QuestionResult(
                question_text="How to configure inbound rules for RDS access?",
                accuracy=0.05,
                completeness=0.10,
                answer_snippet="Security groups act as...",
            ),
        ],
    )


@pytest.fixture
def documents():
    return [
        Document(
            id="networking-guide",
            title="Networking Guide",
            source="networking.md",
            corpus="test",
            sections=[
                Section(
                    id="s1",
                    title="VPC Peering",
                    content="VPC peering allows you to connect two VPCs. "
                    "Traffic between peered VPCs stays on the AWS backbone.",
                ),
                Section(
                    id="s2",
                    title="Security Groups",
                    content="Security groups act as virtual firewalls.",
                ),
            ],
        ),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.extract = AsyncMock(
        return_value=MagicMock(
            text=json.dumps(
                {
                    "diagnosis": "Missing VPC peering setup steps",
                    "suggested_content": "To configure VPC peering between Lambda and RDS, "
                    "first create a peering connection in the VPC console.",
                    "placement": "After the 'VPC Peering' heading",
                    "estimated_impact": "How-To and Integration questions would improve",
                }
            )
        )
    )
    return llm


def test_fix_report_structure():
    report = FixReport(corpus="test", total_fixes=0)
    assert report.corpus == "test"
    assert report.total_fixes == 0
    assert report.recommendations == []


def test_fix_recommendation_fields():
    rec = FixRecommendation(
        priority=1,
        section_title="VPC Peering",
        doc_id="networking-guide",
        diagnosis="Missing setup steps",
        suggested_content="Draft content here.",
        placement="After VPC Peering heading",
        estimated_impact="How-To questions",
        failing_questions=["How to configure VPC peering?"],
        current_accuracy=0.40,
    )
    assert rec.priority == 1
    assert rec.current_accuracy == 0.40
    assert len(rec.failing_questions) == 1


@pytest.mark.asyncio
async def test_fixes_sorted_by_priority(weak_section, gap_section, documents, mock_llm):
    report = AuditReport(
        corpus="test",
        total_sections=2,
        total_questions=5,
        overall_accuracy=0.25,
        weak=[weak_section],
        gaps=[gap_section],
    )
    fix_report = await generate_fixes(report, documents, mock_llm, max_fixes=10)
    assert fix_report.total_fixes == 2
    assert fix_report.recommendations[0].priority == 1
    assert fix_report.recommendations[1].priority == 2


@pytest.mark.asyncio
async def test_max_fixes_respected(weak_section, gap_section, documents, mock_llm):
    report = AuditReport(
        corpus="test",
        total_sections=2,
        total_questions=5,
        overall_accuracy=0.25,
        weak=[weak_section],
        gaps=[gap_section],
    )
    fix_report = await generate_fixes(report, documents, mock_llm, max_fixes=1)
    assert fix_report.total_fixes == 1


@pytest.mark.asyncio
async def test_fix_with_no_gaps(documents, mock_llm):
    report = AuditReport(
        corpus="test",
        total_sections=2,
        total_questions=5,
        overall_accuracy=0.90,
        strong=[],
        weak=[],
        gaps=[],
    )
    fix_report = await generate_fixes(report, documents, mock_llm)
    assert fix_report.total_fixes == 0
    assert fix_report.recommendations == []


@pytest.mark.asyncio
async def test_fix_handles_llm_error(weak_section, documents):
    llm = MagicMock()
    llm.extract = AsyncMock(side_effect=Exception("API error"))
    report = AuditReport(
        corpus="test",
        total_sections=1,
        total_questions=3,
        overall_accuracy=0.40,
        weak=[weak_section],
    )
    fix_report = await generate_fixes(report, documents, llm)
    assert fix_report.total_fixes == 0  # failed gracefully
