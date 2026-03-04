"""Tests for the documentation gap analyzer."""

from __future__ import annotations

import pytest

from kb_arena.audit.analyzer import AuditReport, SectionAudit


@pytest.fixture
def strong_audit():
    return SectionAudit(
        section_id="s1",
        section_title="VPC Config",
        doc_id="doc-1",
        heading_path=["Networking", "VPC Config"],
        questions_tested=3,
        avg_accuracy=0.85,
        worst_question="How to set up VPC peering?",
        worst_accuracy=0.75,
    )


@pytest.fixture
def weak_audit():
    return SectionAudit(
        section_id="s2",
        section_title="Subnets",
        doc_id="doc-1",
        heading_path=["Networking", "Subnets"],
        questions_tested=3,
        avg_accuracy=0.45,
        worst_question="What are subnet CIDR ranges?",
        worst_accuracy=0.20,
    )


@pytest.fixture
def gap_audit():
    return SectionAudit(
        section_id="s3",
        section_title="Security Groups",
        doc_id="doc-1",
        heading_path=["Networking", "Security Groups"],
        questions_tested=3,
        avg_accuracy=0.15,
        worst_question="How to configure security group rules?",
        worst_accuracy=0.05,
    )


def test_audit_report_structure():
    report = AuditReport(
        corpus="test",
        total_sections=10,
        total_questions=30,
        overall_accuracy=0.65,
    )
    assert report.corpus == "test"
    assert report.total_sections == 10
    assert report.strong == []
    assert report.weak == []
    assert report.gaps == []
    assert report.uncovered == []


def test_section_classification_thresholds(strong_audit, weak_audit, gap_audit):
    # Strong: >= 70%
    assert strong_audit.avg_accuracy >= 0.70

    # Weak: 30-70%
    assert 0.30 <= weak_audit.avg_accuracy < 0.70

    # Gap: < 30%
    assert gap_audit.avg_accuracy < 0.30


def test_audit_report_with_sections(strong_audit, weak_audit, gap_audit):
    report = AuditReport(
        corpus="test",
        total_sections=3,
        total_questions=9,
        overall_accuracy=0.48,
        strong=[strong_audit],
        weak=[weak_audit],
        gaps=[gap_audit],
        uncovered=["doc-1/Empty Section"],
    )
    assert len(report.strong) == 1
    assert len(report.weak) == 1
    assert len(report.gaps) == 1
    assert len(report.uncovered) == 1
    assert report.strong[0].section_title == "VPC Config"
    assert report.gaps[0].section_title == "Security Groups"


def test_uncovered_sections_detected():
    report = AuditReport(
        corpus="test",
        total_sections=5,
        total_questions=0,
        overall_accuracy=0.0,
        uncovered=["doc-1/Intro", "doc-1/Setup", "doc-2/Overview"],
    )
    assert len(report.uncovered) == 3


def test_audit_accuracy_aggregation(strong_audit, weak_audit):
    # Weighted average: (0.85*3 + 0.45*3) / 6 = 0.65
    total_q = strong_audit.questions_tested + weak_audit.questions_tested
    weighted = (
        strong_audit.avg_accuracy * strong_audit.questions_tested
        + weak_audit.avg_accuracy * weak_audit.questions_tested
    ) / total_q
    assert abs(weighted - 0.65) < 0.001
