"""Tests for documentation tools — async generators and API endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from kb_arena.models.document import Document, Section

# ── helpers ──


def _make_section_audit(**kwargs):
    """Build a SectionAudit with sensible defaults."""
    from kb_arena.audit.analyzer import QuestionResult, SectionAudit

    defaults = dict(
        section_id="s1",
        section_title="Overview",
        doc_id="doc1",
        heading_path=["Overview"],
        questions_tested=2,
        avg_accuracy=0.75,
        worst_question="What is X?",
        worst_accuracy=0.6,
        question_results=[
            QuestionResult(
                question_text="What is X?",
                accuracy=0.6,
                completeness=0.7,
                answer_snippet="X is...",
            ),
            QuestionResult(
                question_text="How does Y work?",
                accuracy=0.9,
                completeness=0.8,
                answer_snippet="Y works by...",
            ),
        ],
    )
    defaults.update(kwargs)
    return SectionAudit(**defaults)


def _make_document(
    doc_id="doc1", section_id="s1", section_title="Overview", content="Some content here."
):
    return Document(
        id=doc_id,
        source="test",
        corpus="test",
        title="Test Guide",
        sections=[
            Section(
                id=section_id,
                title=section_title,
                content=content,
                heading_path=[section_title],
                level=2,
            )
        ],
        raw_token_count=100,
    )


# ── audit_sections_iter ──


async def test_audit_sections_iter_yields_tuples(sample_documents, mock_llm_client):
    from kb_arena.audit.analyzer import SectionAudit, audit_sections_iter

    mock_audit = _make_section_audit()

    with (
        patch("kb_arena.audit.analyzer.load_documents", return_value=sample_documents),
        patch("kb_arena.audit.analyzer.LLMClient", return_value=mock_llm_client),
        patch(
            "kb_arena.audit.analyzer._audit_section",
            new_callable=AsyncMock,
            return_value=mock_audit,
        ),
    ):
        results = []
        async for idx, total, audit in audit_sections_iter("test-corpus"):
            results.append((idx, total, audit))

    assert len(results) > 0
    for idx, total, audit in results:
        assert isinstance(idx, int)
        assert isinstance(total, int)
        assert isinstance(audit, SectionAudit)
        assert total > 0
        assert 0 <= idx < total


async def test_audit_sections_iter_empty_corpus():
    from kb_arena.audit.analyzer import audit_sections_iter

    with patch("kb_arena.audit.analyzer.load_documents", return_value=[]):
        results = []
        async for item in audit_sections_iter("empty-corpus"):
            results.append(item)

    assert results == []


async def test_audit_sections_iter_respects_max_sections(sample_documents, mock_llm_client):
    from kb_arena.audit.analyzer import audit_sections_iter

    mock_audit = _make_section_audit(questions_tested=1, avg_accuracy=0.5, worst_accuracy=0.5)

    with (
        patch("kb_arena.audit.analyzer.load_documents", return_value=sample_documents),
        patch("kb_arena.audit.analyzer.LLMClient", return_value=mock_llm_client),
        patch(
            "kb_arena.audit.analyzer._audit_section",
            new_callable=AsyncMock,
            return_value=mock_audit,
        ),
    ):
        results = []
        async for item in audit_sections_iter("test-corpus", max_sections=1):
            results.append(item)

    assert len(results) == 1
    # total reported should match the cap
    assert results[0][1] == 1


async def test_audit_sections_iter_index_sequence(sample_documents, mock_llm_client):
    """Indices emitted are sequential starting at 0."""
    from kb_arena.audit.analyzer import audit_sections_iter

    mock_audit = _make_section_audit()

    with (
        patch("kb_arena.audit.analyzer.load_documents", return_value=sample_documents),
        patch("kb_arena.audit.analyzer.LLMClient", return_value=mock_llm_client),
        patch(
            "kb_arena.audit.analyzer._audit_section",
            new_callable=AsyncMock,
            return_value=mock_audit,
        ),
    ):
        indices = [idx async for idx, _total, _audit in audit_sections_iter("test-corpus")]

    assert indices == list(range(len(indices)))


# ── run_audit wrapper ──


async def test_run_audit_returns_report(sample_documents, mock_llm_client):
    from kb_arena.audit.analyzer import AuditReport, run_audit

    mock_audit = _make_section_audit(avg_accuracy=0.8, worst_accuracy=0.7)

    with (
        patch("kb_arena.audit.analyzer.load_documents", return_value=sample_documents),
        patch("kb_arena.audit.analyzer.LLMClient", return_value=mock_llm_client),
        patch(
            "kb_arena.audit.analyzer._audit_section",
            new_callable=AsyncMock,
            return_value=mock_audit,
        ),
    ):
        report = await run_audit("test-corpus")

    assert isinstance(report, AuditReport)
    assert report.corpus == "test-corpus"
    assert report.total_sections > 0
    # avg_accuracy=0.8 → sections should land in strong bucket
    assert len(report.strong) > 0


async def test_run_audit_empty_corpus():
    from kb_arena.audit.analyzer import AuditReport, run_audit

    with patch("kb_arena.audit.analyzer.load_documents", return_value=[]):
        report = await run_audit("missing-corpus")

    assert isinstance(report, AuditReport)
    assert report.total_sections == 0
    assert report.total_questions == 0


async def test_run_audit_gap_sections(sample_documents, mock_llm_client):
    """Sections with very low accuracy land in gaps bucket."""
    from kb_arena.audit.analyzer import run_audit

    mock_audit = _make_section_audit(avg_accuracy=0.2, worst_accuracy=0.1)

    with (
        patch("kb_arena.audit.analyzer.load_documents", return_value=sample_documents),
        patch("kb_arena.audit.analyzer.LLMClient", return_value=mock_llm_client),
        patch(
            "kb_arena.audit.analyzer._audit_section",
            new_callable=AsyncMock,
            return_value=mock_audit,
        ),
    ):
        report = await run_audit("test-corpus")

    assert len(report.gaps) > 0


# ── generate_fixes_iter ──


async def test_generate_fixes_iter_yields_tuples(mock_llm_client):
    from kb_arena.audit.analyzer import AuditReport, QuestionResult, SectionAudit
    from kb_arena.audit.fixer import FixRecommendation, generate_fixes_iter

    weak_section = SectionAudit(
        section_id="s1",
        section_title="VPC Config",
        doc_id="doc1",
        heading_path=["VPC"],
        questions_tested=3,
        avg_accuracy=0.4,
        worst_question="How to configure VPC?",
        worst_accuracy=0.2,
        question_results=[
            QuestionResult(
                question_text="How to configure VPC?",
                accuracy=0.2,
                completeness=0.3,
                answer_snippet="VPC...",
            ),
        ],
    )
    audit_report = AuditReport(
        corpus="test",
        total_sections=5,
        total_questions=10,
        overall_accuracy=0.4,
        weak=[weak_section],
        gaps=[],
    )

    from kb_arena.llm.client import LLMResponse

    fix_json = json.dumps(
        {
            "diagnosis": "Missing VPC setup steps",
            "suggested_content": "To configure VPC...",
            "placement": "After overview",
            "estimated_impact": "How-To questions",
        }
    )
    mock_llm_client.extract.return_value = LLMResponse(
        text=fix_json,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    documents = [_make_document(doc_id="doc1", section_id="s1", section_title="VPC Config")]

    results = []
    async for idx, total, rec in generate_fixes_iter(
        audit_report, documents, mock_llm_client, max_fixes=5
    ):
        results.append((idx, total, rec))

    assert len(results) == 1
    idx, total, rec = results[0]
    assert isinstance(rec, FixRecommendation)
    assert rec.diagnosis == "Missing VPC setup steps"
    assert rec.priority == 1


async def test_generate_fixes_iter_empty_report(mock_llm_client):
    from kb_arena.audit.analyzer import AuditReport
    from kb_arena.audit.fixer import generate_fixes_iter

    audit_report = AuditReport(
        corpus="test",
        total_sections=5,
        total_questions=10,
        overall_accuracy=0.9,
        strong=[],
        weak=[],
        gaps=[],
    )

    results = []
    async for item in generate_fixes_iter(audit_report, [], mock_llm_client):
        results.append(item)

    assert results == []


async def test_generate_fixes_iter_respects_max_fixes(mock_llm_client):
    """Stops after max_fixes recommendations."""
    from kb_arena.audit.analyzer import AuditReport, QuestionResult, SectionAudit
    from kb_arena.audit.fixer import generate_fixes_iter

    weak_sections = [
        SectionAudit(
            section_id=f"s{i}",
            section_title=f"Section {i}",
            doc_id="doc1",
            heading_path=[f"Section {i}"],
            questions_tested=2,
            avg_accuracy=0.3,
            worst_question="Q?",
            worst_accuracy=0.1,
            question_results=[
                QuestionResult(
                    question_text="Q?", accuracy=0.1, completeness=0.2, answer_snippet="..."
                )
            ],
        )
        for i in range(5)
    ]
    audit_report = AuditReport(
        corpus="test",
        total_sections=5,
        total_questions=10,
        overall_accuracy=0.3,
        weak=weak_sections,
        gaps=[],
    )

    from kb_arena.llm.client import LLMResponse

    fix_json = json.dumps(
        {
            "diagnosis": "Thin coverage",
            "suggested_content": "Add more detail",
            "placement": "End",
            "estimated_impact": "How-To",
        }
    )
    mock_llm_client.extract.return_value = LLMResponse(
        text=fix_json,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    documents = [
        _make_document(doc_id="doc1", section_id=f"s{i}", section_title=f"Section {i}")
        for i in range(5)
    ]

    results = []
    async for item in generate_fixes_iter(audit_report, documents, mock_llm_client, max_fixes=2):
        results.append(item)

    assert len(results) == 2


# ── generate_fixes wrapper ──


async def test_generate_fixes_wrapper(mock_llm_client):
    from kb_arena.audit.analyzer import AuditReport, SectionAudit
    from kb_arena.audit.fixer import FixReport, generate_fixes

    audit_report = AuditReport(
        corpus="test",
        total_sections=1,
        total_questions=2,
        overall_accuracy=0.3,
        weak=[
            SectionAudit(
                section_id="s1",
                section_title="IAM",
                doc_id="doc1",
                heading_path=["IAM"],
                questions_tested=2,
                avg_accuracy=0.3,
                worst_question="How to set IAM?",
                worst_accuracy=0.1,
                question_results=[],
            )
        ],
        gaps=[],
    )

    from kb_arena.llm.client import LLMResponse

    fix_json = json.dumps(
        {
            "diagnosis": "Missing IAM steps",
            "suggested_content": "Configure IAM by...",
            "placement": "End of section",
            "estimated_impact": "How-To",
        }
    )
    mock_llm_client.extract.return_value = LLMResponse(
        text=fix_json,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )

    documents = [_make_document(doc_id="doc1", section_id="s1", section_title="IAM")]

    report = await generate_fixes(audit_report, documents, mock_llm_client)

    assert isinstance(report, FixReport)
    assert report.total_fixes == 1
    assert report.recommendations[0].diagnosis == "Missing IAM steps"


async def test_generate_fixes_empty(mock_llm_client):
    from kb_arena.audit.analyzer import AuditReport
    from kb_arena.audit.fixer import FixReport, generate_fixes

    audit_report = AuditReport(
        corpus="test",
        total_sections=0,
        total_questions=0,
        overall_accuracy=1.0,
        strong=[],
        weak=[],
        gaps=[],
    )

    report = await generate_fixes(audit_report, [], mock_llm_client)

    assert isinstance(report, FixReport)
    assert report.total_fixes == 0
    assert report.recommendations == []


# ── qa-pairs endpoint ──


async def test_get_qa_pairs_returns_stored():
    """GET /api/tools/qa-pairs returns stored JSONL pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qa_dir = Path(tmpdir) / "test-corpus" / "qa-pairs"
        qa_dir.mkdir(parents=True)
        pairs = [
            {
                "question": "What is Lambda?",
                "answer": "A serverless compute service.",
                "source_id": "doc1",
                "section_id": "s1",
            },
            {
                "question": "What is EC2?",
                "answer": "A virtual server.",
                "source_id": "doc1",
                "section_id": "s2",
            },
        ]
        (qa_dir / "qa_pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs) + "\n")

        with patch("kb_arena.chatbot.tools_api.settings") as mock_settings:
            mock_settings.datasets_path = tmpdir
            from kb_arena.chatbot.tools_api import get_qa_pairs

            result = await get_qa_pairs("test-corpus")

    assert result["total"] == 2
    assert result["pairs"][0]["question"] == "What is Lambda?"
    assert result["pairs"][1]["question"] == "What is EC2?"


async def test_get_qa_pairs_missing_corpus():
    """GET /api/tools/qa-pairs returns empty dict for unknown corpus."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("kb_arena.chatbot.tools_api.settings") as mock_settings:
            mock_settings.datasets_path = tmpdir
            from kb_arena.chatbot.tools_api import get_qa_pairs

            result = await get_qa_pairs("nonexistent")

    assert result["total"] == 0
    assert result["pairs"] == []


async def test_get_qa_pairs_empty_file():
    """Empty JSONL file returns zero pairs without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qa_dir = Path(tmpdir) / "test-corpus" / "qa-pairs"
        qa_dir.mkdir(parents=True)
        (qa_dir / "qa_pairs.jsonl").write_text("")

        with patch("kb_arena.chatbot.tools_api.settings") as mock_settings:
            mock_settings.datasets_path = tmpdir
            from kb_arena.chatbot.tools_api import get_qa_pairs

            result = await get_qa_pairs("test-corpus")

    assert result["total"] == 0
    assert result["pairs"] == []


# ── corpora endpoint includes hasQaPairs ──


async def test_corpora_includes_qa_pairs_info():
    """list_corpora() includes hasQaPairs and qaPairCount."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "test-corpus"
        (corpus_dir / "processed").mkdir(parents=True)
        (corpus_dir / "processed" / "docs.jsonl").write_text('{"id": "test"}\n')
        qa_dir = corpus_dir / "qa-pairs"
        qa_dir.mkdir()
        (qa_dir / "qa_pairs.jsonl").write_text('{"q": "test?"}\n{"q": "test2?"}\n')

        with (
            patch("kb_arena.chatbot.api.settings") as mock_settings,
        ):
            mock_settings.datasets_path = tmpdir
            mock_settings.results_path = str(Path(tmpdir) / "results")
            from kb_arena.chatbot.api import list_corpora

            result = await list_corpora()

    corpora = result["corpora"]
    assert len(corpora) == 1
    corpus = corpora[0]
    assert corpus["hasQaPairs"] is True
    assert corpus["qaPairCount"] == 2


async def test_corpora_no_qa_pairs():
    """list_corpora() reports hasQaPairs=False when no qa-pairs dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "test-corpus"
        (corpus_dir / "processed").mkdir(parents=True)
        (corpus_dir / "processed" / "docs.jsonl").write_text('{"id": "test"}\n')

        with (
            patch("kb_arena.chatbot.api.settings") as mock_settings,
        ):
            mock_settings.datasets_path = tmpdir
            mock_settings.results_path = str(Path(tmpdir) / "results")
            from kb_arena.chatbot.api import list_corpora

            result = await list_corpora()

    corpus = result["corpora"][0]
    assert corpus["hasQaPairs"] is False
    assert corpus["qaPairCount"] == 0


async def test_corpora_multiple_qa_pairs():
    """qaPairCount counts all non-blank lines in qa_pairs.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "docs-corpus"
        (corpus_dir / "processed").mkdir(parents=True)
        (corpus_dir / "processed" / "docs.jsonl").write_text('{"id": "d1"}\n')
        qa_dir = corpus_dir / "qa-pairs"
        qa_dir.mkdir()
        lines = [json.dumps({"q": f"Q{i}?"}) for i in range(10)]
        (qa_dir / "qa_pairs.jsonl").write_text("\n".join(lines) + "\n")

        with patch("kb_arena.chatbot.api.settings") as mock_settings:
            mock_settings.datasets_path = tmpdir
            mock_settings.results_path = str(Path(tmpdir) / "results")
            from kb_arena.chatbot.api import list_corpora

            result = await list_corpora()

    assert result["corpora"][0]["qaPairCount"] == 10


# ── corpus path traversal validation ──


class TestCorpusValidation:
    """Corpus name validation rejects path traversal characters."""

    def test_generate_rejects_traversal(self):
        import pytest

        from kb_arena.chatbot.tools_api import GenerateRequest

        with pytest.raises(Exception):
            GenerateRequest(corpus="../../etc")

    def test_audit_rejects_traversal(self):
        import pytest

        from kb_arena.chatbot.tools_api import AuditRequest

        with pytest.raises(Exception):
            AuditRequest(corpus="../secrets")

    def test_fix_rejects_slash(self):
        import pytest

        from kb_arena.chatbot.tools_api import FixRequest

        with pytest.raises(Exception):
            FixRequest(corpus="foo/bar")

    def test_chat_rejects_traversal(self):
        import pytest

        from kb_arena.models.api import ChatRequest

        with pytest.raises(Exception):
            ChatRequest(query="test", corpus="../../etc")

    def test_valid_corpus_accepted(self):
        from kb_arena.chatbot.tools_api import GenerateRequest

        req = GenerateRequest(corpus="aws-compute")
        assert req.corpus == "aws-compute"
