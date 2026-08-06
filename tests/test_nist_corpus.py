"""Validation for the NIST SP 800-171 Revision 3 evidence corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from kb_arena.models.document import Document

CORPUS = Path("datasets/nist-800-171-r3")
SOURCE = CORPUS / "raw/nist-sp-800-171r3.html"
SOURCE_SHA256 = "7ff0e0301a248f820b0779509bfbdd1369c5fb10592c4d8abc0d4b769bee0acf"


def _documents() -> list[Document]:
    path = CORPUS / "processed/documents.jsonl"
    return [Document.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _questions() -> list[dict]:
    questions: list[dict] = []
    for path in sorted((CORPUS / "questions").glob("*.yaml")):
        if path.name == "expected_chunks.yaml":
            continue
        questions.extend(yaml.safe_load(path.read_text()) or [])
    return questions


def test_nist_source_manifest_matches_the_snapshot():
    manifest = json.loads((CORPUS / "source-manifest.json").read_text())
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    assert manifest["title"] == "NIST SP 800-171 Revision 3"
    assert manifest["doi"] == "10.6028/NIST.SP.800-171r3"
    assert manifest["publication_date"] == "2024-05"
    assert manifest["retrieved_date"] == "2026-08-05"
    assert manifest["source_sha256"] == SOURCE_SHA256 == digest
    assert manifest["license"] == "Not subject to copyright in the United States"
    assert manifest["source_url"].startswith("https://nvlpubs.nist.gov/")


def test_nist_processed_documents_have_stable_control_provenance():
    documents = _documents()
    ids = [document.id for document in documents]

    assert len(documents) >= 90
    assert len(ids) == len(set(ids))
    assert all(document.corpus == "nist-800-171-r3" for document in documents)
    assert all(document.metadata["control_id"] for document in documents)
    assert all(document.metadata["source_anchor"].startswith("sec-sec_") for document in documents)
    assert all(document.sections for document in documents)
    assert all(document.sections[0].id == document.metadata["control_id"] for document in documents)


def test_nist_controls_exclude_the_document_bibliography():
    """The final control must stop at the bibliography instead of absorbing all 84 entries."""
    documents = _documents()
    contents = {
        document.id: "\n".join(section.content for section in document.sections)
        for document in documents
    }

    assert not any(re.search(r"^\[\d+\] ", text, re.MULTILINE) for text in contents.values())
    assert "Executive Order 13556" not in contents["nist-03.17.03"]

    counts = {document.id: document.raw_token_count for document in documents}
    largest = max(counts.values())
    assert counts["nist-03.17.03"] < 1000
    assert largest < 1000


def test_nist_source_controls_keep_control_enhancements():
    """AC-02(03) is a distinct source control and must not collapse to AC-02."""
    documents = _documents()
    by_id = {document.id: document for document in documents}
    account_management = by_id["nist-03.01.01"]

    assert "AC-02(03)" in account_management.metadata["source_controls"]
    assert "AC-02" in account_management.metadata["source_controls"]

    for document in documents:
        text = "\n".join(section.content for section in document.sections)
        for enhancement in set(re.findall(r"\b[A-Z]{2}-\d{2}\(\d{2}\)", text)):
            assert (
                enhancement in document.metadata["source_controls"]
            ), f"{document.id} dropped {enhancement}"


def test_nist_questions_match_the_approved_type_and_split_counts():
    questions = _questions()

    assert len(questions) == 80
    assert Counter(item["type"] for item in questions) == {
        "direct": 20,
        "paraphrased": 20,
        "scenario": 20,
        "boundary": 10,
        "multihop": 10,
    }
    assert Counter(item["split"] for item in questions) == {
        "development": 48,
        "validation": 12,
        "holdout": 20,
    }


def test_nist_question_loader_preserves_holdout_boundary(monkeypatch):
    from kb_arena.benchmark.questions import load_questions

    monkeypatch.setattr("kb_arena.benchmark.questions.settings.datasets_path", str(CORPUS.parent))

    questions = load_questions("nist-800-171-r3")
    holdout = load_questions("nist-800-171-r3", split="holdout")

    assert Counter(q.split for q in questions) == {
        "development": 48,
        "validation": 12,
        "holdout": 20,
    }
    assert len(holdout) == 20


def test_nist_questions_have_traceable_draft_reviews_and_no_duplicates():
    questions = _questions()
    normalized = [" ".join(item["question"].lower().split()) for item in questions]

    assert len(normalized) == len(set(normalized))
    assert all(item["review_status"] == "machine-assisted-draft" for item in questions)
    assert all(item["reviewed_by"] for item in questions)
    assert all(item["rationale"] for item in questions)
    assert all(item["source_anchors"] for item in questions)
    assert all(item["ground_truth"]["answer"] for item in questions)
    assert all(item["ground_truth"]["source_refs"] for item in questions)


def test_nist_qrels_are_nonempty_and_resolve_to_processed_sections():
    documents = _documents()
    qrels = yaml.safe_load((CORPUS / "questions/expected_chunks.yaml").read_text())
    question_ids = {item["id"] for item in _questions()}
    section_ids = {
        f"{document.id}::{section.id}" for document in documents for section in document.sections
    }

    assert set(qrels) == question_ids
    assert all(qrels[question_id] for question_id in question_ids)
    assert all(target in section_ids for targets in qrels.values() for target in targets)
